#pragma once
// =============================================================================
// numerics/split_explicit.hpp — barotropic/baroclinic mode splitting (DESIGN ADR-9).
//
// WHY: unsplit, dt is pinned by the external gravity wave c_ext=√(g(H1+H2))≈140 m/s,
// but the eddies move at c_int≈2.7 m/s — we step ~50× finer than the slow dynamics
// need. Split-explicit subcycles the fast 2D barotropic mode so the expensive
// LAYERED update takes a ~20–50× larger dt. This is the north star.
//
// ALGORITHM: the MOM6/Hallberg split, ported from ../rakali_dc (battle-tested there;
// NOT the ROMS/Shchepetkin–McWilliams weighted scheme). Forward-Backward Euler
// subcycle; a PLAIN UNIFORM time-mean (no cosine/power weights, no over-integration);
// Hallberg DUAL ANCHORING — mean transport for continuity, end-step velocity for
// momentum. See ADR-9 for the full derivation and the rakali file map.
//
// WHY IT IS NOT a generic `Integrator` policy: SSP-RK3 treats the state as an opaque
// `State`. The split stepper must UNDERSTAND the layered structure — form the depth
// mean, vertically integrate the forcing, subcycle a 2D system, couple back into
// every layer. So it is a CORE-LEVEL stepper (a MultilayerCore policy) built from a
// reused 2D barotropic sub-solver, not a drop-in for `advance(state,scratch,rhs,bc)`.
//
// The barotropic momentum REUSES SadournyEnstrophy + FvPgf + ContinuityFlux on the
// 2D `(η, U=Σₖhₖuₖ, V=Σₖhₖvₖ)` state — that state is a 2D `BaroState`. The only new
// code is the FB substep loop, the depth-mean forcing, and the couple-back.
//
// STATUS: scaffold. Structure/interfaces/coupling settled (ADR-9 + rakali); kernel
// bodies are TODO(M3.5), each landing with its own test. The unsplit `SSPRK3` path
// stays selectable and is this stepper's ORACLE: the split must reproduce the
// unsplit eddy field minus fast transients (and collapse to it at M=1).
// =============================================================================

#include <algorithm>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/mesh.hpp"
#include "physics/state/baro_state.hpp"
#include "physics/state/layered_state.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// ── Mode split: layers → barotropic (rakali derive_bt_from_layers) ───────────────
// Total column thickness at centres  bt.η = Σₖ hₖ  (= H_ref + η), and the DEPTH-MEAN
// face velocity  bt.u = Σₖ (h_faceₖ·uₖ) / Σₖ h_faceₖ  (transport-weighted; h_faceₖ is
// the layer thickness averaged to the face). This is the state the barotropic
// subcycle starts from, and its inverse (the couple-back) reinjects the average.
template <int NL, Mesh M>
inline void derive_bt_from_layers(LayeredState<NL> s, BaroState bt, const M& mesh) {
    const M m = mesh;
    // centres: total thickness
    const Field2 bte = bt.eta;
    for_each_cell(m.extent_x(Loc::Center), m.extent_y(Loc::Center), [=](Index i, Index j) {
        Real h = 0;
        for (int l = 0; l < NL; ++l) h += s.layer[l].eta[i, j];
        bte[i, j] = h;
    });
    // x-faces: transport-weighted depth-mean of u (h_faceₖ = ½(hₖ[i-1]+hₖ[i]))
    const Field2 btu = bt.u;
    for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) {
        Real num = 0, den = 0;
        for (int l = 0; l < NL; ++l) {
            const Real hf = Real(0.5) * (bc_at(m, s.layer[l].eta, i - 1, j) + bc_at(m, s.layer[l].eta, i, j));
            num += hf * s.layer[l].u[i, j]; den += hf;
        }
        btu[i, j] = den > Real(0) ? num / den : Real(0);
    });
    // y-faces: mirror (h_faceₖ = ½(hₖ[j-1]+hₖ[j]))
    const Field2 btv = bt.v;
    for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) {
        Real num = 0, den = 0;
        for (int l = 0; l < NL; ++l) {
            const Real hf = Real(0.5) * (bc_at(m, s.layer[l].eta, i, j - 1) + bc_at(m, s.layer[l].eta, i, j));
            num += hf * s.layer[l].v[i, j]; den += hf;
        }
        btv[i, j] = den > Real(0) ? num / den : Real(0);
    });
}

// ── n_inner (M): CFL-derived barotropic substep count (rakali bt_auto_n_inner) ───
// δt_safe = cfl_safety·l_cfl/c_ext,  l_cfl = 1/√(1/dx²+1/dy²) (=dx/√2 uniform),
// c_ext = √(g·H_max);  M = ⌈Δt/δt_safe⌉.  Forward-Backward Euler is stable for
// c_ext·δt·√(1/dx²+1/dy²) ≤ 1; cfl_safety=0.65 keeps margin. Latch once at setup.
inline int bt_n_inner(Real dt_outer, Real c_ext, Real dx, Real dy, Real cfl_safety = Real(0.65)) {
    const Real l_cfl   = Real(1) / std::sqrt(Real(1) / (dx * dx) + Real(1) / (dy * dy));
    const Real dt_safe = cfl_safety * l_cfl / c_ext;
    return std::max(1, int(std::ceil(dt_outer / dt_safe)));
}

// ── Outer time-integration schemes over the split STAGE Φ (SSP, Shu–Osher form) ──
// Φ (a nullary callable) advances the state in place by one dt — the whole split
// stage (FE slow momentum + FB barotropic subcycle + couple-back). These wrap it in
// an SSP state-combination so the SLOW modes inherit the scheme's stability; `s`
// aliases the live state, `s0` is the saved-sⁿ scratch. On the internal-gravity-wave
// (imaginary) axis: FwdEuler is UNCONDITIONALLY unstable (grows as (ωΔt)²); SSP-RK2
// (Heun, rakali's outer) grows only as (ωΔt)⁴ — fine WITH real viscosity; SSP-RK3 is
// bounded for |ωΔt|<1.73 — stable without leaning on dissipation. A compile-time
// policy (Integrator-axis, ADR-9): swap the outer scheme with one template arg.
struct OuterFwdEuler {                                   // 1 stage — demonstrates the blow-up
    static constexpr int stages = 1;
    template <class State, class Stage> static void advance(State s, State s0, Stage stage) {
        (void)s; (void)s0; stage();
    }
};
struct OuterSSPRK2 {                                     // Heun — matches rakali
    static constexpr int stages = 2;
    template <class State, class Stage> static void advance(State s, State s0, Stage stage) {
        axpby(s0, Real(1), s, Real(0), s);               // s0 = sⁿ
        stage();                                          // u1 = Φ(sⁿ)
        stage();                                          // Φ(u1)
        axpby(s, Real(0.5), s0, Real(0.5), s);            // sⁿ⁺¹ = ½sⁿ + ½Φ(u1)
    }
};
struct OuterSSPRK3 {                                     // imaginary-axis-stable to |ωΔt|<1.73
    static constexpr int stages = 3;
    template <class State, class Stage> static void advance(State s, State s0, Stage stage) {
        axpby(s0, Real(1), s, Real(0), s);               // s0 = sⁿ
        stage(); stage();                                 // Φ(u1)
        axpby(s, Real(0.75), s0, Real(0.25), s);          // u2 = ¾sⁿ + ¼Φ(u1)
        stage();                                          // Φ(u2)
        axpby(s, Real(1) / 3, s0, Real(2) / 3, s);        // sⁿ⁺¹ = ⅓sⁿ + ⅔Φ(u2)
    }
};

// ── Thickness-weighted depth-mean of a per-layer FACE field (rakali face_depth_mean)
// out.u = Σₖ h_faceₖ·fld.uₖ / Σₖ h_faceₖ (h_face from `h`; fld supplies the u/v being
// averaged). out.eta untouched. Forms the barotropic forcing from the slow tendencies.
template <int NL, Mesh M>
inline void depth_mean_faces(LayeredState<NL> fld, LayeredState<NL> h, BaroState out, const M& mesh) {
    const M m = mesh;
    const Field2 ou = out.u;
    for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) {
        Real num = 0, den = 0;
        for (int l = 0; l < NL; ++l) {
            const Real hf = Real(0.5) * (bc_at(m, h.layer[l].eta, i - 1, j) + bc_at(m, h.layer[l].eta, i, j));
            num += hf * fld.layer[l].u[i, j]; den += hf;
        }
        ou[i, j] = den > Real(0) ? num / den : Real(0);
    });
    const Field2 ov = out.v;
    for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) {
        Real num = 0, den = 0;
        for (int l = 0; l < NL; ++l) {
            const Real hf = Real(0.5) * (bc_at(m, h.layer[l].eta, i, j - 1) + bc_at(m, h.layer[l].eta, i, j));
            num += hf * fld.layer[l].v[i, j]; den += hf;
        }
        ov[i, j] = den > Real(0) ? num / den : Real(0);
    });
}

// ── The split-explicit stepper (skeleton) ────────────────────────────────────────
// Parameterised by NL layers and the barotropic sub-solver `Baro` (the FB substep +
// the 2D operators over a BaroState). Owns arena-backed scratch: the live barotropic
// state, the running transport sum → time-mean (continuity leg), the end-step
// snapshot (momentum leg), the fast forcing F_bt_fast, and the entry depth-mean.
template <int NL, class Baro>
class SplitExplicit {
    int       M_ = 1;              // n_inner, latched at init (CFL-derived)
    BaroState bt_{};              // live barotropic (η, U, V) during the subcycle
    BaroState bt_sum_{};          // Σ over substeps → uniform time-mean (transports)
    BaroState bt_end_{};          // end-of-subcycle snapshot (η_end, U_end, V_end)
    BaroState bt_frc_{};          // F_bt_fast: depth-mean slow forcing − BT PGF projection
    BaroState ubt_at_n_{};        // entry depth-mean barotropic velocity (for Δu)

public:
    template <Mesh Msh>
    void init(Arena& a, const Msh& m, Real dt, Real c_ext) {
        M_       = bt_n_inner(dt, c_ext, m.dx(Loc::XFace, 0, 0), m.dy(Loc::YFace, 0, 0));
        bt_      = allocate_baro_state(a, m);
        bt_sum_  = allocate_baro_state(a, m);
        bt_end_  = allocate_baro_state(a, m);
        bt_frc_  = allocate_baro_state(a, m);
        ubt_at_n_ = allocate_baro_state(a, m);
    }

    int n_inner() const { return M_; }

    // One outer baroclinic step (the split lives INSIDE each outer RK2 stage in the
    // rakali driver; the caller supplies the layered slow-RHS, the barotropic ops,
    // and the BC). See ADR-9 for the three stages; kernels are TODO(M3.5).
    template <Mesh Msh, class SlowRhs, class BtOps, class BcOp>
    void step(LayeredState<NL> s, const Msh& mesh, Params p,
              SlowRhs slow_rhs, BtOps bt_ops, BcOp bc) {
        (void)s; (void)mesh; (void)p; (void)slow_rhs; (void)bt_ops; (void)bc; (void)M_;
        // ── Stage 1 — slow RHS + barotropic forcing ────────────────────────────────
        // TODO(M3.5): k = slow_rhs(s)  (per-layer baroclinic PGF, Coriolis+adv, visc, drag)
        //             F_bt   = face_depth_mean(k.momentum)          (Σ F·h_face / Σ h_face)
        //             bt_frc_ = F_bt − face_depth_mean(∇p)          (drop the BT PGF part —
        //                       else the substep double-counts it and √(gH)→√(2gH))
        //             derive bt_ from the layers: η=Σₖhₖ−H_ref, U=Σₖhₖuₖ/Σₖhₖ; save ubt_at_n_.
        //
        // ── Stage 2 — Forward-Backward Euler subcycle (the cheap fast part) ─────────
        // TODO(M3.5): zero bt_sum_; δt = p.dt/M_
        //             for m in [0,M_):
        //               bt_ops.eta_step(bt_, mesh, δt);                 // η first, uses Uⁿ
        //               bt_ops.uv_step (bt_, bt_frc_, mesh, δt);        // then U,V: Sadourny+KE
        //                                                               //   + backward −g∇η + F_bt_fast
        //               bt_sum_ += bt_ (accumulate transports)
        //             bt_end_ = bt_;                                    // end-step snapshot
        //             bt_sum_ *= 1/M_;                                  // uniform time-mean transport
        //
        // ── Stage 3 — couple the two anchors back into the layers ──────────────────
        // TODO(M3.5): (a) continuity: advance hₖ with the slow flux renormalised so
        //                 Σₖ hₖuₖ = bt_sum_ (time-mean)  ⇒  Σₖ hₖ = H + η_end (mass).
        //             (b) momentum: Δu = bt_end_.U − ubt_at_n_.U − p.dt·F_bt  into every layer.
        //             (c) Eulerian h-rescale to Σₖ hₖ = H + η_end; bc(s).
        //
        // ORACLE: M_=1 collapses to unsplit; M_≫1 reproduces the unsplit bc_inst eddies.
    }
};

} // namespace tc
