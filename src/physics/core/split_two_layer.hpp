#pragma once
// =============================================================================
// physics/core/split_two_layer.hpp — SPLIT-EXPLICIT two-layer core (M3.5, DESIGN ADR-9),
// the rakali run_stage_split flow. Per baroclinic step Δt:
//   1. entry barotropic state (η=Σₖhₖ, U=depth-mean); save U at entry (ubt_n).
//   2. slow momentum tendencies: per-layer Coriolis+adv (kcor) + 2-layer PGF (kpgf).
//   3. barotropic forcing: F_fast = depth-mean(kcor);  F_full = F_fast + depth-mean(kpgf).
//      (the subcycle owns −g∇η, so only the NON-PGF part forces it — no double-count.)
//   4. apply the full slow momentum to the layers (forward).
//   5. FB barotropic subcycle (M substeps of δt=Δt/M) forced by F_fast; keep the
//      uniform time-MEAN velocity (continuity) AND the END-step snapshot (momentum).
//   6. Δu = U_end − ubt_n − Δt·F_full injected into every layer (fast increment only).
//   7. thickness update: per-layer continuity with the MEAN-anchored velocity (dual
//      anchor, Hallberg); then Eulerian h-rescale to Σₖhₖ = η_end (mass consistency).
//
// The barotropic momentum reuses SadournyEnstrophy+FvPgf; continuity reuses
// ContinuityFlux — see BarotropicSolver. Oracle: a two-layer barotropic gravity wave
// through this stepper recovers √(g(H₁+H₂)).
// =============================================================================

#include <array>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/state/layered_state.hpp"
#include "physics/continuity/continuity.hpp"
#include "physics/momentum/coriolis.hpp"
#include "physics/momentum/pgf.hpp"
#include "physics/momentum/two_layer_pgf.hpp"
#include "physics/core/barotropic.hpp"
#include "bc/bc.hpp"
#include "numerics/parallel.hpp"
#include "numerics/split_explicit.hpp"

namespace tc {

template <Mesh Msh, ContinuityModule Cont, CoriolisModule Cor, class Pgf2L,
          BoundaryCondition Bc, int M = 30, class Outer = OuterSSPRK3>
class SplitTwoLayerCore {
    static constexpr int NL = 2;
    Msh    mesh_;
    Arena& arena_;
    Params p_{};
    int    M_ = M;

    Cont  cont_{};                                  // per-layer continuity (PPM: accurate thickness/tracers)
    Cor   cor_{};                                   // per-layer Coriolis + advection
    Pgf2L pgf_{};                                   // two-layer reduced-gravity PGF
    Bc    bc_{};
    // The barotropic mode is the FAST surface gravity wave, not a tracer — it needs no
    // PPM reconstruction (rakali uses a plain centred face thickness). Running PPM's
    // 5-cell window+limiter M×/step was the nsys hot path; cheap 1st-order Pcm on the
    // subcycle guts it while the LAYERS keep PPM. (The FB gravity wave is set by the
    // PGF↔continuity coupling, not advection, so upwind diffusion barely touches it.)
    BarotropicSolver<ContinuityFlux<Pcm>, Cor, FvPgf> bt_solver_{};

    LayeredState<NL> state_{};
    LayeredState<NL> s0_{};                         // saved s^n for the SSP-RK3 outer
    LayeredState<NL> kcor_{}, kpgf_{}, kh_{};       // slow-tendency / thickness scratch
    BaroState bt_{}, bt_mean_{}, bt_end_{}, ubt_n_{}, f_fast_{}, f_full_{}, dm_pgf_{};

    // small field-wise helpers (u,v only unless noted)
    void add_uv(BaroState dst, Real a, BaroState x, Real b, BaroState y) const {
        const Msh m = mesh_; const Field2 du = dst.u, xu = x.u, yu = y.u, dv = dst.v, xv = x.v, yv = y.v;
        for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { du[i, j] = a * xu[i, j] + b * yu[i, j]; });
        for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { dv[i, j] = a * xv[i, j] + b * yv[i, j]; });
    }
    void accum_uv(BaroState acc, BaroState x) const {   // acc += x
        const Msh m = mesh_; const Field2 au = acc.u, xu = x.u, av = acc.v, xv = x.v;
        for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { au[i, j] += xu[i, j]; });
        for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { av[i, j] += xv[i, j]; });
    }

public:
    // m_inner overrides the barotropic substep count at RUNTIME (default = template M);
    // nothing is sized by it, so sweeping dt/M needs no recompile.
    SplitTwoLayerCore(Msh mesh, Arena& a, Params p, int m_inner = M)
        : mesh_(mesh), arena_(a), p_(p), M_(m_inner) {}

    void init() {
        state_ = allocate_layered_state<NL>(arena_, mesh_);
        s0_   = allocate_layered_state<NL>(arena_, mesh_);
        kcor_ = allocate_layered_state<NL>(arena_, mesh_);
        kpgf_ = allocate_layered_state<NL>(arena_, mesh_);
        kh_   = allocate_layered_state<NL>(arena_, mesh_);
        for (BaroState* b : { &bt_, &bt_mean_, &bt_end_, &ubt_n_, &f_fast_, &f_full_, &dm_pgf_ })
            *b = allocate_baro_state(arena_, mesh_);
        cont_.init(arena_, mesh_); cor_.init(arena_, mesh_); pgf_.init(arena_, mesh_);
        bt_solver_.init(arena_, mesh_);
    }

    LayeredState<NL>& state() { return state_; }
    const Msh& mesh() const { return mesh_; }
    int n_inner() const { return M_; }

    // Outer time step: the `Outer` SSP policy (default SSP-RK3) wraps the split STAGE
    // Φ. Φ acts as one forward step of the slow coupled system (FE slow momentum + FB
    // barotropic subcycle + couple-back), so the outer scheme sets the SLOW modes'
    // stability — the internal gravity wave in particular. FwdEuler is unconditionally
    // unstable on it; SSP-RK2 (rakali) grows as (ωΔt)⁴ and wants some viscosity; SSP-RK3
    // is bounded for |ωΔt|<1.73. The barotropic subcycle re-runs once per stage (as in
    // the rakali RK driver); a convex SSP combination of mass-consistent states (each
    // Φ closes Σₖhₖ=η_end) stays mass-consistent.
    void step() {
        Outer::advance(state_, s0_, [this] { this->split_stage(); });
    }

    // One split STAGE Φ: advances state_ by p_.dt via the rakali run_stage_split flow.
    void split_stage() {
        const Msh  m  = mesh_;
        const Real dt = p_.dt, dtbt = dt / Real(M_);

        // 1. entry barotropic state; save entry depth-mean velocity
        derive_bt_from_layers<NL>(state_, bt_, m);
        axpby(ubt_n_, Real(1), bt_, Real(0), bt_);                 // ubt_n_ = bt_ (η, U at t^n)

        // 2. slow momentum tendencies
        zero_layered_state<NL>(kcor_, m);
        zero_layered_state<NL>(kpgf_, m);
        for (int l = 0; l < NL; ++l) cor_.compute(state_.layer[l], kcor_.layer[l], m, p_);
        pgf_.compute(state_, kpgf_, m, p_);

        // 3. barotropic forcing:  F_fast = <kcor>,  F_full = <kcor> + <kpgf>
        depth_mean_faces<NL>(kcor_, state_, f_fast_, m);
        depth_mean_faces<NL>(kpgf_, state_, dm_pgf_, m);
        add_uv(f_full_, Real(1), f_fast_, Real(1), dm_pgf_);

        // 4. apply the full slow momentum to the layers (forward Euler)
        for (int l = 0; l < NL; ++l) {
            const Field2 u = state_.layer[l].u, ku = kcor_.layer[l].u, pu = kpgf_.layer[l].u;
            const Field2 v = state_.layer[l].v, kv = kcor_.layer[l].v, pv = kpgf_.layer[l].v;
            for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { u[i, j] += dt * (ku[i, j] + pu[i, j]); });
            for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { v[i, j] += dt * (kv[i, j] + pv[i, j]); });
        }

        // 5. FB barotropic subcycle (from the entry state), keep mean + end
        add_uv(bt_mean_, Real(0), bt_, Real(0), bt_);              // zero the accumulators
        for (int mm = 0; mm < M_; ++mm) { bt_solver_.substep(bt_, f_fast_, m, p_, dtbt); accum_uv(bt_mean_, bt_); }
        axpby(bt_end_, Real(1), bt_, Real(0), bt_);                // end snapshot (η_end, U_end)
        add_uv(bt_mean_, Real(1) / Real(M_), bt_mean_, Real(0), bt_mean_);   // mean velocity

        // 6. Δu correction into every layer:  Δu = U_end − U_n − Δt·F_full
        for (int l = 0; l < NL; ++l) {
            const Field2 u = state_.layer[l].u, ue = bt_end_.u, un = ubt_n_.u, fu = f_full_.u;
            const Field2 v = state_.layer[l].v, ve = bt_end_.v, vn = ubt_n_.v, fv = f_full_.v;
            for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { u[i, j] += ue[i, j] - un[i, j] - dt * fu[i, j]; });
            for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { v[i, j] += ve[i, j] - vn[i, j] - dt * fv[i, j]; });
        }

        // 7. thickness: continuity with MEAN-anchored velocity (dual anchor), then rescale.
        //    Shift each layer velocity by (U_mean − U_end) for continuity, then shift back.
        for (int l = 0; l < NL; ++l) {
            const Field2 u = state_.layer[l].u, um = bt_mean_.u, ue = bt_end_.u;
            const Field2 v = state_.layer[l].v, vm = bt_mean_.v, ve = bt_end_.v;
            for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { u[i, j] += um[i, j] - ue[i, j]; });
            for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { v[i, j] += vm[i, j] - ve[i, j]; });
        }
        zero_layered_state<NL>(kh_, m);
        for (int l = 0; l < NL; ++l) cont_.compute(state_.layer[l], kh_.layer[l], m, p_);
        for (int l = 0; l < NL; ++l) {
            const Field2 h = state_.layer[l].eta, kh = kh_.layer[l].eta;
            const Field2 u = state_.layer[l].u, um = bt_mean_.u, ue = bt_end_.u;   // shift back to end-anchor
            const Field2 v = state_.layer[l].v, vm = bt_mean_.v, ve = bt_end_.v;
            for_each_cell(m.extent_x(Loc::Center), m.extent_y(Loc::Center), [=](Index i, Index j) { h[i, j] += dt * kh[i, j]; });
            for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace), [=](Index i, Index j) { u[i, j] -= um[i, j] - ue[i, j]; });
            for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace), [=](Index i, Index j) { v[i, j] -= vm[i, j] - ve[i, j]; });
        }
        // 8. h-rescale: Σₖ hₖ = η_end (mass consistency)
        {
            const Field2 h0 = state_.layer[0].eta, h1 = state_.layer[1].eta, ee = bt_end_.eta;
            for_each_cell(m.extent_x(Loc::Center), m.extent_y(Loc::Center), [=](Index i, Index j) {
                const Real tot = h0[i, j] + h1[i, j];
                const Real f = tot > Real(0) ? ee[i, j] / tot : Real(1);
                h0[i, j] *= f; h1[i, j] *= f;
            });
        }

        for (int l = 0; l < NL; ++l) bc_.fill_halos(state_.layer[l], m);
    }
};

// bc_inst split instantiation (M=30 barotropic substeps by default).
using SplitTwoLayerPoC =
    SplitTwoLayerCore<CartesianMesh, PpmContinuity, SadournyEnstrophy,
                      TwoLayerReducedGravityPgf, WallBC>;

} // namespace tc
