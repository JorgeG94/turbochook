#pragma once
// =============================================================================
// numerics/split_explicit.hpp — barotropic/baroclinic mode splitting (DESIGN ADR-9).
//
// WHY: unsplit, dt is pinned by the external gravity wave c_ext=√(g(H1+H2))≈140 m/s,
// but the eddies move at c_int≈2.7 m/s — we step ~50× finer than the slow dynamics
// need. Split-explicit subcycles the fast 2D barotropic mode so the expensive
// LAYERED update takes a ~20–50× larger dt. This is the north star.
//
// WHY IT IS NOT a generic `Integrator` policy: SSP-RK3 treats the state as an
// opaque `State` and only needs axpby + an RHS. The split stepper must UNDERSTAND
// the layered structure — it forms the depth-mean transport, vertically integrates
// the momentum forcing, subcycles a 2D system, and couples the average back into
// every layer. So it is a CORE-LEVEL stepper (a MultilayerCore policy) built from a
// reused 2D barotropic sub-solver, not a drop-in for `advance(state,scratch,rhs,bc)`.
//
// The barotropic state is `η, U=Σₖhₖuₖ, V=Σₖhₖvₖ` — exactly a 2D `BaroState` reused.
//
// STATUS: scaffold. The stage structure, interfaces, and coupling math are settled
// here (ADR-9); the kernel bodies are TODO(M3.5), each landing with its own test.
// The unsplit `SSPRK3` path stays selectable and is this stepper's VALIDATION
// ORACLE: the split run must reproduce the unsplit eddy field minus fast transients.
// =============================================================================

#include <span>
#include <vector>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "physics/baro_state.hpp"
#include "physics/layered_state.hpp"

namespace tc {

// ── Barotropic time-averaging weights (Shchepetkin & McWilliams 2005) ────────────
// The barotropic subcycle runs PAST the baroclinic step (to ~1.2–2·M substeps) and
// the coupled-back (η̄,Ū,V̄) is a weighted average over it, shaped so the fast surface
// mode is filtered while the two moment constraints hold:
//     Σ wᵢ = 1            (consistency)         Σ (i/M) wᵢ = 1   (centred at t^{n+1})
// Getting these right is what makes the split STABLE (ADR-9 subtlety a). This fills
// `w` (length = number of substeps ≥ M) and returns the substep count actually used.
// TODO(M3.5): implement the SM2005 power shape  w(τ)=A τ^p(1-(τ/τ0)^q)-B τ  with A,B
// fixed by the two moments; for now a documented placeholder so callers compile.
inline int barotropic_weights(int M, std::span<Real> w) {
    // placeholder: flat over [0,M) — NOT stable, replace with SM2005 (test the moments).
    const int n = M;
    for (int i = 0; i < n; ++i) w[i] = Real(1) / Real(n);
    return n;
}

// ── The split-explicit stepper (skeleton) ────────────────────────────────────────
// Parameterised by NL layers, the barotropic sub-solver `Baro` (a 2D integrator over
// a BaroState reusing the M2 continuity/Coriolis/PGF), and the substep count M. It
// owns arena-backed scratch: the barotropic state + its running average + the
// per-step barotropic FORCING (the vertically integrated slow momentum tendency).
template <int NL, class Baro, int M = 40>
class SplitExplicit {
    BaroState        bt_{};        // barotropic (η, U, V) — the fast 2D mode
    BaroState        bt_avg_{};    // time-averaged (η̄, Ū, V̄) coupled back to the layers
    BaroState        bt_frc_{};    // barotropic forcing: ∫ (slow momentum tendency) dz
    std::vector<Real> w_{};        // averaging weights (host; small)

public:
    template <Mesh Msh>
    void init(Arena& a, const Msh& m) {
        bt_     = allocate_baro_state(a, m);
        bt_avg_ = allocate_baro_state(a, m);
        bt_frc_ = allocate_baro_state(a, m);
        w_.assign(2 * M, Real(0));
        // (void) until the stages land
        (void)barotropic_weights;
    }

    // One baroclinic step of size dt = p.dt. `slow_rhs` computes the LAYERED slow
    // tendencies (advection, Coriolis, baroclinic PGF, dissipation) EXCLUDING the
    // barotropic surface-pressure part (ADR-9 subtlety c); `bt_ops` advances the 2D
    // barotropic system one substep. `bc` fills halos.
    template <Mesh Msh, class SlowRhs, class BtOps, class BcOp>
    void step(LayeredState<NL> s, const Msh& mesh, Params p,
              SlowRhs slow_rhs, BtOps bt_ops, BcOp bc) {
        (void)s; (void)mesh; (void)p; (void)slow_rhs; (void)bt_ops; (void)bc;
        // ── Stage 1 — baroclinic RHS + barotropic forcing ──────────────────────────
        // TODO(M3.5): k = slow_rhs(s)   (per-layer slow tendencies)
        //             bt_frc_ = Σₖ hₖ · (k.layer[k] momentum)   — vertically integrate.
        //             Advance the layered momentum with k at the big dt (predictor).
        //
        // ── Stage 2 — barotropic subcycle (the cheap fast part) ────────────────────
        // TODO(M3.5): const int n = barotropic_weights(M, w_);
        //             bt_ = <depth-integrated s at t^n>;  zero bt_avg_
        //             for i in [0,n):  bt_ops.substep(bt_, bt_frc_, mesh, p.dt/M)
        //                              accumulate bt_avg_ += w_[i] · bt_   (η̄,Ū,V̄)
        //             This is the M2 barotropic solver reused; 3 fields, no reconstruction.
        //
        // ── Stage 3 — couple the average back into the layers ──────────────────────
        // TODO(M3.5): for each layer, REPLACE its depth-mean transport so that
        //             Σₖ hₖ uₖ^{n+1} = Ū (and V), i.e. uₖ += (Ū − Σₖ hₖ uₖ)/H  weighted;
        //             set the free surface from η̄, preserving Σₖ hₖ = H + η̄ (subtlety b).
        //             bc(s).
        //
        // ORACLE: with M=1 and flat weights this must collapse to the unsplit step;
        // with M≫1 it must reproduce the unsplit two-layer bc_inst eddy field.
    }
};

} // namespace tc
