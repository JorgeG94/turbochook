#pragma once
// =============================================================================
// physics/barotropic.hpp — the 2D barotropic Forward-Backward Euler sub-solver
// (DESIGN ADR-9, ported from rakali_dc `rki_barotropic_substep.F90`). This is the
// FAST-MODE engine that the split-explicit stepper subcycles.
//
// One FB substep, in the stability-preserving order (η first — consumes uⁿ — then
// u,v with the JUST-UPDATED η in the pressure gradient):
//     η  ← η + δt·(−∇·(h u))                         [continuity, forward, uses uⁿ]
//     u  ← u + δt·((ζ+f)v − ∂KE/∂x − g∂η/∂x + Fᵤ)     [momentum, backward η]
//     v  ← v + δt·(−(ζ+f)u − ∂KE/∂y − g∂η/∂y + Fᵥ)
// Stable on the gravity-wave eigenmode for  c·δt·√(1/dx²+1/dy²) ≤ 1  (c=√(gH)).
//
// REUSE: the barotropic momentum IS our SadournyEnstrophy + FvPgf, and the
// free-surface continuity IS our ContinuityFlux — run on a 2D BaroState whose `eta`
// field carries the TOTAL column thickness H_ref+η. Then ∇·(h u) uses h = H_ref+η
// (free surface) and −g∇(eta) = −g∇η (flat bottom ⇒ ∇H_ref = 0). The only new code
// is the FB *ordering* + the constant slow forcing F — everything else is validated.
//
// `frc.u/frc.v` is the constant slow (baroclinic) forcing held across the subcycle
// (ADR-9 stage 2); pass a ZEROED BaroState for the pure gravity-wave / unforced case.
// =============================================================================

#include "core/types.hpp"
#include "lib/arena.hpp"
#include "physics/baro_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/pgf.hpp"
#include "numerics/parallel.hpp"

namespace tc {

template <ContinuityModule Cont, CoriolisModule Cor, PgfModule Pgf>
class BarotropicSolver {
    Cont      cont_{};
    Cor       cor_{};
    Pgf       pgf_{};
    BaroState k_{};                 // tendency scratch (arena-backed)

public:
    template <Mesh M> void init(Arena& a, const M& m) {
        cont_.init(a, m);
        cor_ .init(a, m);
        pgf_ .init(a, m);
        k_ = allocate_baro_state(a, m);
    }

    // One Forward-Backward Euler substep of size `dt`. `s.eta` holds the TOTAL column
    // thickness (H_ref+η); `frc` supplies the constant slow momentum forcing.
    template <Mesh M>
    void substep(BaroState s, BaroState frc, const M& mesh, Params p, Real dt) const {
        const M m = mesh;

        // ── A) continuity: η FIRST (consumes uⁿ) ────────────────────────────────────
        zero_baro_state(k_, mesh);
        cont_.compute(s, k_, mesh, p);                          // k.eta = −∇·(h uⁿ)
        {
            const Field2 e = s.eta, ke = k_.eta;
            for_each_cell(m.extent_x(Loc::Center), m.extent_y(Loc::Center),
                          [=](Index i, Index j) { e[i, j] += dt * ke[i, j]; });
        }

        // ── B) momentum: backward −g∇η (new η) + Sadourny (ζ+f, KE) + forcing ───────
        zero_baro_state(k_, mesh);
        pgf_.compute(s, k_, mesh, p);                           // −g∇η into k.u,k.v (η^{n+1})
        cor_.compute(s, k_, mesh, p);                           // (ζ+f)·⊥ − ∇KE into k.u,k.v
        {
            const Field2 u = s.u, ku = k_.u, fu = frc.u;
            for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace),
                          [=](Index i, Index j) { u[i, j] += dt * (ku[i, j] + fu[i, j]); });
        }
        {
            const Field2 v = s.v, kv = k_.v, fv = frc.v;
            for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace),
                          [=](Index i, Index j) { v[i, j] += dt * (kv[i, j] + fv[i, j]); });
        }
    }
};

} // namespace tc
