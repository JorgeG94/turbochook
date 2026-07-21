#pragma once
// =============================================================================
// physics/lateral/dissipation.hpp — horizontal dissipation as a policy axis.
//
// Bleeds the enstrophy cascade at the grid scale so the eddy field stays clean.
// Two siblings on the axis:
//   ShapiroFilter — the M3.5 demo's inline 5-point smoother, factored out + tested.
//   LaplacianVisc — ∇·(A ∇u), with A from Leith (|∇ζ|) or biharmonic, for a
//                   SCALE-SELECTIVE closure that doesn't over-diffuse the broad jet.
//
// This is NUMERICAL dissipation (grid-scale hygiene), distinct from lateral_mix.hpp
// (physical mesoscale closures). STATUS's open quality issue: uniform Shapiro diffuses
// the deformation-scale jet ~10× too hard → the scale-selective sibling is the fix.
// Same {init; compute} shape as the other operators, applied in-place after the step.
//
// TODO(M3.5/M4): factor Shapiro out of demo_baroclinic (shapiro_center); add Leith.
// rakali north-star: src/parameterizations/lateral/structured/rki_ocean_horizontal_viscosity.F90,
//                    rki_ocean_varmix.F90 (Leith), rki_ml_horizontal_viscosity.F90
// =============================================================================

#include <concepts>
#include "lib/arena.hpp"
#include "physics/state/baro_state.hpp"
#include "numerics/parallel.hpp"

namespace tc {

template <class D>
concept Dissipation =
    requires(D d, Arena& a, const CartesianMesh& mesh, BaroState s, Params p) {
        { d.init(a, mesh) };
        { d.apply(s, mesh, p) };   // in-place smoothing / diffusion of the state
    };

class ShapiroFilter {
    Field2 tmp_{};   // out-of-place scratch (read old / write new)
public:
    template <Mesh M> void init(Arena& a, const M& m) {
        tmp_ = a.alloc2d(m.extent_x(Loc::Center), m.extent_y(Loc::Center));
    }
    template <Mesh M> void apply(BaroState s, const M& mesh, Params p) const {
        // TODO(M3.5): one topology-aware 5-point sweep of s.eta into tmp_ then back:
        //   f ← (1-ε)f + ¼ε·Σ neighbours (bc_at for wrap/clamp). Only h is filtered.
        (void)s; (void)mesh; (void)p; (void)tmp_;
    }
};

// TODO(M4): class LaplacianVisc { Field2 A_; ... };  // Leith / biharmonic viscosity.

static_assert(Dissipation<ShapiroFilter>);

} // namespace tc
