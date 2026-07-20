#pragma once
// =============================================================================
// bc/bc.hpp — boundary conditions as compile-time policies.
//
// Strategy (DESIGN decision #3): a BC fills the HALO/ghost rows of the state
// BEFORE each RK stage's RHS. Then the interior kernel is BRANCH-FREE — no
// per-cell `if (boundary)` inside the hot loop, which is exactly what you want on
// a GPU (branch divergence is a tax). `nghost` is a property of the scheme's
// reconstruction stencil (wider for PPM), so the halo is sized to fit it.
//
// Two M2 policies: WallBC (no-normal-flow reflection) and PeriodicBC (wrap). Each
// is a class satisfying the `BoundaryCondition` concept; the dispatch picks one.
// =============================================================================

#include <concepts>
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"

namespace tc {

template <class B>
concept BoundaryCondition =
    requires(B b, BaroState s, const CartesianMesh& mesh) {
        { b.fill_halos(s, mesh) };
    };

class WallBC {
public:
    template <Mesh M> void fill_halos(BaroState s, const M& m) const {
        // TODO(M2): reflect into ghost rows so ∂/∂n = 0 and normal face velocity
        // = 0 at walls (no-normal-flow). Runs before each stage → interior is
        // branch-free.
        (void)s; (void)m;
    }
};

class PeriodicBC {
public:
    template <Mesh M> void fill_halos(BaroState s, const M& m) const {
        // TODO(M2): copy the opposite interior edge into each ghost row (wrap).
        (void)s; (void)m;
    }
};

static_assert(BoundaryCondition<WallBC> && BoundaryCondition<PeriodicBC>);

} // namespace tc
