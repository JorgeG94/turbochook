#pragma once
// =============================================================================
// physics/coriolis.hpp — the Coriolis operator (a compile-time policy slot).
//
//   ∂u/∂t += (f v)|u ,   ∂v/∂t += -(f u)|v      (plus the momentum advection)
//
// The C-grid ocean way is a POTENTIAL-VORTICITY-conserving flux form (Sadourny
// 1975), evaluated at cell CORNERS: q = (f + ζ)/h, then a PV-flux into the u/v
// tendencies. `SadournyEnstrophy` is the enstrophy-conserving variant. An
// energy-conserving sibling would be a second class filling the same slot —
// that's the whole point of the policy axis.
// =============================================================================

#include <concepts>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"

namespace tc {

template <class M>
concept CoriolisModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, p) };
    };

class SadournyEnstrophy {
    Field2 q_{};      // potential vorticity at corners — persistent workspace
public:
    void init(Arena& a, const CartesianMesh& m) {
        q_ = a.alloc2d(m.nx() + 1, m.ny() + 1);    // corners: (nx+1) × (ny+1)
    }

    void compute(BaroState s, BaroState k, Params p) const {
        Field2 q = q_;      // hoist member → local; capture [=], NEVER `this`
        // TODO(M2): implement the Sadourny enstrophy scheme. Sketch:
        //     for_each_corner : q[i,j] = (f + zeta(s,i,j)) / h_corner(s,i,j)
        //     for_each_face_x : k.u[i,j] += pv_flux_u(q, s, i, j)
        //     for_each_face_y : k.v[i,j] += pv_flux_v(q, s, i, j)
        (void)s; (void)k; (void)p; (void)q;
    }
};

static_assert(CoriolisModule<SadournyEnstrophy>);

} // namespace tc
