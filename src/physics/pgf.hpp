#pragma once
// =============================================================================
// physics/pgf.hpp — the pressure-gradient force operator (policy slot).
//
//   ∂u/∂t += -g ∂η/∂x ,   ∂v/∂t += -g ∂η/∂y      (barotropic; on faces)
//
// `FvPgf` is the finite-volume barotropic gradient. Later slots on this axis:
// Montgomery, FV-Wright, gprime (2-layer reduced gravity, arrives M3). This
// operator needs no persistent workspace — init() is a no-op, kept so it still
// satisfies the module concept (uniform {init; compute} interface).
// =============================================================================

#include <concepts>
#include "core/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"

namespace tc {

template <class M>
concept PgfModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, p) };
    };

class FvPgf {
public:
    // No workspace needed, but the slot's contract is {init; compute}, so we keep
    // a trivial init. (Unused params cast to void to silence warnings.)
    void init(Arena& a, const CartesianMesh& m) { (void)a; (void)m; }

    void compute(BaroState s, BaroState k, Params p) const {
        // TODO(M2): -g ∂η/∂x onto u-faces, -g ∂η/∂y onto v-faces.
        //     for_each_face_x : k.u[i,j] += -p.g * (s.eta[i,j] - s.eta[i-1,j]) / p.dx
        //     for_each_face_y : k.v[i,j] += -p.g * (s.eta[i,j] - s.eta[i,j-1]) / p.dy
        // No `this` capture needed — this op reads/writes only the BaroState views
        // that arrive by value, so its kernels are already boundary-clean.
        (void)s; (void)k; (void)p;
    }
};

static_assert(PgfModule<FvPgf>);

} // namespace tc
