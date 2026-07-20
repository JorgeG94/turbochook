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
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "mesh/iterate.hpp"
#include "physics/baro_state.hpp"

namespace tc {

template <class M>
concept PgfModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, mesh, p) };
    };

class FvPgf {
public:
    // No workspace needed, but the slot's contract is {init; compute}, so we keep
    // a trivial init. (Unused params cast to void to silence warnings.)
    template <Mesh M> void init(Arena& a, const M& m) { (void)a; (void)m; }

    // ∂u/∂t += -g ∂η/∂x on u-faces, ∂v/∂t += -g ∂η/∂y on v-faces. A per-face write
    // (each face written once → no race). Neighbours + metric come from the mesh
    // via the FaceView seam (li/ri, span) — NEVER a hardcoded i-1 or a scalar dx
    // (ADR-7), so stretched/spherical drop in unchanged. Accumulates into k, so
    // baro_rhs must zero k before the operator sum.
    template <Mesh M> void compute(BaroState s, BaroState k, const M& mesh, Params p) const {
        const Field2 eta = s.eta;             // hoist views → capture [=], never `this`
        const Field2 ku = k.u, kv = k.v;
        const Real   g  = p.g;

        for_each_x_face(mesh, [=](FaceView f) {
            ku[f.i, f.j] += -g * (eta[f.ri, f.rj] - eta[f.li, f.lj]) / f.span;
        });
        for_each_y_face(mesh, [=](FaceView f) {
            kv[f.i, f.j] += -g * (eta[f.ri, f.rj] - eta[f.li, f.lj]) / f.span;
        });
    }
};

static_assert(PgfModule<FvPgf>);

} // namespace tc
