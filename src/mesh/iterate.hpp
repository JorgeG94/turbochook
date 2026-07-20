#pragma once
// =============================================================================
// mesh/iterate.hpp — the mesh-owned face-iteration seam (DESIGN ADR-7).
//
// "The mesh owns iteration." An operator never writes a raw (i,j) loop with a
// hardcoded i±1 neighbour; it asks the mesh to walk its faces and hands it a
// `FaceView` per face. For a structured Cartesian mesh the connectivity is
// computed (li = i-1 …); for a tripolar/unstructured sibling it would come from
// the fold map / a table — and the OPERATOR does not change. That is what makes
// spherical/tripolar/masked drop-in.
//
// TWO patterns share this file:
//   • per-FACE write (PGF, face-state reconstruction): each face is written by
//     exactly one thread → no race → a plain face loop (here, now).
//   • flux-DIVERGENCE (continuity): a face flux updates BOTH its cells, so the
//     safe shape is per-cell-gather (loop cells, gather their faces). That lands
//     with continuity — its first consumer — so we shape it against real use.
// =============================================================================

#include <execution>
#include <algorithm>
#include <ranges>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// A face between two centre cells: the connectivity + metric a face-local operator
// needs. The mesh FILLS it, so operators reference neighbours via li/ri — never a
// hardcoded offset — and read the metric via `span`, never a scalar dx (ADR-7).
struct FaceView {
    Index i,  j;      // the face's own index (on its staggered grid) — the write target
    Index li, lj;     // minus-side centre cell
    Index ri, rj;     // plus-side  centre cell
    Real  span;       // centre-to-centre distance (the gradient denominator)
};

// x-faces (u-points). Range and connectivity are TOPOLOGY-driven (ADR-7):
//   • wall-x     → interior faces i ∈ [1, nx-1]; the wall faces 0,nx are the BC's
//     job (no normal flow), left untouched → stay 0.
//   • periodic-x → faces i ∈ [0, nx], with wrapped li/ri (shift_x). Faces 0 and nx
//     are the SAME wrap face and get identical li=nx-1, ri=0 → whatever an operator
//     writes to u[0] and u[nx] matches, so the duplicated storage stays in sync
//     with no explicit halo copy.
template <Mesh M, class F>
void for_each_x_face(const M& mesh, F f) {
    const M m = mesh;                              // POD copy → capture by value, never `this`
    const Index nx = m.nx(), ny = m.ny();
    const bool per = (m.edge(Edge::West) == EdgeConn::Periodic);
    const Index i0 = per ? 0 : 1, i1 = per ? nx : nx - 1;   // inclusive face range
    const Index nif = i1 - i0 + 1;
    if (nif <= 0) return;
    auto ids = std::views::iota(Index{0}, nif * ny);
    std::for_each(par, ids.begin(), ids.end(), [=](Index n) {
        const Index i = i0 + n % nif;
        const Index j = n / nif;
        f(FaceView{ .i = i, .j = j,
                    .li = shift_x(m, i - 1, 0), .lj = j,
                    .ri = shift_x(m, i, 0),     .rj = j,
                    .span = m.dx(Loc::XFace, i, j) });
    });
}

// y-faces (v-points), the meridional mirror: wall-y → j ∈ [1, ny-1]; periodic-y →
// j ∈ [0, ny] with wrapped lj/rj.
template <Mesh M, class F>
void for_each_y_face(const M& mesh, F f) {
    const M m = mesh;
    const Index nx = m.nx(), ny = m.ny();
    const bool per = (m.edge(Edge::South) == EdgeConn::Periodic);
    const Index j0 = per ? 0 : 1, j1 = per ? ny : ny - 1;
    const Index njf = j1 - j0 + 1;
    if (njf <= 0) return;
    auto ids = std::views::iota(Index{0}, nx * njf);
    std::for_each(par, ids.begin(), ids.end(), [=](Index n) {
        const Index i = n % nx;
        const Index j = j0 + n / nx;
        f(FaceView{ .i = i, .j = j,
                    .li = i, .lj = shift_y(m, j - 1, 0),
                    .ri = i, .rj = shift_y(m, j, 0),
                    .span = m.dy(Loc::YFace, i, j) });
    });
}

} // namespace tc
