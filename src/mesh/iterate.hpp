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

// Interior x-faces (u-points with a centre cell on both sides): i ∈ [1, nx-1].
// The boundary faces i=0,nx are the BC's job (wall = no normal flow, M2).
template <Mesh M, class F>
void for_each_x_face(const M& mesh, F f) {
    const M m = mesh;                              // POD copy → capture by value, never `this`
    const Index nx = m.nx(), ny = m.ny();
    const Index nif = nx - 1;                       // interior u-faces per row
    if (nif <= 0) return;
    auto ids = std::views::iota(Index{0}, nif * ny);
    std::for_each(par, ids.begin(), ids.end(), [=](Index n) {
        const Index i = 1 + n % nif;                // i ∈ [1, nx-1]
        const Index j = n / nif;
        f(FaceView{ .i = i, .j = j, .li = i - 1, .lj = j, .ri = i, .rj = j,
                    .span = m.dx(Loc::XFace, i, j) });
    });
}

// Interior y-faces (v-points): j ∈ [1, ny-1].
template <Mesh M, class F>
void for_each_y_face(const M& mesh, F f) {
    const M m = mesh;
    const Index nx = m.nx(), ny = m.ny();
    const Index njf = ny - 1;                       // interior v-faces per column
    if (njf <= 0) return;
    auto ids = std::views::iota(Index{0}, nx * njf);
    std::for_each(par, ids.begin(), ids.end(), [=](Index n) {
        const Index i = n % nx;
        const Index j = 1 + n / nx;                 // j ∈ [1, ny-1]
        f(FaceView{ .i = i, .j = j, .li = i, .lj = j - 1, .ri = i, .rj = j,
                    .span = m.dy(Loc::YFace, i, j) });
    });
}

} // namespace tc
