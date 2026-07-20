#pragma once
// =============================================================================
// mesh/mesh.hpp — the grid vocabulary + the `Mesh` concept (DESIGN ADR-7).
//
// ADR-7 separates the three things "grid" conflates, because different code
// consumes each:
//   • METRIC geometry (dx/dy/area, per staggering LOCATION) — consumed by the
//     operators, in flux form so Cartesian↔spherical is a model swap, not a
//     rewrite.
//   • TOPOLOGY (how boundaries connect: wall / periodic / tripolar fold) —
//     consumed by the halo fill, NOT the operators.
//   • MASK (wet/dry) — a non-wetting-drying core makes this a STATIC partition,
//     so it is an optional, compile-time mesh trait (`wet()`), invisible to
//     operators (a dense mesh returns constexpr 1 → the ×wet compiles away).
//
// The `Mesh` is a CONCEPT, not a base class; CartesianMesh / (future) spherical,
// tripolar, masked, compact are SIBLING models of it, never parent/child.
// =============================================================================

#include <concepts>
#include <type_traits>
#include "core/types.hpp"

namespace tc {

// Arakawa C-grid staggering locations (the C-grid T/U/V/Q points):
//   Center = η/tracers, XFace = u, YFace = v, Corner = ζ/PV.
enum class Loc { Center, XFace, YFace, Corner };

// A domain edge and how it connects — TOPOLOGY, consumed by the halo fill.
// `Fold` is the tripolar north fold (index-reversal + a vector sign-flip).
enum class Edge { West, East, South, North };
enum class EdgeConn { Wall, Periodic, Fold };

// How a field transforms across a fold: scalars keep sign, vector components
// flip. Carried by a field's halo spec once the fold BC lands; defined here so
// the whole grid vocabulary lives in one place.
enum class Parity { Scalar, Vector };

// Staggering predicates: a location is x-staggered if it sits on the x-face LINE
// (integer x = i·dx) rather than the x-CENTRE ((i+½)·dx). Corner is staggered in
// both axes; Center in neither. These drive extents and coordinates uniformly.
constexpr bool x_staggered(Loc l) { return l == Loc::XFace || l == Loc::Corner; }
constexpr bool y_staggered(Loc l) { return l == Loc::YFace || l == Loc::Corner; }

// The Mesh concept: geometry answered PER LOCATION so one contract serves
// centres, faces and corners. Metrics (dx/dy/area) feed operators; edge() feeds
// the halo fill; wet() is the masking trait. A missing accessor → not a Mesh
// (crisp error at the model's static_assert, not deep in a kernel).
//
// TRIVIALLY COPYABLE is a hard requirement, not a hope: a Mesh is captured BY
// VALUE into device kernels (for_each_face, diagnostics). A future masked/compact
// mesh that held an OWNING member (a std::vector mask, a connectivity table) would
// otherwise satisfy the shape, then hard-crash on the GPU with the exact
// capture-an-owner failure the design exists to avoid — and pass on the host.
// Masks/tables must be non-owning views (Field), so the mesh stays a POD.
template <class M>
concept Mesh = std::is_trivially_copyable_v<M> &&
    requires(const M m, Index i, Index j, Loc loc, Edge e) {
    { m.nx() }                -> std::convertible_to<Index>;  // interior cells, x
    { m.ny() }                -> std::convertible_to<Index>;  // interior cells, y
    { m.extent_x(loc) }       -> std::convertible_to<Index>;  // #points along x at loc
    { m.extent_y(loc) }       -> std::convertible_to<Index>;  // #points along y at loc
    { m.x(loc, i, j) }        -> std::convertible_to<Real>;   // physical x (metres / lon)
    { m.y(loc, i, j) }        -> std::convertible_to<Real>;   // physical y (metres / lat)
    { m.dx(loc, i, j) }       -> std::convertible_to<Real>;   // edge length, x (metres)
    { m.dy(loc, i, j) }       -> std::convertible_to<Real>;   // edge length, y (metres)
    { m.area(loc, i, j) }     -> std::convertible_to<Real>;   // control-volume area
    { m.coriolis(loc, i, j) } -> std::convertible_to<Real>;   // f where the term lives
    { m.wet(loc, i, j) }      -> std::convertible_to<Real>;   // 1 wet / 0 land (mask trait)
    { m.edge(e) }             -> std::convertible_to<EdgeConn>;// topology (halo fill)
};

// ── Topology-aware neighbour indexing (ADR-7: "the mesh owns iteration") ─────────
// An operator asks the mesh how to reach a neighbour instead of hardcoding i±1 and
// a wall clamp. A Periodic edge wraps (mod n); a Wall edge clamps-to-edge (exactly
// the old `clamp_at` — so wall runs are bit-identical). `s` is the signed offset;
// the returned index is always in [0, n). This is THE seam that makes periodic-x
// (the zonal-jet channel) a mesh property, not an operator branch. Fold is treated
// as a wall here until the tripolar halo lands (documented gap).
template <Mesh M> inline Index shift_x(const M& m, Index i, int s) {
    const Index n = m.nx(); const Index k = i + s;
    if (m.edge(Edge::West) == EdgeConn::Periodic) return (k % n + n) % n;
    return k < 0 ? 0 : (k >= n ? n - 1 : k);
}
template <Mesh M> inline Index shift_y(const M& m, Index j, int t) {
    const Index n = m.ny(); const Index k = j + t;
    if (m.edge(Edge::South) == EdgeConn::Periodic) return (k % n + n) % n;
    return k < 0 ? 0 : (k >= n ? n - 1 : k);
}
// Centre-field access honouring topology — supersedes clamp_at for h/tracer reads
// whose stencil window overhangs the domain. Face fields (u,v) are indexed with
// shift_x on their x-index directly; their staggered (wall) index keeps the stored
// boundary value (0 at a wall), which a clamp would wrongly replicate.
template <Mesh M> inline Real bc_at(const M& m, Field2 h, Index i, Index j) {
    return h[shift_x(m, i, 0), shift_y(m, j, 0)];
}

} // namespace tc
