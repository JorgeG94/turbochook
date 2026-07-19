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
template <class M>
concept Mesh = requires(const M m, Index i, Index j, Loc loc, Edge e) {
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

} // namespace tc
