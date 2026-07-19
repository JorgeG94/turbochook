#pragma once
// =============================================================================
// mesh/cartesian_mesh.hpp — the grid, as a compile-time policy.
//
// TWO new ideas here:
//
//  1. A C++20 CONCEPT (`Mesh`). A concept is a named, compile-time predicate over
//     a type: "does type M provide nx()/ny()/dx()/dy()/coriolis_at()?". It
//     replaces the old abstract-base-class-with-virtuals approach — but with ZERO
//     runtime cost, because the compiler checks it at compile time and inlines
//     the concrete calls. This is the heart of DESIGN's ADR-4 (compile-time
//     dispatch) and the prime directive (no virtuals in kernels).
//
//  2. `CartesianMesh` is a concrete MODEL of that concept. Structured geometry is
//     COMPUTED (dx is a stored scalar; f is an analytic beta-plane function) —
//     zero per-cell storage, everything inlines to plain arithmetic. A future
//     `TriMesh` would be a *sibling* model that READS connectivity tables. They
//     are independent models of one concept, NOT parent/child (DESIGN §7): making
//     structured a subclass of unstructured would force the fast path to carry
//     tables it doesn't need.
//
// For M2 we only build CartesianMesh. The `Mesh` concept EXISTS so the seam is
// there; we don't build the second backend (DESIGN §11 "leave the seam").
// =============================================================================

#include <concepts>
#include "core/types.hpp"

namespace tc {

// The concept: a Mesh must answer these queries. `-> std::convertible_to<Index>`
// constrains not just that the call is valid but what it returns. Read it as a
// contract the compiler enforces on any type claiming to be a Mesh.
template <class M>
concept Mesh = requires(const M m, Index j) {
    { m.nx() }            -> std::convertible_to<Index>;   // interior cells, x
    { m.ny() }            -> std::convertible_to<Index>;   // interior cells, y
    { m.dx() }            -> std::convertible_to<Real>;    // cell size, x  (metric)
    { m.dy() }            -> std::convertible_to<Real>;    // cell size, y
    { m.coriolis_at(j) }  -> std::convertible_to<Real>;    // f on a beta-plane, per row
};

// A uniform Cartesian beta-plane grid: f(y) = f0 + beta*(y - y0). Everything is a
// scalar or a closed-form function → nothing stored per cell, all inlined.
class CartesianMesh {
    Index nx_, ny_;
    Real  dx_, dy_;
    Real  f0_, beta_;         // Coriolis: f = f0 + beta*y  (beta=0 ⇒ f-plane)
public:
    CartesianMesh(Index nx, Index ny, Real dx, Real dy, Real f0 = 1.0e-4, Real beta = 0.0)
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), f0_(f0), beta_(beta) {}

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Real  dx() const { return dx_; }
    Real  dy() const { return dy_; }

    // f at cell-row j's y-coordinate. A real staggered scheme evaluates f at the
    // location the term lives (corner for PV) — refine when Coriolis is ported.
    Real coriolis_at(Index j) const { return f0_ + beta_ * (Real(j) * dy_); }
};

// Compile-time proof the class satisfies the concept. If you break CartesianMesh's
// interface, THIS line fails with a crisp message — not some deep template error
// at the first use site. A cheap, high-value habit.
static_assert(Mesh<CartesianMesh>);

} // namespace tc
