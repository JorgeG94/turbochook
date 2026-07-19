#pragma once
// =============================================================================
// mesh/cartesian_mesh.hpp — the M2 grid model: a uniform Cartesian beta-plane,
// DENSE (all-wet). This is ADR-7's "DenseMesh".
//
// It MODELS the `Mesh` concept (mesh.hpp). Structured geometry is COMPUTED — dx
// is a stored scalar, coordinates and `f` are closed-form — so there is zero
// per-cell storage and every accessor inlines to plain arithmetic. Masking is
// `constexpr 1` (the ×wet in an operator compiles away). A spherical / tripolar /
// masked / compact grid is a SIBLING model of the same concept (ADR-7 §7), NOT a
// subclass: making structured a subclass of unstructured would force the fast
// path to carry tables it doesn't need.
//
// The discipline that makes those siblings drop-in (ADR-7): operators read metrics
// through the LOCATION-aware accessors `dx(loc,i,j)`/`area(loc,i,j)` — never a bare
// scalar — so uniform→stretched→spherical is a model swap. The scalar `dx()`/`dy()`
// below are convenience for the `Params` scaffolding only.
// =============================================================================

#include <array>
#include "core/types.hpp"
#include "mesh/mesh.hpp"

namespace tc {

class CartesianMesh {
    Index nx_, ny_;
    Real  dx_, dy_;
    Real  f0_, beta_;                    // Coriolis: f = f0 + beta*y  (beta=0 ⇒ f-plane)
    std::array<EdgeConn, 4> edges_;      // topology, indexed by Edge

    // Ordinate of a location's (i,j): on a face LINE it is integer·spacing, at a
    // CENTRE it is (index+½)·spacing.
    Real xr(Loc l, Index i) const { return (x_staggered(l) ? Real(i) : Real(i) + Real(0.5)) * dx_; }
    Real yr(Loc l, Index j) const { return (y_staggered(l) ? Real(j) : Real(j) + Real(0.5)) * dy_; }

public:
    CartesianMesh(Index nx, Index ny, Real dx, Real dy,
                  Real f0 = 1.0e-4, Real beta = 0.0,
                  EdgeConn west = EdgeConn::Wall, EdgeConn east  = EdgeConn::Wall,
                  EdgeConn south = EdgeConn::Wall, EdgeConn north = EdgeConn::Wall)
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), f0_(f0), beta_(beta),
          edges_{west, east, south, north} {}

    Index nx() const { return nx_; }     // interior cell counts (loop bounds)
    Index ny() const { return ny_; }

    // Uniform-spacing convenience — feeds the Params scaffolding. The location-aware
    // dx(loc,i,j)/dy(loc,i,j) below are the real contract operators must use (ADR-7).
    Real dx() const { return dx_; }
    Real dy() const { return dy_; }

    // ── Mesh concept surface (all closed-form) ───────────────────────────────────
    Index extent_x(Loc l) const { return nx_ + (x_staggered(l) ? 1 : 0); }
    Index extent_y(Loc l) const { return ny_ + (y_staggered(l) ? 1 : 0); }

    Real x(Loc l, Index i, Index /*j*/) const { return xr(l, i); }   // Cartesian: x ⟂ j
    Real y(Loc l, Index /*i*/, Index j) const { return yr(l, j); }

    Real dx(Loc /*l*/, Index /*i*/, Index /*j*/) const { return dx_; }   // uniform
    Real dy(Loc /*l*/, Index /*i*/, Index /*j*/) const { return dy_; }
    Real area(Loc /*l*/, Index /*i*/, Index /*j*/) const { return dx_ * dy_; }

    // f at the LOCATION the term lives (PV wants Corner → integer j·dy).
    Real coriolis(Loc l, Index /*i*/, Index j) const { return f0_ + beta_ * yr(l, j); }

    // Masking trait: a dense mesh is all ocean → constexpr 1 (the ×wet compiles away).
    static constexpr Real wet(Loc, Index, Index) { return Real(1); }

    EdgeConn edge(Edge e) const { return edges_[static_cast<int>(e)]; }
};

// Compile-time proof the model satisfies the concept — break the interface and
// THIS fails crisply, not some deep template error at the first use site.
static_assert(Mesh<CartesianMesh>);

} // namespace tc
