#pragma once
// =============================================================================
// mesh/masked_mesh.hpp — a Cartesian mesh with a land/sea mask (ADR-7 masking).
//
// Composition: CartesianMesh geometry + a cell-centred wet mask (1 ocean / 0 land),
// held as a NON-OWNING Field view into the arena — so the whole mesh stays
// trivially copyable and captures by value into kernels (the exact reason the Mesh
// concept requires trivial-copyability: a mask must be a view, never an owner).
//
// Geometry delegates to the base mesh; wet(loc) derives the face/corner masks from
// the cell mask (a face is wet iff BOTH the cells it separates are wet). The
// operators multiply their fluxes by wet(face), so no flux crosses a wet–dry face —
// the wet–dry boundary IS the coastline. A non-wetting-drying core ⇒ the mask is a
// static partition set once at init.
// =============================================================================

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "mesh/cartesian_mesh.hpp"

namespace tc {

class MaskedMesh {
    CartesianMesh geom_;
    Field2        wet_;                          // cell-centred wet mask (non-owning)

    Real wc(Index i, Index j) const {            // clamped cell wet
        const Index nx = geom_.nx(), ny = geom_.ny();
        i = i < 0 ? 0 : (i >= nx ? nx - 1 : i);
        j = j < 0 ? 0 : (j >= ny ? ny - 1 : j);
        return wet_[i, j];
    }

public:
    MaskedMesh(CartesianMesh geom, Field2 wet) : geom_(geom), wet_(wet) {}

    // ── geometry: delegate to the base Cartesian mesh ──
    Index nx() const { return geom_.nx(); }
    Index ny() const { return geom_.ny(); }
    Index extent_x(Loc l) const { return geom_.extent_x(l); }
    Index extent_y(Loc l) const { return geom_.extent_y(l); }
    Real x(Loc l, Index i, Index j) const { return geom_.x(l, i, j); }
    Real y(Loc l, Index i, Index j) const { return geom_.y(l, i, j); }
    Real dx(Loc l, Index i, Index j) const { return geom_.dx(l, i, j); }
    Real dy(Loc l, Index i, Index j) const { return geom_.dy(l, i, j); }
    Real area(Loc l, Index i, Index j) const { return geom_.area(l, i, j); }
    Real coriolis(Loc l, Index i, Index j) const { return geom_.coriolis(l, i, j); }
    EdgeConn edge(Edge e) const { return geom_.edge(e); }

    // ── the masking trait: face/corner wet = AND of the cells it touches ──
    Real wet(Loc l, Index i, Index j) const {
        switch (l) {
            case Loc::Center: return wc(i, j);
            case Loc::XFace:  return wc(i - 1, j) * wc(i, j);                        // cells (i-1,j)|(i,j)
            case Loc::YFace:  return wc(i, j - 1) * wc(i, j);                        // cells (i,j-1)|(i,j)
            case Loc::Corner: return wc(i - 1, j - 1) * wc(i, j - 1) * wc(i - 1, j) * wc(i, j);
        }
        return Real(1);
    }
};

static_assert(Mesh<MaskedMesh>);   // incl. trivially copyable (mask is a view)

} // namespace tc
