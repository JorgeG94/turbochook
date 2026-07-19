#pragma once
// =============================================================================
// physics/baro_state.hpp — the Arakawa C-grid barotropic state + run params.
//
// This is the DATA the whole M2 solver pushes around. On a C-grid the pieces of
// state live on DIFFERENT, staggered sub-grids — that's the defining feature of
// the scheme (it's what makes geostrophic balance and wave propagation behave):
//
//        v(i,j+1)                     η : cell CENTRES   (nx   × ny  )
//     ┌─────┴─────┐                   u : x-FACES        (nx+1 × ny  )
//     │           │                   v : y-FACES        (nx   × ny+1)
//  u(i,j)  η(i,j)  u(i+1,j)           ζ/PV : CORNERS     (nx+1 × ny+1)  (workspace)
//     │           │
//     └─────┬─────┘
//        v(i,j)
//
// So instead of one co-located SystemView<N>, BaroState is a BUNDLE of Fields,
// each on its own extent. It is a plain POD of non-owning views ⇒ trivially
// copyable ⇒ it captures BY VALUE into a kernel (the owner is the Arena).
//
// (Co-located tracers S/T will ride a SystemView<N> at centres — that's M3.)
// =============================================================================

#include "core/types.hpp"
#include "core/arena.hpp"
#include "mesh/cartesian_mesh.hpp"

namespace tc {

// The prognostic barotropic state. Copying a BaroState copies three view handles
// (cheap); the doubles live in the Arena. RK stages are just more BaroStates
// (extra register sets) pointing at their own arena slices — see the Integrator.
struct BaroState {
    Field2 eta;   // sea-surface height / free-surface thickness, at centres
    Field2 u;     // x-velocity (or transport), on x-faces
    Field2 v;     // y-velocity (or transport), on y-faces
};

// The trivially-copyable scalar bundle that rides alongside the views into every
// kernel (DESIGN §6 "the boundary: Views + Params"). POD, by value, no owners.
// Geometry SHOULD flow in through the Mesh accessor, not raw dx/dy, so the Mesh
// seam exists (DESIGN §7) — dx/dy kept here for the M2 stub's convenience.
struct Params {
    Index nx, ny;      // interior cell counts
    Real  dx, dy;      // metrics (mirror the mesh; prefer the mesh accessor)
    Real  dt;          // time step
    Real  g;           // gravity
    Real  H;           // mean/reference depth (barotropic wave speed = sqrt(g·H))
};

// Allocate a full C-grid state from the Arena, each field on its staggered
// extent. One call, one owner. (Halo/ghost rows — decision #3 — get added once
// the reconstruction's `nghost` and the BC fill are wired; the M2 stub uses
// interior extents to keep the skeleton readable.)
inline BaroState allocate_baro_state(Arena& a, const CartesianMesh& m) {
    const Index nx = m.nx(), ny = m.ny();
    return BaroState{
        .eta = a.alloc2d(nx,     ny    ),
        .u   = a.alloc2d(nx + 1, ny    ),
        .v   = a.alloc2d(nx,     ny + 1),
    };
}

} // namespace tc
