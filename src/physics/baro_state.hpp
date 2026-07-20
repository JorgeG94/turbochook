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
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// Clamp-to-edge centre access — repeats the boundary value when a stencil window
// overhangs the domain (the no-ghost interim's boundary handling, DESIGN #3; a real
// halo fill supersedes it). Shared by continuity + Coriolis. Metric-free.
inline Real clamp_at(Field2 h, Index i, Index j, Index nx, Index ny) {
    i = i < 0 ? 0 : (i >= nx ? nx - 1 : i);
    j = j < 0 ? 0 : (j >= ny ? ny - 1 : j);
    return h[i, j];
}

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
    Real  H1{}, H2{};    // two-layer rest thicknesses (M3)
    Real  rho1{}, rho2{};// two-layer densities  (reduced gravity g' = g(ρ2-ρ1)/ρ1)
};

// Allocate a full C-grid state from the Arena, each field on its staggered
// extent. One call, one owner. (Halo/ghost rows — decision #3 — get added once
// the reconstruction's `nghost` and the BC fill are wired; the M2 stub uses
// interior extents to keep the skeleton readable.)
template <Mesh M>
inline BaroState allocate_baro_state(Arena& a, const M& m) {
    return BaroState{
        .eta = a.alloc2d(m.extent_x(Loc::Center), m.extent_y(Loc::Center)),
        .u   = a.alloc2d(m.extent_x(Loc::XFace),  m.extent_y(Loc::XFace)),
        .v   = a.alloc2d(m.extent_x(Loc::YFace),  m.extent_y(Loc::YFace)),
    };
}

// Zero a state across every staggered field's full extent. baro_rhs calls this
// before the operator SUM, because each operator ACCUMULATES (+=) into k — without
// it, k holds the previous stage/step's tendencies and the RHS is silently wrong
// (the arena starts zero, so only the FIRST call would accidentally be correct).
template <Mesh M>
inline void zero_baro_state(BaroState k, const M& m) {
    const Field2 eta = k.eta, u = k.u, v = k.v;   // hoist views → capture [=], never `this`
    for_each_cell(m.extent_x(Loc::Center), m.extent_y(Loc::Center),
                  [=](Index i, Index j) { eta[i, j] = Real(0); });
    for_each_cell(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace),
                  [=](Index i, Index j) { u[i, j] = Real(0); });
    for_each_cell(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace),
                  [=](Index i, Index j) { v[i, j] = Real(0); });
}

// dst = a·x + b·y over one field's OWN extent (no mesh needed — the Field carries
// its extents). Elementwise, so aliasing (dst == x or y) is safe.
inline void combine_field(Field2 dst, Real a, Field2 x, Real b, Field2 y) {
    const Index nx = dst.extent(0), ny = dst.extent(1);
    for_each_cell(nx, ny, [=](Index i, Index j) { dst[i, j] = a * x[i, j] + b * y[i, j]; });
}

// The RK combine over a whole staggered state: dst = a·x + b·y per field. The one
// primitive the integrator needs (predictor, stage add, final average).
inline void axpby(BaroState dst, Real a, BaroState x, Real b, BaroState y) {
    combine_field(dst.eta, a, x.eta, b, y.eta);
    combine_field(dst.u,   a, x.u,   b, y.u);
    combine_field(dst.v,   a, x.v,   b, y.v);
}

} // namespace tc
