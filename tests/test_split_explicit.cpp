// =============================================================================
// tests/test_split_explicit.cpp — split-explicit building blocks (M3.5, ADR-9).
//   • bt_n_inner: CFL-derived substep count.
//   • derive_bt_from_layers: the layers→barotropic mode split — total thickness
//     Σₖhₖ and the TRANSPORT-WEIGHTED depth-mean velocity Σₖhₖuₖ/Σₖhₖ (exact).
// Host-serial; no doctest main (test_m0 owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/layered_state.hpp"
#include "numerics/split_explicit.hpp"

using tc::Real;
using tc::Index;

TEST_CASE("bt_n_inner: CFL-derived barotropic substep count") {
    // dt=6s already at the barotropic CFL (dx=4km, c_ext=140) → no subcycling
    CHECK(tc::bt_n_inner(6.0, 140.0, 4000.0, 4000.0) == 1);
    // a large split baroclinic step needs many fast substeps; monotonic in dt
    CHECK(tc::bt_n_inner(300.0, 140.0, 4000.0, 4000.0) > 10);
    CHECK(tc::bt_n_inner(600.0, 140.0, 4000.0, 4000.0) >
          tc::bt_n_inner(300.0, 140.0, 4000.0, 4000.0));
    // finer grid (smaller dx) ⇒ more substeps for the same dt
    CHECK(tc::bt_n_inner(300.0, 140.0, 2000.0, 2000.0) >
          tc::bt_n_inner(300.0, 140.0, 4000.0, 4000.0));
}

TEST_CASE("derive_bt_from_layers: total thickness + transport-weighted depth-mean") {
    const Index nx = 12, ny = 8;
    const Real H1 = 200.0, H2 = 800.0, u1 = 0.5, u2 = 0.1, v1 = -0.2, v2 = 0.3;
    tc::CartesianMesh mesh(nx, ny, 1000.0, 1000.0);
    tc::Arena arena(16u << 20);
    tc::LayeredState<2> s = tc::allocate_layered_state<2>(arena, mesh);
    tc::BaroState bt = tc::allocate_baro_state(arena, mesh);

    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { s.layer[0].eta[i, j] = H1; s.layer[1].eta[i, j] = H2; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace),
                      [=](Index i, Index j) { s.layer[0].u[i, j] = u1; s.layer[1].u[i, j] = u2; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace),
                      [=](Index i, Index j) { s.layer[0].v[i, j] = v1; s.layer[1].v[i, j] = v2; });

    tc::derive_bt_from_layers<2>(s, bt, mesh);

    const Real ubar = (H1 * u1 + H2 * u2) / (H1 + H2);      // = 0.18
    const Real vbar = (H1 * v1 + H2 * v2) / (H1 + H2);      // = (−40+240)/1000 = 0.20
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) CHECK(bt.eta[i, j] == doctest::Approx(H1 + H2));
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx; ++i) CHECK(bt.u[i, j] == doctest::Approx(ubar));   // interior faces
    for (Index j = 1; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) CHECK(bt.v[i, j] == doctest::Approx(vbar));
}
