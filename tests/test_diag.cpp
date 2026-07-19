// =============================================================================
// tests/test_diag.cpp — device-resident reduction diagnostics (ADR-8).
// Host-serial (par → seq); no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "diag/diagnostics.hpp"

TEST_CASE("total_mass = Σ η·area for a uniform free surface") {
    const tc::Index nx = 6, ny = 4;
    const tc::Real  dx = 10.0, dy = 20.0, eta0 = 3.0;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);

    const tc::Field2 eta = s.eta;                          // hoist, capture by value
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) { eta[i, j] = eta0; });

    // analytic: η·(dx·dy)·(nx·ny)
    CHECK(tc::total_mass(s, m) == doctest::Approx(eta0 * dx * dy * nx * ny));
}

TEST_CASE("total_mass sums a non-uniform surface exactly (area-weighted)") {
    const tc::Index nx = 4, ny = 3;
    const tc::Real  dx = 2.0, dy = 2.0;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);

    // η[i,j] = i + j  →  Σ over cells, times cell area dx·dy.
    const tc::Field2 eta = s.eta;
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) { eta[i, j] = tc::Real(i + j); });

    tc::Real want = 0;
    for (tc::Index j = 0; j < ny; ++j)
        for (tc::Index i = 0; i < nx; ++i) want += tc::Real(i + j);
    want *= dx * dy;
    CHECK(tc::total_mass(s, m) == doctest::Approx(want));
}

TEST_CASE("any_nonfinite: clean state passes, a poked NaN is caught") {
    const tc::Index nx = 4, ny = 4;
    tc::CartesianMesh m(nx, ny, 1.0, 1.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);

    // zero all three staggered fields explicitly (don't lean on arena zero-init)
    const tc::Field2 eta = s.eta, u = s.u, v = s.v;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center),
                      [=](tc::Index i, tc::Index j) { eta[i, j] = 0; });
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace),
                      [=](tc::Index i, tc::Index j) { u[i, j] = 0; });
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace),
                      [=](tc::Index i, tc::Index j) { v[i, j] = 0; });
    CHECK(tc::any_nonfinite(s, m) == false);

    // poke a NaN into one u-face → the gate must fire
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace),
                      [=](tc::Index i, tc::Index j) { if (i == 1 && j == 2) u[i, j] = std::nan(""); });
    CHECK(tc::any_nonfinite(s, m) == true);
}
