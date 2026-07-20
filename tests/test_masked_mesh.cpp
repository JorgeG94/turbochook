// =============================================================================
// tests/test_masked_mesh.cpp — land/sea masking (ADR-7). Oracles: face wet = AND of
// its cells; no mass crosses a coast; land cells stay frozen; mass conserved over
// the wet region; lake-at-rest is zero on a masked (circular) domain.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "mesh/masked_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/pgf.hpp"

using tc::Real;
using tc::Index;

TEST_CASE("MaskedMesh: face wet = AND of adjacent cells") {
    const Index nx = 6, ny = 4;
    tc::CartesianMesh geom(nx, ny, 100.0, 100.0);
    tc::Arena arena(1u << 20);
    tc::Field2 mask = arena.alloc2d(nx, ny);
    tc::for_each_cell(nx, ny, [=](Index i, Index j) { mask[i, j] = i < nx / 2 ? Real(1) : Real(0); });
    tc::MaskedMesh m(geom, mask);

    CHECK(m.wet(tc::Loc::Center, 1, 1) == doctest::Approx(1.0));   // ocean cell
    CHECK(m.wet(tc::Loc::Center, 4, 1) == doctest::Approx(0.0));   // land cell
    CHECK(m.wet(tc::Loc::XFace, 3, 1) == doctest::Approx(0.0));    // coast face (cell2 ocean | cell3 land)
    CHECK(m.wet(tc::Loc::XFace, 2, 1) == doctest::Approx(1.0));    // interior ocean face
}

TEST_CASE("Masking: land frozen, no flux across the coast, mass conserved over ocean") {
    const Index nx = 16, ny = 10;
    tc::CartesianMesh geom(nx, ny, 100.0, 200.0);
    tc::Arena arena(4u << 20);
    tc::Field2 mask = arena.alloc2d(nx, ny);
    tc::for_each_cell(nx, ny, [=](Index i, Index j) { mask[i, j] = i < nx / 2 ? Real(1) : Real(0); });
    tc::MaskedMesh m(geom, mask);

    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);

    const tc::Field2 eta = s.eta, u = s.u, v = s.v;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center), [=](Index i, Index j) { eta[i, j] = Real(1000); });
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = Real(0.3); });
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = Real(0); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 200, .dt = 1, .g = 9.81, .H = 1000 };
    cont.compute(s, k, m, p);

    const tc::Field2 ke = k.eta;
    for (Index j = 0; j < ny; ++j)
        for (Index i = nx / 2; i < nx; ++i) CHECK(ke[i, j] == doctest::Approx(0.0));  // land frozen

    Real net = 0, activity = 0;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) {
            const Real r = m.area(tc::Loc::Center, i, j) * ke[i, j];
            net += r; activity += std::abs(r);
        }
    CHECK(activity > 0.0);
    CHECK(std::abs(net) / activity < 1e-12);          // no leak across coast → conserved
}

TEST_CASE("Masking: lake-at-rest on a circular basin ⇒ zero RHS") {
    const Index nx = 14, ny = 14;
    tc::CartesianMesh geom(nx, ny, 100.0, 100.0);
    tc::Arena arena(4u << 20);
    tc::Field2 mask = arena.alloc2d(nx, ny);
    tc::for_each_cell(nx, ny, [=](Index i, Index j) {
        const Real ci = Real(i) - Real(nx - 1) / 2, cj = Real(j) - Real(ny - 1) / 2;
        mask[i, j] = (ci * ci + cj * cj) < Real(nx * nx) / 9 ? Real(1) : Real(0);   // radius ≈ nx/3
    });
    tc::MaskedMesh m(geom, mask);

    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);
    tc::SadournyEnstrophy cor; cor.init(arena, m);
    tc::FvPgf pgf;

    const tc::Field2 eta = s.eta, u = s.u, v = s.v;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center), [=](Index i, Index j) { eta[i, j] = Real(1000); });
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = Real(0); });
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = Real(0); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 100, .dt = 1, .g = 9.81, .H = 1000 };
    cont.compute(s, k, m, p);
    pgf.compute(s, k, m, p);
    cor.compute(s, k, m, p);

    const tc::Field2 ke = k.eta, ku = k.u, kv = k.v;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) CHECK(ke[i, j] == doctest::Approx(0.0));
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx; ++i) CHECK(ku[i, j] == doctest::Approx(0.0));
    for (Index j = 1; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) CHECK(kv[i, j] == doctest::Approx(0.0));
}
