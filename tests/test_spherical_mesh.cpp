// =============================================================================
// tests/test_spherical_mesh.cpp — the lon/lat spherical C-grid, and proof that the
// (mesh-generic) physics operators run correctly on it — the genericity payoff.
// Oracles: dy constant, dx = dy·cosφ, f = 2Ω sinφ; lake-at-rest ⇒ zero RHS on the
// sphere (well-balanced); PPM continuity conserves mass with varying cell area.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/spherical_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/pgf.hpp"

using tc::Real;
using tc::Index;

TEST_CASE("SphericalMesh: metrics — dy constant, dx = dy·cosφ, f = 2Ω sinφ") {
    tc::SphericalMesh m(20, 20, /*lon*/0.0, /*lat*/50.0, 20.0, 70.0);   // 1°×1° cells

    const Real dy = m.dy(tc::Loc::Center, 0, 0);
    CHECK(dy == doctest::Approx(tc::EARTH_RADIUS * 1.0 * tc::DEG2RAD));  // R·Δφ
    CHECK(m.dy(tc::Loc::YFace, 5, 19) == doctest::Approx(dy));           // latitude-independent

    for (Index j : {Index(0), Index(7), Index(19)}) {                   // dx/dy = cosφ
        const Real phi = m.y(tc::Loc::Center, 0, j) * tc::DEG2RAD;
        CHECK(m.dx(tc::Loc::Center, 0, j) == doctest::Approx(dy * std::cos(phi)));
    }
    CHECK(m.dx(tc::Loc::Center, 0, 0) > m.dx(tc::Loc::Center, 0, 19));   // shrinks poleward

    const Real phi_c = m.y(tc::Loc::Corner, 0, 10) * tc::DEG2RAD;        // 60°N corner row
    CHECK(m.coriolis(tc::Loc::Corner, 0, 10) == doctest::Approx(2 * tc::EARTH_OMEGA * std::sin(phi_c)));
    CHECK(m.coriolis(tc::Loc::Corner, 0, 15) > m.coriolis(tc::Loc::Corner, 0, 5));  // f grows poleward

    CHECK(m.extent_x(tc::Loc::XFace) == 21);
    CHECK(m.extent_y(tc::Loc::Corner) == 21);
}

TEST_CASE("SphericalMesh: lake-at-rest ⇒ zero RHS (well-balanced on the sphere)") {
    const Index nx = 16, ny = 16;
    tc::SphericalMesh m(nx, ny, 0.0, 50.0, 20.0, 70.0);
    tc::Arena arena(8u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);
    tc::SadournyEnstrophy cor; cor.init(arena, m);
    tc::FvPgf pgf;

    const tc::Field2 eta = s.eta, u = s.u, v = s.v;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center), [=](Index i, Index j) { eta[i, j] = Real(1000); });
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0; });
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });
    tc::zero_baro_state(k, m);

    // geometry comes entirely from the mesh — Params carries only run scalars (dx/dy unused)
    tc::Params p{ .nx = nx, .ny = ny, .dx = 0, .dy = 0, .dt = 1, .g = 9.81, .H = 1000 };
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

TEST_CASE("SphericalMesh: PpmContinuity conserves mass with varying cell area") {
    const Index nx = 20, ny = 16;
    tc::SphericalMesh m(nx, ny, 0.0, 50.0, 20.0, 66.0);
    tc::Arena arena(8u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);

    const tc::Field2 eta = s.eta, u = s.u, v = s.v;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { eta[i, j] = Real(1000) + Real(0.1) * Real(i) - Real(0.05) * Real(j); });
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = Real(0.3); });
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = Real(0.2); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 0, .dy = 0, .dt = 1, .g = 9.81, .H = 1000 };
    cont.compute(s, k, m, p);

    const tc::Field2 ke = k.eta;
    Real net = 0, activity = 0;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) {
            const Real r = m.area(tc::Loc::Center, i, j) * ke[i, j];   // area varies with latitude
            net += r; activity += std::abs(r);
        }
    CHECK(activity > 0.0);
    CHECK(std::abs(net) / activity < 1e-12);
}
