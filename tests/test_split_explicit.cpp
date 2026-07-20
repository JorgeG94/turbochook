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
#include "physics/split_two_layer.hpp"

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

TEST_CASE("split two-layer: barotropic gravity wave at sqrt(g(H1+H2)) with big outer dt") {
    const Index nx = 64, ny = 4;
    const Real dx = 10000.0, dy = 10000.0, g = 9.81, H1 = 200.0, H2 = 800.0, A = 0.5;
    const Real H = H1 + H2, PI = std::acos(Real(-1));
    tc::CartesianMesh mesh(nx, ny, dx, dy, /*f0*/0.0);
    const Real c = std::sqrt(g * H), L = Real(nx) * dx, kx = PI / L;
    const Real T = Real(2) * PI / (kx * c);
    const int  Msub = 20;
    const Real dt = Real(200.0);                       // dtbt=10s (FB CFL 0.1); OUTER CFL≈2 (unsplit-illegal)

    tc::Arena arena(64u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H,
                  .H1 = H1, .H2 = H2, .rho1 = 1025, .rho2 = 1025 };   // ρ1=ρ2 → pure barotropic
    tc::SplitTwoLayerCore<tc::CartesianMesh, tc::PpmContinuity, tc::SadournyEnstrophy,
                          tc::TwoLayerReducedGravityPgf, tc::WallBC, Msub> core(mesh, arena, p);
    core.init();

    const tc::Field2 e0 = core.state().layer[0].eta, e1 = core.state().layer[1].eta;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center), [=](Index i, Index j) {
        const Real s = A * std::cos(kx * mesh.x(tc::Loc::Center, i, j));
        e0[i, j] = H1 + (H1 / H) * s; e1[i, j] = H2 + (H2 / H) * s;
    });
    for (int l = 0; l < 2; ++l) {
        const tc::Field2 u = core.state().layer[l].u, v = core.state().layer[l].v;
        tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0; });
        tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });
    }

    const Index ai = 0, aj = ny / 2;
    auto surf = [&]() { return core.state().layer[0].eta[ai, aj] + core.state().layer[1].eta[ai, aj] - H; };
    Real t_prev = 0, e_prev = surf(), t_cross = -1;
    for (int n = 1; n <= 400; ++n) {
        core.step();
        const Real t = Real(n) * dt, e = surf();
        REQUIRE(std::isfinite(e));                     // split must stay stable at outer CFL≈2
        if (e < 0 && e_prev >= 0) { t_cross = t_prev + (t - t_prev) * e_prev / (e_prev - e); break; }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);
    CHECK(Real(4) * t_cross == doctest::Approx(T).epsilon(0.05));   // subcycle recovers √(g(H1+H2))
}
