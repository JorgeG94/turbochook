// =============================================================================
// tests/test_two_layer.cpp — the M3 two-layer machinery. Oracles:
//   • reduced-gravity PGF: linear h₁,h₂ ⇒ exact -∇p₁, -∇p₂ (the coupling math);
//   • two-layer lake-at-rest: flat layers, no flow ⇒ zero RHS (well-balanced);
//   • barotropic gravity wave through the 2-layer stepper: with ρ₁=ρ₂ the stack is
//     one layer of depth H₁+H₂, so the surface mode travels at √(g(H₁+H₂)).
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/layered_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/two_layer_pgf.hpp"
#include "physics/multilayer_core.hpp"

using tc::Real;
using tc::Index;

namespace {
void fill_center(tc::Field2 f, const tc::CartesianMesh& m, auto g) {
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { f[i, j] = g(i, j); });
}
}  // namespace

TEST_CASE("TwoLayer PGF: linear thicknesses give exact -∇p₁, -∇p₂ (coupling)") {
    const Index nx = 16, ny = 6;
    const Real dx = 1.0e4, dy = 1.0e4, g = 9.81, rho1 = 1025.0, rho2 = 1027.0;
    const Real a1 = 1.0e-4, a2 = -5.0e-5;                 // h₁,h₂ slopes in x
    const Real r = rho2 / rho1;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(8u << 20);
    tc::LayeredState<2> s = tc::allocate_layered_state<2>(arena, m);
    tc::LayeredState<2> k = tc::allocate_layered_state<2>(arena, m);
    tc::TwoLayerReducedGravityPgf pgf;

    fill_center(s.layer[0].eta, m, [=](Index i, Index j) { return a1 * m.x(tc::Loc::Center, i, j); });
    fill_center(s.layer[1].eta, m, [=](Index i, Index j) { return a2 * m.x(tc::Loc::Center, i, j); });
    for (int l = 0; l < 2; ++l) {
        tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { s.layer[l].u[i, j] = 0; });
        tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { s.layer[l].v[i, j] = 0; });
    }
    tc::zero_layered_state<2>(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = 1, .g = g, .H = 1000,
                  .H1 = 200, .H2 = 800, .rho1 = rho1, .rho2 = rho2 };
    pgf.compute(s, k, m, p);

    const tc::Field2 ku1 = k.layer[0].u, ku2 = k.layer[1].u;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx; ++i) {
            CHECK(ku1[i, j] == doctest::Approx(-g * (a1 + a2)));         // -∂/∂x g(h₁+h₂)
            CHECK(ku2[i, j] == doctest::Approx(-g * (a1 + r * a2)));     // -∂/∂x (g h₁ + r g h₂)
        }
}

TEST_CASE("TwoLayer lake-at-rest: flat layers ⇒ zero RHS") {
    const Index nx = 12, ny = 10;
    const Real H1 = 200.0, H2 = 800.0;
    tc::CartesianMesh m(nx, ny, 100.0, 100.0, /*f0*/1.0e-4);
    tc::Arena arena(8u << 20);
    tc::LayeredState<2> s = tc::allocate_layered_state<2>(arena, m);
    tc::LayeredState<2> k = tc::allocate_layered_state<2>(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);
    tc::SadournyEnstrophy cor; cor.init(arena, m);
    tc::TwoLayerReducedGravityPgf pgf;

    fill_center(s.layer[0].eta, m, [=](Index, Index) { return H1; });
    fill_center(s.layer[1].eta, m, [=](Index, Index) { return H2; });
    for (int l = 0; l < 2; ++l) {
        tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace), [=](Index i, Index j) { s.layer[l].u[i, j] = 0; });
        tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace), [=](Index i, Index j) { s.layer[l].v[i, j] = 0; });
    }
    tc::zero_layered_state<2>(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 100, .dt = 1, .g = 9.81, .H = 1000,
                  .H1 = H1, .H2 = H2, .rho1 = 1025, .rho2 = 1027 };
    for (int l = 0; l < 2; ++l) cont.compute(s.layer[l], k.layer[l], m, p);
    for (int l = 0; l < 2; ++l) cor.compute(s.layer[l], k.layer[l], m, p);
    pgf.compute(s, k, m, p);

    for (int l = 0; l < 2; ++l) {
        const tc::Field2 ke = k.layer[l].eta, ku = k.layer[l].u, kv = k.layer[l].v;
        for (Index j = 0; j < ny; ++j)
            for (Index i = 0; i < nx; ++i) CHECK(ke[i, j] == doctest::Approx(0.0));
        for (Index j = 0; j < ny; ++j)
            for (Index i = 1; i < nx; ++i) CHECK(ku[i, j] == doctest::Approx(0.0));
        for (Index j = 1; j < ny; ++j)
            for (Index i = 0; i < nx; ++i) CHECK(kv[i, j] == doctest::Approx(0.0));
    }
}

TEST_CASE("TwoLayer barotropic wave: ρ₁=ρ₂ ⇒ surface mode at √(g(H₁+H₂))") {
    const Index nx = 64, ny = 4;
    const Real dx = 10000.0, dy = 10000.0, g = 9.81, H1 = 200.0, H2 = 800.0, A = 0.5;
    const Real H = H1 + H2, PI = std::acos(Real(-1));
    tc::CartesianMesh mesh(nx, ny, dx, dy, /*f0*/0.0);
    const Real c = std::sqrt(g * H), L = Real(nx) * dx, kx = PI / L;
    const Real T = Real(2) * PI / (kx * c), dt = Real(0.35) * dx / c;

    tc::Arena arena(48u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H,
                  .H1 = H1, .H2 = H2, .rho1 = 1025, .rho2 = 1025 };   // ρ₁=ρ₂ → pure barotropic
    tc::TwoLayerPoC core(mesh, arena, p);
    core.init();

    // surface anomaly A·cos(kx), distributed by layer depth; both layers at rest.
    // (hoist state views — state() is non-const, so read it outside the kernels)
    const tc::Field2 e0 = core.state().layer[0].eta, e1 = core.state().layer[1].eta;
    fill_center(e0, mesh, [=](Index i, Index j) { return H1 + A * (H1 / H) * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    fill_center(e1, mesh, [=](Index i, Index j) { return H2 + A * (H2 / H) * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    for (int l = 0; l < 2; ++l) {
        const tc::Field2 u = core.state().layer[l].u, v = core.state().layer[l].v;
        tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0; });
        tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });
    }

    const Index ai = 0, aj = ny / 2;
    auto surf = [&]() { return core.state().layer[0].eta[ai, aj] + core.state().layer[1].eta[ai, aj] - H; };
    Real t_prev = 0, e_prev = surf(), t_cross = -1;
    for (int n = 1; n <= 4000; ++n) {
        core.step();
        const Real t = Real(n) * dt, e = surf();
        if (e < 0 && e_prev >= 0) { t_cross = t_prev + (t - t_prev) * e_prev / (e_prev - e); break; }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);
    CHECK(Real(4) * t_cross == doctest::Approx(T).epsilon(0.03));   // c = √(g(H₁+H₂))
}

TEST_CASE("TwoLayer baroclinic wave: internal mode at c' = √(g'·H₁H₂/(H₁+H₂))") {
    const Index nx = 32, ny = 4;
    const Real dx = 20000.0, dy = 20000.0, g = 9.81, H1 = 200.0, H2 = 800.0;
    const Real rho1 = 1025.0, rho2 = 1027.0, A = 1.0;
    const Real H = H1 + H2, PI = std::acos(Real(-1));
    const Real eps = (rho2 - rho1) / rho1, gp = g * eps;            // reduced gravity g'
    const Real cbc = std::sqrt(gp * H1 * H2 / H);                    // baroclinic speed (~1.75 m/s)
    const Real R   = eps * H2 / H - Real(1);                         // baroclinic eigenvector η₂/η₁

    tc::CartesianMesh mesh(nx, ny, dx, dy, /*f0*/0.0);   // no rotation → pure internal gravity wave
    const Real L = Real(nx) * dx, kx = PI / L;
    const Real T  = Real(2) * PI / (kx * cbc);                       // baroclinic period (~8.5 days)
    const Real dt = Real(0.3) * dx / std::sqrt(g * H);              // dt limited by the BAROTROPIC CFL

    tc::Arena arena(48u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H,
                  .H1 = H1, .H2 = H2, .rho1 = rho1, .rho2 = rho2 };
    tc::TwoLayerPoC core(mesh, arena, p);
    core.init();

    // exact baroclinic eigenvector (η₂ = R·η₁, surface ≈ flat), at rest → pure internal mode
    const tc::Field2 e0 = core.state().layer[0].eta, e1 = core.state().layer[1].eta;
    fill_center(e0, mesh, [=](Index i, Index j) { return H1 + A * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    fill_center(e1, mesh, [=](Index i, Index j) { return H2 + R * A * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    for (int l = 0; l < 2; ++l) {
        const tc::Field2 u = core.state().layer[l].u, v = core.state().layer[l].v;
        tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0; });
        tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });
    }

    // Project η₁ onto the m=1 basin mode cos(πx/L): orthogonal to m=3, so this
    // isolates the gravest mode (a raw antinode probe beats m=1 against m=3, which —
    // both being baroclinic — keeps the vertical ratio fixed but modulates the peak).
    const Index aj = ny / 2;
    auto mode1 = [&]() {
        const tc::Field2 e = core.state().layer[0].eta;
        Real a = 0;
        for (Index i = 0; i < nx; ++i) a += (e[i, aj] - H1) * std::cos(kx * mesh.x(tc::Loc::Center, i, aj));
        return a;
    };
    Real t_prev = 0, e_prev = mode1(), t_cross = -1;
    for (int n = 1; n <= 8000; ++n) {
        core.step();
        const Real t = Real(n) * dt, e = mode1();
        if (e < 0 && e_prev >= 0) { t_cross = t_prev + (t - t_prev) * e_prev / (e_prev - e); break; }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);
    CHECK(Real(4) * t_cross == doctest::Approx(T).epsilon(0.03));   // the SLOW internal-mode speed
}
