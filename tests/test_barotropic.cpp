// =============================================================================
// tests/test_barotropic.cpp — the 2D Forward-Backward barotropic sub-solver
// (split-explicit Stage 1, DESIGN ADR-9). It is the fast-mode engine the split
// stepper subcycles, so it must, on its own, get the gravity wave right:
//   • standing mode-1 (η=H+A cos kx, u=0) oscillates at ω=k√(gH) — FB reproduces the
//     surface-gravity-wave speed and conserves mass;
//   • lake-at-rest (flat η, no flow, no forcing) stays flat forever.
// Same oracle as the unsplit gravity-wave test, so a pass means the FB ordering +
// operator reuse are sound before we build the subcycle/coupling on top.
// Host-serial; no doctest main (test_m0 owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/state/baro_state.hpp"
#include "physics/core/barotropic.hpp"

using tc::Real;
using tc::Index;

namespace {
using BtSolver = tc::BarotropicSolver<tc::PpmContinuity, tc::SadournyEnstrophy, tc::FvPgf>;
}

TEST_CASE("barotropic FB solver: standing gravity wave oscillates at sqrt(gH)") {
    const Index nx = 64, ny = 4;
    const Real dx = 10000.0, dy = 10000.0, g = 9.81, H = 1000.0, A = 1.0;
    const Real PI = std::acos(Real(-1));
    tc::CartesianMesh mesh(nx, ny, dx, dy, /*f0*/0.0);        // no rotation → pure gravity wave
    const Real c = std::sqrt(g * H), L = Real(nx) * dx, kx = PI / L;
    const Real T = Real(2) * PI / (kx * c), dt = Real(0.5) * dx / c;   // FB CFL ≈ 0.5

    tc::Arena arena(32u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    BtSolver bt; bt.init(arena, mesh);
    tc::BaroState s   = tc::allocate_baro_state(arena, mesh);
    tc::BaroState frc = tc::allocate_baro_state(arena, mesh);   // zero forcing (arena zero-inits)

    // η = H + A cos(kx)  (TOTAL thickness), u = v = 0
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { s.eta[i, j] = H + A * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });

    // total mass diagnostic (Σ η over centres — uniform mesh)
    auto mass = [&]() { Real mm = 0; for (Index j = 0; j < ny; ++j) for (Index i = 0; i < nx; ++i) mm += s.eta[i, j]; return mm; };
    const Real mass0 = mass();

    const Index ai = 0, aj = ny / 2;
    Real t_prev = 0, e_prev = s.eta[ai, aj], t_cross = -1;
    for (int n = 1; n <= 4000; ++n) {
        bt.substep(s, frc, mesh, p, dt);
        const Real t = Real(n) * dt, e = s.eta[ai, aj];
        if (e < H && e_prev >= H) { t_cross = t_prev + (t - t_prev) * (e_prev - H) / (e_prev - e); break; }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);
    CHECK(Real(4) * t_cross == doctest::Approx(T).epsilon(0.03));       // FB recovers √(gH)
    CHECK(std::abs(mass() - mass0) / mass0 < 1e-10);                    // free-surface continuity conserves mass
}

TEST_CASE("barotropic FB solver: lake-at-rest stays flat") {
    const Index nx = 16, ny = 12;
    const Real H = 1000.0;
    tc::CartesianMesh mesh(nx, ny, 100.0, 100.0, /*f0*/1.0e-4);
    tc::Arena arena(16u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 100, .dt = 1, .g = 9.81, .H = H };
    BtSolver bt; bt.init(arena, mesh);
    tc::BaroState s   = tc::allocate_baro_state(arena, mesh);
    tc::BaroState frc = tc::allocate_baro_state(arena, mesh);
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { s.eta[i, j] = H; });     // flat, u=v=0

    for (int n = 0; n < 200; ++n) bt.substep(s, frc, mesh, p, Real(0.02));

    Real de = 0, sp = 0;
    for (Index j = 0; j < ny; ++j) for (Index i = 0; i < nx; ++i) de = std::max(de, std::abs(s.eta[i, j] - H));
    for (Index j = 0; j < ny; ++j) for (Index i = 0; i <= nx; ++i) sp = std::max(sp, std::abs(s.u[i, j]));
    for (Index j = 0; j <= ny; ++j) for (Index i = 0; i < nx; ++i) sp = std::max(sp, std::abs(s.v[i, j]));
    CHECK(de == doctest::Approx(0.0));                                  // η never leaves H
    CHECK(sp == doctest::Approx(0.0));                                  // no spurious flow
}
