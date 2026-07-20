// =============================================================================
// tests/test_periodic.cpp — the periodic-x boundary (mesh-owns-indexing, ADR-7).
// The zonal-jet channel needs a re-entrant x, so this validates that continuity +
// PGF wrap correctly at the seam (x=0 ≡ x=L), NOT reflect like a wall.
//
//   • standing wave at k = 2π/L: a full-wavelength mode is only clean on a PERIODIC
//     domain (a wall box reflects it), and its crest sits ON the seam — so getting
//     c = √(gH) here proves the wrap flux/gradient are right.
//   • travelling wave: a rightward pulse started at the seam must CROSS it at speed
//     c (a wall would bounce it back).
// f=0 (pure gravity wave). Host-serial; no doctest main (test_m0 owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/ocean_core.hpp"
#include "diag/diagnostics.hpp"

using tc::Real;
using tc::Index;

namespace {
// A periodic-x, wall-y channel (the bc_inst topology), f=0.
tc::CartesianMesh periodic_channel(Index nx, Index ny, Real dx, Real dy) {
    return tc::CartesianMesh(nx, ny, dx, dy, /*f0*/0.0, /*beta*/0.0,
                             tc::EdgeConn::Periodic, tc::EdgeConn::Periodic,
                             tc::EdgeConn::Wall,     tc::EdgeConn::Wall);
}
}  // namespace

TEST_CASE("periodic-x: full-wavelength standing wave oscillates at sqrt(gH)") {
    const Index nx = 64, ny = 4;
    const Real dx = 10000.0, dy = 10000.0, g = 9.81, H = 1000.0, A = 1.0;
    const Real PI = std::acos(Real(-1));
    tc::CartesianMesh mesh = periodic_channel(nx, ny, dx, dy);
    const Real c = std::sqrt(g * H), L = Real(nx) * dx;
    const Real kx = Real(2) * PI / L;                    // full wavelength (periodic mode-1)
    const Real T = Real(2) * PI / (kx * c), dt = Real(0.4) * dx / c;

    tc::Arena arena(32u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    tc::BarotropicPoC core(mesh, arena, p);
    core.init();

    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { eta[i, j] = H + A * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });

    const Real mass0 = tc::total_mass(core.state(), mesh);
    const Index ai = 0, aj = ny / 2;                     // antinode ON the seam
    Real t_prev = 0, e_prev = core.state().eta[ai, aj], t_cross = -1;
    for (int n = 1; n <= 4000; ++n) {
        core.step();
        const Real t = Real(n) * dt, e = core.state().eta[ai, aj];
        if (e < H && e_prev >= H) { t_cross = t_prev + (t - t_prev) * (e_prev - H) / (e_prev - e); break; }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);
    CHECK(Real(4) * t_cross == doctest::Approx(T).epsilon(0.03));   // wrap flux/gradient ⇒ correct c
    CHECK(std::abs(tc::total_mass(core.state(), mesh) - mass0) / mass0 < 1e-10);
}

TEST_CASE("periodic-x: rightward pulse crosses the seam at speed c") {
    const Index nx = 128, ny = 4;
    const Real dx = 10000.0, dy = 10000.0, g = 9.81, H = 1000.0, A = 1.0;
    const Real PI = std::acos(Real(-1));
    tc::CartesianMesh mesh = periodic_channel(nx, ny, dx, dy);
    const Real c = std::sqrt(g * H), L = Real(nx) * dx;
    const Real kx = Real(2) * PI / L, dt = Real(0.3) * dx / c;

    tc::Arena arena(48u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    tc::BarotropicPoC core(mesh, arena, p);
    core.init();

    // Rightward linear gravity wave: η'=A cos(k(x−x0)), u=(c/H)η'. Crest starts at
    // x0 = 3L/4 so it must cross the x=0 seam within the run.
    const Real x0 = Real(0.75) * L;
    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { eta[i, j] = H + A * std::cos(kx * (mesh.x(tc::Loc::Center, i, j) - x0)); });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace),
                      [=](Index i, Index j) { u[i, j] = (c / H) * A * std::cos(kx * (mesh.x(tc::Loc::XFace, i, j) - x0)); });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });

    const Real t_run = Real(0.4) * L / c;                // crest travels 0.4L → past the seam
    const int nsteps = int(t_run / dt);
    for (int n = 0; n < nsteps; ++n) core.step();

    // locate the crest (argmax η along the row)
    const Index aj = ny / 2;
    Index icrest = 0; Real emax = -1e30;
    for (Index i = 0; i < nx; ++i) { const Real e = core.state().eta[i, aj]; if (e > emax) { emax = e; icrest = i; } }
    const Real x_expected = std::fmod(x0 + c * (Real(nsteps) * dt), L);
    const Real x_crest = mesh.x(tc::Loc::Center, icrest, aj);
    Real dxsep = std::abs(x_crest - x_expected);
    dxsep = std::min(dxsep, L - dxsep);                  // periodic distance
    CHECK(dxsep < Real(3) * dx);                         // crest crossed the seam, ≈ c·t
    CHECK(emax > H + Real(0.5) * A);                     // pulse survived (not reflected/cancelled)
}
