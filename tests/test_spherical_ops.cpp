// =============================================================================
// tests/test_spherical_ops.cpp — the operators on the SPHERE (Stage B). The mesh
// geometry is checked in test_spherical_mesh.cpp; here we prove continuity +
// Coriolis + PGF are metric-correct with varying dx=R·cosφ·Δλ and f=2Ω sinφ, so
// the bc_inst run can trust them:
//   • mass conservation on a spherical periodic-x channel → continuity metrics OK;
//   • lake-at-rest → flat state gives exactly zero RHS (well-balanced on the sphere);
//   • geostrophically-balanced zonal flow: the ONLY nonzero continuous tendency is
//     kv = -g∂η/∂y - f·u = 0, so max|kv| is pure discretization residual — small for
//     a broad flow, and O(1) if any metric factor were wrong.
// Host-serial; no doctest main (test_m0 owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/spherical_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/ocean_core.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/pgf.hpp"
#include "diag/diagnostics.hpp"

using tc::Real;
using tc::Index;

namespace {
// The same policy stack as BarotropicPoC, but over the spherical mesh.
using SphCore = tc::OceanCore<tc::SphericalMesh, tc::PpmContinuity, tc::SadournyEnstrophy,
                              tc::FvPgf, tc::WallBC, tc::SSPRK3>;
// The bc_inst band; caller picks resolution + topology.
tc::SphericalMesh band(Index nx, Index ny, tc::EdgeConn we, tc::EdgeConn sn) {
    return tc::SphericalMesh(nx, ny, -193.75, 53.625, -171.25, 64.875,
                             tc::EARTH_RADIUS, tc::EARTH_OMEGA, we, we, sn, sn);
}
Real amax_center_v(tc::Field2 f, Index nx, Index ny) {
    Real m = 0;
    for (Index j = 0; j < ny; ++j) for (Index i = 0; i < nx; ++i) m = std::max(m, std::abs(f[i, j]));
    return m;
}
}  // namespace

TEST_CASE("spherical: mass conserved on a periodic-x channel (continuity metrics)") {
    const Index nx = 48, ny = 48;
    tc::SphericalMesh mesh = band(nx, ny, tc::EdgeConn::Periodic, tc::EdgeConn::Wall);
    tc::Arena arena(64u << 20);
    const Real g = 9.81, H = 1000.0, PI = std::acos(Real(-1));
    tc::Params p{ .nx = nx, .ny = ny, .dx = mesh.dx(tc::Loc::Center, 0, ny / 2),
                  .dy = mesh.dy(tc::Loc::Center, 0, 0), .dt = 20.0, .g = g, .H = H };
    SphCore core(mesh, arena, p);
    core.init();

    // a lumpy but smooth free surface + a gentle zonal flow; let it slosh.
    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center), [=](Index i, Index j) {
        const Real lon = mesh.x(tc::Loc::Center, i, j), lat = mesh.y(tc::Loc::Center, i, j);
        eta[i, j] = H + std::cos(Real(2) * PI * (lon + 193.75) / 22.5) * std::sin(PI * (lat - 53.625) / 11.25);
    });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { u[i, j] = 0.05; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { v[i, j] = 0; });

    const Real mass0 = tc::total_mass(core.state(), mesh);
    for (int n = 0; n < 300; ++n) core.step();
    const Real mass1 = tc::total_mass(core.state(), mesh);
    CHECK(std::abs(mass1 - mass0) / std::abs(mass0) < 1e-11);   // flux-form telescoping on the sphere
}

TEST_CASE("spherical: lake-at-rest gives exactly zero RHS") {
    const Index nx = 24, ny = 24;
    tc::SphericalMesh mesh = band(nx, ny, tc::EdgeConn::Periodic, tc::EdgeConn::Wall);
    tc::Arena arena(32u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, mesh), k = tc::allocate_baro_state(arena, mesh);
    tc::PpmContinuity cont; cont.init(arena, mesh);
    tc::SadournyEnstrophy cor; cor.init(arena, mesh);
    tc::FvPgf pgf;
    const Real H = 1000.0;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center), [=](Index i, Index j) { s.eta[i, j] = H; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) { s.u[i, j] = 0; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { s.v[i, j] = 0; });
    tc::zero_baro_state(k, mesh);
    tc::Params p{ .nx = nx, .ny = ny, .dx = 1, .dy = 1, .dt = 1, .g = 9.81, .H = H };
    cont.compute(s, k, mesh, p); cor.compute(s, k, mesh, p); pgf.compute(s, k, mesh, p);

    CHECK(amax_center_v(k.eta, mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center)) == doctest::Approx(0.0));
    CHECK(amax_center_v(k.u, mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace)) == doctest::Approx(0.0));
    CHECK(amax_center_v(k.v, mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace)) == doctest::Approx(0.0));
}

TEST_CASE("spherical: geostrophically-balanced zonal flow has tiny kv residual") {
    const Index nx = 32, ny = 96;                     // resolve y; broad flow
    tc::SphericalMesh mesh = band(nx, ny, tc::EdgeConn::Periodic, tc::EdgeConn::Wall);
    tc::Arena arena(48u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, mesh), k = tc::allocate_baro_state(arena, mesh);
    tc::PpmContinuity cont; cont.init(arena, mesh);
    tc::SadournyEnstrophy cor; cor.init(arena, mesh);
    tc::FvPgf pgf;

    const Real g = 9.81, H = 1000.0, R = tc::EARTH_RADIUS, del = tc::DEG2RAD;
    const Real phi0 = 59.25 * del, W = 3.0e5, Deta = 1.0;    // W=300 km ≫ dy ⇒ broad
    auto yy = [=](Real lat_deg) { return R * (lat_deg * del - phi0); };   // metres from band centre
    // η(y)=H − Δη tanh(y/W);  u = −(g/f)∂η/∂y = (gΔη/(fW)) sech²(y/W)
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center), [=](Index i, Index j) {
        const Real y = yy(mesh.y(tc::Loc::Center, i, j));
        s.eta[i, j] = H - Deta * std::tanh(y / W);
    });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](Index i, Index j) {
        const Real lat = mesh.y(tc::Loc::XFace, i, j), y = yy(lat);
        const Real f = Real(2) * tc::EARTH_OMEGA * std::sin(lat * del);
        const Real sech2 = Real(1) / std::cosh(y / W);
        s.u[i, j] = (g * Deta / (f * W)) * sech2 * sech2;
    });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](Index i, Index j) { s.v[i, j] = 0; });
    tc::zero_baro_state(k, mesh);
    tc::Params p{ .nx = nx, .ny = ny, .dx = 1, .dy = 1, .dt = 1, .g = g, .H = H };
    cont.compute(s, k, mesh, p); cor.compute(s, k, mesh, p); pgf.compute(s, k, mesh, p);

    // scale for kv: the PGF term magnitude |g ∂η/∂y| ~ g·Δη/W (each big term ≈ this).
    const Real scale = g * Deta / W;
    const Real kv_max = amax_center_v(k.v, mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace));
    MESSAGE("kv_max=", kv_max, " scale=", scale, " ratio=", kv_max / scale);
    CHECK(kv_max / scale < 0.05);        // residual ≪ the balancing terms ⇒ PGF↔Coriolis consistent
    // ku and ∂η/∂t vanish identically for a zonal state (x-derivatives = 0, v = 0):
    CHECK(amax_center_v(k.u, mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace)) < 1e-12);
    CHECK(amax_center_v(k.eta, mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center)) < 1e-12);
}
