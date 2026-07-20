// =============================================================================
// tests/test_gravity_wave.cpp — the first END-TO-END physics test: the whole
// stack (PPM continuity + Fv PGF + SSP-RK2, driven by OceanCore) integrating a
// linear shallow-water gravity wave. Coriolis/BC are no-op stubs here → f=0, and
// closed-box walls are handled in-kernel, so this is pure gravity-wave dynamics:
//     ∂η/∂t = -∇·(hu),  ∂u/∂t = -g∇η   ⇒   waves at c = √(gH).
//
// Oracle 1 (physics): a closed-channel standing mode-1 (η = H + A·cos kx, u=0)
// oscillates at ω = k√(gH); the antinode returns to the mean at T/4. We measure
// that quarter-period and compare to 2π/(4ω) — validates the wave SPEED through
// the full coupled stepper.
// Oracle 2 (conservation): total mass Σ η·area is invariant to machine-eps over
// the whole run (flux-form continuity + RK2 preserve it exactly).
// Host-serial; no doctest main (test_m0.cpp owns it).
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

TEST_CASE("gravity wave: closed-channel standing mode-1 oscillates at sqrt(gH)") {
    const Index nx = 64, ny = 4;                 // 1D channel (uniform in y)
    const Real dx = 10000.0, dy = 10000.0;       // 10 km
    const Real g = 9.81, H = 1000.0, A = 1.0;    // A/H = 1e-3 → linear regime
    const Real PI = std::acos(Real(-1));

    tc::CartesianMesh mesh(nx, ny, dx, dy);
    const Real c   = std::sqrt(g * H);           // gravity-wave speed
    const Real L   = Real(nx) * dx;              // channel length
    const Real kx  = PI / L;                      // mode-1 (u=0 at both walls)
    const Real T   = Real(2) * PI / (kx * c);    // analytic period
    const Real dt  = Real(0.4) * dx / c;         // CFL ≈ 0.4

    tc::Arena arena(32u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    tc::BarotropicPoC core(mesh, arena, p);
    core.init();

    // IC: η = H + A·cos(kx), u = v = 0.
    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { eta[i, j] = H + A * std::cos(kx * mesh.x(tc::Loc::Center, i, j)); });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace),
                      [=](Index i, Index j) { u[i, j] = 0; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace),
                      [=](Index i, Index j) { v[i, j] = 0; });

    const Real mass0 = tc::total_mass(core.state(), mesh);

    // Step until the antinode (i=0) crosses the mean H — that first crossing is T/4.
    const Index ai = 0, aj = ny / 2;
    Real t_prev = 0, e_prev = core.state().eta[ai, aj], t_cross = -1;
    for (int n = 1; n <= 4000; ++n) {
        core.step();
        const Real t = Real(n) * dt;
        const Real e = core.state().eta[ai, aj];
        if (e < H && e_prev >= H) {                       // downward crossing of the mean
            t_cross = t_prev + (t - t_prev) * (e_prev - H) / (e_prev - e);
            break;
        }
        t_prev = t; e_prev = e;
    }
    REQUIRE(t_cross > 0);

    const Real T_measured = Real(4) * t_cross;
    CHECK(T_measured == doctest::Approx(T).epsilon(0.03));   // wave speed = √(gH), 3%

    const Real mass1 = tc::total_mass(core.state(), mesh);
    CHECK(std::abs(mass1 - mass0) / mass0 < 1e-10);          // exact conservation over the run
}
