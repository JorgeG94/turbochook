// =============================================================================
// tests/test_m0.cpp — the first tests. Host-serial (par_unseq → seq), so they
// run anywhere with no TBB and are deterministic. doctest is the framework
// (single-header, fetched by CMake), one runner via CTest.
//
// The rule that carries over (docs/GPU_STDPAR_NOTES.md): CPU-green ≠ GPU-correct.
// These prove LOGIC. Offload/data-motion correctness needs the analytical suite
// run under the -stdpar=gpu build periodically (later).
// =============================================================================

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>
#include <numeric>
#include <ranges>
#include <cmath>

#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/ocean_core.hpp"

TEST_CASE("Arena hands out usable layout_left Field views") {
    tc::Arena arena(1u << 20);              // 1 MiB
    const tc::Index nx = 8, ny = 5;
    tc::Field2 f = arena.alloc2d(nx, ny);

    // Write f[i,j] = i + 100*j, then read it back — checks the column-major
    // index math end to end (whether backed by std::mdspan or tc::MdView).
    for (tc::Index j = 0; j < ny; ++j)
        for (tc::Index i = 0; i < nx; ++i)
            f[i, j] = tc::Real(i) + 100 * tc::Real(j);

    CHECK(f[0, 0] == doctest::Approx(0.0));
    CHECK(f[7, 4] == doctest::Approx(407.0));
    CHECK(f.extent(0) == nx);
    CHECK(f.extent(1) == ny);
}

TEST_CASE("Arena refuses to overflow (fails loud)") {
    tc::Arena tiny(64);                     // 64 bytes = 8 doubles
    CHECK_THROWS_AS(tiny.alloc2d(1000, 1000), tc::Error);
}

TEST_CASE("saxpy via tc::par matches the serial result") {
    const tc::Index N = 10'000;
    const tc::Real  a = 3.0;
    std::vector<tc::Real> x(N), y(N);
    tc::Real* xp = x.data(); tc::Real* yp = y.data();

    tc::for_each_index(N, [=](tc::Index i) { xp[i] = tc::Real(i); yp[i] = tc::Real(2 * i); });
    tc::for_each_index(N, [=](tc::Index i) { yp[i] = a * xp[i] + yp[i]; });

    for (tc::Index i = 0; i < N; ++i)
        CHECK(y[i] == doctest::Approx(a * tc::Real(i) + tc::Real(2 * i)));
}

TEST_CASE("parallel reduction equals the serial reduction") {
    const tc::Index N = 100'000;
    std::vector<tc::Real> y(N);
    tc::Real* yp = y.data();
    tc::for_each_index(N, [=](tc::Index i) { yp[i] = tc::Real(i % 13); });

    auto ids = std::views::iota(tc::Index{0}, N);
    const tc::Real par_sum = std::transform_reduce(
        tc::par, ids.begin(), ids.end(), tc::Real(0),
        std::plus<tc::Real>{}, [=](tc::Index i) { return yp[i]; });

    tc::Real ser_sum = 0;
    for (tc::Index i = 0; i < N; ++i) ser_sum += tc::Real(i % 13);

    CHECK(par_sum == doctest::Approx(ser_sum));
}

TEST_CASE("the compile-time policy stack composes and runs (stubs)") {
    tc::CartesianMesh mesh(16, 8, 500.0, 500.0);
    tc::Arena arena(8u << 20);
    tc::Params p{ .nx = mesh.nx(), .ny = mesh.ny(), .dx = mesh.dx(), .dy = mesh.dy(),
                  .dt = 5.0, .g = 9.81, .H = 500.0 };
    tc::BarotropicPoC core(mesh, arena, p);
    core.init();
    core.step();                            // operators are stubs → must not throw

    // The staggered extents are what a C-grid demands.
    CHECK(core.state().eta.extent(0) == 16);
    CHECK(core.state().u.extent(0)   == 17);   // x-faces = nx+1
    CHECK(core.state().v.extent(1)   == 9);    // y-faces = ny+1
}
