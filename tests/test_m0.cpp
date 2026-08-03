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
#include "physics/state/baro_state.hpp"
#include "physics/core/ocean_core.hpp"

// TU-UNIQUE NAMESPACE -- load-bearing, not cosmetic. SYCL names an unnamed-lambda
// kernel by its C++ mangled type, and doctest's TEST_CASE expands to a `static
// void DOCTEST_ANON_FUNC_<n>` whose counter restarts in every file, so kernels
// from different test files collide on one name. See docs/GPU_STDPAR_NOTES.md.
namespace tu_test_m0 {

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

TEST_CASE("saxpy via tc::do_concurrent matches the serial result") {
    const tc::Index N = 10'000;
    const tc::Real  a = 3.0;
    // ARENA, not std::vector: a device kernel writes these. Plain host heap is
    // reachable from the device only under `nvc++ -stdpar`, which promotes it to
    // managed; hipstdpar faults and oneDPL fails. The arena is managed memory, so
    // it is the portable place for anything a kernel touches. (Same root cause as
    // the arena's own fix -- see lib/device_alloc.hpp.)
    tc::Arena arena(4u << 20);
    tc::Real* xp = arena.alloc2d(N, 1).data_handle();
    tc::Real* yp = arena.alloc2d(N, 1).data_handle();

    tc::for_each_index(N, [=](tc::Index i) { xp[i] = tc::Real(i); yp[i] = tc::Real(2 * i); });
    tc::for_each_index(N, [=](tc::Index i) { yp[i] = a * xp[i] + yp[i]; });

    for (tc::Index i = 0; i < N; ++i)
        CHECK(yp[i] == doctest::Approx(a * tc::Real(i) + tc::Real(2 * i)));
}

TEST_CASE("parallel reduction equals the serial reduction") {
    const tc::Index N = 100'000;
    tc::Arena arena(4u << 20);                     // managed: a kernel writes it
    tc::Real* yp = arena.alloc2d(N, 1).data_handle();
    tc::for_each_index(N, [=](tc::Index i) { yp[i] = tc::Real(i % 13); });

    // Through tc::do_reduce, not a bare std::transform_reduce: oneDPL needs its own
    // algorithm and device policy, and do_reduce is the one place that knows.
    const tc::Real par_sum = tc::do_reduce(
        N, tc::Real(0), std::plus<tc::Real>{}, [=](tc::Index i) { return yp[i]; });

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

}  // namespace tu_test_m0
