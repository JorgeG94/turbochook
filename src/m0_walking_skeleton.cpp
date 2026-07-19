// =============================================================================
// src/m0_walking_skeleton.cpp — Milestone 0: prove the toolchain BEFORE physics.
//
// What this exercises (ROADMAP M0):
//   • the tc::par execution-policy seam + for_each_index  (does it offload?)
//   • std::transform_reduce over a big array via tc::par   (parallel reduction)
//   • the Arena + Field views                              (the memory model)
//   • the logger + profiler                                (host infrastructure)
//   • that the whole compile-time policy stack COMPOSES + runs (BarotropicPoC)
//
// Correctness gate: the parallel reduction must equal the serial one. Offload
// gate (on the GPU build): this must run FAR faster than the host build on a big
// N (nsys is down → verify-by-speed, DESIGN §8).
// =============================================================================

#include <vector>
#include <numeric>       // std::transform_reduce, std::accumulate
#include <ranges>
#include <cmath>

#include "lib/log.hpp"
#include "lib/profiler.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "physics/ocean_core.hpp"

int main() try {
    tc::logger().set_level(tc::LogLevel::info);
    tc::logger().info("TurboChook M0 walking skeleton");

    // ── 1. saxpy + parallel reduction over a big array ───────────────────────
    const tc::Index N = 8'000'000;
    const tc::Real  a = 2.5;
    std::vector<tc::Real> x(N), y(N);          // managed memory under nvc++ -stdpar
    tc::Real* xp = x.data();                   // capture raw pointers, never the vectors
    tc::Real* yp = y.data();

    {
        TC_PROFILE("init");
        tc::for_each_index(N, [=](tc::Index i) {
            xp[i] = tc::Real(1) + tc::Real(i % 7);
            yp[i] = tc::Real(i % 3);
        });
    }
    {
        TC_PROFILE("saxpy");
        tc::for_each_index(N, [=](tc::Index i) { yp[i] = a * xp[i] + yp[i]; });
    }

    tc::Real dev_sum;
    {
        TC_PROFILE("reduce");
        auto ids = std::views::iota(tc::Index{0}, N);
        dev_sum = std::transform_reduce(tc::par, ids.begin(), ids.end(), tc::Real(0),
                                        std::plus<tc::Real>{}, [=](tc::Index i) { return yp[i]; });
    }

    // Serial reference on the host — the correctness oracle.
    tc::Real ser_sum = 0;
    for (tc::Index i = 0; i < N; ++i) ser_sum += a * (tc::Real(1) + tc::Real(i % 7)) + tc::Real(i % 3);

    const tc::Real rel_err = std::abs(dev_sum - ser_sum) / std::abs(ser_sum);
    tc::logger().info("saxpy sum: parallel={:.6e} serial={:.6e} rel_err={:.2e}",
                      dev_sum, ser_sum, rel_err);
    if (rel_err > 1e-12) tc::fail(tc::Errc::nan_detected, "parallel reduction != serial");

    // ── 2. prove the whole policy stack composes + runs (stubs, but real types) ─
    {
        TC_PROFILE("ocean_core");
        tc::CartesianMesh mesh(/*nx*/64, /*ny*/32, /*dx*/1000.0, /*dy*/1000.0);
        tc::Arena arena(64ull << 20);          // 64 MiB, sized once
        tc::Params p{ .nx = mesh.nx(), .ny = mesh.ny(),
                      .dx = mesh.dx(), .dy = mesh.dy(),
                      .dt = 10.0, .g = 9.81, .H = 1000.0 };
        tc::BarotropicPoC core(mesh, arena, p);   // PPM + Sadourny + FV-PGF + Wall + RK2
        core.init();
        for (int s = 0; s < 5; ++s) core.step();   // stubs no-op, but the machinery runs
        tc::logger().info("ocean core ran 5 steps (operators are stubs — M2 fills them in)");
    }

    tc::profiler().report(/*tree=*/true);
    tc::logger().info("M0 OK");
    return 0;
}
catch (const tc::Error& e)      { tc::logger().error("{}", e.what()); return 1; }
catch (const std::exception& e) { tc::logger().error("unhandled: {}", e.what()); return 1; }
