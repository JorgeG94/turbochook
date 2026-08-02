// =============================================================================
// spike 05b — does the DEFAULT path (stdpar) reach Aurora?
//
// THE QUESTION, and it is the most consequential one for Aurora: does plain
// `std::for_each(std::execution::par_unseq, ...)` offload to PVC via oneDPL when
// built with icpx's PSTL-offload flag? If YES, Aurora is covered by the same
// source everything else runs, and native SYCL is a selective escape hatch
// exactly like native CUDA. If NO, Aurora needs hand-written SYCL kernels for
// every operator — a much more expensive commitment, and one worth discovering
// now rather than after fifty operators exist.
//
// The kernel body here is BYTE-IDENTICAL to spike 01's (which ran on V100 and
// GH200 through nvc++). That is deliberate: the only variable is the toolchain.
//
// A BUILD FAILURE IS A RESULT. `-fsycl-pstl-offload=gpu` may not exist under this
// icpx version, may be spelled differently, or may require oneapi::dpl execution
// policies rather than std:: ones. Whatever the compiler says goes in
// build/s05b_build.log and answers the question either way.
//
// HOW TO TELL IT ACTUALLY OFFLOADED (a green run proves nothing on its own — the
// same lesson as DESIGN §8 "verify offload by speed"):
//   • compare the GB/s here against spike 05's native SYCL number on the SAME
//     grid. Same order of magnitude => offloaded. ~10-40 GB/s => it silently ran
//     on the host CPU.
//   • or run with ONEAPI_DEVICE_SELECTOR unset/altered and watch it change.
// =============================================================================

#include <execution>
#include <algorithm>
#include <numeric>
#include <functional>
#include <ranges>
#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

// Identical to spike 01. No SYCL in sight — that is the whole point.
static void sweep(View2 a, View2 b, Index nx, Index ny) {
    auto ids = std::views::iota(0, (nx - 2) * (ny - 2));
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](int n) {
        const Index i = 1 + n % (nx - 2);
        const Index j = 1 + n / (nx - 2);
        b[i, j] = Real(0.25) * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]);
    });
}

static void fill(View2 f, Index nx, Index ny, Real v) {
    auto ids = std::views::iota(0, nx * ny);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(),
                  [=](int n) { f[n % nx, n / nx] = v; });
}

static long bad_count(const Real* dev, std::size_t n) {
    std::vector<Real> host(n);
    tc::stage_to_host(host.data(), dev, n * sizeof(Real));
    long bad = 0;
    for (Real x : host) if (x != Real(1)) ++bad;
    return bad;
}

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;

    std::printf("\n=== spike 05b: stdpar (par_unseq) on Aurora via oneDPL ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters\n\n", int(N), int(N), iters);

    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;
    const std::size_t ncell = std::size_t(N) * N;

    // ── A: shared USM pool, host hands off ───────────────────────────────────
    double gA = 0;
    {
        Real* pool = tc::pool_managed_prefetched(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();

        fill(a, N, N, Real(1)); fill(b, N, N, Real(1));
        tc::device_sync();

        tc::Timer t;
        for (long it = 0; it < iters; ++it) { sweep(a, b, N, N); std::swap(a, b); }
        tc::device_sync();
        const double secs = t.s();
        gA = tc::gbs(iters, N, N, secs);

        tc::check(bad_count(a.data_handle(), ncell) == 0, "A: shared USM pool, bitwise exact");
        std::printf("       A: %8.3f s   %8.2f GB/s\n", secs, gA);
        tc::pool_free(pool);
    }

    // ── B: the strict device-only pool under stdpar — the Aurora twin of spike
    //      02. On NVIDIA this passed (a par_unseq lambda may dereference a raw
    //      device pointer). Whether oneDPL tolerates the same is independent. ──
    {
        Real* pool = tc::pool_device(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();

        fill(a, N, N, Real(1)); fill(b, N, N, Real(1));
        tc::device_sync();

        tc::Timer t;
        for (long it = 0; it < iters; ++it) { sweep(a, b, N, N); std::swap(a, b); }
        tc::device_sync();
        const double secs = t.s();

        tc::check(bad_count(a.data_handle(), ncell) == 0,
                  "B: STRICT malloc_device pool under par_unseq");
        std::printf("       B: %8.3f s   %8.2f GB/s   (device-only pool)\n", secs,
                    tc::gbs(iters, N, N, secs));
        tc::pool_free(pool);
    }

    std::printf("\n  NOW COMPARE variant A against spike 05's native SYCL GB/s on the same\n"
                "  grid. Comparable => par_unseq really offloaded. An order of magnitude\n"
                "  slower => it silently ran on the CPU, which is a FAIL dressed as a PASS.\n\n");

    return tc::verdict("spike 05b");
}
