// =============================================================================
// spike 05 — SYCL / Level Zero on Aurora (PVC), NATIVE queue.parallel_for.
//
// The third backend, and the one that makes the abstraction honest: CUDA and HIP
// are near-isomorphic (hipify is essentially textual), so supporting both proves
// very little. SYCL is genuinely different — explicit queue, no chevrons, its own
// USM model — so if the same `common.hpp` pool verbs and the same Jacobi kernel
// body work here unchanged, the device seam is real.
//
// THIS FILE USES NO EXOTIC FLAGS: plain `icpx -fsycl`. It is therefore expected to
// work regardless of what spike 05b finds about PSTL offload, and it is the
// fallback data point if 05b fails.
//
// QUESTIONS:
//   1. Does the whole pool/staging vocabulary port to SYCL unchanged? (it should
//      — malloc_device/malloc_shared/prefetch/memcpy map 1:1)
//   2. Is PVC coherent or discrete? `usm_system_allocations` is the portable
//      analogue of CUDA's pageableMemoryAccess. PREDICTION: discrete (0), so the
//      A/B ratio should look like the V100's ~4x rather than the GH200's ~1.1x.
//      A prediction is not a result — that is the whole point of running it.
//   3. Does std::mdspan survive capture into a SYCL kernel? (trivially copyable,
//      so it should; if not, the MdView fallback in common.hpp takes over and
//      that is worth knowing before it is load-bearing.)
//   4. Does a strict malloc_device pool work, with the host unable to touch it?
//
// Numbers are directly comparable to spikes 01/02/03: same grid, same iteration
// count, same 5-doubles-per-cell bandwidth convention, same bitwise-exact check
// (a constant field under a Jacobi average stays exactly 1.0 forever).
// =============================================================================

#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

// The kernel body is IDENTICAL to the stdpar and CUDA versions in spikes 01-03.
// Only the launcher differs — which is the claim the device layer rests on.
static void sweep(View2 a, View2 b, Index nx, Index ny) {
    const Index total = (nx - 2) * (ny - 2);
    tc::q().parallel_for(sycl::range<1>(std::size_t(total)), [=](sycl::id<1> idx) {
        const Index n = Index(idx[0]);
        const Index i = 1 + n % (nx - 2);
        const Index j = 1 + n / (nx - 2);
        b[i, j] = Real(0.25) * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]);
    });
}

static void fill(View2 f, Index nx, Index ny, Real v) {
    tc::q().parallel_for(sycl::range<1>(std::size_t(nx) * ny), [=](sycl::id<1> idx) {
        const Index n = Index(idx[0]);
        f[n % nx, n / nx] = v;
    });
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

    std::printf("\n=== spike 05: SYCL native queue.parallel_for (Aurora/PVC) ===\n");
    tc::report_device();
    std::printf("  mdspan backend    : %s\n", TC_SPIKE_STD_MDSPAN ? "std::mdspan" : "tc::MdView fallback");
    std::printf("  grid %d x %d, %ld iters\n\n", int(N), int(N), iters);

    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;
    const std::size_t ncell = std::size_t(N) * N;

    // ── A: malloc_shared + prefetch, host hands off ──────────────────────────
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

        tc::check(bad_count(a.data_handle(), ncell) == 0, "A: shared pool, field bitwise exact");
        std::printf("       A: %8.3f s   %8.2f GB/s   (arena %zu MiB, sealed)\n",
                    secs, gA, arena.bytes_used() >> 20);
        tc::pool_free(pool);
    }

    // ── B: same, but the host reads one element per iteration ────────────────
    double gB = 0;
    {
        Real* pool = tc::pool_managed_prefetched(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();

        fill(a, N, N, Real(1)); fill(b, N, N, Real(1));
        tc::device_sync();

        volatile Real sink = 0;
        tc::Timer t;
        for (long it = 0; it < iters; ++it) {
            sweep(a, b, N, N); std::swap(a, b);
            tc::device_sync();
            sink = a[N / 2, N / 2];              // <-- the host touch (legal on shared USM)
        }
        tc::device_sync();
        const double secs = t.s();
        gB = tc::gbs(iters, N, N, secs);
        (void)sink;

        tc::check(bad_count(a.data_handle(), ncell) == 0, "B: shared pool, field bitwise exact");
        std::printf("       B: %8.3f s   %8.2f GB/s   (host touched 1 element/iter)\n", secs, gB);
        tc::pool_free(pool);
    }

    std::printf("\n  migration penalty A/B = %.1fx"
                "   (compare: V100 4.1x discrete, GH200 1.1x coherent)\n\n",
                gB > 0 ? gA / gB : 0.0);

    // ── C: STRICT malloc_device pool — host cannot dereference it at all ──────
    {
        Real* pool = tc::pool_device(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();

        fill(a, N, N, Real(1)); fill(b, N, N, Real(1));   // setup on DEVICE only
        tc::device_sync();

        tc::Timer t;
        for (long it = 0; it < iters; ++it) { sweep(a, b, N, N); std::swap(a, b); }
        tc::device_sync();
        const double secs = t.s();

        tc::check(bad_count(a.data_handle(), ncell) == 0,
                  "C: strict malloc_device pool, exact via explicit staging");
        std::printf("       C: %8.3f s   %8.2f GB/s   (device-only pool)\n\n",
                    secs, tc::gbs(iters, N, N, secs));
        tc::pool_free(pool);
    }

    return tc::verdict("spike 05");
}
