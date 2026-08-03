// =============================================================================
// spike 01 — MANAGED pool + advise + prefetch, driven by stdpar.
//
// THE QUESTION: can we own the pool ourselves (cudaMallocManaged, sized once,
// prefetched to the device at init) instead of leaning on -stdpar's allocator
// interception — and does it stay device-resident for the whole run?
//
// WHY IT MATTERS: if yes, the Arena becomes an explicit, sealed, device-resident
// pool that ALSO works under native CUDA/HIP, and we never depend on the
// compiler quietly replacing operator new. That is the row of the backend table
// we expect to carry the design.
//
// WHAT IT MEASURES:
//   A. prefetched managed pool, host never touches it during the loop   <- target
//   B. same pool, but the host reads one element every iteration        <- the footgun
//   The ratio B/A is the migration penalty. On a DISCRETE GPU expect a large
//   ratio (rakali/STATUS measured ~100-140x for a full-state copy; one element
//   per step is a page fault per step, so expect a big but smaller number).
//   On a COHERENT machine (pageableMemoryAccess=1) expect the ratio near 1.
//
// PASS looks like: both variants numerically exact (every cell == 1.0), variant
// A's bandwidth in the right order of magnitude for the card, and the ratio
// reported so we can read unified-vs-discrete off it.
// =============================================================================

#include <execution>
#include <algorithm>
#include <numeric>
#include <functional>
#include <ranges>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

// The Jacobi sweep. Conservative by construction on a constant field: init
// everything to 1.0 and every cell stays EXACTLY 1.0 forever, so correctness is
// a bitwise check with no tolerance and no drift, however many iterations run.
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

// Device-side exactness check: count cells != 1.0 without copying the field to
// the host (a host-side check would itself trigger the migration we are measuring).
static long count_bad(View2 f, Index nx, Index ny) {
    auto ids = std::views::iota(0, nx * ny);
    return std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0L,
                                 std::plus<long>{},
                                 [=](int n) { return f[n % nx, n / nx] == Real(1) ? 0L : 1L; });
}

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;

    std::printf("\n=== spike 01: managed pool + prefetch, stdpar ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters\n\n", int(N), int(N), iters);

    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;   // 2 fields + slack

    // ── A: prefetched managed pool, host hands off entirely ──────────────────
    double gA = 0;
    {
        Real* pool = tc::pool_managed_prefetched(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();                       // no allocation past setup, enforced

        fill(a, N, N, Real(1));
        fill(b, N, N, Real(1));
        tc::device_sync();

        tc::Timer t;
        for (long it = 0; it < iters; ++it) { sweep(a, b, N, N); std::swap(a, b); }
        tc::device_sync();
        const double secs = t.s();
        gA = tc::gbs(iters, N, N, secs);

        tc::check(count_bad(a, N, N) == 0, "A: field bitwise exact after the run");
        std::printf("       A: %8.3f s   %8.2f GB/s   (arena %zu MiB, sealed)\n",
                    secs, gA, arena.bytes_used() >> 20);
        tc::pool_free(pool);
    }

    // ── B: same, but the host reads ONE element per iteration ────────────────
    //     This is the stray-diagnostic-copy footgun in its smallest form.
    double gB = 0;
    {
        Real* pool = tc::pool_managed_prefetched(bytes);
        tc::Arena arena(pool, bytes);
        View2 a = arena.alloc2d(N, N, "a");
        View2 b = arena.alloc2d(N, N, "b");
        arena.seal();

        fill(a, N, N, Real(1));
        fill(b, N, N, Real(1));
        tc::device_sync();

        volatile Real sink = 0;
        tc::Timer t;
        for (long it = 0; it < iters; ++it) {
            sweep(a, b, N, N);
            std::swap(a, b);
            tc::device_sync();
            sink = a[N / 2, N / 2];          // <-- the host touch
        }
        tc::device_sync();
        const double secs = t.s();
        gB = tc::gbs(iters, N, N, secs);
        (void)sink;

        tc::check(count_bad(a, N, N) == 0, "B: field bitwise exact after the run");
        std::printf("       B: %8.3f s   %8.2f GB/s   (host touched 1 element/iter)\n",
                    secs, gB);
        tc::pool_free(pool);
    }

    std::printf("\n  migration penalty A/B = %.1fx"
                "   (>>1 => DISCRETE, data really moved; ~1 => COHERENT)\n\n",
                gB > 0 ? gA / gB : 0.0);

    return tc::verdict("spike 01");
}
