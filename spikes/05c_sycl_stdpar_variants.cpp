// =============================================================================
// spike 05c — is a stdpar-SHAPED API reachable on Intel at all?
//
// Spike 05b established that `std::for_each(par_unseq, iota.begin(), iota.end(), f)`
// — the idiom parallel.hpp calls "the reliably-offloading shape" — does NOT compile
// under oneDPL. It routes into oneapi::dpl::__pattern_walk1, which for a
// non-contiguous iterator wraps the range in a sycl::buffer and calls
// set_final_data() to copy results BACK INTO the iterator. iota_view's iterator is a
// read-only proxy, so it dies in libstdc++ with "expression is not assignable".
//
// THE QUESTION HERE: was that about `iota` specifically, or about the whole
// std::-algorithm path on Intel? The answer decides whether the launcher chokepoint
// can present ONE spelling across all backends, or whether Intel simply uses the
// native SYCL launcher (which spike 05 already proved works perfectly).
//
// FOUR VARIANTS, each isolating one thing:
//   1. std::for_each + oneapi::dpl::counting_iterator   — is it iota, or all proxies?
//   2. std::for_each + a real USM int* index array      — contiguous, no proxy at all
//   3. oneapi::dpl::for_each + counting_iterator        — bypass std:: entirely
//   4. native q.parallel_for                            — the baseline that works
//
// Variants 1 and 2 need -fsycl-pstl-offload=gpu (build target s05c_pstl); 3 and 4
// need only -fsycl (target s05c). A BUILD FAILURE OF ANY VARIANT IS A RESULT —
// they are separately #ifdef'd so one failure does not hide the others.
//
// Numbers comparable to spike 05: same grid, same iters, same 5-doubles/cell.
// =============================================================================

// oneDPL asks that its headers precede the standard ones.
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include <execution>
#include <algorithm>
#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

// One kernel body, four launchers. If the body needs editing per launcher, the
// abstraction is not real — so it is written once, here, as a plain callable.
struct Sweep {
    View2 a, b; Index nx, ny;
    void operator()(int n) const {
        const Index i = 1 + n % (nx - 2);
        const Index j = 1 + n / (nx - 2);
        b[i, j] = Real(0.25) * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]);
    }
};
struct Fill {
    View2 f; Index nx; Real v;
    void operator()(int n) const { f[n % nx, n / nx] = v; }
};

static void fill_native(View2 f, Index nx, Index ny, Real v) {
    tc::q().parallel_for(sycl::range<1>(std::size_t(nx) * ny), [=](sycl::id<1> k) {
        Fill{f, nx, v}(int(k[0]));
    });
    tc::device_sync();
}

static long bad_count(const Real* dev, std::size_t n) {
    std::vector<Real> host(n);
    tc::stage_to_host(host.data(), dev, n * sizeof(Real));
    long bad = 0;
    for (Real x : host) if (x != Real(1)) ++bad;
    return bad;
}

// Run `launch` for `iters` sweeps over a fresh field, verify, report.
template <class Launch>
static void run_variant(const char* label, Index N, long iters, Launch launch) {
    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;
    const std::size_t ncell = std::size_t(N) * N;

    Real* pool = tc::pool_device(bytes);
    tc::Arena arena(pool, bytes);
    View2 a = arena.alloc2d(N, N, "a");
    View2 b = arena.alloc2d(N, N, "b");
    arena.seal();

    fill_native(a, N, N, Real(1));
    fill_native(b, N, N, Real(1));

    tc::Timer t;
    for (long it = 0; it < iters; ++it) { launch(Sweep{a, b, N, N}, (N - 2) * (N - 2)); std::swap(a, b); }
    tc::device_sync();
    const double secs = t.s();

    const bool ok = bad_count(a.data_handle(), ncell) == 0;
    tc::check(ok, label);
    std::printf("       %-34s %8.3f s   %8.2f GB/s\n", label, secs, tc::gbs(iters, N, N, secs));
    tc::pool_free(pool);
}

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;

    std::printf("\n=== spike 05c: stdpar-shaped launchers on Intel ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters\n", int(N), int(N), iters);
#if defined(TC_TRY_PSTL)
    std::printf("  built WITH -fsycl-pstl-offload=gpu\n\n");
#else
    std::printf("  built WITHOUT pstl-offload (variants 1-2 compiled out)\n\n");
#endif

#if defined(TC_TRY_PSTL)
    // ── 1. std::for_each over oneDPL's counting_iterator ─────────────────────
    //    A random-access counting iterator oneDPL itself provides. If THIS works,
    //    the 05b failure was about iota_view's proxy specifically, and the
    //    chokepoint can keep one std:: spelling with a per-backend index source.
    run_variant("1 std::for_each + counting_iter", N, iters, [](Sweep s, int n) {
        auto first = oneapi::dpl::counting_iterator<int>(0);
        std::for_each(std::execution::par_unseq, first, first + n, s);
    });

    // ── 2. std::for_each over a REAL contiguous USM int array ────────────────
    //    No proxy iterator anywhere — a plain pointer range oneDPL cannot object
    //    to. Costs one extra 4-byte read per work-item, so if it is the only
    //    thing that works, the price is visible in the GB/s column.
    {
        const int n = (N - 2) * (N - 2);
        int* idx = sycl::malloc_device<int>(std::size_t(n), tc::q());
        tc::q().parallel_for(sycl::range<1>(std::size_t(n)),
                             [=](sycl::id<1> k) { idx[k[0]] = int(k[0]); }).wait();
        run_variant("2 std::for_each + USM int*", N, iters, [=](Sweep s, int m) {
            std::for_each(std::execution::par_unseq, idx, idx + m, s);
        });
        sycl::free(idx, tc::q());
    }
#endif

    // ── 3. oneDPL directly, bound to OUR queue ───────────────────────────────
    //    Bypasses the std:: entry point and the pstl-offload flag entirely. This
    //    is the "Intel dialect" answer: portable in shape, not in spelling.
    run_variant("3 dpl::for_each + counting_iter", N, iters, [](Sweep s, int n) {
        auto policy = oneapi::dpl::execution::make_device_policy(tc::q());
        auto first  = oneapi::dpl::counting_iterator<int>(0);
        oneapi::dpl::for_each(policy, first, first + n, s);
    });

    // ── 4. native SYCL — the baseline spike 05 already proved ────────────────
    run_variant("4 native q.parallel_for", N, iters, [](Sweep s, int n) {
        tc::q().parallel_for(sycl::range<1>(std::size_t(n)),
                             [=](sycl::id<1> k) { s(int(k[0])); });
    });

    std::printf("\n  Read the GB/s column against spike 05's 391 GB/s. A variant that\n"
                "  builds but runs an order of magnitude slower did NOT offload.\n\n");

    return tc::verdict("spike 05c");
}
