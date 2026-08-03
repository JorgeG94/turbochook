// =============================================================================
// spike 05d — why is PVC only delivering ~392 GB/s against a ~3.2 TB/s part?
//
// Spike 05 passed correctness everywhere but ran at roughly an eighth of nominal
// peak, where the NVIDIA runs comfortably exceeded nominal peak on cache reuse.
// That gap is unexplained, and no performance claim about Intel should be made
// until it is. FIVE variants, each isolating exactly one suspect:
//
//   A  range<1>, MdView, %/÷ index math   — the spike-05 baseline, for reference
//   B  nd_range<1> with an explicit workgroup — is the runtime picking badly?
//   C  range<2>, MdView                    — kills the integer % and ÷ per work-item
//   D  range<1>, RAW pointers              — isolates the tc::MdView fallback cost
//   E  A, but on ONE explicitly-partitioned tile — is the root device using both?
//
// SUSPECT E IS THE PRIME ONE. The node reports 6 devices with max_sub_devices=2,
// i.e. COMPOSITE hierarchy: a "device" is a whole 2-tile card. One queue on the
// root device may or may not implicitly scale across both tiles. If E matches A,
// we were only ever using one tile and the real per-tile efficiency is ~2x what
// spike 05 reported. If E is ~half of A, implicit scaling is working and the gap
// is elsewhere.
//
// SUSPECT D matters beyond Intel: Aurora's gcc 13.4 has no <mdspan>, so tc::MdView
// is LIVE on production hardware. If D is materially faster than A, that fallback
// is costing real throughput and needs work rather than sunsetting.
//
// Also worth trying from the shell, no rebuild needed:
//     ZE_FLAT_DEVICE_HIERARCHY=FLAT ./build/s05d
// which should make `visible gpus` report 12 and each device be a single tile.
// =============================================================================

#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

static long bad_count(const Real* dev, std::size_t n, sycl::queue& que) {
    std::vector<Real> host(n);
    que.memcpy(host.data(), dev, n * sizeof(Real)).wait();
    long bad = 0;
    for (Real x : host) if (x != Real(1)) ++bad;
    return bad;
}

// Every variant is timed the same way: allocate on `que`, fill to 1.0, sweep,
// verify bitwise, report. `sweep` does one Jacobi pass and must be async.
template <class Sweep>
static double time_variant(const char* label, sycl::queue& que, Index N, long iters, Sweep sweep) {
    const std::size_t ncell = std::size_t(N) * N;
    const std::size_t bytes = ncell * sizeof(Real) * 4;

    Real* pool = sycl::malloc_device<Real>(bytes / sizeof(Real), que);
    Real* pa = pool;
    Real* pb = pool + ((ncell + 15) & ~std::size_t(15));
    que.fill(pa, Real(1), ncell);
    que.fill(pb, Real(1), ncell);
    que.wait();

    tc::Timer t;
    for (long it = 0; it < iters; ++it) { sweep(que, pa, pb, N); std::swap(pa, pb); }
    que.wait();
    const double secs = t.s();

    const bool ok = bad_count(pa, ncell, que) == 0;
    const double bw = tc::gbs(iters, N, N, secs);
    tc::check(ok, label);
    std::printf("       %-40s %8.3f s   %8.2f GB/s\n", label, secs, bw);
    sycl::free(pool, que);
    return bw;
}

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;
    const int   wg    = argc > 3 ? std::atoi(argv[3])        : 256;

    std::printf("\n=== spike 05d: PVC performance probe ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters, workgroup %d\n\n", int(N), int(N), iters, wg);

    sycl::queue& q = tc::q();

    // ── A: the spike-05 baseline. MdView subscript + %/÷ per work-item. ──────
    const double bwA = time_variant("A range<1>  MdView  %/div", q, N, iters,
        [](sycl::queue& que, Real* a, Real* b, Index nx) {
            const View2 va = tc::make_view(a, nx, nx), vb = tc::make_view(b, nx, nx);
            que.parallel_for(sycl::range<1>(std::size_t(nx - 2) * (nx - 2)), [=](sycl::id<1> k) {
                const Index n = Index(k[0]);
                const Index i = 1 + n % (nx - 2);
                const Index j = 1 + n / (nx - 2);
                vb[i, j] = Real(0.25) * (va[i - 1, j] + va[i + 1, j] + va[i, j - 1] + va[i, j + 1]);
            });
        });

    // ── B: explicit workgroup. Does the runtime's default choice hurt? ───────
    time_variant("B nd_range<1> explicit workgroup", q, N, iters,
        [wg](sycl::queue& que, Real* a, Real* b, Index nx) {
            const View2 va = tc::make_view(a, nx, nx), vb = tc::make_view(b, nx, nx);
            const std::size_t total = std::size_t(nx - 2) * (nx - 2);
            const std::size_t pad   = ((total + wg - 1) / wg) * wg;   // nd_range must divide evenly
            que.parallel_for(sycl::nd_range<1>(sycl::range<1>(pad), sycl::range<1>(wg)),
                             [=](sycl::nd_item<1> it) {
                const std::size_t g = it.get_global_id(0);
                if (g >= total) return;
                const Index n = Index(g);
                const Index i = 1 + n % (nx - 2);
                const Index j = 1 + n / (nx - 2);
                vb[i, j] = Real(0.25) * (va[i - 1, j] + va[i + 1, j] + va[i, j - 1] + va[i, j + 1]);
            });
        });

    // ── C: 2-D range. No integer division per work-item at all. Note SYCL's
    //      range<2> is row-major in the LAST index, so dim 1 is the fast one —
    //      map it to i to keep layout_left coalescing. ────────────────────────
    time_variant("C range<2>  MdView  no div", q, N, iters,
        [](sycl::queue& que, Real* a, Real* b, Index nx) {
            const View2 va = tc::make_view(a, nx, nx), vb = tc::make_view(b, nx, nx);
            que.parallel_for(sycl::range<2>(std::size_t(nx - 2), std::size_t(nx - 2)),
                             [=](sycl::id<2> id) {
                const Index j = 1 + Index(id[0]);
                const Index i = 1 + Index(id[1]);
                vb[i, j] = Real(0.25) * (va[i - 1, j] + va[i + 1, j] + va[i, j - 1] + va[i, j + 1]);
            });
        });

    // ── D: raw pointers, no MdView. Isolates the fallback's index arithmetic,
    //      which is LIVE on Aurora because gcc 13.4 has no <mdspan>. ──────────
    time_variant("D range<2>  raw ptr  no div", q, N, iters,
        [](sycl::queue& que, Real* a, Real* b, Index nx) {
            que.parallel_for(sycl::range<2>(std::size_t(nx - 2), std::size_t(nx - 2)),
                             [=](sycl::id<2> id) {
                const Index j = 1 + Index(id[0]);
                const Index i = 1 + Index(id[1]);
                b[i + nx * j] = Real(0.25) * (a[(i - 1) + nx * j] + a[(i + 1) + nx * j]
                                            + a[i + nx * (j - 1)] + a[i + nx * (j + 1)]);
            });
        });

    // ── E: THE PRIME SUSPECT. Explicitly partition the card into tiles and run
    //      variant A on ONE of them. E ≈ A  =>  the root device was only ever
    //      using one tile. E ≈ A/2  =>  implicit scaling works, look elsewhere. ─
    try {
        auto tiles = q.get_device().create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
                sycl::info::partition_affinity_domain::next_partitionable);
        std::printf("\n  card partitions into %zu tiles; running variant A on tile 0 only\n",
                    tiles.size());
        sycl::queue qt{tiles[0]};
        const double bwE = time_variant("E range<1> MdView, ONE TILE", qt, N, iters,
            [](sycl::queue& que, Real* a, Real* b, Index nx) {
                const View2 va = tc::make_view(a, nx, nx), vb = tc::make_view(b, nx, nx);
                que.parallel_for(sycl::range<1>(std::size_t(nx - 2) * (nx - 2)), [=](sycl::id<1> k) {
                    const Index n = Index(k[0]);
                    const Index i = 1 + n % (nx - 2);
                    const Index j = 1 + n / (nx - 2);
                    vb[i, j] = Real(0.25) * (va[i - 1, j] + va[i + 1, j] + va[i, j - 1] + va[i, j + 1]);
                });
            });
        std::printf("\n  one-tile / whole-card = %.2f\n"
                    "     ~1.0 => the root device was using ONE tile all along\n"
                    "             (real per-tile efficiency is ~2x what spike 05 reported)\n"
                    "     ~0.5 => implicit scaling works; the gap is elsewhere\n",
                    bwA > 0 ? bwE / bwA : 0.0);
    } catch (const sycl::exception& e) {
        std::printf("\n  sub-device partition unavailable: %s\n"
                    "  (try  ZE_FLAT_DEVICE_HIERARCHY=FLAT ./build/s05d  instead —\n"
                    "   `visible gpus` should then report 12, one per tile)\n", e.what());
    }

    std::printf("\n");
    return tc::verdict("spike 05d");
}
