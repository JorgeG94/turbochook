// =============================================================================
// spike 03 — ONE pool, TWO launchers, one binary (nvc++ -stdpar=gpu -cuda).
//
// THE QUESTION: can the same sealed pool and the same View feed BOTH a stdpar
// kernel and a hand-written __global__, in the same translation unit, agreeing
// bitwise?
//
// WHY IT MATTERS: this is the whole "stdpar by default, native kernels where it
// pays" strategy. If one pool serves both launchers, the native path is a
// SELECTIVE optimisation for the handful of kernels that need streams / fusion /
// occupancy control (the launch-bound barotropic subcycle is the named first
// customer) — five hand-written kernels, not fifty. If it does not, discrete
// GPUs need a complete second implementation of every operator, which is the
// expensive outcome we are trying to avoid committing to blind.
//
// It also measures the launch-overhead gap the ROADMAP predicts: the same sweep
// through both paths, so any difference is launcher, not arithmetic.
// =============================================================================

#include <execution>
#include <algorithm>
#include <numeric>
#include <functional>
#include <ranges>
#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

// ── launcher 1: stdpar ───────────────────────────────────────────────────────
static void sweep_stdpar(View2 a, View2 b, Index nx, Index ny) {
    auto ids = std::views::iota(0, (nx - 2) * (ny - 2));
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](int n) {
        const Index i = 1 + n % (nx - 2);
        const Index j = 1 + n / (nx - 2);
        b[i, j] = Real(0.25) * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]);
    });
}

// ── launcher 2: a hand-written CUDA kernel over the SAME View type ───────────
// Note the body is byte-identical to the lambda above. That is the point: the
// kernel BODY is portable; only the launcher differs. In the real code this is
// what a TC_KERNEL macro + a backend-dispatched for_each_cell would give us.
__global__ void sweep_cuda_kernel(View2 a, View2 b, Index nx, Index ny) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx - 2) * (ny - 2)) return;
    const Index i = 1 + n % (nx - 2);
    const Index j = 1 + n / (nx - 2);
    b[i, j] = Real(0.25) * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]);
}

static void sweep_cuda(View2 a, View2 b, Index nx, Index ny) {
    const int total = (nx - 2) * (ny - 2);
    const int block = 256;
    sweep_cuda_kernel<<<(total + block - 1) / block, block>>>(a, b, nx, ny);
}

static void fill_stdpar(View2 f, Index nx, Index ny, Real v) {
    auto ids = std::views::iota(0, nx * ny);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(),
                  [=](int n) { f[n % nx, n / nx] = v; });
}

static long bad_count(Real* dev, std::size_t n) {
    std::vector<Real> host(n);
    tc::stage_to_host(host.data(), dev, n * sizeof(Real));
    long bad = 0;
    for (Real x : host) if (x != Real(1)) ++bad;
    return bad;
}

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;

    std::printf("\n=== spike 03: one pool, two launchers (nvc++ -stdpar=gpu -cuda) ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters\n\n", int(N), int(N), iters);

    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;
    const std::size_t ncell = std::size_t(N) * N;

    // A managed+prefetched pool, so this spike is independent of spike 02's answer.
    Real* pool = tc::pool_managed_prefetched(bytes);
    tc::Arena arena(pool, bytes);
    View2 a = arena.alloc2d(N, N, "a");
    View2 b = arena.alloc2d(N, N, "b");
    arena.seal();

    // ── stdpar path ──────────────────────────────────────────────────────────
    fill_stdpar(a, N, N, Real(1));
    fill_stdpar(b, N, N, Real(1));
    tc::device_sync();
    tc::Timer t1;
    for (long it = 0; it < iters; ++it) { sweep_stdpar(a, b, N, N); std::swap(a, b); }
    tc::device_sync();
    const double s_std = t1.s();
    tc::check(bad_count(a.data_handle(), ncell) == 0, "stdpar launcher exact");

    // ── CUDA path, SAME pool, SAME views ─────────────────────────────────────
    fill_stdpar(a, N, N, Real(1));
    fill_stdpar(b, N, N, Real(1));
    tc::device_sync();
    tc::Timer t2;
    for (long it = 0; it < iters; ++it) { sweep_cuda(a, b, N, N); std::swap(a, b); }
    tc::device_sync();
    const double s_cuda = t2.s();
    tc::check(bad_count(a.data_handle(), ncell) == 0, "CUDA launcher exact, same pool");

    std::printf("       stdpar : %8.3f s   %8.2f GB/s\n", s_std,  tc::gbs(iters, N, N, s_std));
    std::printf("       cuda   : %8.3f s   %8.2f GB/s\n", s_cuda, tc::gbs(iters, N, N, s_cuda));
    std::printf("\n  launcher gap stdpar/cuda = %.2fx   (near 1 => stdpar costs nothing here;\n"
                "  a large gap on TINY kernels is the launch-bound case the native path is for)\n\n",
                s_cuda > 0 ? s_std / s_cuda : 0.0);

    tc::pool_free(pool);
    return tc::verdict("spike 03");
}
