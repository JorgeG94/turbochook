// =============================================================================
// spike 02 — DEVICE-ONLY pool (cudaMalloc) driven by stdpar.
//
// THE QUESTION: may a stdpar kernel capture, by value, a pointer into memory the
// host cannot touch? Semantically there is nothing to map — the pointer's value
// IS a device address, so the dereference is an ordinary device load, exactly as
// if it had been handed to a __global__. The open part is whether nvc++ ALLOWS
// the configuration: -stdpar=gpu is documented around managed/unified memory
// (hence its "no stack, no global variables in kernels" restriction), so
// -gpu=mem:separate may be rejected, or may work, or may build and fault.
//
// WHY IT MATTERS: if this works we can drop managed memory entirely and the
// type system alone enforces the host/device split. If it does NOT work,
// nothing above the Arena changes — spike 01's managed+prefetch pool carries the
// design instead. So this is the nice-to-have, not the load-bearing one.
//
// A COMPILE FAILURE OR A CLEAN RUNTIME ERROR IS A RESULT, NOT A BUG. Capture the
// exact compiler/driver message — that message is the answer.
//
// NOTE: the host may not initialise or read the field directly here. Setup runs
// as a device kernel; verification stages back through an explicit cudaMemcpy —
// which is precisely the discipline a device-only pool would impose on every
// diagnostic, NetCDF write, restart, and unit test.
// =============================================================================

#include <execution>
#include <algorithm>
#include <numeric>
#include <functional>
#include <ranges>
#include <vector>
#include "common.hpp"

using tc::Real; using tc::Index; using tc::View2;

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

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 1024;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 100;

    std::printf("\n=== spike 02: device-only pool (cudaMalloc) + stdpar ===\n");
    tc::report_device();
    std::printf("  grid %d x %d, %ld iters\n\n", int(N), int(N), iters);

    const std::size_t bytes = std::size_t(N) * N * sizeof(Real) * 4;

    Real* pool = tc::pool_device(bytes);          // host CANNOT dereference this
    tc::Arena arena(pool, bytes);
    View2 a = arena.alloc2d(N, N, "a");
    View2 b = arena.alloc2d(N, N, "b");
    arena.seal();

    // Setup on the DEVICE — there is no host path to this memory.
    fill(a, N, N, Real(1));
    fill(b, N, N, Real(1));
    tc::device_sync();

    tc::Timer t;
    for (long it = 0; it < iters; ++it) { sweep(a, b, N, N); std::swap(a, b); }
    tc::device_sync();
    const double secs = t.s();

    // Explicit staging — the ONLY way to look at the data. This is what every
    // host consumer would have to do under a device-only pool.
    std::vector<Real> host(std::size_t(N) * N);
    tc::stage_to_host(host.data(), a.data_handle(), std::size_t(N) * N * sizeof(Real));

    long bad = 0;
    for (std::size_t k = 0; k < host.size(); ++k) if (host[k] != Real(1)) ++bad;

    tc::check(bad == 0, "stdpar kernel wrote correct values through a device pointer");
    std::printf("       %8.3f s   %8.2f GB/s\n\n", secs, tc::gbs(iters, N, N, secs));

    tc::pool_free(pool);
    return tc::verdict("spike 02");
}
