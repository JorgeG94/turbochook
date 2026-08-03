// =============================================================================
// spikes/alpha/main.cpp — backend-agnostic BY CONSTRUCTION.
//
// There is not one #ifdef below this line. If a backend ever needs one here,
// device::do_concurrent has failed at its only job and THAT is the finding.
//
// WHY CORRECTNESS PROVES OFFLOAD, with no timing heuristic:
//
//   The pointers come from cudaMalloc / hipMalloc / sycl::malloc_device -- NOT
//   managed, NOT migratable. The host cannot dereference them. So if a
//   do_concurrent writes the right values and a device->host copy reads them
//   back correctly, the loop body MUST have run on the device: host execution
//   over that pointer would fault or produce garbage, not the right answer.
//
//   A "PASS" here therefore means BOTH "stdpar accepted our hand-managed device
//   memory" AND "it actually offloaded". On the host build it just means the
//   algorithms work, which is why the host target is the free sanity check and
//   not evidence of anything.
// =============================================================================
#include "device.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using tc::Index;
namespace dev = tc::device;

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    std::printf("  [%s] %-34s %s\n", ok ? "PASS" : "FAIL", what, detail);
    if (!ok) ++failures;
}

} // namespace

int main(int argc, char** argv) {
    const Index nx = (argc > 1) ? Index(std::atoi(argv[1])) : 512;
    const Index ny = (argc > 2) ? Index(std::atoi(argv[2])) : 256;
    const Index n  = nx * ny;

    std::printf("\n=== alpha: device::do_concurrent over hand-managed memory ===\n");
    std::printf("  backend           : %s\n", TC_BACKEND_NAME);
    std::printf("  parallel algorithm: %s\n", TC_PAR_NAME);
    std::printf("  grid              : %d x %d  (%d cells)\n", nx, ny, n);
    dev::initialize();
    std::printf("\n");

    // ---- allocate: OURS, explicit, no managed memory anywhere ---------------
    double* a = dev::alloc<double>(std::size_t(n));
    double* b = dev::alloc<double>(std::size_t(n));
    if (!a || !b) {
        std::printf("  [FAIL] device allocation returned null -- nothing else can run\n");
        return 2;
    }
    std::printf("  [ok ] allocated 2 x %.1f MiB of device memory\n",
                double(n) * sizeof(double) / (1024.0 * 1024.0));

    // ---- 1. elementwise fill -----------------------------------------------
    dev::do_concurrent(n, [=] (Index t) { a[t] = 2.0 * double(t) + 1.0; });
    dev::sync();

    std::vector<double> h(std::size_t(n), -12345.0);
    dev::copy_to_host(h.data(), a, std::size_t(n));

    bool fill_ok = true;
    for (Index t = 0; t < n && fill_ok; ++t)
        if (h[std::size_t(t)] != 2.0 * double(t) + 1.0) fill_ok = false;
    check(fill_ok, "elementwise fill",
          fill_ok ? "stdpar wrote to hand-allocated device memory" : "wrong values read back");

    // ---- 2. 2-D flattened indexing + neighbour reads ------------------------
    // The launcher owns flattening (i fastest), so the body sees (i, j). A
    // wrong i/j split still produces a defined answer -- just the wrong one --
    // which is exactly why this is checked rather than assumed.
    dev::do_concurrent(n, [=] (Index t) {
        const Index i = t % nx;
        const Index j = t / nx;
        const bool interior = (i > 0 && i < nx - 1 && j > 0 && j < ny - 1);
        b[t] = interior ? (a[t - 1] + a[t + 1] + a[t - nx] + a[t + nx] - 4.0 * a[t])
                        : 0.0;
    });
    dev::sync();
    dev::copy_to_host(h.data(), b, std::size_t(n));

    // a is linear in t with slope 2 in i and 2*nx in j, so the 5-point
    // Laplacian of it is identically zero on the interior.
    bool stencil_ok = true;
    for (Index j = 1; j < ny - 1 && stencil_ok; ++j)
        for (Index i = 1; i < nx - 1 && stencil_ok; ++i)
            if (std::fabs(h[std::size_t(j * nx + i)]) > 1e-9) stencil_ok = false;
    check(stencil_ok, "2-D flattened stencil",
          stencil_ok ? "neighbour reads + i/j split correct" : "Laplacian of a linear field != 0");

    // ---- 3. reduction -------------------------------------------------------
    const double got  = dev::reduce<double>(n, 0.0, [=] (Index t) { return a[t]; });
    const double want = double(n) * double(n);          // sum of 2t+1, t=0..n-1
    const bool   red_ok = std::fabs(got - want) <= 1e-6 * want;
    char buf[128];
    std::snprintf(buf, sizeof buf, "got %.6g want %.6g", got, want);
    check(red_ok, "reduction (transform_reduce)", buf);

    // ---- 4. the same lambda body over both precisions ------------------------
    float* f = dev::alloc<float>(std::size_t(n));
    if (f) {
        dev::do_concurrent(n, [=] (Index t) { f[t] = float(t) * 0.5f; });
        dev::sync();
        std::vector<float> hf(std::size_t(n), -1.0f);
        dev::copy_to_host(hf.data(), f, std::size_t(n));
        bool prec_ok = true;
        for (Index t = 0; t < n && prec_ok; ++t)
            if (hf[std::size_t(t)] != float(t) * 0.5f) prec_ok = false;
        check(prec_ok, "float as well as double", "same shape, both precisions");
        dev::free_(f);
    } else {
        check(false, "float as well as double", "allocation failed");
    }

    dev::free_(a);
    dev::free_(b);

    std::printf("\n  RESULT: %s  [%s]\n\n",
                failures == 0 ? "ALL PASS" : "FAILURES",
                TC_BACKEND_NAME);

    if (failures == 0) {
#if defined(TC_BACKEND_HOST)
        std::printf("  => the SHAPE compiles and is correct. This says NOTHING about\n"
                    "     offload or about hand-managed device memory -- there is no\n"
                    "     device here. Run a vendor target for that.\n\n");
#else
        std::printf("  => stdpar accepted explicitly-allocated device memory on this\n"
                    "     toolchain, AND the kernels really ran on the device: the\n"
                    "     pointers are not host-dereferenceable, so a host-side\n"
                    "     execution could not have produced these values.\n\n");
#endif
    }
    return failures == 0 ? 0 : 1;
}
