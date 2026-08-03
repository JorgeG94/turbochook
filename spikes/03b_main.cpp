// =============================================================================
// spike 03b (host half) — compiled by g++ (or clang++), with NO CUDA headers.
//
// THE QUESTION: does the "gcc host + nvcc device layer" split actually hold, with
// the host side ignorant of CUDA entirely?
//
// WHY IT MATTERS: it is the fallback that makes the project independent of NVHPC.
// If stdpar-on-GPU is only available through nvc++ (and on AMD only through
// clang --hipstdpar, on Intel only through icpx), then this path is what lets the
// SAME host code target a GPU with an ordinary system compiler — the device layer
// is the only thing that gets swapped. It is also the honest test of whether the
// `device_alloc` seam is thin enough: everything CUDA below the line, nothing
// above it.
//
// Note what is ABSENT here: no <cuda_runtime.h>, no __global__, no nvcc. Just an
// extern "C" surface. That absence IS the result.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <chrono>
#include <utility>

// The entire device layer, as seen by host code. Swap the .cu for a .hip twin and
// this declaration block does not change.
extern "C" {
void*  tcs_pool_alloc(std::size_t bytes);
void   tcs_pool_free(void* p);
void   tcs_to_host(double* dst, const void* src, std::size_t bytes);
void   tcs_sync();
void   tcs_device_report();
void   tcs_fill(double* f, int nx, int ny, double v);
void   tcs_sweep(const double* a, double* b, int nx, int ny);
}

int main(int argc, char** argv) {
    const int  N     = argc > 1 ? std::atoi(argv[1]) : 2048;
    const long iters = argc > 2 ? std::atol(argv[2]) : 200;

    std::printf("\n=== spike 03b: g++ host + nvcc device layer (no CUDA above the seam) ===\n");
    tcs_device_report();
    std::printf("  grid %d x %d, %ld iters\n\n", N, N, iters);

    const std::size_t ncell = std::size_t(N) * N;
    const std::size_t bytes = ncell * sizeof(double);

    // One pool, carved by hand into two fields — the Arena in its crudest form.
    char*   pool = static_cast<char*>(tcs_pool_alloc(bytes * 2));
    double* a    = reinterpret_cast<double*>(pool);
    double* b    = reinterpret_cast<double*>(pool + bytes);

    tcs_fill(a, N, N, 1.0);
    tcs_fill(b, N, N, 1.0);
    tcs_sync();

    const auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < iters; ++it) { tcs_sweep(a, b, N, N); std::swap(a, b); }
    tcs_sync();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<double> host(ncell);
    tcs_to_host(host.data(), a, bytes);
    long bad = 0;
    for (double x : host) if (x != 1.0) ++bad;

    const double gbs = double(iters) * double(ncell) * 5.0 * sizeof(double) / secs / 1e9;
    std::printf("  [%s] host built without CUDA, device layer ran correctly\n",
                bad == 0 ? "PASS" : "FAIL");
    std::printf("       %8.3f s   %8.2f GB/s\n\n", secs, gbs);

    tcs_pool_free(pool);
    std::printf("  ---- spike 03b: %s ----\n\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
