// =============================================================================
// spike 03b (device half) — the "gcc host + nvcc kernels" path.
//
// This file is the ONLY thing nvcc compiles, and it is the prototype of the
// `device_alloc` layer: every CUDA call in the project lives behind this surface,
// exposed as a plain C ABI so the host side can be built by g++, clang++, or
// nvc++ without knowing CUDA exists. Swap this file for a HIP twin and the host
// side does not change by one character — that is the portability claim, tested
// rather than asserted.
//
// Deliberately NO std::mdspan across this boundary: nvcc's device C++23 support
// lags, and the whole point of a C ABI seam is that it carries only pointers and
// PODs. The host side reconstructs whatever view type it likes over the pointer.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" {

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "  CUDA FAIL: %s -> %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// ── the pool: one allocation, device-only, never grows ───────────────────────
void* tcs_pool_alloc(size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}
void tcs_pool_free(void* p) { cudaFree(p); }

void tcs_to_host(double* dst, const void* src, size_t bytes) {
    ck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "D2H");
}
void tcs_sync() { ck(cudaDeviceSynchronize(), "sync"); }

void tcs_device_report() {
    int dev = 0; cudaDeviceProp pr{};
    ck(cudaGetDevice(&dev), "getDevice");
    ck(cudaGetDeviceProperties(&pr, dev), "getDeviceProperties");
    std::printf("  device            : %s (cc%d%d)\n", pr.name, pr.major, pr.minor);
    std::printf("  pageableMemAccess : %d\n", pr.pageableMemoryAccess);
}

// ── the kernels — layout_left indexing by hand (i + nx*j), matching Field2 ───
__global__ void k_fill(double* f, int nx, int ny, double v) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < nx * ny) f[n] = v;
}

__global__ void k_sweep(const double* a, double* b, int nx, int ny) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx - 2) * (ny - 2)) return;
    const int i = 1 + n % (nx - 2);
    const int j = 1 + n / (nx - 2);
    b[i + nx * j] = 0.25 * (a[(i - 1) + nx * j] + a[(i + 1) + nx * j]
                          + a[i + nx * (j - 1)] + a[i + nx * (j + 1)]);
}

void tcs_fill(double* f, int nx, int ny, double v) {
    const int total = nx * ny, block = 256;
    k_fill<<<(total + block - 1) / block, block>>>(f, nx, ny, v);
}

void tcs_sweep(const double* a, double* b, int nx, int ny) {
    const int total = (nx - 2) * (ny - 2), block = 256;
    k_sweep<<<(total + block - 1) / block, block>>>(a, b, nx, ny);
}

} // extern "C"
