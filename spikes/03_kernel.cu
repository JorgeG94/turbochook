// =============================================================================
// spike 03 (device half) — the CUDA launcher, in its OWN translation unit.
//
// WHY SEPARATE: nvc++ -cuda gives CUDA *interoperability* (cudaMalloc, cudaMemcpy,
// linking libcudart) but is NOT a CUDA compiler — `__global__` + `<<<>>>` is
// rejected with "CUDA C++ compilation is unsupported in nvc++" (verified, nvc++
// 26.5). So the stdpar TU and the CUDA TU must be compiled by DIFFERENT compilers
// and linked, which is the real shape of the "stdpar by default, native where it
// pays" strategy:
//
//     nvc++ -stdpar=gpu   physics.cpp   ─┐
//                                         ├─ link ─> one binary, one pool
//     nvcc                launchers.cu  ─┘
//
// The boundary carries raw pointers + extents (a C ABI), never std::mdspan —
// nvcc's device C++23 support lags, and a POD boundary is what makes the HIP twin
// a drop-in replacement.
// =============================================================================

#include <cuda_runtime.h>

extern "C" {

// layout_left indexing by hand (i + nx*j) — identical addressing to Field2, so
// the two launchers walk memory the same way and the comparison is fair.
__global__ void k_sweep(const double* a, double* b, int nx, int ny) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx - 2) * (ny - 2)) return;
    const int i = 1 + n % (nx - 2);
    const int j = 1 + n / (nx - 2);
    b[i + nx * j] = 0.25 * (a[(i - 1) + nx * j] + a[(i + 1) + nx * j]
                          + a[i + nx * (j - 1)] + a[i + nx * (j + 1)]);
}

void tcs3_sweep_cuda(const double* a, double* b, int nx, int ny) {
    const int total = (nx - 2) * (ny - 2), block = 256;
    k_sweep<<<(total + block - 1) / block, block>>>(a, b, nx, ny);
}

} // extern "C"
