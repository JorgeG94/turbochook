#pragma once
// =============================================================================
// lib/device_alloc.hpp — the arena's backing store, per backend.
//
// WHY THIS EXISTS. The arena used to be a std::vector<std::byte>, and that works
// on exactly one toolchain: `nvc++ -stdpar` silently promotes the whole heap to
// MANAGED memory, so ordinary host allocations are device-reachable. Nobody else
// does that for you:
//
//   nvc++ -stdpar   heap is promoted to managed      -> vector worked
//   --hipstdpar     needs HMM/XNACK or interposition -> MEMORY ACCESS FAULT
//   oneDPL/SYCL     no promotion mechanism at all    -> runtime failure
//
// Both non-NVIDIA failures were observed on real hardware, and both had the same
// root cause. It is the concrete case
// for CONTRACT_MEMORY §0.0's second line: the language expresses parallelism, but
// WE express memory. Leaning on a vendor to fix your allocations is precisely the
// dependency that does not port.
//
// WHY MANAGED AND NOT DEVICE-ONLY. The demos read fields from the HOST -- the PPM
// render loop does `e[i, j]` directly on arena memory (demo_baroclinic.cpp:161).
// Device-only storage is the eventual design (with an explicit mirror()), but it
// would break every existing consumer today. Managed keeps one pointer valid on
// both sides, which is the drop-in. Spike 01 measured managed + prefetch to stay
// device-resident, so this is not a performance concession either.
// =============================================================================

#include <cstddef>
#include <cstdlib>
#include <new>

// Keyed on the EXPLICIT backend define from CMake, never on a compiler macro.
// __NVCOMPILER is also true for -stdpar=multicore, which has no GPU and must take
// the host branch; __HIPCC__ is true for host TUs under hipcc. Anything not named
// here (multicore, off) falls through to ordinary aligned host memory, which is
// correct for both.
#if   defined(TC_STDPAR_SYCL)
#  include <sycl/sycl.hpp>
#elif defined(TC_STDPAR_HIP)
#  include <hip/hip_runtime.h>
#elif defined(TC_STDPAR_CUDA)
#  include <cuda_runtime.h>
#endif

namespace tc::detail {

#if defined(TC_STDPAR_SYCL)
// THE queue for the process -- allocation AND launch must share it.
//
// SYCL USM is CONTEXT-BOUND: memory from sycl::malloc_shared(q1) is not valid in a
// kernel submitted to q2 unless the two queues share a context, and two separately
// constructed queues from gpu_selector_v need not. numerics/parallel.hpp therefore
// launches on THIS queue rather than making its own -- an earlier revision had one
// each, which is invalid on Intel and harmless everywhere else, i.e. exactly the
// bug a single-vendor CI cannot see.
inline sycl::queue& device_queue() {
    static sycl::queue q{sycl::gpu_selector_v};
    return q;
}
#endif

// Managed/shared bytes, host- AND device-addressable. Returns nullptr on failure;
// the caller reports, because it knows the requested size.
inline void* managed_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
#if   defined(TC_STDPAR_SYCL)
    return sycl::malloc_shared(bytes, device_queue());
#elif defined(TC_STDPAR_HIP)
    void* p = nullptr;
    return (hipMallocManaged(&p, bytes) == hipSuccess) ? p : nullptr;
#elif defined(TC_STDPAR_CUDA)
    // nvc++ -stdpar would promote a plain heap allocation anyway; going through
    // cudaMallocManaged makes that EXPLICIT rather than a compiler favour, and
    // keeps all four backends on the same code path.
    void* p = nullptr;
    return (cudaMallocManaged(&p, bytes) == cudaSuccess) ? p : nullptr;
#else
    // Host build: ordinary aligned memory. 128 B to match the arena's field
    // alignment so the very first field starts aligned too.
    const std::size_t rounded = ((bytes + 127) / 128) * 128;
    return std::aligned_alloc(128, rounded);
#endif
}

inline void managed_free(void* p) noexcept {
    if (!p) return;
#if   defined(TC_STDPAR_SYCL)
    sycl::free(p, device_queue());
#elif defined(TC_STDPAR_HIP)
    hipFree(p);
#elif defined(TC_STDPAR_CUDA)
    cudaFree(p);
#else
    std::free(p);
#endif
}

} // namespace tc::detail
