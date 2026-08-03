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
#include <exception>
#include <cstdio>
#include <new>
#include "lib/gpu_assert.hpp"

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
    static sycl::queue q = [] {
        sycl::queue qq{sycl::gpu_selector_v};
        const auto d = qq.get_device();
        // Announce the device ONCE. gpu_selector_v picks *a* GPU: on a box with an
        // integrated device alongside the discrete one it may not be the one you
        // meant, and fp64 support differs between them -- which shows up as wrong
        // NUMBERS in reductions rather than as an error. Print it so a
        // wrong-device run is visible instead of mysterious.
        std::fprintf(stderr,
            "[sycl] device: %s | fp64=%d | max_sub_devices=%u\n",
            d.get_info<sycl::info::device::name>().c_str(),
            (int)d.has(sycl::aspect::fp64),
            d.get_info<sycl::info::device::partition_max_sub_devices>());
        return qq;
    }();
    return q;
}
#endif

// Managed/shared bytes, host- AND device-addressable.
//
// THROWS on failure, naming the call and the driver's own message -- the previous
// version collapsed every failure to a nullptr, so "out of memory" and "invalid
// device" were indistinguishable by the time the arena saw them.
inline void* managed_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
#if   defined(TC_STDPAR_SYCL)
    // USM allocation reports failure by returning nullptr rather than throwing,
    // so it needs its own check -- TC_GPU_CHECK is for return-code runtimes.
    void* p = sycl::malloc_shared(bytes, device_queue());
    if (!p)
        fail(Errc::out_of_memory,
             "sycl::malloc_shared failed for " + std::to_string(bytes) + " bytes");
    return p;
#elif defined(TC_STDPAR_HIP)
    void* p = nullptr;
    TC_GPU_CHECK(hipMallocManaged(&p, bytes));   // throws, naming the call + code
    return p;
#elif defined(TC_STDPAR_CUDA)
    // nvc++ -stdpar would promote a plain heap allocation anyway; going through
    // cudaMallocManaged makes that EXPLICIT rather than a compiler favour, and
    // keeps all four backends on the same code path.
    void* p = nullptr;
    TC_GPU_CHECK(cudaMallocManaged(&p, bytes));
    return p;
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
    // sycl::free THROWS on error, and this runs from ~Arena, which is
    // noexcept(true) -- an escaping exception here is std::terminate. Swallow and
    // log, exactly as the CUDA/HIP paths do via TC_GPU_CHECK_NOTHROW.
    try {
        sycl::free(p, device_queue());
    } catch (const std::exception& e) {
        try { logger().error("sycl::free failed: {}", e.what()); } catch (...) {}
    } catch (...) {}
#elif defined(TC_STDPAR_HIP)
    // NOTHROW: managed_free runs from ~Arena, and a destructor is noexcept(true).
    // It also silences the "ignoring return value of hipFree" warning honestly --
    // by CHECKING the code, not by casting it to void.
    TC_GPU_CHECK_NOTHROW(hipFree(p));
#elif defined(TC_STDPAR_CUDA)
    TC_GPU_CHECK_NOTHROW(cudaFree(p));
#else
    std::free(p);
#endif
}

} // namespace tc::detail
