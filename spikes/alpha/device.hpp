// =============================================================================
// spikes/alpha/device.hpp — the whole thesis in one header.
//
//   The LANGUAGE expresses parallelism.   -> device::do_concurrent / reduce
//   WE express memory.                    -> device::alloc / free / copy
//   Native kernels INSERT, not replace.   -> not exercised here; this spike is
//                                            about whether the stdpar path
//                                            works over hand-managed memory.
//
// Every per-backend dialect lives in THIS FILE and nowhere else. main.cpp is
// backend-agnostic and must stay that way -- if you find yourself adding an
// #ifdef there, the abstraction has failed and that is the result.
//
// THE QUESTION: does a parallel algorithm accept pointers WE allocated with the
// native device allocator (cudaMalloc / hipMalloc / sycl::malloc_device),
// with NO managed memory and NO implicit migration?
//
// ANSWERED, 2026-08-03 -- yes, on every toolchain asked:
//
//   nvc++ -stdpar=gpu -gpu=mem:separate + cudaMalloc      PASS (cc70 and cc90)
//   hipcc --hipstdpar + hipMalloc                         PASS
//   icpx -fsycl + oneDPL + sycl::malloc_device            PASS
//
// The AMD cell was the one in genuine doubt. hipstdpar is NOT hipified CUDA: it
// leans on HMM/XNACK to make ORDINARY HOST allocations device-reachable, so its
// design assumption is that you did NOT hand-manage memory. Had it required
// interposing malloc, the arena would fight it rather than compose with it. It
// accepts hipMalloc'd pointers -- so "CUDA is basically HIP" extends to the
// stdpar path too, not only to __global__ kernels.
//
// Keep this file runnable: it is the regression test for the thesis, and the
// first thing to re-run when a toolchain moves.
// =============================================================================
#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <iterator>

// ── backend selection ────────────────────────────────────────────────────────
// These includes MUST be at global scope. Putting them inside `namespace tc`
// makes the standard library resolve as `tc::std::...` and the errors point at
// libc++ internals rather than at this file. (Cost me one build to rediscover.)
#if   defined(TC_BACKEND_CUDA)
#  define TC_BACKEND_NAME "cuda (nvc++ -stdpar=gpu -gpu=mem:separate)"
#  include <cuda_runtime.h>
#  include <algorithm>
#  include <execution>
#  include <numeric>
#elif defined(TC_BACKEND_HIP)
#  define TC_BACKEND_NAME "hip (--hipstdpar)"
#  include <hip/hip_runtime.h>
#  include <algorithm>
#  include <execution>
#  include <numeric>
#elif defined(TC_BACKEND_SYCL)
#  define TC_BACKEND_NAME "sycl (icpx -fsycl + oneDPL)"
#  include <oneapi/dpl/execution>
#  include <oneapi/dpl/algorithm>
#  include <oneapi/dpl/numeric>
#  include <oneapi/dpl/iterator>
#  include <sycl/sycl.hpp>
#else
#  define TC_BACKEND_HOST 1
#  define TC_BACKEND_NAME "host (no device; ordinary memory) -- SHAPE CHECK ONLY"
#  include <algorithm>
#  include <execution>
#  include <numeric>
#  include <cstdlib>
#endif

// Not every stdlib ships parallel execution policies -- Apple's libc++ has none
// at all (verified). The HOST target is allowed to degrade to a serial loop,
// because it was never evidence about offload; it only proves the shape
// compiles. A DEVICE backend without policies is a hard error: running the
// spike serially there would report PASS while measuring nothing.
// TC_FORCE_SERIAL is set by the Makefile when libstdc++ is present but TBB is
// not: <execution> then COMPILES and fails to LINK, which is a worse failure
// than not offering the policy at all.
#if defined(TC_BACKEND_SYCL)
#  define TC_PAR_NAME "oneDPL device_policy"
#elif defined(TC_FORCE_SERIAL)
#  if defined(TC_BACKEND_CUDA) || defined(TC_BACKEND_HIP)
#    error "TC_FORCE_SERIAL on a GPU backend -- that would report PASS while measuring nothing"
#  endif
#  define TC_PAR_NAME "SERIAL FALLBACK (forced: no TBB for libstdc++ par_unseq)"
#elif defined(__cpp_lib_parallel_algorithm) && __cpp_lib_parallel_algorithm >= 201603L
#  define TC_HAS_PAR  1
#  define TC_PAR_NAME "std::execution::par_unseq"
#elif defined(TC_BACKEND_CUDA) || defined(TC_BACKEND_HIP)
#  error "this stdlib has no parallel execution policies -- a GPU stdpar build cannot be meaningful"
#else
#  define TC_PAR_NAME "SERIAL FALLBACK (no <execution> policies in this stdlib)"
#endif

namespace tc {

using Index = std::int32_t;

namespace device {

// ── the index source ─────────────────────────────────────────────────────────
// oneDPL REJECTS std::views::iota -- it is a read-only proxy iterator (measured,
// spike 05c). So SYCL uses oneDPL's own counting_iterator and everyone else uses
// this one. This divergence is the entire reason do_concurrent exists: it is
// three lines here instead of a decision at fifty call sites.
#if !defined(TC_BACKEND_SYCL)
struct counting_iterator {
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = Index;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const Index*;
    using reference         = Index;             // by value: no proxy object

    Index i = 0;

    reference operator*() const { return i; }
    reference operator[](difference_type n) const { return i + Index(n); }

    counting_iterator& operator++()    { ++i; return *this; }
    counting_iterator  operator++(int) { auto t = *this; ++i; return t; }
    counting_iterator& operator--()    { --i; return *this; }
    counting_iterator  operator--(int) { auto t = *this; --i; return t; }

    counting_iterator& operator+=(difference_type n) { i += Index(n); return *this; }
    counting_iterator& operator-=(difference_type n) { i -= Index(n); return *this; }

    friend counting_iterator operator+(counting_iterator a, difference_type n) { return a += n; }
    friend counting_iterator operator+(difference_type n, counting_iterator a) { return a += n; }
    friend counting_iterator operator-(counting_iterator a, difference_type n) { return a -= n; }
    friend difference_type   operator-(counting_iterator a, counting_iterator b) { return a.i - b.i; }

    friend bool operator==(counting_iterator a, counting_iterator b) { return a.i == b.i; }
    friend bool operator!=(counting_iterator a, counting_iterator b) { return a.i != b.i; }
    friend bool operator< (counting_iterator a, counting_iterator b) { return a.i <  b.i; }
    friend bool operator> (counting_iterator a, counting_iterator b) { return a.i >  b.i; }
    friend bool operator<=(counting_iterator a, counting_iterator b) { return a.i <= b.i; }
    friend bool operator>=(counting_iterator a, counting_iterator b) { return a.i >= b.i; }
};
inline counting_iterator index_begin(Index n) { (void)n; return counting_iterator{0}; }
inline counting_iterator index_end  (Index n) { return counting_iterator{n}; }
#endif

// ── context ──────────────────────────────────────────────────────────────────
// Exists BECAUSE of SYCL: malloc_device and parallel_for both need a queue,
// while CUDA/HIP have an implicit default. Threading it through later would
// touch every call site, so it exists from the start and CUDA/HIP ignore theirs.
#if defined(TC_BACKEND_SYCL)
inline sycl::queue& q() {
    static sycl::queue queue{sycl::gpu_selector_v};
    return queue;
}
#endif

inline void initialize() {
#if defined(TC_BACKEND_SYCL)
    auto d = q().get_device();
    std::printf("  device            : %s\n",
                d.get_info<sycl::info::device::name>().c_str());
    // fp64 matters here: gpu_selector_v may pick an INTEGRATED device whose
    // double support is absent or emulated. That breaks reductions and physics
    // while simple stores still look fine -- wrong NUMBERS, not an error.
    std::printf("  fp64 aspect       : %d %s\n", (int)d.has(sycl::aspect::fp64),
                d.has(sycl::aspect::fp64) ? "" : "  <-- doubles unsupported on this device!");
    // The PVC trap: a COMPOSITE root device is 6.3x slower than one tile.
    // Reported here so a wrong-device run is visible rather than mysterious.
    std::printf("  max_sub_devices   : %u\n",
                d.get_info<sycl::info::device::partition_max_sub_devices>());
#elif defined(TC_BACKEND_CUDA) || defined(TC_BACKEND_HIP)
    int dev = 0;
#  if defined(TC_BACKEND_CUDA)
    cudaGetDevice(&dev);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
#  else
    hipGetDevice(&dev);
    hipDeviceProp_t p{}; hipGetDeviceProperties(&p, dev);
#  endif
    std::printf("  device            : %s\n", p.name);
#endif
}

inline void sync() {
#if   defined(TC_BACKEND_CUDA)
    cudaDeviceSynchronize();
#elif defined(TC_BACKEND_HIP)
    hipDeviceSynchronize();
#elif defined(TC_BACKEND_SYCL)
    q().wait();
#endif
}

// ── memory: OURS, explicitly, with no managed fallback ───────────────────────
template <class T> T* alloc(std::size_t count) {
    void* p = nullptr;
#if   defined(TC_BACKEND_CUDA)
    if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) return nullptr;
#elif defined(TC_BACKEND_HIP)
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess) return nullptr;
#elif defined(TC_BACKEND_SYCL)
    p = sycl::malloc_device(count * sizeof(T), q());
#else
    p = std::aligned_alloc(128, ((count * sizeof(T) + 127) / 128) * 128);
#endif
    return static_cast<T*>(p);
}

template <class T> void free_(T* p) {
    if (!p) return;
#if   defined(TC_BACKEND_CUDA)
    cudaFree(p);
#elif defined(TC_BACKEND_HIP)
    hipFree(p);
#elif defined(TC_BACKEND_SYCL)
    sycl::free(p, q());
#else
    std::free(p);
#endif
}

template <class T> void copy_to_host(T* dst, const T* src, std::size_t count) {
#if   defined(TC_BACKEND_CUDA)
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
#elif defined(TC_BACKEND_HIP)
    hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost);
#elif defined(TC_BACKEND_SYCL)
    q().memcpy(dst, src, count * sizeof(T)).wait();
#else
    std::copy_n(src, count, dst);
#endif
}

// ── parallelism: the LANGUAGE's job ──────────────────────────────────────────
// One shape. Callers never learn which backend they are on.
template <class F>
void do_concurrent(Index n, F f) {
#if   defined(TC_BACKEND_SYCL)
    oneapi::dpl::for_each(oneapi::dpl::execution::make_device_policy(q()),
                          oneapi::dpl::counting_iterator<Index>(0),
                          oneapi::dpl::counting_iterator<Index>(n), f);
#elif defined(TC_HAS_PAR)
    std::for_each(std::execution::par_unseq, index_begin(n), index_end(n), f);
#else
    for (Index t = 0; t < n; ++t) f(t);
#endif
}

// Explicit binary op -- global_max's shape. std::plus is special-cased by some
// implementations, so a custom associative op is a genuinely different path.
template <class T, class Binop, class Unary>
T reduce_op(Index n, T init, Binop binop, Unary unary) {
#if   defined(TC_BACKEND_SYCL)
    return oneapi::dpl::transform_reduce(
        oneapi::dpl::execution::make_device_policy(q()),
        oneapi::dpl::counting_iterator<Index>(0),
        oneapi::dpl::counting_iterator<Index>(n), init, binop, unary);
#elif defined(TC_HAS_PAR)
    return std::transform_reduce(std::execution::par_unseq,
                                 index_begin(n), index_end(n), init, binop, unary);
#else
    T acc = init;
    for (Index t = 0; t < n; ++t) acc = binop(acc, unary(t));
    return acc;
#endif
}

template <class T, class F>
T reduce(Index n, T init, F unary) {
#if   defined(TC_BACKEND_SYCL)
    return oneapi::dpl::transform_reduce(
        oneapi::dpl::execution::make_device_policy(q()),
        oneapi::dpl::counting_iterator<Index>(0),
        oneapi::dpl::counting_iterator<Index>(n), init, std::plus<T>{}, unary);
#elif defined(TC_HAS_PAR)
    return std::transform_reduce(std::execution::par_unseq,
                                 index_begin(n), index_end(n), init,
                                 std::plus<T>{}, unary);
#else
    T acc = init;
    for (Index t = 0; t < n; ++t) acc = acc + unary(t);
    return acc;
#endif
}

} // namespace device
} // namespace tc
