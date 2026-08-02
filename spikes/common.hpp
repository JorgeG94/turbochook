#pragma once
// =============================================================================
// spikes/common.hpp — the shared sliver for the memory-model spikes.
//
// Deliberately NOT src/ code. These spikes exist to answer toolchain questions
// before the answers get baked into the Arena. Everything here is the smallest
// thing that can carry the question:
//   • Space           — the compile-time memory-space tag (Host | Device)
//   • View2           — a layout_left 2D view (std::mdspan where available)
//   • pool_*          — the pinned-once pool, one call per allocation strategy
//   • Timer / check   — reporting
//
// The CUDA runtime calls are guarded by TC_SPIKE_CUDA so the host-only spike
// (04) compiles under plain g++ with no CUDA in sight.
// =============================================================================

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <array>

#if defined(TC_SPIKE_CUDA)
#include <cuda_runtime.h>
#endif

namespace tc {

using Real  = double;
using Index = int;

// ── the memory-space tag: the thing that makes host-touches-device a COMPILE
//    error rather than a fault. This is the property we want from Kokkos
//    without Kokkos. ──────────────────────────────────────────────────────────
enum class Space { Host, Device };

// On a unified/coherent machine Device memory IS host-readable; on a discrete
// one it is not. The spikes verify this at runtime; the TYPE system stays
// conservative (Device is never host-subscriptable) so one source is correct
// on both.
constexpr bool host_accessible(Space s) { return s == Space::Host; }

// =============================================================================
// View2 — non-owning, column-major, trivially copyable. Same __has_include seam
// as src/core/types.hpp, so the spike exercises real std::mdspan wherever the
// stdlib has it (that is itself part of the question: does mdspan-over-a-device
// -pointer survive the capture into a stdpar kernel?).
// =============================================================================
#if __has_include(<mdspan>)
  #define TC_SPIKE_STD_MDSPAN 1
#else
  #define TC_SPIKE_STD_MDSPAN 0
#endif

#if TC_SPIKE_STD_MDSPAN
} // namespace tc
#include <mdspan>
namespace tc {
using View2 = std::mdspan<Real, std::dextents<Index, 2>, std::layout_left>;
inline View2 make_view(Real* p, Index nx, Index ny) { return View2(p, nx, ny); }
#else
class View2 {
    Real* p_ = nullptr;
    Index nx_ = 0, ny_ = 0;
public:
    constexpr View2() = default;
    constexpr View2(Real* p, Index nx, Index ny) : p_(p), nx_(nx), ny_(ny) {}
    constexpr Real& operator[](Index i, Index j) const { return p_[i + nx_ * j]; }
    constexpr Index extent(int r) const { return r == 0 ? nx_ : ny_; }
    constexpr Real* data_handle() const { return p_; }
};
inline View2 make_view(Real* p, Index nx, Index ny) { return View2(p, nx, ny); }
#endif

// =============================================================================
// The pool. ONE allocation for the whole run, four strategies. Which one is
// viable under which compiler/flag combination is exactly what the spikes ask.
// =============================================================================
#if defined(TC_SPIKE_CUDA)

inline void cuda_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "  CUDA FAIL: %s -> %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// (a) MANAGED + advise + prefetch: stdpar-compatible AND device-resident. The
//     100x penalty comes from host-triggered page faults, not from managed
//     memory itself — so a pool that is prefetched once and never touched by
//     the host behaves like device memory for the whole run.
inline Real* pool_managed_prefetched(std::size_t bytes, int dev = 0) {
    void* p = nullptr;
    cuda_ck(cudaMallocManaged(&p, bytes), "cudaMallocManaged");
    cuda_ck(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, dev), "cudaMemAdvise");
    cuda_ck(cudaMemPrefetchAsync(p, bytes, dev), "cudaMemPrefetchAsync");
    cuda_ck(cudaDeviceSynchronize(), "sync after prefetch");
    return static_cast<Real*>(p);
}

// (b) MANAGED, no prefetch — the current turbochook behaviour (std::vector under
//     -stdpar allocator interception). The control for (a).
inline Real* pool_managed_plain(std::size_t bytes) {
    void* p = nullptr;
    cuda_ck(cudaMallocManaged(&p, bytes), "cudaMallocManaged");
    return static_cast<Real*>(p);
}

// (c) DEVICE-ONLY: the strict pool. Host cannot dereference it at all. Whether a
//     stdpar kernel may capture a pointer into this is spike 02's whole question.
inline Real* pool_device(std::size_t bytes) {
    void* p = nullptr;
    cuda_ck(cudaMalloc(&p, bytes), "cudaMalloc");
    return static_cast<Real*>(p);
}

inline void pool_free(Real* p) { cudaFree(p); }

// Explicit staging — what every host consumer (Reporter, NetCDF, restart, tests)
// would call under a device-only pool.
inline void stage_to_host(Real* dst, const Real* src, std::size_t bytes) {
    cuda_ck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "D2H");
}
inline void stage_to_device(Real* dst, const Real* src, std::size_t bytes) {
    cuda_ck(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "H2D");
}
inline void device_sync() { cuda_ck(cudaDeviceSynchronize(), "deviceSynchronize"); }

// Report what the device says about coherence — this is what decides
// unified-vs-discrete, and therefore which row of the backend table applies.
inline void report_device() {
    int dev = 0; cudaDeviceProp pr{};
    cuda_ck(cudaGetDevice(&dev), "cudaGetDevice");
    cuda_ck(cudaGetDeviceProperties(&pr, dev), "cudaGetDeviceProperties");
    std::printf("  device            : %s (cc%d%d)\n", pr.name, pr.major, pr.minor);
    std::printf("  unifiedAddressing : %d\n", pr.unifiedAddressing);
    std::printf("  managedMemory     : %d\n", pr.managedMemory);
    std::printf("  concurrentManaged : %d\n", pr.concurrentManagedAccess);
    std::printf("  pageableMemAccess : %d   <-- 1 => COHERENT (Grace-Hopper class)\n",
                pr.pageableMemoryAccess);
}

#else  // host-only build: the pool is plain aligned memory, staging is a no-op.

inline Real* pool_managed_prefetched(std::size_t bytes, int = 0) {
    return static_cast<Real*>(std::aligned_alloc(128, ((bytes + 127) / 128) * 128));
}
inline Real* pool_managed_plain(std::size_t bytes) { return pool_managed_prefetched(bytes); }
inline Real* pool_device(std::size_t bytes)        { return pool_managed_prefetched(bytes); }
inline void  pool_free(Real* p)                    { std::free(p); }
inline void  device_sync() {}
inline void  report_device() { std::printf("  device            : (host build, no CUDA)\n"); }

// Staging is a memcpy when there is no device — so every spike still COMPILES and
// RUNS host-only. That is not just convenience: it is the mirror-is-a-no-op rule
// from spike 04 applied to the pool, and it means these sources can be
// syntax-checked (and debugged) on a laptop before they reach a GPU box.
inline void stage_to_host(Real* dst, const Real* src, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}
inline void stage_to_device(Real* dst, const Real* src, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

#endif

// =============================================================================
// A bump arena over the pool — sized once, sealed after setup, never grows.
// Thirty lines, and it is the whole memory model.
// =============================================================================
class Arena {
    Real*       base_ = nullptr;
    std::size_t cap_  = 0;      // in Reals
    std::size_t top_  = 0;
    bool        sealed_ = false;
public:
    Arena(Real* base, std::size_t bytes) : base_(base), cap_(bytes / sizeof(Real)) {}

    View2 alloc2d(Index nx, Index ny, const char* label = "") {
        if (sealed_) { std::fprintf(stderr, "  arena SEALED, refusing '%s'\n", label); std::exit(3); }
        const std::size_t need = std::size_t(nx) * std::size_t(ny);
        const std::size_t al   = (top_ + 15) & ~std::size_t(15);   // 128B for Real
        if (al + need > cap_) { std::fprintf(stderr, "  arena OVERFLOW on '%s'\n", label); std::exit(3); }
        Real* p = base_ + al;
        top_ = al + need;
        return make_view(p, nx, ny);
    }
    void seal() { sealed_ = true; }                 // no allocation past setup
    std::size_t bytes_used() const { return top_ * sizeof(Real); }
};

// ── reporting ────────────────────────────────────────────────────────────────
struct Timer {
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    double s() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

inline int g_fails = 0;
inline void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fails;
}
inline int verdict(const char* name) {
    std::printf("  ---- %s: %s ----\n\n", name, g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}

// Effective bandwidth of the Jacobi sweep: 4 reads + 1 write per interior cell.
inline double gbs(long iters, Index nx, Index ny, double secs) {
    const double bytes = double(iters) * double(nx) * double(ny) * 5.0 * double(sizeof(Real));
    return bytes / secs / 1.0e9;
}

} // namespace tc
