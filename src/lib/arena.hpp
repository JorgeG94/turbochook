#pragma once
// =============================================================================
// lib/arena.hpp — the memory Arena (DESIGN ADR-3).
//
// ONE monotonic bump allocator over ONE flat buffer. It owns the bytes; it hands
// out non-owning `Field` views (mdspans) INTO itself. This is the whole memory
// model of the solver.
//
// Why an arena instead of one std::vector per field:
//   • Under `nvc++ -stdpar`, that single std::vector<std::byte> is MANAGED memory
//     → one migration surface (few big page migrations, not many small ones).
//   • Snapshot / ping-pong / restart = one std::copy of the whole buffer.
//   • Truthful byte accounting: the high-water mark IS the memory used.
//   • Zero mid-run allocation — everything is carved out up front.
//   • It slots UNDER the kernel boundary (ADR-1): kernels only ever see `Field`
//     views, so swapping the owner (owning-vector → arena) changes ZERO kernels.
//
// THE HARD RULE: size it once, never let it grow. If the backing vector
// reallocated, every `Field` view into it would dangle. So: compute total bytes
// up front, construct the Arena once, then only ever hand out views.
//
// Two-tier use:
//   • monotonic  alloc*()      → persistent fields (never freed for the run)
//   • mark()/restore()         → a stack discipline for transient per-step scratch
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <utility>
#include "core/types.hpp"
#include "lib/error.hpp"
#include "lib/device_alloc.hpp"

namespace tc {

class Arena {
    std::byte*  buf_ = nullptr;           // MANAGED bytes -- see lib/device_alloc.hpp
    std::size_t cap_ = 0;
    std::size_t top_ = 0;                 // bump pointer (bytes used so far)

    static std::size_t align_up(std::size_t n, std::size_t a) {
        return (n + a - 1) & ~(a - 1);    // round n up to a multiple of a (a = power of two)
    }

public:
    // Size ONCE, here. After this ctor the buffer never reallocates.
    //
    // NOT a std::vector any more: plain host heap is device-reachable only under
    // `nvc++ -stdpar`, which promotes it to managed for you. hipstdpar faults and
    // oneDPL fails outright. See lib/device_alloc.hpp.
    explicit Arena(std::size_t bytes)
        : buf_(static_cast<std::byte*>(detail::managed_alloc(bytes))), cap_(bytes) {
        if (bytes != 0 && !buf_)
            fail(Errc::out_of_memory, "arena: managed allocation failed");
    }

    // Owns the bytes, so: move-only, and the destructor must not throw. An
    // implicit copy would double-free the pool the first time one was passed by
    // value -- the defect the DeviceAllocation design calls out explicitly.
    ~Arena() { detail::managed_free(buf_); }
    Arena(Arena&& o) noexcept
        : buf_(std::exchange(o.buf_, nullptr)),
          cap_(std::exchange(o.cap_, 0)),
          top_(std::exchange(o.top_, 0)) {}
    Arena& operator=(Arena&& o) noexcept {
        if (this != &o) {
            detail::managed_free(buf_);
            buf_ = std::exchange(o.buf_, nullptr);
            cap_ = std::exchange(o.cap_, 0);
            top_ = std::exchange(o.top_, 0);
        }
        return *this;
    }
    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    // Carve out an (nx × ny) Real field and return a layout_left view over it.
    // `align` (≥128 B default) keeps each field's base address aligned so the
    // fast axis stays coalescing-friendly on the GPU.
    Field2 alloc2d(Index nx, Index ny, std::size_t align = 128) {
        top_ = align_up(top_, align);
        const std::size_t need = sizeof(Real) * std::size_t(nx) * std::size_t(ny);
        if (top_ + need > cap_)
            fail(Errc::out_of_memory, "arena overflow — size it once, up front");
        // Treat the byte cursor as Real storage. (C++23 blesses this via
        // std::start_lifetime_as<Real[]>; reinterpret_cast is the portable
        // stand-in and works on every compiler we target.)
        Real* p = reinterpret_cast<Real*>(buf_ + top_);
        top_ += need;
        return Field2(p, nx, ny);
    }

    // 3D twin (arrives with M3 layers). Same discipline.
    Field3 alloc3d(Index nx, Index ny, Index nz, std::size_t align = 128) {
        top_ = align_up(top_, align);
        const std::size_t need = sizeof(Real) * std::size_t(nx) * std::size_t(ny) * std::size_t(nz);
        if (top_ + need > cap_)
            fail(Errc::out_of_memory, "arena overflow — size it once, up front");
        Real* p = reinterpret_cast<Real*>(buf_ + top_);
        top_ += need;
        return Field3(p, nx, ny, nz);
    }

    // Stack marker for transient scratch: mark() the top, hand out scratch,
    // restore(mark) to reclaim it at the end of the step. (Persistent fields are
    // allocated before the first mark and never restored past.)
    std::size_t mark() const { return top_; }
    void restore(std::size_t m) { top_ = m; }

    std::size_t bytes_used() const { return top_; }        // the truthful number
    std::size_t bytes_capacity() const { return cap_; }
};

} // namespace tc
