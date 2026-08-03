#pragma once
// =============================================================================
// numerics/parallel.hpp — the execution-policy seam + the iteration idioms.
//
// This is where "run on the GPU" actually happens. Two pieces:
//
//   1. tc::par — the execution policy. Normally std::execution::par_unseq (which
//      nvc++ -stdpar=gpu offloads to the GPU, and -stdpar=multicore runs on CPU
//      threads). Under the TC_STDPAR_OFF define (the g++ host build) it becomes
//      std::execution::seq — sequential, deterministic, and crucially needs NO
//      TBB. So the same source compiles three ways (gpu / multicore / host).
//
//   2. for_each_cell / for_each_face_* — the loop idioms. The reliably-offloading
//      shape (verified on nvc++ 26.5 / V100) is: build a FLAT 1D index range
//      with std::views::iota, hand it to std::for_each(tc::par, …), and unflatten
//      (i,j) INSIDE the lambda. One 1D parallel range is what the runtime likes;
//      a nested 2D loop is not the idiom.
//
// COALESCING (DESIGN ADR-2): we make the FAST axis (i, index 0 of a layout_left
// Field) the fast-varying part of the flat index (`i = n % nx`). Adjacent thread
// ids → adjacent i → adjacent memory — the classic coalescing rule (put the
// contiguous index on the fast-varying thread axis).
//
// KERNEL RULES (the prime directive) apply to the callable `f`: capture BY VALUE,
// only Field views + POD scalars may cross in, never `this`, no allocation, no
// virtual, no throw. See DESIGN §8.
// =============================================================================

#include <execution>
#include <algorithm>
#include <ranges>
#include "core/types.hpp"

namespace tc {

// The policy seam. `inline constexpr` so it's a single shared value with no ODR
// fuss across TUs.
//
// NB: bind by REFERENCE, not by value. libstdc++ makes the policy objects
// copyable, but libc++ (Apple Clang) marks their copy ctor `= delete`, so
// `auto par = std::execution::seq` fails to compile there — a const reference to
// the inline-constexpr policy object copies nothing and works on both stdlibs.
// (On libc++ the host build also needs -fexperimental-library to expose the PSTL
// policies at all; CMake adds it for Clang. See CMakeLists.txt, TC_STDPAR=off.)
#if defined(TC_STDPAR_OFF)
inline constexpr const auto& par = std::execution::seq;        // host build: no TBB needed
#else
inline constexpr const auto& par = std::execution::par_unseq;  // gpu / multicore
#endif
// ^ BOTH bind by reference. The comment above explains why `seq` must, and then
// the original bound `par_unseq` BY VALUE two lines later -- the same libc++
// deleted-copy-ctor bug it warns about, latent only because the host build takes
// the other branch.

// ── the index source ─────────────────────────────────────────────────────────
// `std::views::iota` was the original spelling and it is NOT portable: oneDPL
// REJECTS it (read-only proxy iterator), so no code routed through here could
// ever compile for Intel. A plain random-access counting iterator is accepted by
// libstdc++, libc++ and nvc++ alike, and oneDPL's own `counting_iterator` is the
// drop-in for the SYCL build. See spikes/alpha/device.hpp, where the same type
// carries a two-layer-equivalent kernel on all four toolchains.
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

// ── do_concurrent — THE launch site. Every loop idiom below goes through it. ──
// One shape, so the per-backend dialect has exactly one place to live. This is
// the seam CONTRACT_MEMORY §0.0 describes: the language expresses parallelism,
// and fifty kernels never learn which backend they are on.
template <class F>
void do_concurrent(Index count, F f) {
    std::for_each(par, counting_iterator{0}, counting_iterator{count}, f);
}

// ── 1D: run f(n) for n in [0, count) ─────────────────────────────────────────
// The bedrock. saxpy, reductions setup, anything flat.
template <class F>
void for_each_index(Index count, F f) {
    do_concurrent(count, [=](Index n) { f(n); });
}

// ── 2D cell-centred: run f(i,j) over an nx×ny grid ───────────────────────────
// Flatten to a single range, unflatten inside. `i` (fast axis) = n % nx so
// adjacent threads hit adjacent memory in a layout_left Field.
template <class F>
void for_each_cell(Index nx, Index ny, F f) {
    do_concurrent(nx * ny, [=](Index n) { f(n % nx, n / nx); });
}

// ── Staggered (Arakawa C-grid) twins — arrive in M2 ──────────────────────────
// On a C-grid the pieces of state live on DIFFERENT grids: η at cell centres
// (nx × ny), u on x-faces (nx+1 × ny), v on y-faces (nx × ny+1), vorticity/PV at
// corners (nx+1 × ny+1). Each needs its own iteration extent — same flatten/
// unflatten idiom, different bounds. (Bodies identical to for_each_cell; named
// separately so call sites read as intent and the extents can't be mixed up.)
template <class F>
void for_each_face_x(Index nx, Index ny, F f) {   // x-faces: (nx+1) × ny
    do_concurrent((nx + 1) * ny, [=](Index n) { f(n % (nx + 1), n / (nx + 1)); });
}
template <class F>
void for_each_face_y(Index nx, Index ny, F f) {   // y-faces: nx × (ny+1)
    do_concurrent(nx * (ny + 1), [=](Index n) { f(n % nx, n / nx); });
}
template <class F>
void for_each_corner(Index nx, Index ny, F f) {   // corners: (nx+1) × (ny+1)
    do_concurrent((nx + 1) * (ny + 1), [=](Index n) { f(n % (nx + 1), n / (nx + 1)); });
}

} // namespace tc
