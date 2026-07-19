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
#if defined(TC_STDPAR_OFF)
inline constexpr auto par = std::execution::seq;          // host build: no TBB needed
#else
inline constexpr auto par = std::execution::par_unseq;    // gpu / multicore
#endif

// ── 1D: run f(n) for n in [0, count) ─────────────────────────────────────────
// The bedrock. saxpy, reductions setup, anything flat.
template <class F>
void for_each_index(Index count, F f) {
    auto ids = std::views::iota(Index{0}, count);
    std::for_each(par, ids.begin(), ids.end(), [=](Index n) { f(n); });
}

// ── 2D cell-centred: run f(i,j) over an nx×ny grid ───────────────────────────
// Flatten to a single iota range, unflatten inside. `i` (fast axis) = n % nx so
// adjacent threads hit adjacent memory in a layout_left Field.
template <class F>
void for_each_cell(Index nx, Index ny, F f) {
    auto ids = std::views::iota(Index{0}, nx * ny);
    std::for_each(par, ids.begin(), ids.end(),
                  [=](Index n) { f(n % nx, n / nx); });
}

// ── Staggered (Arakawa C-grid) twins — arrive in M2 ──────────────────────────
// On a C-grid the pieces of state live on DIFFERENT grids: η at cell centres
// (nx × ny), u on x-faces (nx+1 × ny), v on y-faces (nx × ny+1), vorticity/PV at
// corners (nx+1 × ny+1). Each needs its own iteration extent — same flatten/
// unflatten idiom, different bounds. (Bodies identical to for_each_cell; named
// separately so call sites read as intent and the extents can't be mixed up.)
template <class F>
void for_each_face_x(Index nx, Index ny, F f) {   // x-faces: (nx+1) × ny
    auto ids = std::views::iota(Index{0}, (nx + 1) * ny);
    std::for_each(par, ids.begin(), ids.end(),
                  [=](Index n) { f(n % (nx + 1), n / (nx + 1)); });
}
template <class F>
void for_each_face_y(Index nx, Index ny, F f) {   // y-faces: nx × (ny+1)
    auto ids = std::views::iota(Index{0}, nx * (ny + 1));
    std::for_each(par, ids.begin(), ids.end(),
                  [=](Index n) { f(n % nx, n / nx); });
}
template <class F>
void for_each_corner(Index nx, Index ny, F f) {   // corners: (nx+1) × (ny+1)
    auto ids = std::views::iota(Index{0}, (nx + 1) * (ny + 1));
    std::for_each(par, ids.begin(), ids.end(),
                  [=](Index n) { f(n % (nx + 1), n / (nx + 1)); });
}

} // namespace tc
