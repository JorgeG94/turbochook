#pragma once
// =============================================================================
// numerics/parallel.hpp — the execution-policy seam + the iteration idioms.
//
// This is where "run on the GPU" actually happens. Two pieces:
//
//   1. tc::par — the execution policy. Normally std::execution::par_unseq (which
//      nvc++ -stdpar=gpu offloads to the GPU, and -stdpar=multicore runs on CPU
//      threads). Under the TC_STDPAR_OFF define (the g++/clang host build) it
//      becomes std::execution::seq — sequential, deterministic, and crucially
//      needs NO TBB. Under TC_STDPAR_SYCL there is no execution policy at all:
//      the launcher calls sycl::queue::parallel_for directly. So the same source
//      compiles FIVE ways:
//      gpu / multicore / hip / sycl / host.
//
//   2. do_concurrent / do_reduce — THE launch sites, and the only two. Every
//      loop idiom below is a thin wrapper over them, so the per-backend dialect
//      lives in one place. The reliably-offloading shape (verified on nvc++ 26.5
//      / V100, and by spikes/alpha on cc70, cc90, hipstdpar and oneDPL) is: a
//      FLAT 1D index range, unflattened to (i,j) INSIDE the lambda. One 1D
//      parallel range is what every runtime likes; a nested 2D loop is not the
//      idiom.
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

#if defined(TC_STDPAR_SYCL)
#  include <sycl/sycl.hpp>          // NO oneDPL -- see the SYCL branch below
#else
#  include <execution>
#  include <ranges>
#  include <algorithm>
#  include <numeric>
#endif
#include "core/types.hpp"
#include "lib/error.hpp"
#include "lib/device_alloc.hpp"   // THE queue: USM is context-bound, so allocation
                                   // and launch must share one (see that header)

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
//
// Under TC_STDPAR_SYCL there is NO policy object at all: <execution> is not even
// included on that path (the launchers below call sycl::queue::parallel_for), so
// naming std::execution here would not compile. The seam is do_concurrent /
// do_reduce; `par` is an implementation detail of the std-algorithm backends.
#if !defined(TC_STDPAR_SYCL)
#  if defined(TC_STDPAR_OFF)
inline constexpr const auto& par = std::execution::seq;        // host build: no TBB needed
#  else
inline constexpr const auto& par = std::execution::par_unseq;  // gpu / multicore
#  endif
#endif
// ^ BOTH bind by reference. The comment above explains why `seq` must, and then
// the original bound `par_unseq` BY VALUE two lines later -- the same libc++
// deleted-copy-ctor bug it warns about, latent only because the host build takes
// the other branch.


// ── do_concurrent / reduce — THE launch sites. ───────────────────────────────
// Every loop and every reduction in the codebase goes through these two, so the
// per-backend dialect has exactly one place to live. This is the seam
// CONTRACT_MEMORY §0.0 describes: the language expresses parallelism, and fifty
// kernels never learn which backend they are on.
//
// Intel is the reason these are functions rather than a bare `std::for_each(par,
// ...)` at each call site: it needs sycl::queue::parallel_for and sycl::reduction
// rather than a standard algorithm at all, so the choice has to be decidable in
// ONE place instead of at fifty.
template <class F>
void do_concurrent(Index count, F f) {
#if defined(TC_STDPAR_SYCL)
    // RAW SYCL, not oneDPL. oneDPL derives its kernel name from the execution
    // POLICY's name parameter, so a bare make_device_policy(q) hands every call
    // site the same name and kernels get launched with the wrong functor's
    // arguments -- ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE where the sizes
    // differ, silently wrong numbers where they match. Naming the policy per call
    // site is the documented remedy and it fights the language: a kernel name must
    // be forward-declarable and a lambda closure type is not.
    //
    // q.parallel_for with an unnamed lambda has no naming problem at all, needs no
    // oneDPL, and is the same one-flat-range shape every other backend uses. One
    // dependency fewer, for a launcher that is six lines.
    if (count <= 0) return;
    detail::device_queue()
        .parallel_for(sycl::range<1>(std::size_t(count)),
                      [=](sycl::id<1> it) { f(Index(it[0])); })
        .wait();                       // host reads follow immediately
#else
    // std::views::iota, NOT the hand-rolled counting_iterator below. This is the
    // shape verified to offload on nvc++/V100, and there is no reason to move off
    // it: only oneDPL ever rejected iota_view, and oneDPL is handled above.
    //
    // The hand-rolled type is also not a conforming C++17 ForwardIterator -- its
    // `reference` is a prvalue `Index`, where the requirement is a true reference.
    // libstdc++ and libc++ tolerate that; an offloading implementation need not.
    auto ids = std::views::iota(Index{0}, count);
    std::for_each(par, ids.begin(), ids.end(), f);
#endif
}

// Σ over [0,count) of unary(n), combined with `binop`. `T` is explicit and may be
// wider than the field type.
//
// NOT named `reduce`: an argument like std::plus<int> drags namespace std in by
// ADL, so an unqualified `reduce(...)` in namespace tc is AMBIGUOUS against
// std::reduce. Pairs with do_concurrent instead. (In its eventual home,
// tc::device::reduce, the qualification makes the plain name safe again.)
template <class T, class Binop, class Unary>
T do_reduce(Index count, T init, Binop binop, Unary unary) {
#if defined(TC_STDPAR_SYCL)
    // sycl::reduction takes an arbitrary ASSOCIATIVE binop plus an identity --
    // `init` is that identity (0 for plus, lowest() for max), which is exactly
    // what every caller already passes.
    auto& q = detail::device_queue();
    if (count <= 0) return init;

    // ONE scratch cell per result type, allocated once. A malloc_shared per call
    // would run at every diagnostic; and the earlier `if (!res) return init;`
    // was worse than unchecked -- an allocation failure silently returned the
    // IDENTITY, so a failed reduction reported a mass of zero as if it were
    // physics.
    static T* res = [&] {
        T* r = sycl::malloc_shared<T>(1, q);
        if (!r) fail(Errc::out_of_memory, "sycl::malloc_shared failed for the reduction scratch");
        return r;
    }();

    *res = init;
    q.parallel_for(sycl::range<1>(std::size_t(count)),
                   sycl::reduction(res, init, binop),
                   [=](sycl::id<1> it, auto& acc) { acc.combine(unary(Index(it[0]))); })
     .wait();
    return *res;                       // deliberately not freed: one cell per type,
                                       // released with the process, and freeing it
                                       // at exit would race the queue's teardown
#else
    auto ids = std::views::iota(Index{0}, count);
    return std::transform_reduce(par, ids.begin(), ids.end(), init, binop, unary);
#endif
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
