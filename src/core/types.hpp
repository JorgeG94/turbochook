#pragma once
// =============================================================================
// core/types.hpp — the vocabulary types every other header speaks in.
//
// This is the single most load-bearing file in the project, so it's worth
// reading slowly. It establishes two things:
//   1. `tc::Real` / `tc::Index` — the scalar + index types (one place to change).
//   2. `tc::Field<Rank>` — the "grid array" type: a NON-OWNING, column-major,
//      contiguous view over memory somebody else owns (the Arena, ADR-3).
//
// Design rules encoded here (DESIGN.md ADR-2, §9):
//   • A grid field IS a std::mdspan with `layout_left` (column-major = Fortran
//     order). mdspan's DEFAULT is layout_right (C order) — we always override.
//   • mdspan is C++23. g++13's libstdc++ doesn't ship <mdspan> yet, and the host
//     build (fast dev loop) uses g++. So we provide a ~40-line hand-rolled
//     fallback, `tc::MdView`, behind a `__has_include(<mdspan>)` seam. Kernels
//     are written ONCE against `Field<Rank>` and compile either way.
// =============================================================================

#include <cstdint>
#include <array>
#include <cstddef>

namespace tc {

// ── Scalars ──────────────────────────────────────────────────────────────────
// `Real` is the working precision. Centralising it means a future float/double
// switch is one line, and every signature reads `Real`, not `double`, so intent
// is explicit. `Index` is signed on purpose: signed arithmetic makes `i-1` at a
// left boundary well-defined (unsigned would wrap to a huge number), and it
// matches the ghost-cell index math `[nghost, nghost+n)`.
using Real  = double;
using Index = int;

// =============================================================================
// The mdspan seam. Two worlds, one name `Field<Rank>`:
//   • real std::mdspan   when the stdlib has <mdspan>  (nvc++ 26.5, libstdc++14+)
//   • tc::MdView<Rank>   otherwise                     (g++13 host build)
//
// IMPORTANT (verified, DESIGN §9): nvc++ 26.5 SHIPS a working <mdspan> but leaves
// the feature-test macro `__cpp_lib_mdspan` UNDEFINED. So we must gate on
// `__has_include(<mdspan>)`, NOT on the macro — the macro would wrongly reject
// nvc++'s perfectly good header.
// =============================================================================
#if __has_include(<mdspan>)
  #define TC_HAS_STD_MDSPAN 1
#else
  #define TC_HAS_STD_MDSPAN 0
#endif

#if TC_HAS_STD_MDSPAN
} // namespace tc  (close it: the include must sit at global scope)
#include <mdspan>
namespace tc {

// `dextents<Index,Rank>` = all-dynamic extents (sizes known at run time, not in
// the type). `layout_left` = first index varies fastest = column-major. So a 2D
// Field indexes as `f[i,j]` with linear offset `i + nx*j` — adjacent `i` are
// adjacent in memory. That is the coalescing contract: put the parallel/
// fast-varying thread index on axis 0 and neighbouring threads touch
// neighbouring bytes (DESIGN ADR-2 "coalescing rule").
template <int Rank>
using Field = std::mdspan<Real, std::dextents<Index, Rank>, std::layout_left>;

#else  // ── hand-rolled fallback: the sliver of mdspan we actually use ─────────

// MdView<Rank> — a non-owning, column-major (layout_left) view over a raw
// pointer. Deliberately tiny: construct from (pointer, extents), subscript with
// the C++23 multidimensional operator[], query an extent, get the data handle.
// No copy of data — copying an MdView copies the pointer+extents (cheap, and
// trivially copyable, so it captures by value into a kernel just like mdspan).
//
// Teaching notes:
//   • `operator[](Index, Index)` is the C++23 *multidimensional subscript*
//     (P2128). GCC 12+ accepts it under -std=c++23. That's why `f[i,j]` (not
//     `f(i,j)`) is the spelling — it matches real std::mdspan exactly, so the
//     kernels don't care which backend is live.
//   • `constexpr` + no virtuals + a plain pointer member ⇒ trivially copyable ⇒
//     kernel-safe.
template <int Rank>
class MdView {
    Real* p_ = nullptr;
    std::array<Index, Rank> ext_{};

    // layout_left linear offset: i0 + n0*(i1 + n1*(i2 + ...)). Written as a
    // Horner fold from the last index inward.
    template <class... I>
    constexpr Index offset(I... idx) const {
        static_assert(sizeof...(I) == Rank, "wrong number of indices");
        const std::array<Index, Rank> ix{ static_cast<Index>(idx)... };
        Index off = 0;
        for (int r = Rank - 1; r >= 0; --r) off = off * ext_[r] + ix[r];
        return off;
    }

public:
    constexpr MdView() = default;
    // Variadic ctor: MdView<2>(ptr, nx, ny), MdView<3>(ptr, nx, ny, nz).
    template <class... E>
    constexpr explicit MdView(Real* p, E... e) : p_(p), ext_{ static_cast<Index>(e)... } {
        static_assert(sizeof...(E) == Rank, "extent count must equal Rank");
    }

    // Multidim subscript (C++23). Two overloads keep it readable at Rank 2/3;
    // both const and mutable so `Field` works as source and destination.
    constexpr Real& operator[](Index i, Index j)          requires (Rank == 2) { return p_[offset(i, j)]; }
    constexpr Real  operator[](Index i, Index j)    const  requires (Rank == 2) { return p_[offset(i, j)]; }
    constexpr Real& operator[](Index i, Index j, Index k)       requires (Rank == 3) { return p_[offset(i, j, k)]; }
    constexpr Real  operator[](Index i, Index j, Index k) const requires (Rank == 3) { return p_[offset(i, j, k)]; }

    constexpr Index extent(int r) const { return ext_[r]; }   // size along axis r
    constexpr Real* data_handle() const { return p_; }        // raw pointer (host-side only)
};

template <int Rank>
using Field = MdView<Rank>;

#endif  // TC_HAS_STD_MDSPAN

// Convenience aliases — the shapes we actually use.
using Field2 = Field<2>;
using Field3 = Field<3>;

} // namespace tc
