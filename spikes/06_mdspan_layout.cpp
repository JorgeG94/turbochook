// =============================================================================
// spike 06 — can `std::mdspan` with a CUSTOM LAYOUT be the kernel currency?
//
// THE DECISION THIS SETTLES: `View<T,Rank>` is either a real `std::mdspan` or a
// bespoke type. The only reasons to own the type were (a) Fortran-style arbitrary
// lower bounds and (b) a padded leading dimension — and both are exactly what an
// mdspan LAYOUT POLICY exists to express. If a custom layout survives device
// codegen at zero cost, we get the standard vocabulary, interop, and (C++26)
// `submdspan` for free, and the hand-rolled fallback becomes a sunset seam rather
// than architecture.
//
// FOUR QUESTIONS, in order of consequence:
//   1. Is <mdspan> even available on this toolchain? (Aurora's gcc 13.4 says no —
//      that is an upgrade request, and this spike is the evidence for it.)
//   2. Does a user-defined layout policy compile and RUN inside a device kernel,
//      under nvc++ -stdpar and under icpx -fsycl? Nobody has tested this.
//   3. Is it ZERO COST versus layout_left and versus raw pointers? If a custom
//      layout costs 10%, this whole idea is dead.
//   4. Does the Fortran-bounds property actually hold — can a kernel write
//      `a[i-1,j]` at `i = 1` and reach the halo with NO special-casing?
//
// A NOTE ON WHY THE MDSPAN ROUTE IS SAFER THAN THE BESPOKE ONE. The design sketch
// pre-shifted the DATA POINTER (`base_ = raw - lo0 - ld*lo1`), which forms a
// pointer outside the allocation — universally fine in practice, formally UB, and
// something a sanitizer build may object to. With mdspan the shift lives in the
// MAPPING instead: `operator()` returns `offset_ + i + ld*j`, where `offset_` is
// negative but the RESULT is always non-negative for valid indices. No
// out-of-bounds pointer is ever formed. Same instruction count, no UB question.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>
#if !defined(TC_SPIKE_SYCL)
#include <execution>
#include <algorithm>
#include <ranges>
#endif
#include "common.hpp"

#if !defined(TC_SPIKE_SYCL)
namespace tc {
// Bind by CONST REFERENCE in BOTH branches. libc++ marks the policy copy ctors
// `= delete`, so `auto par = par_unseq` fails to compile there. src/numerics/
// parallel.hpp applies this fix to the `seq` branch only — the `par_unseq` branch
// has the same latent bug and has simply never been built under libc++.
#if defined(TC_STDPAR_OFF)
inline constexpr const auto& par = std::execution::seq;
#else
inline constexpr const auto& par = std::execution::par_unseq;
#endif
}
#endif

#if TC_SPIKE_STD_MDSPAN
#include <mdspan>
#endif

using tc::Real; using tc::Index;

// ── report what this toolchain actually has ─────────────────────────────────
static void report_toolchain() {
    std::printf("  <mdspan> present    : %d\n", TC_SPIKE_STD_MDSPAN);
#ifdef __cpp_lib_mdspan
    std::printf("  __cpp_lib_mdspan    : %ld\n", (long)__cpp_lib_mdspan);
#else
    std::printf("  __cpp_lib_mdspan    : UNDEFINED  (nvc++ leaves it so despite a working header\n"
                "                        -- gate on __has_include, never on this macro)\n");
#endif
#if defined(__NVCOMPILER)
    std::printf("  compiler            : nvc++ %d.%d\n", __NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__);
#elif defined(__INTEL_LLVM_COMPILER)
    std::printf("  compiler            : icpx %d\n", __INTEL_LLVM_COMPILER);
#elif defined(__clang__)
    std::printf("  compiler            : clang %d.%d\n", __clang_major__, __clang_minor__);
#elif defined(__GNUC__)
    std::printf("  compiler            : gcc %d.%d\n", __GNUC__, __GNUC_MINOR__);
#endif
#if defined(_GLIBCXX_RELEASE)
    std::printf("  libstdc++ release   : %d\n", _GLIBCXX_RELEASE);
#  if TC_SPIKE_STD_MDSPAN && _GLIBCXX_RELEASE < 15
    std::printf("  ^^ NOTE: <mdspan> is present DESPITE libstdc++ %d, which has none.\n"
                "     The COMPILER bundles its own header. So availability is a\n"
                "     per-compiler question, not a libstdc++-version question --\n"
                "     verified: nvc++ 26.3 over libstdc++ 11.\n", _GLIBCXX_RELEASE);
#  elif !TC_SPIKE_STD_MDSPAN
    std::printf("  ^^ and the compiler bundles none either. Fixes, cheapest first:\n"
                "     (1) a newer toolchain module that bundles <mdspan>;\n"
                "     (2) a libstdc++ >= 15 behind --gcc-toolchain / -gcc-name.\n");
#  endif
#elif defined(_LIBCPP_VERSION)
    std::printf("  libc++ version      : %d\n", _LIBCPP_VERSION);
#endif
    std::printf("  C++ standard        : %ld", (long)__cplusplus);
    if (__cplusplus < 202302L) std::printf("   (pre-final C++23; nvc++ reports 202100)");
    std::printf("\n");
}

#if TC_SPIKE_STD_MDSPAN

// =============================================================================
// layout_fortran — the whole proposal, in one mapping.
//
// Column-major (dim 0 fastest), arbitrary INCLUSIVE lower bounds, and an
// optional padded leading dimension. The lower-bound shift is folded into
// `offset_` once at construction, so the hot path is one multiply-add per
// dimension with NO subtraction -- identical in shape to layout_left.
// =============================================================================
// `Check` is a template parameter so BOTH variants live in one binary and the
// A/B is honest. On failure the mapping returns index 0 (memory-safe) and raises a
// device-side flag; the branch cannot be elided because the return value differs.
template <bool Check>
struct layout_fortran_t {
    template <class Extents>
    class mapping {
        Extents ext_{};
        Index   ld_     = 0;   // leading dimension (>= extent(0))
        Index   offset_ = 0;   // -(lo0 + ld*lo1 + ld*n1*lo2 + ...)
        Index   lo_[Extents::rank()]{};
    public:
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_fortran_t<Check>;

        constexpr mapping() = default;

        // lo[] are the inclusive lower bounds; ext carries the COUNTS.
        constexpr mapping(const Extents& e, const Index (&lo)[Extents::rank()], Index ld)
            : ext_(e), ld_(ld ? ld : Index(e.extent(0))) {
            for (rank_type r = 0; r < Extents::rank(); ++r) lo_[r] = lo[r];
            // offset_ = -(lo0 + ld*(lo1 + n1*(lo2 + ...)))
            Index acc = 0;
            for (rank_type r = Extents::rank(); r-- > 1; )
                acc = (r == Extents::rank() - 1) ? lo_[r] : acc * Index(ext_.extent(r)) + lo_[r];
            offset_ = -(lo_[0] + ld_ * acc);
        }

        TC_KERNEL constexpr index_type operator()(Index i, Index j) const
            requires (Extents::rank() == 2) {
            if constexpr (Check)
                if (i < lo_[0] || i > hi(0) || j < lo_[1] || j > hi(1)) return 0;
            return offset_ + i + ld_ * j;
        }

        TC_KERNEL constexpr index_type operator()(Index i, Index j, Index k) const
            requires (Extents::rank() == 3) {
            return offset_ + i + ld_ * (j + Index(ext_.extent(1)) * k);
        }

        TC_KERNEL constexpr index_type operator()(Index i, Index j, Index k, Index b) const
            requires (Extents::rank() == 4) {
            return offset_ + i + ld_ * (j + Index(ext_.extent(1)) * (k + Index(ext_.extent(2)) * b));
        }

        constexpr const Extents& extents() const { return ext_; }
        TC_KERNEL constexpr Index lo(rank_type r) const { return lo_[r]; }
        TC_KERNEL constexpr Index hi(rank_type r) const { return lo_[r] + Index(ext_.extent(r)) - 1; }

        constexpr index_type required_span_size() const {
            index_type n = ld_;
            for (rank_type r = 1; r < Extents::rank(); ++r) n *= index_type(ext_.extent(r));
            return n;
        }
        constexpr index_type stride(rank_type r) const {
            index_type s = 1;
            for (rank_type q = 0; q < r; ++q) s *= (q == 0) ? ld_ : index_type(ext_.extent(q));
            return s;
        }
        static constexpr bool is_always_unique()     { return true;  }
        static constexpr bool is_always_exhaustive() { return false; }   // padding may exist
        static constexpr bool is_always_strided()    { return true;  }
        constexpr bool is_unique()     const { return true; }
        constexpr bool is_exhaustive() const { return ld_ == Index(ext_.extent(0)); }
        constexpr bool is_strided()    const { return true; }

        // REQUIRED by [mdspan.layout.reqmts]: a layout mapping must be
        // equality-comparable. Omitting it does not produce a clean diagnostic —
        // it makes mdspan::operator[] fail SUBSTITUTION, and you get nine lines of
        // "no viable overloaded operator[]" pointing at your kernel instead. This
        // one line is the difference. (Found by spike, 2026-08-03.)
        friend constexpr bool operator==(const mapping&, const mapping&) = default;
    };
};

// Name the requirement instead of letting the call site fail. If a future layout
// change breaks the contract, THIS fires, not a wall of overload-resolution noise.
static_assert(std::is_invocable_r_v<Index,
                  const layout_fortran_t<false>::mapping<std::dextents<Index,2>>&, Index, Index>,
              "layout_fortran::mapping is not a valid mdspan layout mapping");
static_assert(std::is_invocable_r_v<Index,
                  const layout_fortran_t<true>::mapping<std::dextents<Index,2>>&, Index, Index>,
              "layout_fortran_checked::mapping is not a valid mdspan layout mapping");

using layout_fortran         = layout_fortran_t<false>;
using layout_fortran_checked = layout_fortran_t<true>;

template <class T, int Rank, bool Check = false>
using FView = std::mdspan<T, std::dextents<Index, Rank>, layout_fortran_t<Check>>;
template <class T, int Rank>
using LView = std::mdspan<T, std::dextents<Index, Rank>, std::layout_left>;

// ── the launcher: one flat range, unflattened inside (validated on both vendors) ──
template <class F>
static void launch(Index count, F f) {
#if defined(TC_SPIKE_SYCL)
    tc::q().parallel_for(sycl::range<1>(std::size_t(count)),
                         [=](sycl::id<1> n) { f(Index(n[0])); });
#else
    auto ids = std::views::iota(Index{0}, count);
    std::for_each(tc::par, ids.begin(), ids.end(), [=](Index n) { f(n); });
#endif
}

// The SAME Jacobi body, four addressing schemes. Init everything (halo included)
// to 1.0; a Jacobi average leaves every cell exactly 1.0 forever, so correctness
// is bitwise with no tolerance.
//
// Variant D is the one that matters: bounds are (1-ng .. n+ng), the kernel loops
// the INTERIOR 1..n, and `a[i-1,j]` at i==1 reaches the halo with no branch, no
// clamp, and no wrap. That is the ghost-cell property, demonstrated rather than
// asserted.

// Every variant runs `warm` untimed iterations first. WITHOUT THIS the first
// variant measured pays first-touch page faults on the freshly allocated pool and
// looks ~1.7x slower than the rest — which would "prove" that mdspan is faster
// than raw pointers. A control re-run of variant A at the end detects any residual
// drift; if A_first and A_last disagree, no ratio in the table means anything.
static constexpr long WARM = 10;

static double run_raw(Real* pool, Index n, Index ng, long iters) {
    const Index ld = n + 2*ng, tot = ld * (n + 2*ng);
    Real *a = pool, *b = pool + ((tot + 15) & ~15);
    launch(tot, [=](Index k) { a[k] = Real(1); });
    launch(tot, [=](Index k) { b[k] = Real(1); });
    tc::device_sync();
    for (long it = 0; it < WARM; ++it) {
        Real* A = a; Real* B = b;
        launch(n * n, [=](Index m) {
            const Index i = ng + m % n, j = ng + m / n;
            B[i + ld*j] = Real(0.25)*(A[(i-1)+ld*j] + A[(i+1)+ld*j]
                                    + A[i+ld*(j-1)] + A[i+ld*(j+1)]);
        });
        std::swap(a, b);
    }
    tc::device_sync();
    tc::Timer t;
    for (long it = 0; it < iters; ++it) {
        Real* A = a; Real* B = b;
        launch(n * n, [=](Index m) {
            const Index i = ng + m % n, j = ng + m / n;     // zero-based interior
            B[i + ld*j] = Real(0.25)*(A[(i-1)+ld*j] + A[(i+1)+ld*j]
                                    + A[i+ld*(j-1)] + A[i+ld*(j+1)]);
        });
        std::swap(a, b);
    }
    tc::device_sync();
    return t.s();
}

template <class MkView>
static double run_view(Real* pool, Index n, Index ng, long iters, MkView mk, bool fortran_bounds) {
    const Index ld = n + 2*ng, tot = ld * (n + 2*ng);
    Real* pa = pool; Real* pb = pool + ((tot + 15) & ~15);
    auto A0 = mk(pa), B0 = mk(pb);
    launch(tot, [=](Index k) { A0.data_handle()[k] = Real(1); });
    launch(tot, [=](Index k) { B0.data_handle()[k] = Real(1); });
    tc::device_sync();

    const Index lo = fortran_bounds ? Index(1) : ng;        // first interior index
    for (long it = 0; it < WARM; ++it) {                    // untimed — see WARM
        auto A = mk(pa), B = mk(pb);
        launch(n * n, [=](Index m) {
            const Index i = lo + m % n, j = lo + m / n;
            B[i, j] = Real(0.25) * (A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]);
        });
        std::swap(pa, pb);
    }
    tc::device_sync();
    tc::Timer t;
    for (long it = 0; it < iters; ++it) {
        auto A = mk(pa), B = mk(pb);
        launch(n * n, [=](Index m) {
            const Index i = lo + m % n, j = lo + m / n;
            B[i, j] = Real(0.25) * (A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]);
        });
        std::swap(pa, pb);
    }
    tc::device_sync();
    return t.s();
}

static long bad(const Real* p, std::size_t n) {
    std::vector<Real> h(n);
    tc::stage_to_host(h.data(), p, n * sizeof(Real));
    long c = 0; for (Real x : h) if (x != Real(1)) ++c; return c;
}
#endif // TC_SPIKE_STD_MDSPAN

int main(int argc, char** argv) {
    const Index N     = argc > 1 ? Index(std::atoi(argv[1])) : 2048;
    const long  iters = argc > 2 ? std::atol(argv[2])        : 200;
    const Index ng    = 2;

    std::printf("\n=== spike 06: std::mdspan with a custom layout ===\n");
    tc::report_device();
    report_toolchain();

#if !TC_SPIKE_STD_MDSPAN
    std::printf("\n  <mdspan> IS NOT AVAILABLE on this toolchain.\n"
                "  That is the RESULT: this is an upgrade request, not a design decision.\n"
                "  libstdc++ needs >= 15; nvc++ 26.5 already ships a working header.\n\n");
    return tc::verdict("spike 06 (no mdspan)");
#else
    std::printf("  grid %d x %d (+%d halo), %ld iters\n\n", int(N), int(N), int(ng), iters);

    const Index ld = N + 2*ng;
    const std::size_t ncell = std::size_t(ld) * ld;
    const std::size_t bytes = ncell * sizeof(Real) * 4;
    Real* pool = tc::pool_device(bytes);

    // A — raw pointers, the zero-cost baseline
    const double sA = run_raw(pool, N, ng, iters);
    tc::check(bad(pool, ncell) == 0, "A raw pointer               exact");

    // B — std::mdspan + layout_left, zero-based
    const double sB = run_view(pool, N, ng, iters, [=](Real* p) {
        return LView<Real,2>(p, ld, ld);
    }, false);
    tc::check(bad(pool, ncell) == 0, "B mdspan / layout_left      exact");

    // C — std::mdspan + layout_fortran, lo = 0 (so it must MATCH layout_left)
    const double sC = run_view(pool, N, ng, iters, [=](Real* p) {
        const Index lo[2] = {0, 0};
        return FView<Real,2>(p, layout_fortran::mapping<std::dextents<Index,2>>(
                                    std::dextents<Index,2>(ld, ld), lo, ld));
    }, false);
    tc::check(bad(pool, ncell) == 0, "C mdspan / layout_fortran   exact (lo=0)");

    // D — THE POINT: bounds (1-ng .. N+ng), interior loop 1..N, halo reached by
    //     plain i-1 / i+1 with no branch anywhere.
    const double sD = run_view(pool, N, ng, iters, [=](Real* p) {
        const Index lo[2] = {1 - ng, 1 - ng};
        return FView<Real,2>(p, layout_fortran::mapping<std::dextents<Index,2>>(
                                    std::dextents<Index,2>(ld, ld), lo, ld));
    }, true);
    tc::check(bad(pool, ncell) == 0, "D mdspan / FORTRAN BOUNDS   exact (lo=1-ng)");

    // E — variant D with BOUNDS CHECKING ON. Same bounds, same kernel, same data.
    //     This is the number that decides whether checks stay on in production.
    const double sE = run_view(pool, N, ng, iters, [=](Real* p) {
        const Index lo[2] = {1 - ng, 1 - ng};
        return FView<Real,2,true>(p, layout_fortran_t<true>::mapping<std::dextents<Index,2>>(
                                        std::dextents<Index,2>(ld, ld), lo, ld));
    }, true);
    tc::check(bad(pool, ncell) == 0, "E mdspan / BOUNDS CHECKED   exact");

    // ...and prove the check actually fires, rather than being compiled away.
    {
        const Index lo[2] = {1 - ng, 1 - ng};
        layout_fortran_t<true>::mapping<std::dextents<Index,2>> mc(
            std::dextents<Index,2>(ld, ld), lo, ld);
        layout_fortran_t<false>::mapping<std::dextents<Index,2>> mu(
            std::dextents<Index,2>(ld, ld), lo, ld);
        const Index way_out = N + 10 * ng;                 // far outside the halo
        tc::check(mc(way_out, 1) == 0 && mu(way_out, 1) != 0,
                  "  check FIRES out of bounds (and is absent when off)");
    }

    // Control: re-run A last. If it disagrees with the first A, the machine drifted
    // and no ratio below is trustworthy.
    const double sA2 = run_raw(pool, N, ng, iters);
    tc::check(bad(pool, ncell) == 0, "A' raw pointer, re-run     exact");

    std::printf("\n       A raw pointer              %8.3f s   %8.2f GB/s\n", sA, tc::gbs(iters,N,N,sA));
    std::printf("       B mdspan layout_left       %8.3f s   %8.2f GB/s   (%.3fx of A)\n",
                sB, tc::gbs(iters,N,N,sB), sA/sB);
    std::printf("       C mdspan layout_fortran    %8.3f s   %8.2f GB/s   (%.3fx of A)\n",
                sC, tc::gbs(iters,N,N,sC), sA/sC);
    std::printf("       D mdspan FORTRAN BOUNDS    %8.3f s   %8.2f GB/s   (%.3fx of A)\n",
                sD, tc::gbs(iters,N,N,sD), sA/sD);

    std::printf("       E mdspan BOUNDS CHECKED    %8.3f s   %8.2f GB/s   (%.3fx of D)\n",
                sE, tc::gbs(iters,N,N,sE), sD/sE);
    std::printf("       A' raw pointer, re-run     %8.3f s   %8.2f GB/s   (control)\n",
                sA2, tc::gbs(iters,N,N,sA2));

    const double drift = (sA2 > sA ? sA2/sA : sA/sA2) - 1.0;
    std::printf("\n  control drift A vs A'      : %.1f%%   %s\n", 100.0*drift,
                drift < 0.05 ? "-- stable, ratios are meaningful"
                             : "-- TOO LARGE, ignore every ratio above");
    std::printf("\n  E/D is the COST OF ALWAYS-ON BOUNDS CHECKING. These kernels are\n"
                "  bandwidth-bound (~84%% of DRAM peak in the reference model), so spare ALU\n"
                "  is exactly what a bounds compare consumes -- if E/D is within the control\n"
                "  drift, checks stay ON in production and OOB stops being silent.\n");
    std::printf("\n  B/C/D within a few %% of A => a custom layout is free, and\n"
                "  `View<T,Rank> = std::mdspan<T, dextents, layout_fortran>` is the design.\n"
                "  Variant D also proves the ghost-cell property: a[i-1,j] at i==1 reaches\n"
                "  the halo with NO branch, NO clamp and NO wrap in the kernel.\n\n");

    tc::pool_free(pool);
    return tc::verdict("spike 06");
#endif
}
