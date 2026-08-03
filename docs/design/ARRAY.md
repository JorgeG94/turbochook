# Design — `Array`, `View`, and the vocabulary types

**Status:** proposed, rewritten 2026-08-03 after an adversarial review found six blockers
in the previous draft. This is the interface freeze that unblocks the device, arena, comm
and API workstreams ([`../plans/00_PLAN.md`](../plans/00_PLAN.md) §2).

Contracts: [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) ·
[`../CONTRACT_RUNTIME.md`](../CONTRACT_RUNTIME.md)

> **What changed and why.** The previous draft proposed a custom `layout_fortran` giving
> arbitrary lower bounds. **That cannot be a conforming `std::mdspan` layout.** A layout
> policy controls the *arithmetic* (index → offset); the standard fixes the *domain* as
> `[0, extent(r))`, and `mdspan::operator[]` asserts it before consulting the layout.
> Verified: `v[-1, 1]` traps (SIGTRAP) under `_LIBCPP_HARDENING_MODE_FAST`, libc++'s
> **production** tier, and `m(11,11) = 156` exceeds `required_span_size() = 144`.
>
> **The ghost-cell mechanism never needed lower bounds.** Zero-based, array `(nx+2·ng)`,
> interior `ng .. ng+nx-1`: a kernel loops the interior and reads `v[i-1, j]`, which at
> `i == ng` lands on halo cell `ng-1`. No branch, no clamp, no wrap — identical to the
> Fortran-bounds version. Only the *label* on the interior changes, and no one writes
> that literal by hand: `Region` carries it, derived once from the mesh.
>
> So `View` is a **conforming `std::mdspan` with `layout_left`**, and we get hardening,
> `submdspan`, and generic-algorithm interop that the custom layout would have forfeited.

---

## 1. What these types are, and what they are not

| | `Array<T,Rank,Space,Loc>` | `View<T,Rank>` |
|---|---|---|
| owns | a slice of the arena (the arena owns the bytes) | nothing |
| lives | host side | **crosses into kernels, by value** |
| knows | extents, `Space`, `Loc`, a label | extents |
| subscript | gated by `static_assert` on `Space` | always allowed |

**They are not:** array expressions (no `a + b`), resizable, reference-counted, or
lifetime-managing. Kernels do arithmetic; arrays hold numbers.

**Copying an `Array` copies a handle.** Two `Array`s can name the same storage —
deliberate (RK registers, ping-pong), but surprising, so it is stated here.

---

## 2. Vocabulary

```cpp
namespace tc {

// ── scalars ─────────────────────────────────────────────────────────────────
// A specialisable TRAIT, not `is_floating_point_v` directly: __half, nv_bfloat16,
// sycl::half and _Float16 are NOT is_floating_point_v on the target toolchains, so
// a concept written against it could never admit them. One line now; a concept
// change touching every operator later.
template <class T> struct is_scalar_type : std::is_floating_point<T> {};
template <class T> inline constexpr bool is_scalar_v = is_scalar_type<T>::value;
template <class T> concept Scalar = is_scalar_v<T>;

// Widths are load-bearing (the ~1.06x address-bound measurement, and the 2^31
// global-index argument), so spell them. `long` is 32-bit under LLP64.
using Index       = std::int32_t;   // LOCAL indices only
using GlobalIndex = std::int64_t;   // global HORIZONTAL index and I/O offsets

#if defined(TC_SINGLE_PRECISION)
using Real = float;
#else
using Real = double;
#endif

enum class DType : std::int32_t { F32 = 0, F64 = 1 };   // fixed width: crosses the C ABI
template <Scalar T> constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, float>) return DType::F32;
    else                                    return DType::F64;
}

// ── space ───────────────────────────────────────────────────────────────────
enum class Space : std::int32_t { Host = 0, Device = 1 };

// TWO predicates, because these are two different questions.
//
// (a) MAY host code subscript it? Always strict, regardless of hardware. On a
//     coherent machine a host read of device memory is legal but NOT portable --
//     code that compiles on GH200 would fail on a V100. The gate enforces the
//     discipline, not the hardware.
constexpr bool host_subscriptable(Space s) { return s == Space::Host; }

// (b) Is a mirror FREE? This one does track the hardware, and it is why `mirror`
//     can be the identity on GH200 (measured host-touch penalty 1.1x) while
//     staging on a V100 (4.1x) and PVC (15.4x).
inline constexpr bool kDeviceIsHostCoherent =
#if defined(TC_COHERENT_MEMORY)
    true;
#else
    false;
#endif
constexpr bool mirror_is_identity(Space s) {
    return s == Space::Host || (s == Space::Device && kDeviceIsHostCoherent);
}

// ── C-grid staggering ───────────────────────────────────────────────────────
enum class Loc : std::int32_t { Center = 0, XFace = 1, YFace = 2, Corner = 3 };

} // namespace tc
```

**On `Accum`.** An earlier draft had `Accum<T> = double` "so reductions accumulate wider
than they store". In the default `Real = double` build that is *not* wider and the
production configuration gains nothing. Reproducible conservation totals need
`CONTRACT_RUNTIME` §8.1's **`Efp` fixed-point accumulator**, not a wider float. `Accum`
is therefore only meaningful for `float` fields, and is defined in the reduction header,
not here.

---

## 3. `View` — a conforming `std::mdspan`

```cpp
template <class T, int Rank>
using View = std::mdspan<T, std::dextents<Index, Rank>, std::layout_left>;
```

`layout_left` is column-major: dimension 0 fastest, `v[i,j]` at offset `i + n0*j`. That is
the Fortran memory order and the coalescing contract, unchanged.

### 3.1 Indexing convention

Zero-based, halo included in the extents:

```
Array extent            0 .. nx + 2*ng - 1
interior                ng .. ng + nx - 1        <- Region carries this
halo                    0 .. ng-1  and  ng+nx .. nx+2*ng-1
```

A stencil reads `v[i-1, j]`; at `i == ng` that is halo cell `ng-1`, in bounds, no branch.
**Nobody writes `ng` in a kernel.** The loop bounds come from `Region`, derived once from
the mesh (§6.2), so adding or resizing the halo touches the mesh and nothing else.

Ported MOM6/rakali formulas need a uniform offset at the loop bounds only — the formulas
themselves encode *relative* stencils (`i-1`, `i+1`), which are unchanged.

### 3.2 Bounds checking comes from the standard library, free

Because `View` is a conforming `mdspan`, `_LIBCPP_HARDENING_MODE` and
`_GLIBCXX_ASSERTIONS` check every subscript against the extents — implemented and tested
by the standard library, at no cost to us.

| build | setting |
|---|---|
| debug / CI (host-serial) | `_LIBCPP_HARDENING_MODE_DEBUG` / `_GLIBCXX_ASSERTIONS` — **always** |
| production | `_LIBCPP_HARDENING_MODE_FAST` by default; measure before disabling |

This supersedes the previous draft's hand-rolled per-access checks entirely, and with
them the 8% measurement — which measured one particular check design (returning index 0
injects a data dependency into the address computation, serialising the load behind the
compare) applied only to the rank-2 overload. **Re-measure with hardening before drawing
any conclusion about production cost.**

The launch-time check is still worth having and is complementary: *does this region fit
this array, with room for the stencil?* It catches wrong loop bounds and a too-thin halo
before a single subscript runs. See §6.2 for why it needs the views passed explicitly.

### 3.3 Padding

`std::layout_left_padded` (C++26) supplies an aligned leading dimension when we want one.
Until then `layout_left`, unpadded. This was never measured to matter; it is an option,
not a plan.

### 3.4 Availability

| toolchain | `<mdspan>` | note |
|---|---|---|
| nvc++ 26.3 | **yes** | bundled by the compiler; libstdc++ 11 underneath |
| Apple clang 21 / libc++ | yes | `__cpp_lib_mdspan = 202207` |
| **icpx 2025.03 on Aurora** | **no** | measured; libstdc++ 13, compiler bundles none |
| gcc | **needs libstdc++ 16** | landed in GCC 16, *not* 15 — verified |

Gate on **`__has_include(<mdspan>)`, never `__cpp_lib_mdspan`** — nvc++ leaves the macro
undefined while shipping a working header.

Aurora is the only gap. ALCF installs the latest oneAPI in the US autumn (~October 2026);
since nvc++ bundles its own over libstdc++ 11, a current icpx plausibly will too. See
§10 for the shim — which is now genuinely minimal, because there is no custom layout to
reproduce.

---

## 4. `Array` — the owning handle

```cpp
template <class T, int Rank, Space S, Loc L = Loc::Center>
class Array {
    View<T, Rank> v_{};                       // mdspan IS default-constructible
    const char*   label_ = "";
public:
    using value_type = T;
    static constexpr int   rank_v = Rank;     // `rank_v`, not `rank`: mdspan::rank() is
    static constexpr Space space  = S;        // a FUNCTION, and generic code over both
    static constexpr Loc   loc    = L;        // must not trip over the difference

    constexpr Array() = default;
    constexpr Array(View<T,Rank> v, const char* label) : v_(v), label_(label) {}

    constexpr View<T, Rank> view()  const { return v_; }              // kernels see this
    constexpr View<const T, Rank> cview() const;                      // read-only kernels

    // Host subscript. A COMPILE ERROR on device storage -- not a fault, not a runtime
    // guard. ASCII only: nvc++'s EDG frontend renders non-ASCII as '???'.
    template <class... I>
    constexpr T& operator[](I... idx) const {
        static_assert(host_subscriptable(S),
            "host subscript of Space::Device storage - take a mirror() first");
        return v_[idx...];                    // mdspan has operator[], NOT operator()
    }

    constexpr Index extent(int r) const { return v_.extent(r); }
    constexpr std::size_t bytes() const;
    constexpr const char* label() const { return label_; }
    static constexpr DType dtype() { return dtype_of<T>(); }
};
```

Four things the previous draft got wrong, all verified by the reviewer:

- **`mdspan` has no `operator()`** — P2128 replaced it with `operator[]` before C++23
  shipped.
- **Default constructor**, or `Array` cannot be a member of any state aggregate —
  `LayeredState`, `BaroState`, `VmixContext` are all exactly that shape.
- **A public constructor**, or the Arena (workstream 03) cannot build one.
- **`dtype_of` defined**, not just declared — the C ABI (workstream 05) calls it.

There is no `lo()`/`hi()`. Zero-based means the lower bound is always 0, and the interior
box belongs to `Region`, not to the field.

### 4.1 `Loc` is a template parameter; `Parity` is not

`Loc` earns it: a face field has **one more face than cells** in its normal direction, so
it changes the extents, and it is consumed by halo exchange, output interpolation and
restart layout. `Loc` must therefore be **enforced, not merely declared** — `Arena::alloc`
derives face extents from `Loc` plus the cell dimensions rather than trusting the caller,
because a `Loc::XFace` array allocated without the extra face would make the exchange
silently wrong.

`Parity` (scalars copy across a tripolar fold; vector components sign-flip) is **not** a
template parameter. It changes no layout and no extent — only a sign in one pack loop —
and vector components must exchange **as a pair** regardless, since a 90° rotation *swaps*
them. It belongs at halo-group registration: `group.add(u, v, Parity::Vector)`.

### 4.2 Rank 4 from the start, and the axis order is a rule

Five named consumers, three of them not sea ice:

| use | shape |
|---|---|
| **tracer bundle** | `(i, j, k, tr)` |
| **tendency history** (Adams–Bashforth) | `(i, j, k, hist)` |
| **time-bracketed forcing** | `(i, j, k, 2)` |
| sea-ice ITD | `(i, j, cat, layer)` |
| shortwave bands | `(i, j, k, band)` |

The tracer case is decisive: MOM6's array-of-derived-types-of-pointers registry is the
most GPU-hostile structure in its tracer path, and the fix is one contiguous rank-4 SoA
block. Deferring rank 4 means discovering that after the registry exists.

**Axis priority** — a ranking to be compacted onto the available dimensions, not five
literal dimensions:

```
1. i                                        always fastest -- coalescing
2. j
3. axes iterated in PARALLEL in the kernel   (ice category)
4. axes carrying a SERIAL recurrence         (vertical k)
5. axes looped OUTSIDE the kernel            (tracer, history, bracket, band)
```

Ocean 3-D compacts to `(i,j,k)`; sea ice to `(i,j,cat,layer)`; tracers to `(i,j,k,tr)`,
each tracer a **contiguous rank-3 block** that `submdspan` hands out directly.

Getting it backwards is silent: `(i,j,tr,k)` interleaves tracers and destroys coalescing.

> `CONTRACT_MEMORY.md` invariant 6 has been updated to match — the older "vertical
> slowest" wording forbade both rank-4 shapes above.

### 4.3 Extents are dynamic; `NZ` is not an `Array` concern

Rank is compile-time, every extent runtime. The measured "template on `NZ`" win — a fixed
`NZ_STACK_MAX = 256` against an actual `nz = 75` wastes 3.4× of the per-thread footprint —
is about **`std::array<T,NZ>` column locals declared inside kernels**, templated
separately. Do not put static extents on `Array`.

---

## 5. `mirror` and `copy`

```cpp
// Identity when a host read is free -- host build, OR a coherent device build.
template <class T, int R, Space S, Loc L>
auto mirror(const Array<T,R,S,L>& a, ScratchScope& scratch)
    -> std::conditional_t<mirror_is_identity(S), Array<T,R,S,L>, Array<T,R,Space::Host,L>>;

// SYNCHRONISES before returning. No-op when dst and src share a data handle.
template <class T, int R, Space D, Space Sr, Loc L>
void copy(const Array<T,R,D,L>& dst, const Array<T,R,Sr,L>& src);
```

`copy` **synchronises**. `do_concurrent` is contractually async
([`02_DEVICE.md`](../plans/02_DEVICE.md)), so a read-back that did not sync would be racy
on every discrete device — and this is the sample everyone copies:

```cpp
auto hh = mirror(h, scratch);
copy(hh, h);                                  // syncs
for (Index j = ny_lo; j <= ny_hi; ++j)
    for (Index i = nx_lo; i <= nx_hi; ++i) out << hh[i, j];
```

`mirror_is_identity` is what makes GH200 free (measured host-touch penalty 1.1×) while
V100 (4.1×) and PVC (15.4×) stage. Note it is deliberately **not** the same predicate as
the subscript gate — see §2.

### 5.0 Initialization — on the device, asynchronously, poisoned by default

Three rules, and the reasons are not the obvious ones.

**1. Device arrays are initialized BY A DEVICE KERNEL, never by a host loop plus a copy.**
The obvious reason is avoiding an H2D transfer of the whole arena. The stronger reason is
**first touch**: on a managed pool the first write decides where the pages live, so a
host-side init lands every page on the host and migrates all of them on first device
access — the 4–15× penalty, paid once across gigabytes. Device-side init is what makes
`cudaMemAdvise` + `prefetch` (spike 01) mean anything.

**2. Poison by default. Do not zero.** Zero is a *plausible* value in an ocean model —
`η = 0`, `u = 0`, a zeroed tendency are all legitimate states — so zero-initialisation
silently hides "I forgot to set this field". A signalling NaN propagates and is caught by
the `any_nonfinite` reduction already running at diagnostic cadence, turning a forgotten
initialisation into a loud failure instead of a plausible run.

```cpp
enum class Init { Poison,   // signalling NaN -- the default
                  Zero,     // when zero is genuinely the intended value
                  None };   // "I overwrite this immediately" -- skips a full pass
```

`Init::None` is the escape for a large field about to be filled from numpy or a formula;
it is an optimisation, so it should be justified at the call site, not reached for by
habit.

**3. Every setup fill is ASYNC, and `seal()` is the single barrier.** `alloc` enqueues its
fill and returns. Synchronising per field would give N stalls at startup for no reason,
and it is precisely what stops host-side setup — reading config, opening output files,
building metrics — from overlapping the device work. One `device::sync()` inside
`Arena::seal()` covers all of it, which is a barrier that already had to exist.

```cpp
template <class T, int R, Space S, Loc L>
void fill(const Array<T,R,S,L>&, T value);                   // device kernel, ASYNC

template <class T, int R, Space S, Loc L, class F>
void fill_by(const Array<T,R,S,L>&, Region, F formula);      // device kernel, ASYNC
```

`fill_by` is how an analytic initial condition stays device-side — it is a `TC_KERNEL`
callable over indices, not a host callback, and it is the same three-tier rule as forcing
(`REDESIGN.md` §7.1): array data from numpy is the default path, an analytic menu is
second, and arbitrary per-cell Python is not offered because it cannot run in a kernel.

Data genuinely arriving from a file or numpy still needs one H2D transfer. That is
unavoidable, it happens once, and it should go on a stream so it overlaps the remaining
fills rather than serialising behind them.

> **RAII does not extend to the fill.** `Array` owns nothing — the arena owns the bytes
> — so a constructor cannot meaningfully "initialize on declaration". The initialisation
> policy belongs to `Arena::alloc`, which is the thing that actually creates storage.
> Putting a kernel launch in `Array`'s constructor would also make copying a handle look
> like it might do work, which it must never.

### 5.1 Debug escapes

```cpp
template <class T, int R, Loc L> class HostArray;   // OWNS its memory (vector-backed),
                                                    // so it outlives Arena::seal()
template <class T, int R, Space S, Loc L>
HostArray<T,R,L> debug_snapshot(const Array<T,R,S,L>&);

template <class T, int R, Space S, Loc L>
T debug_peek(const Array<T,R,S,L>&, auto... idx);    // one element, one full sync
```

`HostArray` is a distinct owning type precisely because `Array` owns nothing. Both
escapes allocate, transfer and synchronise, and both are **named to be grep-bannable** —
a pre-commit hook rejecting `debug_` under `src/physics/` costs one line and prevents the
stray per-step host touch that is worth 4× on V100 and 15× on PVC.

We deliberately do **not** build an auto-syncing shadow-copy debug mode. It hides exactly
the cost that must stay visible.

---

## 6. Slicing

`std::submdspan` (C++26) is specified for `layout_left`, so it works on `View` with no
customisation point of our own — one of the concrete gains from conforming.

```cpp
template <class T, int R, Space S, Loc L>
Array<T,R,S,L> subbox(const Array<T,R,S,L>&, Region);     // general case
template <class T, Space S, Loc L>
Array<T,2,S,L> layer (const Array<T,3,S,L>&, Index k);    // rank-reducing
template <class T, Space S, Loc L>
Array<T,3,S,L> tracer(const Array<T,4,S,L>&, Index t);    // CONTIGUOUS -- bundle is slowest
```

Until `submdspan` ships, implement these three by hand against `layout_left` (offset +
extents); the migration is then a genuine one-liner each, which the previous draft claimed
prematurely for a custom layout where it would have been false.

**`column` is the exception**, because it walks the slowest axis and is therefore strided:

```cpp
template <class T> class ColumnView {
    T* base_; Index stride_;
public:
    TC_KERNEL T& operator[](Index k) const { return base_[k * stride_]; }
};
```

Backing scratch is `(cell, level)` — **cell fast** — so adjacent threads touch adjacent
memory at each `k`. The natural-looking `(level, cell)` gives each thread a contiguous
column and destroys coalescing across threads. `stride_` must account for any padding
(`ld * extent(1)`, not `extent(0) * extent(1)`).

### 6.1 `Region` has two uses — do not conflate them

As a **loop bound** it says which cells a kernel visits while the array stays whole; that
is what comm/compute overlap needs, since neighbour reads must reach *outside* the region.
As a **sub-box** it produces a smaller array. Same struct, different verb.

### 6.2 Flattening: the launcher owns it

Measured: a flat 1-D range beat a natural `range<2>` by **1.8×** on PVC (2479 vs 1376
GB/s), *despite* `range<2>` eliminating the per-work-item `%` and `÷`. So flattening is
mandatory; only its location is open, and it belongs to the launcher.

```cpp
// PREFERRED -- launcher flattens, walks, unflattens; the body sees indices.
device::do_concurrent(tag, region, stencil, views, [] TC_KERNEL (Index i, Index j) {...});

// For genuinely elementwise work and reductions.
device::do_concurrent(tag, count, [] TC_KERNEL (Index n) {...});
```

Three reasons: `i = n % nx` is a correctness-for-performance invariant (fifty bodies is
fifty chances to write `n / nx` and lose coalescing with no wrong answer, only a slow
one); it is the only place an integer division by a *runtime* `nx` can be strength-reduced
once for every kernel; and a backend may want to flatten differently (an `nd_range` with
an explicit workgroup is already worth ~4% on PVC).

**The views must be passed explicitly.** A closure's captures are not reflectable, so the
launch-time region check of §3.2 cannot see them otherwise — the previous draft specified
a check the launcher had no way to perform. `stencil` supplies the radius; the launcher
asserts `region ⊆ every view's extents, inset by the radius`.

> `do_concurrent`'s signature is stated three different ways across the frozen documents.
> This one wins; fix `CONTRACT_MEMORY.md` §2 and `00_PLAN.md` §2 in the same commit.

---

## 7. A/B/C staggering — a boundary concern

`Loc` already spans all three Arakawa staggerings: A is `Center`, B is `Corner`, C is
`XFace`/`YFace`. No enum change. What is missing is a **pair** type.

```cpp
enum class Stagger { A, B, C };

template <class T, int Rank, Stagger St, Space S> struct VectorField;

template <class T, int Rank, Space S> struct VectorField<T, Rank, Stagger::C, S> {
    Array<T,Rank,S,Loc::XFace> u;
    Array<T,Rank,S,Loc::YFace> v;
};
// B: both Loc::Corner.  A: both Loc::Center.

template <class T, Space S = Space::Device> using CVector2 = VectorField<T,2,Stagger::C,S>;
template <class T, Space S = Space::Device> using CVector3 = VectorField<T,3,Stagger::C,S>;
```

**`Rank` is a parameter**: D1's first milestone is 2-D barotropic, whose `(ubt, vbt)` pair
is rank 2. A rank-3-only `VectorField` could not hold the demonstrator's own velocities.

It earns its place three times: halo exchange (components go as a pair — they sign-flip
across a 180° fold and *swap* across a 90° rotation), coupling (SIS2 tags its stress with
`flux_uv_stagger`; MOM6 reads it back as `wind_stagger`), and frame rotation.

**The core is C-grid, unconditionally.** A/B appear only at boundaries — forcing ingest
(A→C), ice coupling (B or A↔C), diagnostic output (C→A). Restart stays native.

```cpp
template <Stagger To, class T, int R, Stagger From, Space S>
VectorField<T,R,To,S> restagger(const VectorField<T,R,From,S>&, const Mesh&, ScratchScope&);
```

`ScratchScope&`, not `Arena&`: every crossing happens inside the time loop, and the arena
is sealed by then.

### 7.1 Three orthogonal transforms — keep them separate

1. **Staggering** (A/B/C) — an interpolation.
2. **Vector frame** — geographic (east/north) ↔ grid-aligned (i/j), a **rotation** by the
   local grid angle. On a tripolar grid the i-direction is not east, and near the fold it
   is nowhere near east. Independent of staggering and equally mandatory.
3. **Horizontal mesh** — a different grid entirely. **Out of scope**: runtime regridding
   is ~7k lines in FMS that MOM6 uses only to ingest arbitrary observations. Require
   pre-interpolated inputs.

A single `convert()` doing (1) and (2) together is how you get a field rotated twice, or
not at all, with nothing at the call site to tell you which.

---

## 8. The ABI surface

```cpp
extern "C" {
struct FieldDesc {
    void*        data;
    std::int32_t dtype;        // DType
    std::int32_t space;        // Space  -- §5.1 promised this; the previous draft omitted it
    std::int32_t loc;          // Loc    -- Python needs staggering for output and restart
    std::int32_t rank;         // <= 4
    std::int64_t extent[4];
    std::int64_t stride[4];    // in BYTES (numpy's convention), stride[0] == sizeof(T)
};
}
static_assert(std::is_standard_layout_v<FieldDesc>);
```

Fixed-width members throughout, `extern "C"`, strides in **bytes** because that is what
numpy expects. `static_assert(Rank <= 4)` wherever a `FieldDesc` is produced.

---

## 9. Conventions this type encodes

1. **Column-major, dimension 0 fastest.** The parallel index maps to it.
2. **`k` increases UPWARD: `k = ng` is the bed, `k = ng+nz-1` is the surface layer.**
   MOM6 and SIS2 both use the opposite (`k=1` at the surface; `0` = snow in the ice
   column), so every ported formula flips — two independent surveys named this the top
   porting hazard. Gated by three tests whose failure is unmissable: a dense current
   settles at the bed, a buoyant plume sits at the surface, a positive surface heat flux
   warms the top layer.
3. **Extra dimensions extend outward** per the §4.2 priority.
4. **Face fields carry the extra face**, derived by `Arena::alloc` from `Loc` (§4.1).
   Symmetric memory, unconditionally.

---

## 10. The fallback shim — scope and sunset

Needed only where `<mdspan>` is absent, which today is **icpx on Aurora** (measured:
icpx 2025.03 over libstdc++ 13, neither supplying a header).

Because there is no custom layout, the shim is genuinely minimal: a `layout_left`-shaped
non-owning view with `operator[]` for ranks 2–4, `extent()`, and `data_handle()`. **No
layout policies, no accessors, no `submdspan`** — anything richer is a signal to wait for
the real header. Never `kokkos/mdspan`; the dependency policy forbids it by name.

Critically, **the shim and the real path now have the same semantics** — zero-based
`layout_left` either way. The previous draft's contradiction (physics depending on
Fortran bounds that the shim could not provide) does not arise.

**Sunset: delete it when `__has_include(<mdspan>)` holds on every toolchain in the CI
matrix.** Next review ~October 2026 against the new ALCF oneAPI; `make s06_sycl` prints
the availability block that decides it.

---

## 11. Open decisions

1. **Production hardening tier.** `_LIBCPP_HARDENING_MODE_FAST` by default is proposed.
   The previous 8% figure measured a different, worse check design and does not transfer —
   re-measure with hardening on the GH200 before settling this.
2. **`TC_COHERENT_MEMORY` detection.** Build flag, or a `capabilities()` query latched at
   `device::initialize`? A flag is simpler and matches the per-target build matrix, but a
   mismatch between flag and hardware silently makes `mirror` wrong.
3. **`ColumnView` subscript base.** Whether `k` runs `0..nz-1` or `ng..ng+nz-1` inside a
   column kernel — the surrounding array is halo'd, the scratch need not be. Pick one and
   assert it.

---

## 12. Acceptance

- `View` and every `Array` alias are `std::is_trivially_copyable_v` — `static_assert`ed.
- The whole of §4 **compiles**, instantiated for `Space::Host` and `Space::Device`,
  ranks 2–4, `float` and `double`, and as a member of a default-constructed aggregate.
- `TC_NEGATIVE` build fails with the named `static_assert` under nvc++ **and** any host
  compiler with `<mdspan>` (not "g++", which has none until GCC 16).
- A hardened build (`_LIBCPP_HARDENING_MODE_DEBUG`) runs the full test suite **without
  trapping** — the check that would have caught the previous draft's blocker.
- `mirror` of host-accessible storage returns the same data handle and allocates nothing;
  under `TC_COHERENT_MEMORY` that includes `Space::Device`.
- Signatures published to plans 02–05, and those plans updated in the same commit.
