# The memory & device contract

Everything is built on this. Two layers, one boundary:

```
  device/   the ONLY backend-aware code. Allocation + launch. Nothing else.
  core/     Space · Loc · View (std::mdspan) · Array · Arena · mirror
  ─────────────────────────────────────────────────────────────
  a kernel receives ONLY a trivially-copyable View + POD, by value
```

Every design choice below is either a measured result from [`../spikes/`](../spikes/)
or a direct consequence of one. Nothing here is aspirational.

---

## 0. Precision is a parameter, not a typedef

`Real` is a **default**, never a mandate. Every container and every operator is
generic over the scalar type; `Real` only names what an unqualified call site gets.

```cpp
namespace tc {

// A specialisable TRAIT, not `is_floating_point_v` directly: __half, nv_bfloat16,
// sycl::half and _Float16 are NOT is_floating_point_v on the target toolchains, so a
// concept written against it could never admit them.
template <class T>
struct is_scalar_type : std::is_floating_point<T> {};   // SPECIALISABLE: __half,
template <class T> concept Scalar = is_scalar_type<T>::value;  // bf16 are not
                                                              // is_floating_point_v

#if defined(TC_SINGLE_PRECISION)
using Real = float;
#else
using Real = double;
#endif

// Only meaningful for float fields: Accum<double> == double is NOT wider, so the
// default production build gains nothing from it. Reproducible conservation totals
// need the Efp fixed-point accumulator (CONTRACT_RUNTIME §8.1), not a wider float.
template <Scalar T> using Accum = std::conditional_t<std::is_same_v<T,float>, double, T>;

// Runtime tag for the C ABI / numpy boundary: Python must be told the dtype.
enum class DType { F32, F64 };
template <Scalar T> constexpr DType dtype_of();

} // namespace tc
```

### 0.1 Three different things called "variable precision"

Only the first is a typedef; the other two are why the templating matters.

1. **Whole-program working precision.** One switch, `float` or `double` everywhere.
   Falls out for free once operators are generic.
2. **Mixed precision across fields.** fp32 tracers beside fp64 prognostics; fp32
   diagnostics and output beside fp64 state. **The Arena supports this with no work
   at all** — it hands out bytes, so `alloc<float>` and `alloc<double>` coexist in
   one pool. This is the largest untapped performance lever we have: every kernel we
   have measured is bandwidth-bound, so fp32 storage is worth ~2x — more than the
   9–19% the native-CUDA escape hatch buys.
3. **Mixed precision within one kernel** (fp32 storage, fp64 compute). Leave the
   seam, don't build it:

```cpp
template <Scalar TStore, Scalar TComp = TStore>   // default: identical to today
void apply_divergence(View<TStore,2> kh, ...);
```

### 0.2 The literal rule

**Never write a bare floating literal in a kernel. Use `T(0.25)`.**

In a `float` kernel, `x * 0.25` promotes to `double`, does the arithmetic in double,
and narrows back — silently emitting double-precision ops on hardware where they can
cost 1:32, and quietly defeating the whole point of the fp32 build. This is the same
rule Oceananigans enforces (`zero(grid)`, `one(grid)`, `convert(FT, 1//2)`) and rakali
hit from the other direction with `-Kieee`. It is a lint rule, not a style preference.

### 0.3 The cost, and the discipline

Templating on the scalar multiplies the instantiation count on top of every other
compile-time axis — and §3 of [`REDESIGN.md`](REDESIGN.md) already counts ~2000. It
also doubles the CI matrix.

So: **template the code, instantiate a small set.** The scalar type genuinely must be
compile-time (its polymorphism boundary is per-cell — the §3 rule), but the *set of
combinations we build* is a deliberate, enumerated decision, and the "not built" error
must name what is available.

**Spell the integer widths**, because they are load-bearing and `long` is 32-bit under
LLP64: `Index = std::int32_t` (32-bit addressing measured ~1.06× on address-bound
kernels, neutral on the register-bound column kernels that are 58% of a stage) and
`GlobalIndex = std::int64_t`, used *only* for the global **horizontal** index and I/O
offsets. You never form a global 3-D index; do not let the concept exist.

---

## 1. `core/` — arrays with Fortran memory order and a halo in the extents

### 1.1 The halo lives in the extents; the interior lives in `Region`

```cpp
namespace tc {

// Column-major (Fortran memory order) is kept -- that is the coalescing contract.
// Fortran LOWER BOUNDS are not; see the note below.
//   extent  = nx + 2*ng          interior = ng .. ng+nx-1
struct Dims { Index n[4]; int rank; };

} // namespace tc
```

**Why this retires the ghost-cell reindex.** `DESIGN.md` §7 decision 3 records that
moving to real ghost cells is *"a pervasive reindex … not a drop-in"*. It is neither.
A stencil reads `v[i-1, j]`; at `i == ng` that is halo cell `ng-1`, in bounds, with **no
branch, no clamp, no wrap** — and **no operator body changes**. Only the loop bounds move,
and they come from `Region`, derived once from the mesh:

```cpp
auto h = arena.alloc<Real>("h", nx + 2*ng, ny + 2*ng);   // halo in the extents
// interior kernels loop Region::interior(mesh)          — unchanged forever
// the BC fills the halo strips                          — the only code that knows ng
```

> An earlier draft proposed arbitrary Fortran **lower bounds** (interior `1..nx`) via a
> custom mdspan layout. That cannot conform: the standard fixes the index domain as
> `[0, extent)` and hardened libc++ **traps** on `v[-1, 1]`. The mechanism above never
> needed them — only the interior's *label* differs. See
> [`design/ARRAY.md`](design/ARRAY.md).

### 1.2 `View` — the kernel currency, and it IS `std::mdspan`

```cpp
template <class T, int Rank>
using View = std::mdspan<T, std::dextents<Index, Rank>, std::layout_left>;
```

`layout_left` is column-major — dimension 0 fastest, `v[i,j]` at offset `i + n0*j`. That
is the Fortran memory order and the coalescing contract, unchanged. Subscript is
`v[i, j]` (C++23 multidimensional `operator[]`); **`mdspan` has no `operator()`**.

Conforming buys three things a bespoke type would forfeit: `std::submdspan` (C++26, and
`layout_left` is one of the layouts it is specified for), generic-algorithm interop, and
**bounds checking implemented by the standard library** — `_LIBCPP_HARDENING_MODE` /
`_GLIBCXX_ASSERTIONS`, always on in debug/CI.

Availability is a **per-compiler** question, not a libstdc++ one: nvc++ 26.3 bundles a
working header over libstdc++ 11, while icpx 2025.03 on Aurora bundles none (measured).
Gate on `__has_include(<mdspan>)`, never `__cpp_lib_mdspan`. The fallback shim has the
**same semantics** — zero-based `layout_left` — so no code path diverges.

### 1.3 `Array` — the owning handle, and the gate

```cpp
enum class Space { Host, Device };

template <class T, int Rank, Space S, Loc L = Loc::Center>
class Array {
    View<T, Rank> v_{};              // default-constructible: Array is a member of
    const char*   label_ = "";       // LayeredState, BaroState, VmixContext ...
public:
    static constexpr Space space = S;
    static constexpr Loc   loc   = L;   // changes EXTENTS; enforced by Arena::alloc
    constexpr Array() = default;
    constexpr Array(View<T,Rank> v, const char* label) : v_(v), label_(label) {}

    // The ONLY thing that crosses to a kernel. No space tag: a View only ever
    // exists inside a kernel or inside a host function that already proved access.
    View<T, Rank> view() const { return v_; }

    // Host subscript. A COMPILE ERROR on device storage — not a fault, not a
    // runtime guard. (Kokkos catches most of this at runtime; we can do better
    // because we never select a space dynamically.)
    template <class... I> constexpr T& operator[](I... idx) const {
        static_assert(S == Space::Host,
            "host subscript of Space::Device storage - take a mirror() first");
        return v_[idx...];
    }

    Index lo(int r) const { return v_.lo(r); }
    Index hi(int r) const { return v_.hi(r); }
    const char* label() const { return label_; }
};
```

ASCII only in that message: nvc++'s EDG frontend renders non-ASCII as `???` (verified).

### 1.4 `mirror` — and why it is also the numpy boundary

```cpp
// Identity when the source is already host-accessible: no allocation, no copy,
// and `copy()` below becomes a no-op. That is what lets ONE diagnostic / NetCDF /
// restart / unit-test / numpy path serve the host build, coherent machines, and
// discrete machines with no #ifdef at the call site.
template <class T, int Rank, Space S>
auto mirror(const Array<T,Rank,S>& a, Arena& host_scratch) {
    if constexpr (S == Space::Host) return a;                       // zero cost
    else                            return host_scratch.alloc_like(a);
}

template <class T, int Rank, Space D, Space Src>
void copy(const Array<T,Rank,D>& dst, const Array<T,Rank,Src>& src) {
    if (dst.view().data() == src.view().data()) return;             // the no-op case
    device::memcpy_to_host(dst.view().data(), src.view().data(), bytes(src));
}
```

Read-back is then written once, everywhere:

```cpp
auto hh = mirror(h, scratch);
copy(hh, h);
for (Index j = 1; j <= ny; ++j)
    for (Index i = 1; i <= nx; ++i)
        out << hh(i, j);
```

The same call is the numpy handoff (`Array<Device>` → mirror → buffer protocol), which
is why `Array`/`mirror` must be shaped with extents, strides, dtype and lifetime in
mind now rather than growing a second staging concept for Python later.

### 1.5 `Arena` — one pool, sized once, sealed

```cpp
class Arena {
    std::byte*  pool_ = nullptr;
    std::size_t cap_ = 0, top_ = 0;
    bool        sealed_ = false;
public:
    explicit Arena(std::size_t bytes, Space s = Space::Device);   // ONE device_alloc

    template <class T, Space S = Space::Device, class... Ds>
    Array<T, sizeof...(Ds), S> alloc(const char* label, Ds... dims);

    void seal();                  // after init: any further alloc throws
    ScratchScope scratch();       // RAII; restores the bump pointer on destruction
    std::size_t bytes_used() const;
    void report() const;          // per-label breakdown - what mem_report.hpp was for
};
```

Declaration reads like Fortran:

```cpp
//  real :: h (1-ng:nx+ng, 1-ng:ny+ng)
//  real :: u (1-ng:nx+ng+1, 1-ng:ny+ng)
auto h = arena.alloc<Real>("h", nx + 2*ng, ny + 2*ng);                    // Loc::Center
auto u = arena.alloc<Real, Space::Device, Loc::XFace>("u", nx, ny, ng);  // +1 face DERIVED
```

Rules, all load-bearing:

- **Sized once, never grows.** A reallocation would dangle every `View`.
- **`seal()` after `init()`.** Turns "no allocation in the time loop" from a
  convention into an invariant.
- **A label per allocation.** Cheap now, annoying to retrofit, and it is the honest
  memory report.
- **`ScratchScope` for transients** — RAII restore, exception-safe. This is the C++
  win over the Fortran original, and it finally makes `mark()/restore()` usable.

---

## 2. `device/` — the only backend-aware code

```cpp
namespace tc::device {

// FIVE backends, and note what the split actually is: THREE of them are ISO C++
// parallel algorithms with a different execution policy, and only two leave the
// standard. Serial is not "the non-stdpar build" -- it is stdpar with
// std::execution::seq, so even the test build exercises the standard algorithm
// surface and switching to threads is one token.
//
//   Serial     std::for_each(seq, ...)         deterministic; no TBB; the fast dev loop
//   Multicore  std::for_each(par_unseq, ...)   real CPU target, threaded
//   Cuda/Hip   std::for_each(par_unseq, ...)   nvc++ -stdpar=gpu   <-- still stdpar!
//                 or a __global__ trampoline   only where measured to pay
//   Sycl       oneapi::dpl::for_each(...)      forced off std:: -- see 2.1
enum class Backend { Serial, Multicore, Cuda, Hip, Sycl };
inline constexpr Backend backend = TC_BACKEND;      // build config

struct Caps {
    const char* name;
    bool        coherent;        // pageableMemoryAccess | aspect::usm_system_allocations
    std::size_t global_mem;
    int         wg_default;
};

// initialize() selects ONE DEVICE. On Intel that means one TILE, never a
// COMPOSITE root device -- measured, a stencil on the 2-tile root device runs
// 6.3x SLOWER than the same kernel on a single tile, because implicit scaling
// sends every neighbour read across the inter-tile link. One rank per tile.
void  initialize(int device = 0);
void  finalize();
Caps  capabilities();

void* malloc_device(std::size_t);      // strict: host cannot dereference
void* malloc_shared(std::size_t);      // managed / USM-shared
void  free(void*);
void  memcpy_to_host  (void*, const void*, std::size_t);
void  memcpy_to_device(void*, const void*, std::size_t);
void  prefetch(void*, std::size_t);    // no-op where meaningless
void  sync();

struct LaunchOpts { int workgroup = 0; };   // 0 = backend default; explicit is ~4% on PVC

// THE CHOKEPOINTS. Every kernel and every reduction in the codebase goes through
// these two functions -- nothing else in the tree may name a launcher.
// The views are passed EXPLICITLY: a closure's captures are not reflectable, so the
// launch-time region/stencil check cannot see them otherwise.
template <class Stencil, class Views, class F>
void do_concurrent(KernelTag, Region, Stencil, Views, F, LaunchOpts = {});
template <class F> void do_concurrent(KernelTag, Index count, F, LaunchOpts = {});

// Reductions have the SAME portability problem and need the same treatment:
// std::transform_reduce(par_unseq, ...) on Serial/Multicore/nvc++, oneDPL on SYCL,
// a device reduce on the native paths. `Acc` is deliberately explicit so a float
// field can accumulate in double (§0).
template <class Acc, class Transform, class Combine>
Acc reduce(Index count, Acc init, Transform t, Combine c);

} // namespace tc::device
```

### 2.1 Four implementations, one call site

This is the payoff, and it is why `device/` is built first. The four backends do not
merely differ in spelling — **they disagree on what you are allowed to iterate over**:

```cpp
// ── Serial / Multicore / nvc++-GPU: ONE body, three policies ────────────────
// tc::par is std::execution::seq under TC_STDPAR_OFF, par_unseq otherwise. The
// serial test build and the GPU production build run the SAME standard algorithm
// call; only the policy token differs. (iota_view iterators are accepted by
// libstdc++ and libc++ PSTL -- the rejection below is oneDPL's DEVICE path only.)
auto ids = std::views::iota(Index{0}, count);
std::for_each(tc::par, ids.begin(), ids.end(), f);

// ── SYCL / Intel ────────────────────────────────────────────────────────────
// iota_view is REJECTED by oneDPL (its iterator is a read-only proxy; oneDPL tries
// to wrap the range in a sycl::buffer and copy results back INTO it). counting_iterator
// works, and measured FASTER than native q.parallel_for (418 vs 392 GB/s).
// NEVER -fsycl-pstl-offload: it halves the whole binary, including kernels that
// never touch a PSTL algorithm.
auto first = oneapi::dpl::counting_iterator<Index>(0);
oneapi::dpl::for_each(oneapi::dpl::execution::make_device_policy(queue()),
                      first, first + count, f);

// ── CUDA / HIP (its own .cu TU, compiled by nvcc/hipcc) ─────────────────────
const int wg = o.workgroup ? o.workgroup : 256;
tc_trampoline<<<(count + wg - 1) / wg, wg>>>(count, f);
```

Had those spellings been inlined at 165 call sites the way Oceananigans has them, the
oneDPL discovery would have been a migration. As one function it is four lines.

**Policy: ISO C++ parallel algorithms are the default, and leaving them requires a
measurement.** Three of the five backends are `std::for_each` with a different policy;
`std::transform_reduce` covers the reductions on the same three. We drop to a vendor
launcher only where a number says we must — and the only place a number has said so is
oneDPL's device path, which we should treat as a defect to report rather than a fact to
route around: `std::for_each` **never writes through its iterators** (elements are
passed to the callable), so oneDPL wrapping the range in a `sycl::buffer` and calling
`set_final_data()` to copy results back is unnecessary work that additionally rejects
every read-only random-access range, `iota_view` included. That is worth an upstream
issue. Standard parallelism only matures if real codes lean on it and report what
breaks; being a serious stress case is part of the point of building this way.

**Dependency note.** `std::execution::par_unseq` needs libtbb under GCC/Clang
(nvc++ `-stdpar=multicore` brings its own runtime). That is a *per-configuration
toolchain* requirement for the Multicore build, not a library dependency — the Serial
build, which is what CI and the fast dev loop use, needs nothing. The "stdlib-first,
doctest is the only dependency" policy stands.

### 2.2 The flat-1D launch is measured, not assumed

`do_concurrent` takes a **count**, and the callable unflattens. Verified on both vendors:

- Flattened `range<1>` beat a natural `range<2>` by **1.8x** on PVC (2479 vs 1376 GB/s)
  *despite* `range<2>` eliminating an integer `%` and `÷` per work-item. A 1-D range
  gives consecutive fast-axis indices per group; a 2-D range gets a tiled workgroup
  that breaks full-row coalescing.
- Put the fast axis (`i`, dim 0 of a column-major array) on the fast-varying part of
  the flat index. Adjacent threads then touch adjacent memory.
- Keep the vertical the **slowest**-varying part so columns stay block-local.

### 2.3 `TC_KERNEL`

```cpp
#if defined(__CUDACC__) || defined(__HIPCC__)
#  define TC_KERNEL __host__ __device__
#else
#  define TC_KERNEL                       // stdpar and SYCL need no annotation
#endif
```

Only the CUDA/HIP path needs it. Applied unconditionally; inert elsewhere.

---

## 3. What a kernel looks like

```cpp
// ∂h/∂t -= ∇·F   — flux divergence, one pass, interior only.
// Generic over the scalar: T is deduced from the arguments, and mixing an fp32
// flux with an fp64 tendency is a compile error rather than a silent promotion.
template <Scalar T>
void apply_divergence(View<T,2> kh, View<T,2> fx, View<T,2> fy,
                      Mesh m, T inv_area)
{
    device::do_concurrent(m.n_cells(), [=] TC_KERNEL (Index n) {
        const Index i = m.i_of(n), j = m.j_of(n);
        kh(i, j) -= ( fx(i+1, j) - fx(i, j)
                    + fy(i, j+1) - fy(i, j) ) * inv_area;
    });
}
```

Note what is absent: no `#ifdef`, no backend name, no manual index arithmetic, no
offset for the halo, no launch configuration. And note `fx(i+1,j)` at `i == nx` reads a
halo cell that *exists* because the array was declared with bounds — no clamp, no
branch, no wrap logic in the operator.

Call it with `array.view()`:

```cpp
apply_divergence(kh.view(), fx.view(), fy.view(), mesh, inv_area);
```

Passing `kh` itself does not compile. That is the boundary, enforced.

### 3.1 Loop locals — the `local` / `local_init` specifiers

We borrowed Fortran's name, so be explicit about which of its locality specifiers we
actually offer. Three cases, in increasing cost:

**(a) Scalars — free, and already correct.** A C++ lambda body's locals are
per-invocation by construction. This is `local` with no ceremony, and `local_init` is
just initialising from a captured value.

```cpp
device::do_concurrent(n, [=] TC_KERNEL (Index k) {
    const T inv_h = T(1) / h(k);        // private, register-resident
    T acc = seed;                        // == local_init(acc)
});
```

**(b) Small column locals — `std::array<T, N>` with a COMPILE-TIME extent.** Verified
to offload under nvc++ ([`STATUS.md`](STATUS.md) #1). This is the natural shape for
remap, vmix, and the tridiagonal solve.

```cpp
device::do_concurrent(ncols, [=] TC_KERNEL (Index c) {
    std::array<T, MAX_NZ> dz;            // per-iteration, in local/register memory
    for (Index k = 1; k <= nz; ++k) dz[k-1] = h(c, k);
    // ... serial vertical recurrence ...
});
```

Cost, and it is GPU-fundamental rather than a bug to fix (STATUS #6): fixed-size
per-thread locals plus a serial vertical recurrence make column kernels
**occupancy-bound — roughly 10x lower throughput than a flat map.** Budget for it;
don't try to abstract it away. `MAX_NZ` must be a constant: either a cap, or template
the kernel on `NZ`.

> **THE OCCUPANCY CLIFF — this is where column kernels are won or lost.**
> Column solvers are ~58% of an ocean stage ([`REDESIGN.md`](REDESIGN.md) §1.5), so
> this is the main event, not a corner case. And the cliff is brutal: in the ALE-remap
> benchmark, a first cut that privatised **all ~14 `MAX_NZ`-sized per-column arrays**
> into the loop's private set ran at **0.22x — 4.6x SLOWER than the baseline.** The
> whole private set spilled to local memory and occupancy collapsed. The fix was
> purely structural: privatise only the **six small column vectors** and keep the heavy
> geometry workspace inside a called helper's frame. Same arithmetic, 1.33x instead of
> 0.22x.
>
> The C++ rule that follows: **count per-thread bytes; occupancy is the budget.** Keep
> the kernel lambda's own locals minimal and push bulky workspace into a `TC_KERNEL`
> helper the compiler can scope tightly — and check registers and spills (`ptxas -v` or
> the backend equivalent) rather than guessing. The same benchmark's redi hoist cut
> registers 93→64 and local memory 16.4→14.4 KB, lifting occupancy 34%→50%; **part of
> that kernel's speedup was the occupancy, not the removed FLOPs.**
>
> And the highest-value optimisation for column kernels is not a launch trick, it is
> **removing redundant recomputation of shared column geometry** — the centre column
> rebuilt once instead of four times (redi, 1.24-1.35x), T and S sharing one PPM
> geometry pass instead of two (ALE remap, 1.33x). Both bit-identical, both portable to
> CPU, both worth ~3x more than the language choice.

**(c) Too big for registers, or runtime-sized — a slice of Arena scratch.** Never
allocate inside a kernel (that alone stalls the device on entry/exit; the rakali
lesson). Take a `ScratchScope` outside the launch and hand each iteration its slice.

**The layout here is a real decision, and the obvious choice is wrong.** A column
workspace wants to be indexed `(cell, level)` — **cell fast** — so that at each `k`,
adjacent threads touch adjacent memory. The natural-looking `(level, cell)` gives each
thread a contiguous column and destroys coalescing across threads. GPU-first means
cell-fast; a CPU-tuned build would want the opposite, which is a reason to go through
an accessor rather than raw indices.

```cpp
// (cell, level) — cell is dim 0, therefore fast, therefore coalesced at each k.
auto work = scratch.alloc<T>("col_work", dim(1, ncols), dim(1, nz));

// A strided 1-D accessor so the kernel body still reads like Fortran.
template <class T>
class ColumnView {
    T*    base_;
    Index stride_;                       // = ncols
public:
    // k is 0-based over the column; base_ already points at level 0 of this cell.
    TC_KERNEL T& operator[](Index k) const { return base_[k * stride_]; }
};

device::do_concurrent(ncols, [=] TC_KERNEL (Index c) {
    ColumnView<T> col = work.column(c);  // {&work(c,1), ncols}
    for (Index k = 1; k <= nz; ++k) col(k) = ...;   // coalesced, reads like col(k)
});
```

**Not offered:** `shared` locality. A `do_concurrent` iteration may not write to
anything another iteration reads — that is the no-scatter rule, and it is a hard
precondition rather than a missing feature. Cross-iteration communication means either
a second pass or a `reduce`.

### 3.2 Reductions — the `reduce` specifier

F2018 added `reduce()` as a locality specifier on `do concurrent`, so the pairing is
faithful. It is a separate chokepoint because the portability problem is identical:

```cpp
namespace tc::device {

// `Acc` is EXPLICIT and may be wider than the field type (§0). `t(n) -> Acc` is the
// per-iteration contribution; `c(a,b) -> Acc` must be associative AND commutative,
// because no backend promises an order.
template <class Acc, class Transform, class Combine>
Acc reduce(Index count, Acc init, Transform t, Combine c);

}
```

Four implementations, same shape as `do_concurrent`:

```cpp
// Serial / Multicore / nvc++-GPU
std::transform_reduce(tc::par, ids.begin(), ids.end(), init, c, t);
// SYCL
oneapi::dpl::transform_reduce(device_policy(), first, first + count, init, c, t);
// CUDA/HIP native
cub::DeviceReduce::Reduce(...)   // only if measured to pay
```

A conserved total is then one line, and the integrand is a lambda rather than a new
function — ADR-8's whole claim, now with the accumulator width made explicit:

```cpp
const Accum<T> mass = device::reduce<Accum<T>>(
    m.n_cells(), 0.0,
    [=] TC_KERNEL (Index n) { auto [i,j] = m.ij(n);
                              return Accum<T>(h(i,j)) * m.area(i,j) * m.wet(i,j); },
    [=] TC_KERNEL (Accum<T> a, Accum<T> b) { return a + b; });
```

**Fuse multiple reductions into one pass.** Ocean status lines want max|u|, total mass
and total energy together; three separate calls means three full sweeps and three
device syncs. A struct accumulator costs nothing extra:

```cpp
struct Budget { double mass, ke; Real umax; };
TC_KERNEL Budget join(Budget a, Budget b) {
    return { a.mass + b.mass, a.ke + b.ke, a.umax > b.umax ? a.umax : b.umax };
}
```

(Oceananigans carries a live TODO — *"we should get both the extrema in one single
reduction instead of two"* — precisely because they didn't do this.)

**Two hard rules:**

1. **Every `reduce` ends in a device sync and a scalar copy to host.** That is fine at
   diagnostic cadence and fatal per step — it is the 4–15x host-touch penalty in
   another costume. Reductions are cadence-gated, always.
2. **`Sum::Fast` is not reproducible.** Floating-point addition is not associative, so
   a tree reduction gives different bits run-to-run and, later, across MPI
   decompositions. Since conservation totals are our validation oracle (ADR-8), the
   oracle must not be the thing that wobbles. Seam it now, build it when MPI lands:

```cpp
enum class Sum { Fast, Reproducible };   // Reproducible: fixed-point integer
                                         // accumulation, order-independent by
                                         // construction (the MOM6/FMS approach).
```

Compensated (Kahan/Neumaier) summation improves *accuracy* but is still not
order-independent — it does not solve reproducibility. Only fixed-point does.

---

## 4. The invariants

1. **Only `View` + POD cross into a kernel, by value.** Never an `Array`, never an
   owner, never `this`.
2. **`Space` lives in the type.** A host subscript of device storage is a
   `static_assert`, not a fault.
3. **One pool, sized once, sealed after init.** No allocation in the time loop, ever.
4. **`mirror()` is the only way host code sees device data** — and it is free where
   the data is already reachable.
5. **`device::do_concurrent` is the only launch site in the codebase.** No `for_each`,
   no `<<<>>>`, no `q.parallel_for` outside `device/`.
6. **Column-major, fast axis first**, then a PRIORITY that compacts onto the
   available dimensions: parallel axes (ice category) before serial-recurrence axes
   (vertical k) before axes looped outside the kernel (tracer, history, band). So
   ocean 3-D is `(i,j,k)`, sea ice `(i,j,cat,layer)`, tracers `(i,j,k,tr)` — the
   last two put something *after* the vertical, which the older "vertical slowest"
   wording forbade. See `design/ARRAY.md` §4.2.
   **And `k` increases UPWARD: `k=1` is the bed, `k=nz` is the surface layer.**
   This is rakali's convention. MOM6 and SIS2 both use the opposite (`k=1` at the
   surface, `0` = the snow layer in the ice column), so **every ported formula flips**
   — two independent surveys named this as the single highest porting hazard, and
   rakali's sea-ice notes call the flip "highest hazard" outright. Decide it once,
   here, and gate it with tests whose failure mode is unmissable: a dense current must
   settle at `k=1`, a buoyant plume must sit at `k=nz`, and a positive surface heat
   flux must warm `k=nz` while preserving stratification.
   Extra dimensions extend the same rule outward — a sea-ice thickness distribution is
   `(i, j, cat)` / `(i, j, cat, layer)` with `i` fastest, so the category axis costs
   nothing and is embarrassingly parallel (see §3.1: flattening the launch over
   `cell × category` buys ~5x more independent columns, which is free occupancy for
   exactly the column kernels that are occupancy-bound).
7. **Arrays carry bounds.** Halos are a declaration, not a reindex.

## 5. Acceptance gate for this layer

- All spikes rebuilt on top of the real `device/` + `core/` and still passing on
  V100, GH200, PVC, and host.
- The migration of existing consumers changes **zero operator math**. If it does, the
  boundary is not where we think it is — which is the cheapest possible moment to
  discover that.
- `arena.report()` accounts for every byte, by label.
- A CI gate on kernel launches per step and host-side per-step time, so abstraction
  overhead cannot accrete invisibly.
