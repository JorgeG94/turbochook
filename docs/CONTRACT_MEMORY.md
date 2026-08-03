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

// Runtime tag for the C ABI / numpy boundary: Python must be told the dtype.
// FIXED underlying type -- FieldDesc marshals it through an int32 field, and an
// implementation-defined width there is a live ABI hazard.
enum class DType : std::int32_t { F32 = 0, F64 = 1 };

// DEFINED, not just declared: workstream 05's C ABI calls it, and a bare declaration
// links only while nobody does. The else branch is a static_assert rather than a
// fallthrough to F64 -- `is_scalar_type` is specialisable so __half/bf16 can be
// Scalar, and a half field announcing itself to numpy as F64 gives wrong itemsize,
// wrong strides and a 4x over-read with no diagnostic.
template <class> inline constexpr bool dependent_false = false;
template <Scalar T> constexpr DType dtype_of() {
    if      constexpr (std::is_same_v<T, float>)  return DType::F32;
    else if constexpr (std::is_same_v<T, double>) return DType::F64;
    else static_assert(dependent_false<T>, "dtype_of: no DType for this scalar - extend DType");
}

} // namespace tc
```

**`Accum` is not here.** An earlier draft defined `Accum<T> = double` in this header "so
reductions accumulate wider than they store". In the default `Real = double` build that is
not wider and the production configuration gains nothing; reproducible conservation totals
need `CONTRACT_RUNTIME` §8.1's **`Efp` fixed-point accumulator**, not a wider float. It is
therefore only meaningful for `float` fields and lives in `device/reduce.hpp` (workstream
02), which is also where the reductions that use it live.

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

```
Column-major (Fortran memory ORDER) is kept -- that is the coalescing contract.
Fortran LOWER BOUNDS are not; see the note below.

  extent    0 .. nx + 2*ng - 1
  interior  ng .. ng + nx - 1        <- Region carries this
  halo      0 .. ng-1   and   ng+nx .. nx+2*ng-1
```

*(There is no separate dimension descriptor. Extents live in `std::dextents<Index,Rank>`
inside the `View`; the interior box lives in `Region`. An earlier draft's `struct Dims`
was the stump of the rejected `dim(lo,hi)` API and is gone.)*

**Why this retires the ghost-cell reindex.** `DESIGN.md` §7 decision 3 records that
moving to real ghost cells is *"a pervasive reindex … not a drop-in"*. It is neither.
A stencil reads `v[i-1, j]`; at `i == ng` that is halo cell `ng-1`, in bounds, with **no
branch, no clamp, no wrap** — and **no operator body changes**. Only the loop bounds move,
and they come from `Region`, derived once from the mesh:

```cpp
auto h = arena.alloc<Real>("h", nx, ny, ng);   // cell counts + halo width; alloc adds 2*ng
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

The class is specified in [`design/ARRAY.md`](design/ARRAY.md) §4, which is authoritative
and compiles. Reproduced here only far enough to fix the contract:

```cpp
enum class Space : std::int32_t { Host = 0, Device = 1 };   // fixed width: crosses the ABI
enum class Loc   : std::int32_t { Center = 0, XFace = 1, YFace = 2, Corner = 3 };

template <class T, int Rank, Space S, Loc L = Loc::Center>
class Array {
    View<T, Rank> v_{};              // default-constructible: Array is a member of
    const char*   label_ = "";       // LayeredState, BaroState, VmixContext ...
public:
    using value_type = T;
    static constexpr int   rank_v = Rank;   // NOT `rank`: mdspan::rank() is a FUNCTION
    static constexpr Space space  = S;
    static constexpr Loc   loc    = L;      // changes EXTENTS; enforced by Arena::alloc

    constexpr Array() = default;
    constexpr Array(View<T,Rank> v, const char* label) : v_(v), label_(label) {}

    // The ONLY thing that crosses to a kernel. Named `kernel_view` because it also
    // BYPASSES the gate below -- see design/ARRAY.md §4.0; lint, not types, closes it.
    constexpr View<T, Rank>       kernel_view()  const { return v_; }
    constexpr View<const T, Rank> ckernel_view() const;

    // Host subscript. A COMPILE ERROR on device storage — not a fault, not a
    // runtime guard. Via the NAMED PREDICATE, not `S == Space::Host` spelled out:
    // there are two predicates and they answer different questions (design §2).
    template <class... I> constexpr T& operator[](I... idx) const {
        static_assert(host_subscriptable(S),
            "host subscript of Space::Device storage - take a mirror() first");
        return v_[idx...];
    }

    constexpr Index       extent(int r) const { return v_.extent(r); }
    constexpr std::size_t bytes() const;      // sizeof(T) * required_span_size()
    constexpr const char* label() const { return label_; }
    static constexpr DType dtype() { return dtype_of<T>(); }
};
```

ASCII only in that message: nvc++'s EDG frontend renders non-ASCII as `???` (verified).

**There is no `lo()`/`hi()`.** Zero-based means the lower bound is always 0, and the
interior box belongs to `Region`, not to the field. An earlier draft of this section
carried both accessors delegating to `v_.lo(r)`/`v_.hi(r)` — members `std::mdspan` does
not have, from the rejected Fortran-bounds design.

### 1.4 `mirror` — and why it is also the numpy boundary

```cpp
// ALWAYS returns Space::Host in the TYPE. When the source is already host-accessible
// -- host build, OR a coherent device build -- the identity lives in the BODY: no
// allocation, same data_handle(), and `copy` below becomes a no-op. That is what lets
// ONE diagnostic / NetCDF / restart / unit-test / numpy path serve the host build,
// coherent machines, and discrete machines with no #ifdef at the call site.
template <class T, int R, Space S, Loc L>
Array<T,R,Space::Host,L> mirror(const Array<T,R,S,L>& a, ScratchScope& scratch);

// SYNCHRONISES before returning. No-op when dst and src share a data handle.
template <class T, int R, Space D, Space Sr, Loc L>
void copy(const Array<T,R,D,L>& dst, const Array<T,R,Sr,L>& src);
```

Three things that must not drift, each of which the previous draft got wrong:

- **`ScratchScope&`, not `Arena&`.** Every crossing happens inside the time loop, and the
  arena is sealed by then (§1.5) — so an `Arena&` overload throws at the first diagnostic.
- **`Loc L` is a template parameter.** Without it, deduction fails for every `Loc::XFace` /
  `YFace` / `Corner` field, i.e. for most of the prognostic state.
- **`copy` synchronises.** `do_concurrent` is contractually async, so a read-back that did
  not sync would be racy on every discrete device — and this is the block everyone
  copy-pastes, so the race would ship everywhere. It is declared, not defined, here on
  purpose: a body that unconditionally called `memcpy_to_host` would run an H2D copy
  backwards.

Read-back is then written once, everywhere:

```cpp
auto hh = mirror(h, scratch);
copy(hh, h);                                  // syncs; a no-op if mirror was the identity
for (Index j = ng; j < ng + ny; ++j)
    for (Index i = ng; i < ng + nx; ++i)
        out << hh[i, j];                      // interior, zero-based -- NOT 1..nx
```

Both details in that loop are load-bearing. `hh[i, j]`, because **`mdspan` has no
`operator()`** (§1.2). And `ng .. ng+nx-1`, because under zero-basing a `1 <= i <= nx`
loop reads the halo strip and then one cell past the end — verified to abort under
`_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG`, silently wrong without it.

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
    explicit Arena(MemoryQuantity size, Pool p = Pool::Device);   // ONE device_alloc

    // `Loc L` is a VALUE parameter and must precede the type pack, or it can never be
    // specified. It is also carried into the RETURN type -- an alloc that dropped it
    // would hand back a Loc::Center handle for a face field, defeating the enforcement
    // below. `ng` is the halo WIDTH, not part of the rank: rank is sizeof...(Ds) - 1.
    //
    // `Init` is a TEMPLATE parameter, not a trailing defaulted argument: a function
    // parameter pack must come last, so `alloc(const char*, Ds..., Init = ...)` cannot
    // deduce (verified -- it fails with "no known conversion from int to Init").
    template <class T, Space S = Space::Device, Loc L = Loc::Center,
              Init I = Init::Poison, class... Ds>
    Array<T, sizeof...(Ds) - 1, S, L> alloc(const char* label, Ds... dims_then_ng);

    void seal();                  // after init: any further alloc throws; ALSO the
                                  // single device::sync() covering every async fill
    ScratchScope scratch();       // RAII; restores the bump pointer on destruction
    MemoryQuantity bytes_used() const;
    void report() const;          // per-label breakdown - what mem_report.hpp was for
};
```

Declaration is **cell counts plus the halo width**, zero-based index domain:

```cpp
auto h = arena.alloc<Real>("h", nx, ny, ng);                             // Loc::Center
auto u = arena.alloc<Real, Space::Device, Loc::XFace>("u", nx, ny, ng);  // +1 face DERIVED
```

One convention, not two. A caller that pre-summed `nx + 2*ng` would leave `alloc` unable to
tell halo from interior — and therefore unable to **derive the extra face** that
`Loc::XFace` requires, which is the whole point of putting `Loc` in the type
([`design/ARRAY.md`](design/ARRAY.md) §4.1). A face field allocated without the extra face
makes halo exchange silently wrong.

Rules, all load-bearing:

- **Sized once, never grows.** A reallocation would dangle every `View`.
- **`seal()` after `init()`.** Turns "no allocation in the time loop" from a
  convention into an invariant.
- **A label per allocation.** Cheap now, annoying to retrofit, and it is the honest
  memory report.
- **`ScratchScope` for transients** — RAII restore, exception-safe. This is the C++
  win over the Fortran original, and it finally makes `mark()/restore()` usable.

### 1.6 Three pools, and `Space` is not the discriminator

There are three kinds of storage, and they answer **two different questions**:

| | who can dereference it? | who owns the pages? | allocator |
|---|---|---|---|
| host-local | host | this rank | `aligned_alloc` |
| host-shared | host | every rank on the node | `MPI_Win_allocate_shared` |
| device | device | this rank | `cudaMalloc` / `sycl::malloc_device` |

`Space` answers only the first, because that is the one a kernel body can observe.
Host-local and host-shared are ordinary host pointers and dereference identically, so
promoting the distinction to a third `Space` value would double the instantiation matrix
and turn every `if constexpr (S == Space::Host)` into a two-case test, for a difference
nothing at the point of use can see.

`Pool` selects **which arena you allocate from**, not which overload you call — one
`Arena` per pool, held together, the way `alloc`'s parameter pack requires (a trailing
`Pool` argument cannot follow `Ds...` any more than a trailing `Init` could):

```cpp
enum class Pool { HostLocal, HostShared, Device };

class Arenas {                       // one bump stack per pool
public:
    Arena& device();
    Arena& host();
    Arena& shared();                 // COLLECTIVE -- see below
    void   seal();                   // seals all three; one barrier
};

// Space::Host in the TYPE, the shared pool in the ALLOCATION:
auto tide = arenas.shared().alloc<Real, Space::Host>("tide_amp", nx, ny, /*ng=*/0);
```

**The device pool is one plain `cudaMalloc`.** Not `cudaMallocAsync`: the async allocator's
value is stream-ordered reuse across repeated alloc/free, and the sealed arena exists
precisely so that never happens. Plain `cudaMalloc` also keeps the base pointer stable,
which is what makes CUDA graph capture legal.

> **No separate pinned pool.** Page-locked host memory is what makes `cudaMemcpyAsync`
> genuinely async on a discrete GPU — out of pageable memory the driver stages through its
> own bounce buffer. We are not building a pool for it: on GH200 the C2C coherence makes
> the distinction largely moot, and on the discrete path the cost lands on diagnostic
> read-back at cadence rather than per-step. Revisit only if host-staged MPI halos (i.e.
> CUDA-aware MPI off) ever become the production path — that one *is* per-step.

**Shared allocation is COLLECTIVE on the node communicator**, and that is the real
constraint. Every rank on the node must make the same `alloc` calls, in the same order,
with the same sizes — a single rank-dependent `if` in setup (`if (has_open_boundary)
arena.alloc(...)`) hangs the node with no diagnostic. The sealed-arena design already
satisfies this by construction (allocation happens once, at init, in deterministic program
order), but it must be a **stated invariant** rather than an accident: `seal()` hashes the
allocation sequence and `MPI_Allreduce`s it, turning a hang into an error message for the
cost of one reduction.

Two consequences:

- **`MPI_Win_free` is collective too**, so tearing down the shared pool is a collective
  call. It gets an explicit `finalize()` rather than pure RAII, because one rank unwinding
  out of an exception path while the others do not is a deadlock.
- **`seal()` is already the barrier** the shared pattern needs. Allocate collectively →
  fill once → barrier → read-only forever, and `seal()` is that barrier.

Read-only-after-init shared arrays are typed `Array<const T, …>`. Concurrent unsynchronised
writes into a window is the failure mode worth catching in the type system, and `const`
catches it exactly.

> **Deferred:** using a shared window for on-node halo exchange by direct load/store. That
> is a different mechanism — the window holds *mutable* subdomain data, "read-only forever"
> is gone, and it needs its own synchronisation protocol. It also matters much less on the
> GPU path, where on-node neighbours go GPU-direct and never touch host memory.

### 1.7 Size the pools before allocating any of them

A requirement is **not a number, it is a polynomial in one flexible dimension** — and the
useful operation is inverting it:

```cpp
class MemoryRequirement {
public:
    static MemoryRequirement Static(MemoryQuantity);          // n^0
    static MemoryRequirement FlexLinear(MemoryQuantity);      // n^1  -- halo, edge terms
    static MemoryRequirement FlexQuadratic(MemoryQuantity);   // n^2  -- tile interior

    MemoryRequirement  operator+ (const MemoryRequirement&) const;  // BOTH live at once
    MemoryRequirement  operator| (const MemoryRequirement&) const;  // ALTERNATIVES -> max
    MemoryRequirement  operator* (std::uint32_t) const;             // n copies

    MemoryQuantity total(std::optional<std::size_t> flex_dim = {}) const;
    std::size_t    calculate_flex_dim(MemoryQuantity available) const;   // the inverse
};

class MemoryRequirements {                     // the per-pool, per-label tree
public:
    void add(const char* label, Pool, MemoryRequirement);
    MemoryQuantity total(Pool) const;
    bool fits(const Capabilities&, std::string& why_not) const;
    void report() const;                       // tree_printer: per-label, per-pool
};
```

**`+` versus `|` is the whole point.** Two fields that coexist sum; two that are
alternatives — a scratch buffer used by remap *or* by vmix, never both — take the max.
Without the distinction every requirement is a worst-case sum, which over-reserves the
scratch tier and makes the reported figure useless for deciding anything. `ScratchScope`
nesting is exactly a `|` fold.

**The flexible dimension is a tile edge.** For a tile of edge `n`: interior storage goes as
`n^2`, halo storage as `4*n*ng`, everything else is static. So

```
total(n) = static + (4*ng*per_cell)*n + (per_cell*nz*nfields)*n^2
```

and `calculate_flex_dim(free)` answers *"how large a tile fits in what is left"* — the
question a 1 km global run with 100 layers must answer, because per-column scratch for a
full subdomain will not fit. MOM6 tiles for the same reason with a compile-time block size.

**D1 does not need this.** Every requirement it registers is `Static`, and a polynomial
with zero flex coefficients *is* a plain quantity — so the shape costs nothing now and
retrofitting it later would touch every `add()` call site. That is the opposite trade from
the general units library (§00), where the machinery is large and the second user is
hypothetical.

Setup already walks the operator set to derive `ng = max(halo_width)`; the same pass sums
every field's bytes. Checking that total against `capabilities()` **before the first
`device_alloc`** turns a mid-setup OOM — half a pool live, no useful message — into:

```
required: device 43.2 GiB   available: 31.7 GiB   short by 11.5 GiB
  prognostic  18.4 GiB  (h, u, v, S, T x nz=75)
  diagnostics 15.7 GiB  <-- 8 diagnostics at 3-D cadence
  scratch      9.1 GiB
```

For the shared pool the same sum is `MPI_Allreduce`d over the node, which is the only way
to catch "128 ranks x 340 MB does not fit" before it hangs.

`MemoryQuantity` is a real type, not a `std::size_t`: it carries its own alignment, rounds
up through `round_up()` rather than a `(n+255)/256*256` scattered across call sites, and
prints itself. Alignment belongs in the type because the arena bump-allocates sub-arrays
out of one slab and each one needs its own guarantee — `cudaMalloc` promises 256 B,
`sycl::malloc_device` promises considerably less clearly.

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
    device::do_concurrent(tag, m.interior(), Stencil{1}, Views{kh, fx, fy},
                          [=] TC_KERNEL (Index i, Index j) {
        kh[i, j] -= ( fx[i+1, j] - fx[i, j]
                    + fy[i, j+1] - fy[i, j] ) * inv_area;
    });
}
```

Note what is absent: no `#ifdef`, no backend name, no manual index arithmetic, no
offset for the halo, no launch configuration. And note `fx[i+1,j]` at the last interior
`i` reads a halo cell that *exists because the halo is in the extents* — no clamp, no
branch, no wrap logic in the operator.

**`kh[i, j]`, not `kh(i, j)`** — `mdspan` has no call operator (§1.2). The subscript
takes a comma-separated index list; note that inside a function-like macro it needs an
extra paren layer (`assert((v[i,j] == x))`), because the commas are otherwise read as
macro-argument separators.

Call it with `array.kernel_view()`:

```cpp
apply_divergence(kh.kernel_view(), fx.kernel_view(), fy.kernel_view(), mesh, inv_area);
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
device::do_concurrent(tag, ncols, [=] TC_KERNEL (Index c) {
    std::array<T, MAX_NZ> dz;            // per-iteration, in local/register memory
    for (Index k = 0; k < nz; ++k) dz[k] = h[c, k];
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
// No halo: the column scratch has no vertical stencil, so ng does not appear.
auto work = scratch.alloc<T>("col_work", ncols, nz, /*ng=*/0);

// A strided 1-D accessor. Needs an explicit constructor -- private members make it a
// non-aggregate, so without one it is unconstructible, brace-init included.
template <class T>
class ColumnView {
    T*    base_ = nullptr;
    Index stride_ = 1;                   // = a.extent(0), taken from the mapping
public:
    ColumnView() = default;
    TC_KERNEL ColumnView(T* base, Index stride) : base_(base), stride_(stride) {}
    TC_KERNEL T& operator[](Index k) const { return base_[k * stride_]; }
};

template <class T, Space S, Loc L>
TC_KERNEL ColumnView<T> column(const Array<T,2,S,L>& a, Index c);

device::do_concurrent(tag, ncols, [=] TC_KERNEL (Index c) {
    ColumnView<T> col = column(work, c);            // {&work[c,0], work.extent(0)}
    for (Index k = 0; k < nz; ++k) col[k] = ...;    // coalesced across threads at each k
});
```

**`k` runs `0 .. nz-1` in a column kernel**, and `col[k]`, not `col(k)`. The surrounding
3-D array is halo'd; the column scratch is not, because there is no stencil in the
vertical to need one.

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
   **And `k` increases UPWARD: `k = ng` is the bed, `k = ng+nz-1` is the surface layer**
   (zero-based, halo in the extents — §1.1; in unhalo'd column scratch that is `0` and
   `nz-1`). This is rakali's convention. MOM6 and SIS2 both use the opposite (surface
   first, `0` = the snow layer in the ice column), so **every ported formula flips**
   — two independent surveys named this as the single highest porting hazard, and
   rakali's sea-ice notes call the flip "highest hazard" outright. Decide it once,
   here, and gate it with tests whose failure mode is unmissable: a dense current must
   settle at the bed, a buoyant plume must sit at the surface layer, and a positive
   surface heat flux must warm the surface layer while preserving stratification.
   Extra dimensions extend the same rule outward — a sea-ice thickness distribution is
   `(i, j, cat)` / `(i, j, cat, layer)` with `i` fastest, so the category axis costs
   nothing and is embarrassingly parallel (see §3.1: flattening the launch over
   `cell × category` buys ~5x more independent columns, which is free occupancy for
   exactly the column kernels that are occupancy-bound).
7. **The halo lives in the extents.** Allocating it is a declaration, not a reindex —
   operators read `v[i-1,j]` unchanged and only the loop bounds move, into `Region`.

## 5. Acceptance gate for this layer

- All spikes rebuilt on top of the real `device/` + `core/` and still passing on
  V100, GH200, PVC, and host.
- The migration of existing consumers changes **zero operator math**. If it does, the
  boundary is not where we think it is — which is the cheapest possible moment to
  discover that.
- `arena.report()` accounts for every byte, by label.
- A CI gate on kernel launches per step and host-side per-step time, so abstraction
  overhead cannot accrete invisibly.
