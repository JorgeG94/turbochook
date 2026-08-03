# Foundation plan — the SWE demonstrator

The target that gates everything below, and the interface freeze that lets the
workstreams run in parallel.

Design contracts: [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) ·
[`../CONTRACT_RUNTIME.md`](../CONTRACT_RUNTIME.md) · [`../REDESIGN.md`](../REDESIGN.md)

---

## 1. Two demonstrators, then physics

### D1 — Shallow water (breadth)

Rebuild the **existing, already-validated** C-grid solver on the new foundation:
2D barotropic, then two-layer baroclinic on a spherical periodic-x channel.

Why this and not something new: `main` already runs this to 580 days and rolls the
`bc_inst` jet into a clean eddy field. **Every failure is therefore a foundation
failure, not a physics failure** — which is the entire point of a demonstrator. And
the oracles are analytical and already passing:

| gate | oracle |
|---|---|
| lake-at-rest | flat η stays flat, zero RHS |
| gravity wave | `c = √(gH)` to <0.01% |
| geostrophic balance | residual at machine-eps |
| mass conservation | drift ~1e-12 (telescoping continuity) |
| two-layer modes | barotropic `√(g(H₁+H₂))`, baroclinic `√(g'H₁H₂/H)` to 0.1% |
| **regression** | eddy field vs the retiring `main` at matched settings |
| **multi-rank** | 2×2 decomposition reproduces single-rank |

It exercises: bounded arrays + halos, `Space`/`mirror`, the arena, `do_concurrent`,
`reduce`, slot dispatch (continuity/Coriolis/PGF), the stepper as a combination over Φ
(**including MOM6 split RK2**, which the two-layer case already uses), MPI
decomposition and exchange, the Python API, diagnostics and output.

### D2 — One column kernel (depth)

SWE has **no column kernels**, and those are ~58% of a real ocean stage
(REDESIGN §1.5) and where the occupancy cliff lives — a 4.6× swing. So D1 alone
proves the foundation is *broad* but not that it is *deep*.

Pick one — ALE remap, or an implicit vertical diffusion solve — and build it purely as
a foundation probe: fixed-size column locals, the scratch-slice accessor
(`CONTRACT_MEMORY.md` §3.1), a register/occupancy measurement, and a native-launcher
A/B. Gate: **conservative to machine-eps, and occupancy measured, not assumed.**

**No parameterizations until D1 and D2 are both green.** Closures are where "is it
broken or is it just the physics?" stops being answerable.

---

## 2. The interface freeze

`Array`/`View` is the keystone — the arena hands them out, the comm layer exchanges
them, the API hands them to numpy. **Freeze this first; then the four workstreams are
genuinely independent.** These signatures are the contract between plans 01–05.

> **Superseded — see [`../design/ARRAY.md`](../design/ARRAY.md), which is authoritative.**
> An adversarial review found the earlier bespoke-`View` sketch could not work: a custom
> mdspan layout with Fortran lower bounds is not conforming (the standard fixes the index
> domain as `[0, extent)`, and hardened libc++ traps on the ghost-cell access). `View` is
> now a conforming `std::mdspan` with `layout_left`, zero-based, and `Parity` has moved
> off `Array` to halo-group registration.

```cpp
namespace tc {

using Index       = std::int32_t;   // LOCAL indices
using GlobalIndex = std::int64_t;   // global HORIZONTAL index + I/O offsets

template <class T> struct is_scalar_type : std::is_floating_point<T> {};   // specialisable
template <class T> concept Scalar = is_scalar_type<T>::value;

enum class Space : std::int32_t { Host, Device };
enum class Loc   : std::int32_t { Center, XFace, YFace, Corner };
enum class DType : std::int32_t { F32, F64 };
enum class Parity { Scalar, Vector };      // NOT on Array -- halo-group registration only

constexpr bool host_subscriptable(Space);  // strict always: portability discipline
constexpr bool mirror_is_identity(Space);  // tracks hardware: free on coherent devices

// The kernel currency IS a conforming std::mdspan. Zero-based; the halo is in the
// extents and the interior box lives in Region.
template <class T, int Rank>
using View = std::mdspan<T, std::dextents<Index, Rank>, std::layout_left>;

template <class T, int Rank, Space S, Loc L = Loc::Center>
class Array {                                       // owning handle, host-side
public:
    static constexpr int   rank_v = Rank;           // not `rank`: mdspan::rank() is a fn
    static constexpr Space space  = S;
    static constexpr Loc   loc    = L;
    constexpr Array() = default;
    constexpr Array(View<T,Rank>, const char* label);
    constexpr View<T,Rank>       view()  const;
    constexpr View<const T,Rank> cview() const;
    template <class... I> constexpr T& operator[](I...) const;   // static_assert on Space
    constexpr Index extent(int r) const;
    static constexpr DType dtype();
};

} // namespace tc
```

```cpp
namespace tc::device {

enum class Backend  { Serial, Multicore, Cuda, Hip, Sycl };

// Regime is an ANNOTATION, not a dispatch input -- stdpar is the path and native
// is an insertion at a named kernel, so there is no routine choice to make. The
// byte+flop recorder checks the claim rather than trusting it. `Launcher::Auto` is
// deleted; AB survives because it is how an insertion gets justified.
enum class Regime   { MemoryBound, LaunchBound, RegisterBound };
enum class Launcher { Stdpar, Native, AB };

struct Stream;
struct LaunchOpts { int workgroup = 0; Stream* stream = nullptr; };
struct KernelTag  { const char* name; Regime regime; };
struct Region;                                       // default: interior()

void  initialize(int device = 0);                    // ONE TILE on Intel
void  finalize();
Caps  capabilities();

void* malloc_device(std::size_t);
void* malloc_shared(std::size_t);
void  free(void*);
void  memcpy_to_host(void*, const void*, std::size_t);
void  memcpy_to_device(void*, const void*, std::size_t);
void  prefetch(void*, std::size_t);
void  sync();                                        // do_concurrent MAY be async

template <class F> void do_concurrent(KernelTag, Region, F, LaunchOpts = {});
template <class Acc, class Xf, class Cmb> Acc reduce(Region, Acc init, Xf, Cmb);

} // namespace tc::device
```

Two seams that must exist on day 0 even though nothing uses them yet: the **`Region`
parameter** (comm/compute overlap without touching operators) and **`Loc` on `Array`**,
which drives exchange extents, output interpolation and restart layout — and which
`Arena::alloc` must *enforce* rather than trust, since a `Loc::XFace` array allocated
without its extra face compiles and then makes the exchange silently wrong.

---

## 3. Workstreams and dependencies

The source tree and its one-way dependency rule are
[`../CONTRACT_LAYERS.md`](../CONTRACT_LAYERS.md); every workstream below lands inside
`src/lib/` except 05.

```
        ┌──────────────────────────────┐
        │ 00 VOCAB  (MemoryQuantity,   │  build FIRST — the only workstream
        │  types, Space/Loc, assert)   │  with no unresolved blocker
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ 01 ARRAY   (the keystone)    │  the freeze — steps 1-4
        └──────────────┬───────────────┘
     ┌─────────┬───────┴────────┬──────────────┐
     ▼         ▼                ▼              ▼
 02 DEVICE  03 ARENA        04 COMM        05 API
 (needs 00) (needs 01,02    (needs 01,02)  (needs 01,03)
             alloc sig)
     └─────────┴────────────────┴──────────────┘
                        ▼
                  D1  SWE demonstrator
                        ▼
                  D2  column kernel probe
```

| # | workstream | depends on | plan |
|---|---|---|---|
| 00 | Vocabulary + `MemoryQuantity` | — | [`00_VOCAB.md`](00_VOCAB.md) |
| 01 | Array / View / slice | 00 | [`01_ARRAY.md`](01_ARRAY.md) |
| 02 | Device manager | 00 | [`02_DEVICE.md`](02_DEVICE.md) |
| 03 | Arena (memory stack) | 01 sig, 02 alloc sig | [`03_ARENA.md`](03_ARENA.md) |
| 04 | Comm / MPI wrapper | 01 sig, 02 | [`04_COMM.md`](04_COMM.md) |
| 05 | Python API | 01 sig, 03 | [`05_API.md`](05_API.md) |

**00 is the first thing to build**, and specifically `MemoryQuantity` within it. It is the
only piece of the corpus with nothing to resolve first: `Arena` is constructed from one,
`DeviceAllocation` takes one, `Array::bytes()` returns one and `MemoryRequirement` sums
them — so 01/02/03 all need it and none can supply it. It also depends on no toolchain
feature at all (no `<mdspan>`, no CUDA, no MPI), which makes it buildable and fully
property-testable on the mac dev box.

02 depends only on 00, and its test suite already exists (the five spikes), so it runs in
parallel with the 01 freeze.

---

## 4. Global acceptance gates

1. **The tripwire.** Porting the existing operators onto the new foundation changes
   **zero operator math**. If it doesn't hold, the boundary is misplaced — and a
   mechanical refactor is the cheapest possible place to learn that.
2. **All five spikes rebuilt on the real `device/` + `core/`**, still green on V100,
   GH200, PVC and host.
3. **`arena.report()` accounts for every byte, by label.**
4. **A CI budget** on kernel launches per step and host-side per-step wall time.
   Without a gate, abstraction overhead accretes invisibly.
5. **Bit-identity discipline** for every port: `max rel < 1e-12` normalised by the
   field's **global** max (never per element), and *verify the verifier* — perturb one
   term by 1 ulp and confirm the check trips.

## 5. Deliberately deferred

Parameterizations of any kind, EOS, tracers beyond S/T, tripolar fold, sea ice,
open boundaries, restarts, the I/O server, load balancing. Every one of them slots
into a seam listed above; none of them belongs before D1 and D2.
