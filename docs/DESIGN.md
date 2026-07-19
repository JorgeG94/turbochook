# TurboChook — Design & Architecture Spec

> Read this in full before writing a line of code. It is the settled architecture from a
> long design conversation; the reasoning matters as much as the conclusions. Where a
> decision is still open it is marked **OPEN** with a recommended default.

## 1. What this is

A GPU-native, from-scratch C++23 finite-volume solver for coastal/ocean flows, built on
**ISO C++ standard parallelism** — `std::for_each(std::execution::par_unseq, …)` compiled
with `nvc++ -stdpar=gpu`. It is a ground-up build in idiomatic modern C++, and an exercise
in very-modern-C++ + GPU compute.

**North star — the ocean dynamical core.** An Arakawa **C-grid**, continuity-PPM,
PV-conserving-Coriolis, split-explicit hydrostatic ocean model. **Proof-of-concept on-ramp:
a 2D C-grid *barotropic* shallow-water solver** — which *is* the ocean core's fast mode (the
core minus stratification), so the PoC doubles as the foundation and nothing is thrown away.
This deliberately targets the C-grid / split-explicit **ocean** regime, **not** the coastal
HLL/HLLC **Godunov** regime — a Riemann-solver SWE would build the wrong engine. Layers, EOS,
vertical coordinates, and vertical mixing slot in *on top* of the barotropic foundation
(roadmap M3+) without re-touching it.

Fortran `do concurrent` + NVHPC offload and C++ `stdpar` are the same idea from the same
vendor/runtime — two ISO-standard parallel dialects over one compiler/runtime. This project
works in the C++ dialect; the numerics are standard C-grid ocean methods.

## 2. Prime directive — the constraint that shapes everything

**stdpar-on-GPU forbids runtime polymorphism inside kernels.** A kernel is a callable
handed to a parallel algorithm; everything it touches must be trivially copyable,
device-accessible, and captured **by value**. Inside a parallel region there is:

- **NO** virtual dispatch, `std::function`, RTTI, exceptions, or `dynamic_cast`
- **NO** capturing `this` of a host object, or any owning container (`std::vector`, …)
- **NO** allocation

This kills the reflexive OOP design (abstract base + virtual `compute()`); it either
won't offload or is catastrophically slow. The architecture is therefore forced toward
**value semantics, data-oriented layout, compile-time polymorphism (concepts/templates),
and a hard host/device split.** Everything below follows from this.

## 3. Three-layer architecture

```
┌─ Host orchestration ───────────────────────────────┐
│  config, lifecycle (RAII), I/O, the time loop,      │  full C++ allowed here:
│  dispatch decisions, error handling (exceptions)    │  virtuals, variant, exceptions
├─ The boundary: Views + Params ─────────────────────┤
│  hand kernels cheap POD BY VALUE:                   │  the contract
│  mdspan views + a trivially-copyable params struct  │
├─ Device kernels ───────────────────────────────────┐
│  free functions / lambdas over mdspan + scalars.    │  "dumb" C++ only:
│  No allocation, no virtuals, no exceptions.         │  plain math over views
└─────────────────────────────────────────────────────┘
```

The boundary is sacred. The moment a `virtual`, a `std::function`, or a captured `this`
crosses into the device layer, offload breaks (often silently — it runs on CPU).

## 4. Settled decisions (ADRs)

### ADR-1 — Layering & the view boundary
The three layers above. Kernels take **views** (`mdspan`) + **POD params** by value, never
owners. This makes storage strategy invisible to compute (see ADR-3).

### ADR-2 — Data representation
Two distinct "array-shaped" things, two tools:

- **Grid field** (nx·ny doubles) = `std::mdspan<Real, dextents<int,Rank>, std::layout_left>`
  over a flat contiguous buffer. `layout_left` == column-major == **exactly the Fortran
  array layout** (first index fastest; 2D offset `i + nx*j`). mdspan's *default* is
  `layout_right` (C order) — we always specify `layout_left`. mdspan is **non-owning**
  (the owner is the Arena, ADR-3). `mdspan` **is** the "Tensor that maps indices to the
  1D pointer" — do not hand-roll index math.
- **Per-cell value type** (`Cons`/`Flux`, N conserved vars) = `std::array<Real, N>`
  (guaranteed contiguous, register-resident, trivially copyable). Physics reads names via
  **structured bindings** (`auto [h,hu,hv] = q;`); generic code indexes `q[v]`. This gives
  readability AND N-genericity with zero new machinery. **OPEN/upgrade:** graduate to a
  `Vec<N>` wrapper (array + operators + tuple protocol) when per-layer/per-tracer vector math
  earns operator sugar (arrives with M3 layers). Drop-in, decide later.

- **Storage is SoA**: a system of N variables is `std::array<Field2, N>` (one contiguous
  buffer per component), **not** `mdspan<Cons>` AoS (stride-N reads kill coalescing).

- **Coalescing rule**: `layout_left` makes index 0 contiguous; iterate so the *parallel*
  (fast-varying) thread index maps to index 0 → adjacent threads touch adjacent memory.
  (This is the C++ restatement of the "contiguous index innermost" `do concurrent` rule.)

### ADR-3 — Memory: the Arena
One monotonic bump allocator over a single flat, **managed** buffer; hands out `mdspan`
views into itself. Rationale: one managed allocation → one migration surface (fewer/larger
page migrations under `-stdpar`); mass snapshot/ping-pong/restart via one `std::copy`;
truthful byte accounting (the high-water mark *is* the number); inter-field alignment
control for coalescing; zero mid-run allocation.

- Back it with `std::vector<std::byte>` → portable AND auto-managed under `nvc++ -stdpar`.
- `alloc2d<T>(nx,ny)` aligns the offset (≥128 B), bumps, returns a `layout_left` mdspan.
- Two-tier: **monotonic** for persistent fields + a **stack marker** (`mark`/`restore`)
  for transient per-step scratch.
- **HARD RULE — size once, never grow.** If the backing vector reallocates, every mdspan
  into it dangles. Compute total bytes up front, allocate once, then only hand out views.
- It slots **under** the ADR-1 boundary: milestone-1 may use a simpler owning `Field`
  (owns a `vector`); swapping in the Arena changes **zero kernels**. That swap is the proof
  the boundary is drawn correctly.

### ADR-4 — Dispatch: compile-time policies + a runtime bridge
Selection of scheme (flux / limiter / EOS / integrator / BC) is **compile-time** via C++20
`concept`s (not virtual base classes) — the compiler inlines the policy into the kernel,
zero overhead. Runtime config (a namelist/CLI string) bridges to the compile-time world via
`std::variant` + `std::visit`: string → variant of policy types → `visit` launches the
correct templated kernel. One binary, no virtuals, every scheme fully inlined.

### ADR-5 — Value type baseline
`tc::Vec<N>` (home-baked, `std::array`-backed) is the per-cell/per-layer value type; readable
via structured bindings, generic via `[]`. Introduced at M3 (layers); the M2 barotropic PoC is
scalar-per-face and needs no Vec. See ADR-2 / §7.

### ADR-6 — Generalized vertical coordinate (ALE) is the *shape* of the core, not a bolt-on

The ocean core is **thickness-based, Lagrangian-then-remap** from the ground up (the ALE
lineage of the MOM6 family of models). This is a structural commitment — retrofitting it
means restructuring the state, the stepper, and every operator. It lives in **three places**,
not one "base concept":

1. **State (data shape):** layer thickness `h_layer` is the **prognostic** vertical variable;
   z-interfaces are **diagnostic** (`z(k) = −H + Σ_{k'<k} h`, a `z_from_h` helper). **No fixed
   z-levels are stored.** This is what "GVC lives in the ocean state" means — the *shape*, not a
   coordinate object.
2. **Stepper (a step):** the loop advances layers freely (Lagrangian — they distort), then at
   the remap cadence regrids to the target. The `Integrator` carries a **remap hook**
   (cadence-gated on a thermo-step counter).
3. **Policies (two composable axes):** `Vcoord` (target-thickness generator: z*/sigma/rho/
   hybrid — a module slot) × `Reconstruction` (the conservative remap kernel — the **same**
   policy continuity + tracers use).

**Load-bearing consequence — design for vanishing layers from day one.** A general coordinate
means layers can vanish (h→0: z* thinning, isopycnal outcrop). Every operator that divides by
`h` must carry the thin-layer guard (`H_VANISHED` = dynamic-vanish skip/merge vs `H_DIV_EPS` =
pure 1/0 armour — two constants with documented, distinct roles) from the start. This is a
GPU-FV fundamental that carries over unchanged, and the single discipline separating
"GVC-ready" from "GVC-retrofit-nightmare". **No uniform-`dz` assumptions anywhere.**

**Scope:** M2 (barotropic, single layer) → GVC is degenerate (nothing to remap). "Design with
it in mind" = shape the **M3** layered state (thickness-prognostic), operator interfaces (take
arbitrary `h_layer`, tolerate vanishing), and the stepper (remap hook) so the ALE remap drops
in at **M4** with zero retrofit. Build the seam at M3; build the remap at M4.

## 5. The compile-time policy axes (the payoff)

Independent compile-time axes that compose — painful in Fortran, natural via C++ concepts.
These are the **ocean-core** operators (NOT a Godunov `RiemannSolver` — that's the coastal
regime we are *not* building). The barotropic PoC (M2) exercises the **top subset** (grid +
continuity + Coriolis + PGF + integrator + BC); layers/EOS/vcoord/vmix arrive M3+:

| Axis | Concept | Examples | Swapping it touches… |
|---|---|---|---|
| Grid / mesh | `Mesh` | Cartesian (later tripolar, unstructured) | only the Mesh |
| Continuity | `Continuity` | PPM thickness flux | only continuity |
| Coriolis | `Coriolis` | Sadourny PV-enstrophy, energy form | only Coriolis |
| Pressure gradient | `PGF` | Montgomery, FV, gprime (2-layer) | only the PGF |
| Time | `Integrator` | Fwd-Euler, SSP-RK2 (+ barotropic/baroclinic split) | only the integrator |
| Boundary | `BC` | wall, periodic, Flather | only the BC fill |
| Vert. coord (M4) | `Vcoord` | sigma, z*, ALE remap | only the remap |
| Vert. mixing (M5) | `Vmix` | PP81, KPP | only vmix |

Each is a compile-time policy; the dispatch bridge (ADR-4) maps a runtime config string to
the chosen instantiation — the compile-time form of a namelist + enum dispatch.

## 6. Key type skeletons (the settled design, in code)

```cpp
namespace tc {

using Real = double;

// ── ADR-2: the grid field == the Fortran array (column-major, contiguous) ──
template <int Rank>
using Field = std::mdspan<Real, std::dextents<int, Rank>, std::layout_left>;
using Field2 = Field<2>;

// ── ADR-3: the arena (sized once, never grows; returns mdspans) ──
class Arena {
    std::vector<std::byte> buf_;
    std::size_t top_ = 0;
public:
    explicit Arena(std::size_t bytes) : buf_(bytes) {}
    template <class T>
    auto alloc2d(int nx, int ny, std::size_t align = 128) {
        top_ = (top_ + align - 1) & ~(align - 1);
        T* p = reinterpret_cast<T*>(buf_.data() + top_);   // C++23: std::start_lifetime_as<T[]>
        top_ += sizeof(T) * std::size_t(nx) * ny;
        assert(top_ <= buf_.size() && "arena overflow — size it once, up front");
        return std::mdspan<T, std::dextents<int,2>, std::layout_left>(p, nx, ny);
    }
    std::size_t mark() const { return top_; }
    void restore(std::size_t m) { top_ = m; }
    std::size_t bytes_used() const { return top_; }        // truthful accounting
};

// ── the boundary: POD view bundle + params, captured BY VALUE ──
template <int N> struct SystemView { std::array<Field2, N> q; };   // co-located SoA at CENTRES
                                                                   // (e.g. tracers S/T later)
                                                                   // barotropic momentum is STAGGERED — see BaroState (§6)
struct Params { int nx, ny; Real dx, dy, dt, g; };

// ── the iteration abstraction (reliably-offloading idiom) ──
template <class F>
void for_each_cell(int nx, int ny, F f) {
    auto ids = std::views::iota(0, nx * ny);               // flat index; unflatten inside
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(),
                  [=](int n) { f(n % nx /*fast axis*/, n / nx); });
}

// ── The physics: an Arakawa C-GRID, split-explicit ocean core (NOT Godunov/Riemann) ──
// State is STAGGERED, so instead of a co-located SystemView<N> the barotropic state is a
// bundle of views, each on its OWN grid:
struct BaroState {
    Field2 eta;   // (nx,   ny  )   cell CENTRES  (sea-surface height / free-surface thickness)
    Field2 u;     // (nx+1, ny  )   x-FACES
    Field2 v;     // (nx,   ny+1)   y-FACES
};

// The RHS is a SUM OF OPERATOR TENDENCIES, not the divergence of a Riemann flux. Each
// operator is a compile-time policy (§5), writing into a matching tendency BaroState:
//   ∂η/∂t = -∇·(H u)                      [Continuity : PPM thickness flux, on centres]
//   ∂u/∂t = -g ∂η/∂x + (f v)|u + adv      [PGF + PV-conserving Coriolis, on faces]
//   ∂v/∂t = -g ∂η/∂y - (f u)|v + adv
template <Continuity Cont, Coriolis Cor, PGF Pgf>
void baro_rhs(BaroState s, BaroState k, Params p) {
    Cont::apply(s, k, p);   // η tendency + thickness fluxes   (for_each_cell over centres)
    Pgf ::apply(s, k, p);   // -g grad(η) into u, v            (for_each_face)
    Cor ::apply(s, k, p);   // PV-conserving Coriolis + advection into u, v
}
// Implement the continuity-PPM / PV-Coriolis / FV-PGF numerics (the algorithm) — these are
// standard C-grid ocean methods; see GPU_STDPAR_NOTES.md for the design lessons that shape
// how they're written on the GPU. `for_each_face` is the staggered-grid twin of `for_each_cell`.

// ── RK combine — per staggered field (each on its own grid extent) ──
void axpy_field(Field2 o, Field2 x, Field2 y, Real a, Real b, Params p);   // o = a·x + b·y
// SSP-RK2 (Heun) applied to each of {eta, u, v}. The split-explicit structure — many fast
// barotropic substeps per slow baroclinic step — lives in the Integrator (M2+).

// ── runtime → compile-time bridge (host side, ADR-4) ──
// std::visit picks Continuity / Coriolis / PGF instantiations from config strings — the
// compile-time twin of a namelist + enum dispatch. One binary, no virtuals.

// (Co-located tracers S/T ride the SystemView<N> path at centres — arrives with M3 layers.)

} // namespace tc
```

## 6b. The physics-module pattern (class-per-module, workspace-owning)

The classic "a derived type per physics module" (a continuity module, a Coriolis-advection
module, …) maps to: **a physics module is a class that (a) owns its arena-backed workspace
and (b) satisfies a per-module concept.** Each *scheme variant* is its own such class; the
dispatch (ADR-4) chooses which fills the slot.

```cpp
template <class M>
concept CoriolisModule = requires(M m, Arena& a, const Mesh& mesh, BaroState s, BaroState k, Params p) {
    { m.init(a, mesh) };
    { m.compute(s, k, p) };                 // adds the Coriolis tendency into k, using m's workspace
};

class SadournyEnstrophy {                   // the enstrophy-conserving Coriolis-advection module
    Field2 q_, fcor_;                       // persistent workspace, arena-backed
public:
    void init(Arena& a, const Mesh& m) { q_ = a.alloc2d<Real>(m.nx+1, m.ny+1); fcor_ = m.f_corner(); }
    void compute(BaroState s, BaroState k, Params p) const {
        Field2 q = q_, fcor = fcor_;        // HOIST members to locals — capture [=], NEVER `this`
        for_each_corner(p, [=](int i,int j){ q[i,j] = (fcor[i,j]+zeta(s,i,j))/h_corner(s,i,j); });
        for_each_face_x(p, [=](int i,int j){ k.u[i,j] += pv_flux_u(q,s,i,j); });
        for_each_face_y(p, [=](int i,int j){ k.v[i,j] += pv_flux_v(q,s,i,j); });
    }
};
```

Rules:
- **Workspace lives on the module, allocated from the Arena in `init`.** Because the arena is
  managed memory, `init(Arena&)` subsumes the classic alloc **+ device-map** step — there is
  no separate enter-data orchestration to maintain, and none of its memcpy-explosion footguns.
- **`compute` hoists member views into locals, then the kernel captures `[=]` — never `this`.**
  This is the general rule for passing device data across a value-semantic boundary: pass the
  component views, not an aggregate handle.
- **The god-state `OceanCore<Cont, Cor, Pgf, Integ>`** composes the chosen module classes;
  `baro_rhs` = the sum of `module.compute(s, k)` calls.
- **Dispatch:** `std::variant` / `std::visit` picks the variant class at config time — the
  compile-time form of a namelist selector + enum dispatch.

### Continuity is generic over reconstruction — but PPM only, for now

`Continuity<Reconstruction>` factors the shared FV flux-divergence from the swept-flux
reconstruction (PPM parabola / WENO / PLM). **Build PPM only**; PLM/WENO are drop-in later via
the same concept — leave the seam, don't build it. The `Reconstruction` contract is
load-bearing: **positivity-preserving + monotone** swept flux (thickness ≥ 0; you divide by h
*everywhere*) — order of accuracy is secondary, and a WENO here must be the positivity-
preserving flavour. The same `Reconstruction` also drives tracer advection later
(thickness/tracer consistency).

The module pattern in one line each: a physics module = a class filling a slot; a runtime
config string maps to the chosen class via `std::variant`/`std::visit` (compile-time);
`init(Arena&)` replaces alloc + enter-data because the arena is managed; "hoist members to
locals, capture `[=]`, never `this`" is the device-boundary rule; the composed
`OceanCore<...>` is the god-state slot map.

## 7. Decisions (resolved) + the dimension/mesh seam

1. **Value type — `tc::Vec<N>`** (home-baked, `std::array`-backed, eager, trivially copyable;
   glm-style, NOT Eigen/`std::vector` — see FOUNDATIONS §2b). `Cons`/`Flux`/per-layer state are
   aliases of it. Introduced at **M3** (layers); the M2 barotropic PoC is scalar-per-face and
   needs no Vec.
2. **Integrator — a policy `struct`** (e.g. `SSPRK2`) matching an `Integrator` concept. The
   concept is over `(Workspace&, RhsOp, BcOp, Params)` and knows **nothing** of physics/flux
   — it only calls the RHS op + `combine`s. Hand-code the SSP stages (not a Butcher engine).
   Declares `static constexpr int n_scratch` so the Workspace sizes its register set.
3. **Ghosts — halo rows inside each field.** `nghost` is a **property of the reconstruction/
   scheme** (`Scheme::nghost`; =1 for the barotropic stencil, wider for PPM/higher-order). A `BC` kernel fills them
   **before each RK stage's `rhs`** → the interior kernel is **branch-free** (no boundary
   `if`). 0-based; interior = `[nghost, nghost+n)`.
4. **Workspace owns registers.** A `Workspace` allocates `U + n_scratch` registers from the
   Arena; `System<N>` is one register. Ping-pong lives inside `Integrator::advance`; the flux
   only ever sees a `SystemView`.

### The dimension vs mesh seam — what templating actually buys

- **Templating over spatial `Rank` (2D↔3D, structured)** is largely free: `Field<Rank>`,
  `Vec<N>`, a dimension-generic stencil. Do it.
- **Structured ↔ unstructured is a DIFFERENT axis** — connectivity/iteration, not dimension;
  a `Rank` template does not get you triangles. What is shared for free is the **physics**
  (the `Continuity`/`Coriolis`/`PGF` operators); what differs is **iteration + connectivity**.
  (The natural factoring: shared, mesh-agnostic physics modules over *separate* structured/
  unstructured flux-iteration kernels.)
- **Unification path (deferred):** a `Mesh`/topology **concept** + `for_each_face(mesh, …)`
  handing the flux a `FaceView{left, right, normal, area}`. Structured computes it from
  `(i,j)`; unstructured reads connectivity tables → the flux becomes mesh-agnostic. Caveats:
  must be **compile-time** (concept, not virtual) so the structured path inlines to plain
  index arithmetic (zero-cost); must encode the unstructured **no-scatter double-visit**
  race pattern.
- **Concept, not a base class; siblings, not parent/child.** Do NOT model structured as a
  *subclass* of unstructured. Two reasons: (1) a virtual `Mesh` base = device dispatch inside
  the flux = forbidden (prime directive); (2) even non-virtually, making the structured grid a
  *special case* of unstructured forces the fast path to carry connectivity tables it doesn't
  need — indirection where `i±1` suffices — the performance inversion (and the "special case"
  actually has *less* state, so it doesn't fit inheritance anyway). `CartesianMesh` and
  `TriMesh` are **independent** models of one `Mesh` concept: structured *computes* geometry
  (inlined, zero storage), unstructured *reads* it (tables). The compiler monomorphizes each.
- **For turbochook now (structured only, §11):** flow geometry into the flux via a small
  accessor, **not raw `p.dx`/`p.dy`**, so the Mesh seam *exists* without the second backend.
  Leave the seam; don't build it.

## 8. GPU hard rules (kernel-authoring checklist)

- Capture **by value**; only `mdspan` views + POD params cross the boundary. Never `this`,
  never an owner, never a `std::vector`. **VERIFIED** (nvc++ 26.5 / V100): a member-function
  kernel that captures `this` PASSES on host but hard-crashes on the GPU with
  `cudaErrorIllegalAddress` — hoist member views to locals first (the §6b `compute` discipline).
- No virtual / exceptions / RTTI / `std::function` / allocation inside a kernel.
- Arena **sized once, never reallocates** (else dangling mdspans).
- `layout_left` + parallel index on the contiguous (fast) axis = coalesced.
- **Double-buffer**: read old, write new (no read-neighbour-write-own race — same buffer =
  UB). RK registers make this natural.
- **Keep state device-resident: never touch it on the host inside the time loop.** Map once
  (the arena), ping-pong pointers/registers on device, and do host reads only at output/diag
  cadence. **MEASURED** (nvc++ 26.5 / V100): a per-step host copy of the state forces a
  host↔device migration every step and is **~120–140× slower** than resident (and a 1.0× no-op
  on the host build — the penalty is purely GPU migration). This is the managed-memory form of
  the map-once discipline; a stray per-step diagnostic copy silently reintroduces the 100×
  penalty.
- **Verify offload.** A green CPU/serial run proves correctness, *not* that it offloaded.
  With nsys: confirm kernels in the timeline. When nsys is unavailable, the working substitute
  is **verify-by-speed**: build `gpu` + `host` and require `gpu/host ≫ 1` on a big problem
  (a build can report `backend=gpu` yet run at host speed if it didn't offload). These
  toolchain patterns are verified on nvc++ 26.5 / V100.

## 9. Build & toolchain

- **C++23**, CMake ≥ 3.25.
- Three configs from one source:
  - `nvc++ -std=c++23 -stdpar=gpu -O2`   → GPU offload (production).
  - `nvc++ -std=c++23 -stdpar=multicore` → CPU threads (portable perf).
  - `g++ -std=c++23 -O2` / `clang++`     → host serial (`par_unseq` → seq); for tests/CI.
- `std::mdspan` is C++23. Prefer it; if the stdlib lacks it, hand-roll a minimal
  `layout_left` `tc::mdview` (~40 lines — the sliver we use: ctor from ptr+extents,
  `operator[](i,j[,k])`, `extent()`, `data_handle()`) behind a `__has_include(<mdspan>)` seam
  in `core/types.hpp`. **Never Kokkos.**
  **VERIFIED** (nvc++ 26.5 / V100): `std::mdspan<layout_left>` with the `m[i,j]` subscript
  **offloads and is zero-cost** vs manual index math; the fallback path works on g++13 (no
  `<mdspan>`). **Gate on `__has_include(<mdspan>)` ONLY** — nvc++ 26.5 leaves `__cpp_lib_mdspan`
  UNDEFINED despite a working `<mdspan>`, so the feature-test macro would wrongly reject it.

### Dependency policy

- **stdlib-first.** `src/core/` (numeric types) and `src/lib/` (plumbing) depend on nothing but the C++ stdlib — the stdlib is the
  only dependency the core needs (`std::format`/`print`, `mdspan`, parallel algorithms,
  `chrono`, `source_location`).
- **doctest** is the only dependency, and it is **test-only** (fetched, never shipped).
- **No Kokkos, ever** — not the framework, not `kokkos/mdspan`.
- **Escape valve:** if heavy machinery is ever genuinely needed, prefer a **C-ABI bridge to a
  small external routine** over adopting a heavy C++ framework (Kokkos, Eigen, Trilinos, …).
  Reuse a focused routine across a C boundary before importing someone else's across a C++ one.

## 10. Testing

**Framework: doctest** (single-header, fetched via CMake `FetchContent`, fastest compiles) —
one runner through **CTest**. Two tiers:

- **Unit** — instantiate a few cells, call one kernel, assert against hand-computed values.
  Kernels are pure free functions over views ⇒ testable on host with `std::execution::seq`.
  (`tc::Vec` ops, the `Continuity`/`Coriolis`/`PGF` operators on a tiny grid, `Arena`, the
  `Profiler` self-time math.)
- **Analytical** — the highest-leverage class (each one tends to catch a real bug):
  **geostrophic adjustment** (relaxes to the correct balanced state), **lake-at-rest**
  (well-balanced — flat η stays flat), a **Kelvin/Rossby wave** (right propagation speed),
  **conservation** (total-mass drift ~ machine eps); at M3+, a **baroclinic-instability**
  channel (eddies from physics).

Add a test with (or before) each kernel. Float asserts use a tolerance (`doctest::Approx`).

**The execution-policy seam** (so host tests need no TBB): `src/numerics/parallel.hpp` defines
`tc::par` = `std::execution::par_unseq` normally, but `std::execution::seq` under the
`TC_STDPAR_OFF` define (the host build). `for_each_cell` uses `tc::par`. → the GPU/multicore
builds offload; the host-serial test build is deterministic and dependency-free.

**GPU testing rule (CPU-green ≠ GPU-correct):** the suite compiles in all three configs. CI
runs the host-serial suite (fast); **additionally run the analytical suite under the
`-stdpar=gpu` build periodically** — that is what catches offload/data-motion bugs a host run
is blind to. Never `ctest -jN` on the GPU build (workers share one GPU → spurious failures).

## 11. Non-goals (for now)

Unstructured grids, MPI, implicit solvers, AMR, the coastal Godunov regime. The architecture
leaves seams (custom mdspan layouts, the arena, the policy axes, the `Mesh` seam) but we do
not build for them yet. **Near-term goal: a 2D C-grid barotropic SWE (the ocean core's fast
mode).** North star: the full ocean dynamical core (§1) — a multi-month+ arc, climbed one
roadmap rung at a time.

**Non-hydrostatic is a separate future *regime*, not a from-start requirement.** The core is
hydrostatic (the C-grid split-explicit ALE lineage of §1 is inherently so). NH would *add*:
a global **elliptic (Poisson) solve** for the non-hydrostatic pressure `q`, a **projection /
fractional-step** time integration, and a **prognostic `w`** — the hydrostatic operators
(continuity / Coriolis / hydro-PGF / EOS / vmix) survive unchanged. Seam (unbuilt): the NH
pressure is an extra `PGF`-style operator, the projection stepper is another `Integrator`, and
the elliptic solve is a **bolt-on subsystem** (it does *not* fit the local-stencil policy
pattern — it is globally coupled + iterative + preconditioned, the hardest GPU piece). **Caveat
— NH ⟂ ALE:** a thickness-prognostic Lagrangian grid (ADR-6) and the NH elliptic operator do
not coexist cleanly; an NH regime would use a fixed/sigma vertical grid (a separate NH path).
So NH is a distinct regime — seam-ed, not built — and must not distort the hydrostatic core.
