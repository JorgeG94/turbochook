# TurboChook — Design & Architecture Spec

> Read this in full before writing a line of code. It is the settled architecture from a
> long design conversation; the reasoning matters as much as the conclusions. Where a
> decision is still open it is marked **OPEN** with a recommended default.

## 1. What this is

A GPU-native, from-scratch C++23 finite-volume solver for coastal/ocean flows, built on
**ISO C++ standard parallelism** — `std::for_each(std::execution::par_unseq, …)` compiled
with `nvc++ -stdpar=gpu`. It re-implements the architecture of the Rakali Fortran solver
(`../rakali_dc`) in idiomatic modern C++, as a learning exercise for very-modern-C++ +
GPU compute. First target: 2D shallow-water (SWE). Designed so 3D, Euler, and other
equation sets slot in **without touching compute kernels**.

Rakali's model is `do concurrent` + let NVHPC offload it. C++ stdpar is the same idea from
the same vendor/runtime — this port is a translation between two ISO-standard parallel
dialects, not an analogy.

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
  `Vec<N>` wrapper (array + operators + tuple protocol) when the HLLC star-state math earns
  operator sugar (`sR*FL - sL*FR + sL*sR*(R-L)`). Drop-in, decide later.

- **Storage is SoA**: a system of N variables is `std::array<Field2, N>` (one contiguous
  buffer per component), **not** `mdspan<Cons>` AoS (stride-N reads kill coalescing).

- **Coalescing rule**: `layout_left` makes index 0 contiguous; iterate so the *parallel*
  (fast-varying) thread index maps to index 0 → adjacent threads touch adjacent memory.
  (This is the C++ restatement of Rakali's "contiguous index innermost" `do concurrent`
  rule.)

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
`std::array<Real,N>` now (zero machinery, generic, readable via bindings); `Vec<N>` later
when flux math earns operators. See ADR-2.

## 5. The orthogonal policy axes (the payoff)

Four **independent** compile-time axes that compose. This is what's painful in Fortran and
natural in modern C++:

| Axis | Concept | Examples | Swapping it touches… |
|---|---|---|---|
| Physics | `EquationSet` | SWE, Euler | only the EquationSet |
| Numerics | `RiemannSolver<E>` | HLL, HLLC, Rusanov | only the solver |
| Time | `Integrator` | Fwd-Euler, SSP-RK2, RK3 | only the integrator |
| Boundary | `BC` | reflective, periodic | only the BC fill kernel |

`EquationSet` supplies `Cons`, `Flux`, `phys_flux_{x,y}`, `wave_speeds_{x,y}`, `contact_{x,y}`
(HLLC), and `static constexpr int N`. `RiemannSolver<E>` is generic over any `EquationSet`.

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
template <int N> struct SystemView { std::array<Field2, N> q; };   // SoA, indexable
struct Params { int nx, ny; Real dx, dy, dt, g; };

// ── the iteration abstraction (reliably-offloading idiom) ──
template <class F>
void for_each_cell(int nx, int ny, F f) {
    auto ids = std::views::iota(0, nx * ny);               // flat index; unflatten inside
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(),
                  [=](int n) { f(n % nx /*fast axis*/, n / nx); });
}

// ── ADR-4/5: physics axis — an EquationSet ──
template <class E>
concept EquationSet = requires(typename E::Cons L, typename E::Cons R, Real g) {
    { E::phys_flux_x(L, g)    } -> std::same_as<typename E::Flux>;
    { E::wave_speeds_x(L,R,g) } -> std::same_as<std::pair<Real,Real>>;
};

struct SWE {
    static constexpr int N = 3;
    using Cons = std::array<Real, N>;   // {h, hu, hv}
    using Flux = std::array<Real, N>;
    static Flux phys_flux_x(Cons q, Real g) {
        auto [h, hu, hv] = q; Real u = hu / h;
        return { hu, hu*u + 0.5*g*h*h, hv*u };
    }
    static std::pair<Real,Real> wave_speeds_x(Cons L, Cons R, Real g); // sL, sR
    // + phys_flux_y, wave_speeds_y, contact_x/y (HLLC)
};

// ── numerics axis — a Riemann solver, generic over any EquationSet ──
template <EquationSet E>
struct HLL {
    using Cons = typename E::Cons; using Flux = typename E::Flux;
    static Flux flux_x(Cons L, Cons R, Real g) {
        auto [sL, sR] = E::wave_speeds_x(L, R, g);
        if (sL >= 0) return E::phys_flux_x(L, g);
        if (sR <= 0) return E::phys_flux_x(R, g);
        Flux FL = E::phys_flux_x(L,g), FR = E::phys_flux_x(R,g), out;
        Real inv = 1.0 / (sR - sL);
        for (int v = 0; v < E::N; ++v)                     // N-generic
            out[v] = (sR*FL[v] - sL*FR[v] + sL*sR*(R[v] - L[v])) * inv;
        return out;
    }
    // HLLC adds a contact-wave correction across E::contact_x — same shape.
};

// ── spatial operator L: fills tendency K = -div(F(U)) ──
template <EquationSet E, template<class> class Riemann>
void rhs(SystemView<E::N> U, SystemView<E::N> K, Params p) {
    for_each_cell(p.nx, p.ny, [=](int i, int j) {
        if (i==0 || i==p.nx-1 || j==0 || j==p.ny-1) return;   // interior only (ghosts filled by BC)
        auto gather = [&](int ii, int jj){ typename E::Cons c;
            for (int v=0; v<E::N; ++v) c[v] = U.q[v][ii,jj]; return c; };
        auto C = gather(i,j);
        auto fe = Riemann<E>::flux_x(C, gather(i+1,j), p.g), fw = Riemann<E>::flux_x(gather(i-1,j), C, p.g);
        auto fn = Riemann<E>::flux_y(C, gather(i,j+1), p.g), fs = Riemann<E>::flux_y(gather(i,j-1), C, p.g);
        Real ix = 1/p.dx, iy = 1/p.dy;
        for (int v = 0; v < E::N; ++v)                     // N-generic scatter
            K.q[v][i,j] = -((fe[v]-fw[v])*ix + (fn[v]-fs[v])*iy);
    });
}

// ── the RK combine (over STORAGE — always N-generic, value type irrelevant) ──
template <int N>
void combine(SystemView<N> o, SystemView<N> x, SystemView<N> y, SystemView<N> k,
             Real a, Real b, Real c, Params p) {
    for_each_cell(p.nx, p.ny, [=](int i, int j) {
        for (int v = 0; v < N; ++v)
            o.q[v][i,j] = a*x.q[v][i,j] + b*y.q[v][i,j] + c*p.dt*k.q[v][i,j];
    });
}

// ── time axis: SSP-RK2 (Heun). Reads like the math. ──
// K = L(U);      U1 = U + dt*K
// K = L(U1);     U^{n+1} = 1/2 U + 1/2 U1 + 1/2 dt*K
//   (fill_ghosts before each rhs)

// ── runtime → compile-time bridge (host side) ──
// using FluxChoice = std::variant<HLL<SWE>, Rusanov<SWE>>;
// std::visit([&](auto solver){ rhs<SWE, decltype(solver)::template rebind>(…); }, choice);

} // namespace tc
```

## 7. Open decisions (resolve during implementation)

1. **Value type `array` vs `Vec<N>`** — array baseline (ADR-5); add `Vec<N>` (operators +
   tuple protocol for structured bindings) when HLLC math earns it. Drop-in.
2. **Integrator: concept-struct vs free functions** — recommend a `struct SSPRK2 { static
   void advance(...); }` matching an `Integrator` concept, for the same swappability the
   flux has.
3. **Ghosts** — recommend halo rows inside each field (`nghost`, Rakali-style) filled by a
   `BC` kernel each stage; keeps the interior kernel branch-free (drop the boundary `if` in
   `rhs`).
4. **System vs Workspace ownership** — recommend a `Workspace` type that owns all registers
   (`U`, `U1`, `K`) allocated from the Arena; `System<N>` is just one register.

## 8. GPU hard rules (kernel-authoring checklist)

- Capture **by value**; only `mdspan` views + POD params cross the boundary. Never `this`,
  never an owner, never a `std::vector`.
- No virtual / exceptions / RTTI / `std::function` / allocation inside a kernel.
- Arena **sized once, never reallocates** (else dangling mdspans).
- `layout_left` + parallel index on the contiguous (fast) axis = coalesced.
- **Double-buffer**: read old, write new (no read-neighbour-write-own race — same buffer =
  UB). RK registers make this natural.
- **Verify offload with nsys.** A green CPU/serial run proves correctness, *not* that it
  offloaded. `nvc++ -stdpar=gpu -gpu=…` then confirm kernels in the timeline.

## 9. Build & toolchain

- **C++23**, CMake ≥ 3.25.
- Three configs from one source:
  - `nvc++ -std=c++23 -stdpar=gpu -O2`   → GPU offload (production).
  - `nvc++ -std=c++23 -stdpar=multicore` → CPU threads (portable perf).
  - `g++ -std=c++23 -O2` / `clang++`     → host serial (`par_unseq` → seq); for tests/CI.
- `std::mdspan` is C++23. If a compiler lacks it, vendor `kokkos/mdspan` (reference impl,
  drop-in `std::` shim). Prefer real `std::mdspan`.

## 10. Testing

Analytical tests are the highest-leverage thing here (every one caught a real bug in
Rakali). Kernels are pure free functions over views ⇒ testable on host with
`std::execution::seq`. Target cases: **dam-break** (vs exact SWE Riemann solution),
**lake-at-rest** (well-balancedness — flat stays flat), **radial-symmetry** of a Gaussian
drop, and **conservation** (total mass drift ~ machine eps). Add a test before or with each
kernel.

## 11. Non-goals (for now)

Unstructured grids, MPI, implicit solvers, AMR. The architecture leaves seams (custom
mdspan layouts, the arena, the policy axes) but we do not build for them yet. 2D structured
SWE end-to-end is the near-term goal.
