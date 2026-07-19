# TurboChook — Modern-C++ Primer (for the Fortran/HPC brain)

Three tools carry this whole codebase: **lambda captures**, **`std::mdspan`**, and
**concepts**. If they feel like magic, this is the demystifier — built up from the
actual code, with Fortran analogies since that's the home turf.

The one-sentence frame that ties them together:

> **Everything a GPU kernel touches must be a cheap, copyable *value* that is valid
> no matter which processor runs it.**

That is *value semantics*, and it is why the trio below looks the way it does. This
primer is the "why it works" companion to [`CPP_STYLE.md`](CPP_STYLE.md) (the "how to
write it" rules) and the prime directive in [`DESIGN.md`](DESIGN.md).

---

## 1. Lambdas and `[=]` — the capture that ships to the GPU

A **lambda** is an anonymous function written inline. This, from `FvPgf::compute`:

```cpp
for_each_x_face(mesh, [=](FaceView f) {
    ku[f.i, f.j] += -g * (eta[f.ri, f.rj] - eta[f.li, f.lj]) / f.span;
});
```

is *literally* sugar for the compiler generating a hidden struct — a "closure type":

```cpp
struct __closure {
    Field2 ku;      // ← the "captures" become member variables
    Field2 eta;
    Real   g;
    void operator()(FaceView f) const { ku[f.i, f.j] += -g * (...); }
};
```

The **capture list** `[...]` says *how the lambda gets the outside variables it uses*
— each becomes a **member** of that struct:

| Capture | Meaning | The closure stores… |
|---|---|---|
| `[=]` | by **value** (copy) | a **copy** of each used variable |
| `[&]` | by **reference** | a **reference/pointer** to the original |
| `[this]` | the enclosing object | a copy of the `this` **pointer** |

So `[=]` = *make copies of `ku`, `eta`, `g` and carry them inside the closure.*

### Why by-value is non-negotiable for the GPU

When you hand the closure to `std::for_each(par_unseq, …)`, the runtime **physically
copies the closure object to where the threads run — including onto the GPU.** Think
of it like an MPI message: whatever the struct contains gets shipped to the other side.

- Capture a **reference** (`[&]`) to a host stack variable → the struct carries a
  *host address*; on the GPU that address is garbage → crash. (You would never send a
  raw pointer over MPI and dereference it on another rank.)
- Capture by **value** (`[=]`) → the struct carries **copies**. And the trick is that
  a `Field2` copy is *tiny* — a pointer + a couple of extents (see §2) — and the
  pointer it holds points into the **Arena's managed memory, which the GPU can see**.
  So the copy is valid on the device.

### Why capturing `this` crashes (the classic landmine)

Write the kernel *inside a member function* and touch a member directly, and the
compiler **implicitly captures `this`** — the closure now holds a *host* `this`
pointer, and `this->mass_flux_x_` on the GPU dereferences a host address →
`cudaErrorIllegalAddress` (verified, nvc++ 26.5 / V100). The fix is the discipline in
every physics module: **hoist members to locals first**, then capture those:

```cpp
const Field2 ku = k.u;                    // local copy of the fat pointer (device-valid)
... [=](FaceView f){ ku[f.i, f.j] += ...; }   // closure copies ku, NOT this
```

### The subtlety that explains the `MdView` fix

A lambda's `operator()` is **`const` by default** (see the struct above). So members
captured by value are `const` *inside the body* — `ku` is a `const Field2` there. That
is why `mdspan` must be **shallow-const** (a const view still lets you write elements),
or `ku[i,j] += …` would not compile. (This exact requirement is why `MdView`'s const
`operator[]` returns `Real&` — see `core/types.hpp`.)

---

## 2. `std::mdspan` / `Field` — a non-owning view

The cleanest Fortran analogy: **`mdspan` is an assumed-shape array pointer.**

```fortran
real, pointer :: eta(:,:)   ! carries bounds; points at memory it does NOT own
```

Our `Field2` (an `mdspan`, or the hand-rolled `MdView` fallback on stdlibs without
`<mdspan>`) is exactly that: a small descriptor of **`{ pointer, extents, layout }`**
and nothing else. It **owns no memory.** Copying it copies the *descriptor* (a "fat
pointer"), **not** the array — like copying a Fortran pointer, not its target. That is
what makes it cheap to capture `[=]` into a kernel.

The ownership split in the codebase:

- **`Arena`** owns one big block of memory (one giant `allocate`, sized once).
- **`Field2`** is a *view* into a slice of it. Kernels only ever see views.

`m[i, j]` is the C++23 **multidimensional subscript** — it computes a linear offset
into the flat memory. We use **`layout_left` = column-major = Fortran order**: the
offset is `i + nx*j`, so incrementing `i` moves one element (contiguous). Two payoffs
you will recognize:

- it matches Fortran, so the indexing mental model transfers directly;
- adjacent GPU threads get adjacent `i` → adjacent memory → **coalesced** loads (the
  GPU version of a unit-stride inner loop). `layout_right` (C order, the default) would
  make `j` the fast axis — wrong for this.

**Shallow const:** the view behaves like `double* const` — a *const pointer to mutable
data*. A `const` view can still write its elements; the constness of the *view* is not
the constness of the *data*. This is why a by-value (`const`) capture can still write
the field it points at.

---

## 3. Concepts — a compile-time interface contract

A **concept** is a *named, compile-time yes/no question about a type*: "does type `T`
support these operations?"

```cpp
template <class M>
concept Mesh = requires(const M m, Index i, Index j, Loc loc) {
    { m.nx() }            -> std::convertible_to<Index>;
    { m.area(loc, i, j) } -> std::convertible_to<Real>;
    // …
};
```

Read it as: *"`M` is a `Mesh` if, given an `m` and some indices, all these expressions
compile and return the right types."* The `requires(...)` line introduces hypothetical
variables; each `{ expr } -> constraint;` is a required valid expression plus a
constraint on its result type.

The contrast with the OOP way:

| | Virtual / base class | Concept |
|---|---|---|
| Checked | at **run time** (vtable) | at **compile time** |
| Cost per call | indirect call, no inlining | **zero — fully inlined** |
| Legal inside a GPU kernel? | ❌ (no virtuals on device) | ✅ |
| Relationship | `CartesianMesh : public MeshBase` | `CartesianMesh` just *has* the methods |

`CartesianMesh` inherits from nothing. It simply *has* `nx()`, `area()`, … so
`static_assert(Mesh<CartesianMesh>)` passes. It is **duck typing verified by the
compiler** — closer to a Fortran generic interface than to an abstract class. A
`template <Mesh M> void f(const M&)` accepts *any* type that satisfies the concept,
resolves the exact calls at compile time, and inlines them — so swapping a scheme costs
nothing at run time. When a type does *not* satisfy it, you get a crisp error *at the
call site* ("`Weno5` does not satisfy `WallReconstruction`") instead of a template
avalanche deep inside.

This is the mechanism behind every policy axis in the design — `Mesh`,
`WallReconstruction`, `PgfModule`, `Integrator` — compile-time choice, zero runtime
dispatch, GPU-legal.

---

## 4. Watch them lock together — the `FvPgf` kernel

```cpp
void compute(BaroState s, BaroState k, const CartesianMesh& mesh, Params p) const {
    const Field2 eta = s.eta;              // (2) hoist VIEWS to locals — fat pointers, device-valid
    const Field2 ku = k.u;                 //     (never touch a member ⇒ never capture `this`)
    const Real   g  = p.g;                 //     POD scalar

    for_each_x_face(mesh, [=](FaceView f) {          // (1) [=] copies eta, ku, g into the closure;
        ku[f.i, f.j] += -g * (eta[f.ri, f.rj]        //     the runtime ships that closure to the device
                            - eta[f.li, f.lj]) / f.span;  // (2) m[i,j], shallow-const write
    });
}
// (3) for_each_x_face is `template <Mesh M>` — CartesianMesh qualifies, call inlines, zero overhead
```

- **(3) Concept** — `for_each_x_face` accepts any `Mesh`; the Cartesian calls inline away.
- **(2) mdspan** — `eta`/`ku` are cheap views into the Arena; `m[i,j]` is column-major
  indexing; the `const` write works because of shallow-const.
- **(1) `[=]`** — the closure carries *copies* of those views + `g`, all device-valid,
  nothing pointing back at the host stack or `this`.

**Value semantics is the single through-line.** Concepts make the *algorithm choice* a
compile-time value (no vtable). `mdspan` makes a *field* a cheap copyable value (a fat
pointer, not the data). `[=]` makes the *kernel* carry copies of those values. Nothing
a kernel holds points back at the host — which is exactly what lets `nvc++ -stdpar=gpu`
copy it to the device and run it unchanged.

---

## Where to look in the code

| Idea | File |
|---|---|
| `Field`/`MdView`, `layout_left`, shallow-const `operator[]` | `src/core/types.hpp` |
| `tc::par` policy + the `for_each_*` idioms | `src/numerics/parallel.hpp` |
| The Arena (owner) that hands out views | `src/lib/arena.hpp` |
| The `Mesh` concept + a model | `src/mesh/mesh.hpp`, `src/mesh/cartesian_mesh.hpp` |
| A real kernel using all three | `src/physics/pgf.hpp` |
| Sibling-concept design worked example | `src/physics/reconstruction.hpp` |
