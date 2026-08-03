# 00 — Vocabulary: the layer below `Array`

**Depends on: nothing at all.** No `<mdspan>`, no CUDA, no MPI, no `Mesh`.
**Blocks: 01, 02, 03, 04, 05** — every one of them.

Contracts: [`../CONTRACT_LAYERS.md`](../CONTRACT_LAYERS.md) (where this lives and why) ·
[`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) §0, §1.6–§1.7.

> **Why this exists as its own workstream.** The plan had `Array` at the bottom. It
> isn't. `Array::bytes()` returns a `MemoryQuantity`; `Arena` is *constructed* from one;
> `DeviceAllocation` takes one; `MemoryRequirements` sums them. All three of 01/02/03
> need this layer and none of them can supply it, so it was being implicitly duplicated
> three ways.
>
> It is also the only part of the corpus with **no unresolved blocker**. 01 is currently
> stalled on three collisions with the existing tree; 03 depends on 01; `mirror` needs
> `ScratchScope` from 03, which is a cycle. This layer has no dependency to cycle
> through, needs no toolchain feature, and is testable on a laptop with no GPU.

---

## Scope

```
src/lib/core/quantity.hpp     MemoryQuantity, Alignment, the unit machinery it needs
src/lib/core/types.hpp        Index, GlobalIndex, Real, is_scalar_type, Scalar, DType, dtype_of
src/lib/core/space.hpp        Space, Loc, Parity, host_subscriptable, mirror_is_identity, Init
src/lib/core/assert.hpp       TC_ASSERT / TC_THROW + the exception types
src/lib/constants.hpp         g, rho0, omega, R_earth, ... -- physical constants
src/lib/logging/logger.hpp    levels, rank-aware, leader_only
tests/test_core_vocab.cpp
```

Everything here is a **value type or a compile-time predicate**. Nothing allocates,
nothing launches, nothing touches a backend. That is the entry criterion for living at
this level, and it is what makes the whole layer property-testable.

**`constants.hpp` and `logging/` are here because nothing else can host them.** Both are
needed by every layer above, neither depends on anything, and leaving them unassigned is
how `g = 9.81` ends up redefined in four kernels and how the no-`printf` rule ends up with
nowhere for output to go. The logger is deliberately small — levels, a rank prefix, and
`leader_only` — because unordered output from N ranks is unreadable and genuinely
*ordered* output needs serialisation, which is a debug-only luxury. It takes the rank as
an `int` at construction rather than including anything from `lib/comm/`, so it stays
dependency-free.

## `MemoryQuantity`

```cpp
class MemoryQuantity {
    std::uint64_t bytes_ = 0;
    Alignment     align_ = Alignment::of<std::max_align_t>();
public:
    constexpr MemoryQuantity() = default;
    static constexpr MemoryQuantity bytes(std::uint64_t);
    static constexpr MemoryQuantity kib(double), mib(double), gib(double);
    template <class T> static constexpr MemoryQuantity of(std::uint64_t count);  // count * sizeof(T)

    constexpr std::uint64_t  count()   const;          // raw bytes
    constexpr Alignment      align()   const;
    constexpr MemoryQuantity round_up(Alignment) const;
    constexpr MemoryQuantity operator+(MemoryQuantity) const;   // CHECKED
    std::string              to_string() const;                 // "43.2 GiB"
};
```

Four properties, each of which is why it is a type and not a `std::size_t`:

1. **Alignment travels with the size.** The arena bump-allocates every field out of one
   slab, so each sub-array needs its own guarantee — and the backends do not agree
   (`cudaMalloc` promises 256 B, `sycl::malloc_device` promises considerably less
   clearly). Carrying it in the type is what stops `round_up` from being re-derived, with
   a different constant, at each call site.
2. **Addition is checked.** Summing a requirements tree in `std::uint64_t` and wrapping
   is a silent under-allocation that surfaces as heap corruption thousands of lines away.
   Overflow throws, naming the two operands.
3. **It prints itself.** `report()` and `MemoryRequirements` both need "43.2 GiB", and
   the formatting logic should exist once.
4. **`of<T>(count)` removes the `* sizeof(T)`** that is otherwise written at every
   allocation and forgotten at exactly one.

**Do not build a general units library yet.** `MemoryQuantity` should be *shaped* so a
`Quantity<Unit>` could subsume it later, but the general machinery waits for a second and
third real user. The candidates are known and worth recording, because one of them is a
bug rakali actually shipped:

- `Metres` vs `Degrees` — rakali's `set_bathymetry_seamount` takes a length scale in grid
  units, which is metres on a Cartesian grid and **degrees** on a spherical one. Mixing
  them makes `exp(-r^2/L^2) = 1` everywhere and the basin silently flattens to a constant
  depth. A units type catches it at compile time; a comment did not.
- `Seconds` vs `Days` — the Python API already wants `tc.days(1)`.

## Work order

1. **`Alignment` + `MemoryQuantity`.** Pure arithmetic. Property tests: `round_up` is
   idempotent, never shrinks, always lands on a multiple; addition is associative and
   throws on overflow rather than wrapping.
2. **`types.hpp`.** Fixed-width `Index`/`GlobalIndex`, `is_scalar_type` as a
   **specialisable trait**, `DType` with a fixed underlying type, `dtype_of` **defined**
   with a `static_assert` in the `else` — never a fallthrough to `F64`.
   **This file already exists** (125 lines, 36 dependents, 149 use sites). Append; add
   `using Field = View<Real,Rank>` only once 01 lands. `Index` changes from `int` to
   `std::int32_t` under all 149 sites — a no-op on LP64/arm64, but state it.
3. **`space.hpp`.** `Space`, `Loc`, `Parity`, `Init`, and the **two** predicates —
   `host_subscriptable` (strict always: portability discipline) and `mirror_is_identity`
   (tracks the hardware). They answer different questions and must not be collapsed.
   **`Loc` and `Parity` currently live in `src/ocean_lib/mesh/mesh.hpp:28`** with 266 use sites.
   Move them here, along with `x_staggered`/`y_staggered`, and have `mesh.hpp` include
   this. A second definition in namespace `tc` is a hard redefinition error the moment
   one TU sees both, which is immediately.
4. **`assert.hpp`.** `TC_ASSERT` + the exception hierarchy. Backend-specific asserts
   (`CUDAAssert`, `HIPAssert`, `MPIAssert`) are **not** here — they are backend-aware and
   belong to 02.

## Tests

| test | asserts |
|---|---|
| round_up idempotent | `q.round_up(a).round_up(a) == q.round_up(a)`, over a generated range |
| round_up never shrinks | `q.round_up(a).count() >= q.count()` and is a multiple of `a` |
| addition checked | summing past `UINT64_MAX` **throws**, naming both operands |
| `of<T>` | `MemoryQuantity::of<double>(1000).count() == 8000` |
| formatting | round-trips through `to_string` at KiB/MiB/GiB boundaries |
| `dtype_of` | `float`→F32, `double`→F64; a specialised `is_scalar_type` **fails to compile** |
| predicates | `host_subscriptable` strict in both builds; `mirror_is_identity` tracks `TC_COHERENT_MEMORY` |
| fixed widths | `sizeof(Index)==4`, `sizeof(GlobalIndex)==8`, `sizeof(DType)==4`, `sizeof(Space)==4` |

## Acceptance gate

- Green with **no GPU, no `<mdspan>`, no MPI** — this layer must build on the mac dev box
  under plain `clang++ -std=c++23` with nothing else installed.
- `Loc`/`Parity` have exactly **one** definition in the tree; `mesh.hpp` compiles against
  the moved one with no change to its 266 use sites.
- Signatures published to 01–05. `MemoryQuantity` in particular is load-bearing for
  `Arena`'s constructor and `Array::bytes()`, so it is frozen here or it is frozen
  nowhere.

## Deferred

The general `Quantity<Unit>` template, physical units (`Metres`/`Degrees`/`Seconds`),
and any arithmetic on `MemoryQuantity` beyond checked addition and `round_up` —
subtraction and division have no call site yet and would only invite one.
