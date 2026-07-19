# GPU stdpar design notes — what modern C++ makes easy vs. what stays GPU-fundamental

Design choices for a GPU-native solver split three ways: some constraints are just an **older
language fighting you** (modern C++ erases them), some are **the GPU fighting you** (C++
inherits them *unchanged* — do not "fix" them), and some are **new C++ footguns**. Keeping
these straight stops two mistakes: expecting C++ to dissolve constraints that are actually the
GPU's, and "modernizing away" an idiom that was load-bearing.

## Bucket 1 — Genuine C++ wins (the *language* was the tax)

These are pure ergonomics tax in classic imperative/directive-offload HPC code; a from-scratch
modern-C++/stdpar design avoids them by construction.

| Older imperative-HPC idiom | turbochook (modern C++ + stdpar) |
|---|---|
| **enum + `switch` dispatch** — weak polymorphism → a runtime branch inside the kernel | **compile-time policies** (concepts) — the branch vanishes, the scheme inlines |
| **array-of-struct + indirection shims** for a set of tracers — the kernel can't cheaply index a struct-array with heap-backed members; you *want* a "4D contiguous tracer bank" but can't get it cleanly | **data-oriented layout from the start** — a contiguous tracer bank (`mdspan` / `array<Field3, Ntr>`). The 4D backing you wished for *is* the natural C++ representation |
| **manual device data-mapping orchestrator** — separate host/device memory forces mapping every component by hand; miss one → a 100–1000×+ memcpy explosion | **arena managed memory** — allocation *is* the mapping; no orchestrator exists |
| **explicit-shape array args** — assumed-shape descriptors trigger per-launch descriptor copies | **`mdspan` value semantics** — extents travel with the view for free |
| **structured/unstructured source duplication** + **cross-module `_impl` copy-paste** (the compiler won't inline a device helper across a translation unit) | **templates + header-only inlining** — write once, specialize per backend |
| **module-global lazy scratch** (`ensure`/`cleanup` dances) | **RAII arena-owned member workspace** (DESIGN §6b) — no globals |
| **`stop`/error-code + `is_init` flags** | **exceptions (host) + RAII lifecycle** |

Biggest two: the **tracer backing** and the **structured/unstructured duplication** — pure
language-ergonomics tax that C++ erases entirely.

**On remap / generalized vertical coordinates specifically:** the ALE framework itself
(target-thickness generator + conservative remap) is good design independent of language. The
C++ win is narrow but real — it decomposes into two orthogonal compile-time policies that an
older language can't cheaply share: `Vcoord` (target generator: sigma / z* / rho / hybrid) ×
`Reconstruction` (remap method: PCM / PLM / PPM / PQM). And that **`Reconstruction` is the
*same* policy** continuity and tracer advection use. Such a language ends up carrying parallel PPM
implementations across continuity, remap, and tracer advection because it can't share
compile-time-specialized code; C++ = **one positivity-preserving reconstruction, three
consumers.**

## Bucket 2 — GPU-fundamental (carries over UNCHANGED — not a C++ win)

The honest half. Much "workaround" code in a mature GPU solver is the **GPU** talking, not the
language, and C++ inherits every bit. Do NOT try to "clean these up" — they're structural.

- **Fixed-size column buffers** (a compile-time `MAX_NZ` bound) — device code can't
  heap-allocate or size locals dynamically → `std::array<Real, MAX_NZ>` in C++ too.
- **Column-serial recurrences** (serial in the vertical `k`, parallel in `i,j`: tridiagonal
  solves, remap integrals, PV scans) — algorithmic, language-agnostic.
- **Positivity / monotonicity limiters** on thickness — physics, not language (the
  `Reconstruction` contract: h ≥ 0 is load-bearing; you divide by h everywhere).
- **Capture-by-value / no-`this` / hoist-members-to-locals** — the device execution model;
  identical discipline in any language. (Capturing `this` in a member kernel passes on the host
  and hard-crashes on the GPU with an illegal-address fault.)
- **CPU-green ≠ GPU-correct** — the entire validation burden is unchanged; a passing host build
  proves logic, not offload or data motion.

So the inside of your remap and vcoord kernels will *look* structurally like any mature GPU
solver's — fixed columns, serial-k, conservative integrals — because that's the GPU + the math,
not the language.

## Bucket 3 — New C++ / nvc++ footguns

- **Managed memory ≠ free.** You trade explicit-mapping boilerplate for **page-migration
  overhead** and loss of explicit data-motion control (which directive-based mapping gave you).
  Keep state device-resident by discipline; a callback that touches host data every step can
  thrash (measured: a per-step host copy of the state is ~100×+ slower than staying resident on
  a V100, and a 1.0× no-op on the host build — the penalty is purely GPU migration).
- **Template compile times + binary bloat** — every policy combination is an instantiation.
  Eliminate the runtime branch for *clarity*, not perf: a loop-uniform in-kernel branch is
  *nearly free* on the GPU (measured, nvc++ / V100).
- **Abstraction astronomy** — concepts-all-the-way-down can out-clever the reader. A pragmatic
  enum dispatch is legible; keep the C++ equally so.
- **C++23 stdlib gaps under nvc++** (`std::mdspan` / `std::print` present but with undefined
  feature-test macros, or absent on an older host stdlib) — a real M0 risk; gate on
  `__has_include`, not the feature macro, and keep the hand-rolled fallback.

## The meta-lesson

C++ pays off precisely where the older toolchain has **ceremony** (dispatch, mapping,
duplication, indirection) — that evaporates. It pays off **not at all** where the constraints
are **GPU-shaped** (fixed columns, serial-k, positivity, capture rules) — same in any language.
Net win, but the win is *"less boilerplate,"* not *"the hard parts get easier."* The hard parts
— **core stability, GPU correctness, validation** — are Bucket 2, and they are waiting for you
in C++ just the same.
