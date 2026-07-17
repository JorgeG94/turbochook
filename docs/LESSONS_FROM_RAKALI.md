# Lessons from Rakali / Fortran → turbochook

Rakali's design choices split three ways: some were **Fortran fighting the language** (C++
erases them), some were **the GPU fighting you** (C++ inherits them *unchanged* — do not
"fix" them), and some are **new C++ footguns** Fortran didn't have. Keeping these straight
stops two mistakes: expecting C++ to dissolve constraints that are actually the GPU's, and
"modernizing away" a Rakali-ism that was load-bearing.

## Bucket 1 — Genuine C++ wins (Fortran fought the *language*)

These were pure ergonomics tax; C++ removes them.

| Rakali did X *because Fortran* | turbochook does Y |
|---|---|
| **enum + `select case`** dispatch (`parse_*_variant`) — weak polymorphism → runtime branch | **compile-time policies** (concepts) — branch vanishes, scheme inlines |
| **array-of-DT + outer-shim** for `tracers(:)` — NVHPC can't index a DT-array w/ allocatable components in a kernel; you *wanted* "4D contiguous tracer backing" but couldn't get it cleanly | **data-oriented layout from the start** — a contiguous tracer bank (`mdspan` / `array<Field3,Ntr>`). The 4D backing you wished for *is* the natural C++ representation |
| **`enter_data` orchestrator + map-every-component** — mem:separate forces manual mapping; miss one → 150–1500× memcpy explosion | **arena managed memory** — allocation *is* the mapping; no orchestrator exists |
| **explicit-shape dummies** (`arr(nx,ny,nz)`) — assumed-shape triggers per-launch descriptor memcpys | **`mdspan` value semantics** — extents travel with the view for free |
| **structured/unstructured duplication** (`rki_ml_*` vs `*_unstr`) + **`_impl` copy-paste** (NVHPC won't inline cross-module) | **templates + header-only inlining** — write once, specialize per backend |
| **module-global lazy scratch** (`*_workspace_ensure` / `*_cleanup`) | **RAII arena-owned member workspace** (DESIGN §6b) — no globals |
| **`error stop` + `is_init` flags** | **exceptions (host) + RAII lifecycle** |

Biggest two: the **tracer backing** and the **structured/unstructured duplication** — those
were *pure* Fortran-ergonomics tax and C++ erases them entirely.

**On remap / regrid / generalized vertical coordinates specifically:** the ALE framework
itself (target-thickness generator + conservative remap) is good design that transfers. The
C++ win is narrow but real — it decomposes into two orthogonal compile-time policies Fortran
couldn't cheaply share: `Vcoord` (target generator: sigma / z* / zstar-full / rho / hycom) ×
`Reconstruction` (remap method: PCM/PLM/PPM/PQM). And that **`Reconstruction` is the *same*
policy** used by continuity and tracer advection. Rakali carries parallel PPM implementations
across continuity, remap, and tracer-drain because Fortran can't share compile-time-
specialized code; C++ = **one positivity-preserving reconstruction, three consumers.**

## Bucket 2 — GPU-fundamental (carries over UNCHANGED — not a C++ win)

The honest half. Much of Rakali's "workaround" code was the **GPU** talking, not Fortran, and
C++ inherits every bit. Do NOT try to "clean these up" — they're structural.

- **Fixed-size column buffers** (`NZ_STACK_MAX`) — device code can't heap-allocate or size
  locals dynamically → `std::array<Real, NZ_MAX>` in C++ too.
- **Column-serial remap** (serial in k, parallel in i,j) — algorithmic, language-agnostic.
- **Positivity / monotonicity limiters** on thickness — physics, not language (the
  `Reconstruction` contract: h ≥ 0 is load-bearing; you divide by h everywhere).
- **Capture-by-value / no-`this` / hoist-to-locals** — the device execution model; identical
  discipline in both languages.
- **CPU-green ≠ GPU-correct** — the entire validation burden is unchanged.

So the inside of your remap and vcoord kernels will *look* structurally like Rakali's — fixed
columns, serial-k, conservative integrals — because that's the GPU + the math, not Fortran.

## Bucket 3 — New C++ footguns Fortran didn't have

- **Managed memory ≠ free.** You trade `enter_data` boilerplate for **page-migration
  overhead** and loss of explicit data-motion control. Rakali's mem:separate + hand-placed
  `update self` gives fine control you're giving up. Keep state device-resident by discipline;
  a callback that touches host data every step can thrash.
- **Template compile times + binary bloat** — every policy combination is an instantiation.
  Eliminate the runtime branch for *clarity*, not perf: Rakali's own measurement
  (`dc_uniform_select_case`) found a uniform in-kernel branch is *nearly free* on GPU.
- **Abstraction astronomy** — concepts-all-the-way-down can out-clever the reader. Rakali's
  pragmatic enum dispatch was legible; keep the C++ equally so.
- **C++23 stdlib gaps under nvc++** (`mdspan` / `print`) — a real M0 risk Fortran didn't have.

## The meta-lesson

C++ pays off precisely where Rakali has **ceremony** (dispatch, mapping, duplication,
DT-indirection) — that evaporates. It pays off **not at all** where Rakali has **GPU-shaped
constraints** (fixed columns, serial-k, positivity, capture rules) — same in any language. Net
win, but the win is *"less boilerplate,"* not *"the hard parts get easier."* The hard parts —
**core stability, GPU correctness, validation** — are Bucket 2, and they are waiting for you in
C++ just the same.
