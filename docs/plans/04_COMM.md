# 04 — Comm: the MPI wrapper

**Depends on:** 01 (Array + `Loc`), 02 (device binding, `Region`).

Contract: [`../CONTRACT_RUNTIME.md`](../CONTRACT_RUNTIME.md) §8.

The rule, borrowed from rakali and worth a pre-commit hook: **MPI appears nowhere
outside `src/lib/comm/`, and no operator learns that it exists.** The decomposition
changes the *work region*, not the kernel.

---

## Scope

```
src/lib/comm/comm.hpp          the facade every other file sees
src/lib/comm/decomp.hpp        Decomp: local/global extents, offsets, neighbours
src/lib/comm/exchange.hpp      grouped halo exchange; Loc from the type, Parity at registration
src/lib/comm/collective.hpp    all_reduce, incl. the reproducible mode
src/lib/comm/mpi/*.cpp         real backend
src/lib/comm/single/*.cpp      serial backend — the DEFAULT, and complete
tests/test_comm.cpp
```

## Work order

1. **`single/` first, and make it complete.** Single-rank is the degenerate case of the
   distributed one, not a separate path. Everything below must work at 1×1 before any
   MPI is linked — which also means CI never needs MPI.
2. **`Decomp`.** Local extents, global extents, this rank's offset in global index
   space, neighbour rank per edge, and the edge connectivity (wall / periodic / fold /
   neighbour). `nx_local` derived from `nx_global` and the process grid; auto-factor
   `px*py` when the config leaves them at 1.
3. **Device binding order — build this before the exchange, it is the trap.**
   ```cpp
   comm::bind_device_from_launcher_env();  // local rank from OMPI_COMM_WORLD_LOCAL_RANK etc.
   comm::init();                            // MPI_Init
   device::initialize();                    // now safe
   ```
   - **CUDA:** `CUDA_VISIBLE_DEVICES` per rank **before `MPI_Init`** — UCX creates a
     primary context at `MPI_Init`, so with all GPUs visible every rank *also* lands a
     context on device 0 while the binding diagnostic still looks correct.
   - **Intel:** one rank per **tile** — measured **6.3×**. `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.

   The local rank is readable from the launcher's environment *without* `MPI_Init`,
   which is the only reason this ordering is possible. Encode it; don't document it.
4. **Halo exchange, driven by the type.**
   ```cpp
   // Parity is NOT a template parameter of Array -- it changes no layout and no
   // extent, and vector components must exchange as a PAIR regardless (a 90-degree
   // rotation SWAPS them). It is declared once, at group registration.
   HaloGroup g;
   g.add(eta, Parity::Scalar, /*width=*/1);
   g.add(u, v, Parity::Vector, 1);            // the pair, together
   Request exchange_begin(HaloGroup&, const Decomp&);
   void    exchange_end(Request&);
   ```
   `Loc` comes from the array's type and gives the extents plus the duplicated face
   seam. `Parity` is supplied once at registration and checked there. **Grouping is the
   point**: MOM6 packs its whole barotropic hot loop into one message set, and at global
   1 km message count dominates. Pack buffers come from the **arena** — device-resident,
   so they go straight to a GPU-aware `MPI_Isend`.
5. **Comm/compute overlap via `Region`.** The whole point of the day-0 seam:
   ```cpp
   auto req = exchange_begin(h, decomp);
   do_concurrent(tag, Region::interior_inset(ng), kernel);   // while it flies
   exchange_end(req);
   do_concurrent(tag, Region::halo_strips(ng), kernel);      // the four edges
   ```
   No operator changes. Oceananigans' entire distributed launcher override is six
   lines for exactly this reason.
6. **Collectives.** `all_reduce` over the local `device::reduce` result. Two modes:
   - `Sum::Fast` — `MPI_Allreduce`, not reproducible across rank counts.
   - `Sum::Reproducible` — **fixed-point integer accumulation**, order-independent by
     construction. Conservation totals are the validation oracle (ADR-8), and a tree
     reduction gives different bits on a different decomposition. Compensated
     (Kahan/Neumaier) summation improves accuracy but is **still order-dependent** —
     it does not solve this. Only fixed-point does.

## Decisions to record

- **`GlobalIndex = std::int64_t`, distinct from `Index = std::int32_t`.** Spell the
  widths: `long` is 32-bit under LLP64, and the widths are load-bearing (the ~1.06×
  address-bound measurement, and the global horizontal index at 1 km being under 2³¹ by
  only 2×). Locals stay 32-bit; only decomposition arithmetic and I/O go wide.
- **GPU-aware vs host-staged buffers is a runtime capability query, not a build flag**
  where possible — fall back cleanly, and report which path is live at startup.
- Packing: start with a `do_concurrent` pack into an arena buffer. Note that
  Oceananigans moved *away* from kernels to strided view copies for this; measure
  before assuming.

## Tests (all runnable single-rank, then 2×2)

| test | asserts |
|---|---|
| decomp identity | 1×1 decomposition reproduces global extents exactly |
| exchange round-trip | fill halo with sentinel, exchange, interior untouched, halo correct |
| periodic wrap | a field with a known ramp wraps correctly in x |
| **vector parity** | a `Parity::Vector` field sign-flips across a fold edge; `Scalar` does not |
| overlap equivalence | interior-then-strips == whole-domain-after-exchange, bitwise |
| **reproducible sum** | identical bits at 1, 2 and 4 ranks; `Fast` mode differs |
| binding order | each rank reports a distinct device; on Intel, a tile not a card |

## Acceptance gate

- **The SWE demonstrator's baroclinic run at 2×2 reproduces single-rank.** Eddy field,
  mass drift, and KE within the bit-identity bar — this is the real gate; the unit
  tests only localise failures.
- The reproducible sum is **bitwise identical across rank counts**.
- No `mpi.h` include outside `src/lib/comm/` (enforce with a pre-commit hook).
- CI passes with the `single/` backend and no MPI installed.

## Deferred

Tripolar fold *implementation* (the `Parity` seam must exist; the fold need not),
nested/mosaic grids, an I/O server, load balancing, one-sided/RMA, NCCL-style
comm streams.
