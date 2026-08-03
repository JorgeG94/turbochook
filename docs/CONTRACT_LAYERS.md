# Contract — source layers and the dependency rule

**Status:** normative. This is the one contract that constrains every other file in the
tree, and it is the cheapest one to enforce.

Related: [`CONTRACT_MEMORY.md`](CONTRACT_MEMORY.md) (what lives in `lib/memory` and
`lib/device`) · [`CONTRACT_RUNTIME.md`](CONTRACT_RUNTIME.md) (dispatch, steppers, MPI
seams) · [`plans/00_PLAN.md`](plans/00_PLAN.md).

---

## 1. The rule

```
api  ->  {ocean_physics, ice_physics}  ->  ocean_lib  ->  lib
```

**Dependencies run strictly one way.** `lib/` must never include from `ocean_lib/`;
`ocean_lib/` must never include from `ocean_physics/` or `ice_physics/`; neither physics
tree includes from `api/`. `ice_physics/` and `ocean_physics/` are **peers** and must not
include each other — anything they genuinely share belongs in `ocean_lib/`.

This is the whole point of the split. The directories are a description; the rule is the
contract, and without it the layering rots in eighteen months with nobody noticing.

**Enforced by pre-commit**, because a rule nobody checks is a comment:

```
no-upward-include   for each layer, grep its #includes for a path in a higher layer
```

Cheap to write, and it fails loud at the moment someone reaches upward — which is always
a two-line convenience that turns into a cycle.

### What each layer means

| layer | knows about | does NOT know about |
|---|---|---|
| `lib/` | bytes, devices, ranks, arrays, streams | the ocean, grids, physics |
| `ocean_lib/` | grids, metrics, staggering, reconstruction, remap | closures, EOS, forcing |
| `ocean_physics/` | the ocean | sea ice, the Python API |
| `ice_physics/` | sea ice | the ocean's internals |
| `api/` | everything, and binds it | — |

The test for `lib/`: **could this file compile into a code that solves something other
than an ocean?** If not, it is in the wrong layer.

---

## 2. The tree

```
src/
  lib/                     domain-agnostic infrastructure
    core/                  Array, View, slice, VectorField, the vocabulary
    memory/                MemoryQuantity, MemoryRequirement(s), Arena(s), ScratchScope,
                           allocation/{local,shared,gpu}
    device/                backend, context, alloc, launch, native, capabilities
    comm/                  MPI wrappers, halo groups, collectives, Efp reduction
    containers/            bidirectional_map, cyclic_buffer, reverse_destruct_vector
    io/                    NetCDF / HDF5 plumbing, parallel-IO mechanics
    logging/               the logger, leader_only
    timing/                timers, byte + flop recorder
    constants.hpp          physical constants

  ocean_lib/               ocean-shaped, physics-free
    mesh/                  cartesian, spherical, supergrid, tripolar fold, metrics
    numerics/              tridiagonal, CG, quadrature, root-finding
    reconstruction/        PCM, PLM, PPM, PQM, WENO
    remap/                 regrid + remap
    advection/
    stepping/              SspStep<N>, Adams-Bashforth -- the GENERIC steppers
    diagnostics/           registry, cadence dispatch, derived catalog
    io/                    CF-compliant field writer, restart layout, gauges

  ocean_physics/
    dycore/                continuity, Coriolis, PGF, the split-explicit orchestration
    ALE/
    equation_of_state/
    parameterizations/{vertical,horizontal,stochastic}
    boundary/              wall, Flather, tidal, sponge, radiation
    forcing/               wind stress, heat flux, tides

  ice_physics/             LATER -- a peer of ocean_physics, not a child
    rheology/  itd/  thermo/

  api/                     spec, validate, factory, isolver, the C ABI
  driver/                  the run loop -- testable on its own
app/                       thin entry point
```

---

## 3. The three splits that are easy to get wrong

Each of these looks like one component and is actually two, one per side of a layer
boundary. Getting them wrong is how domain knowledge leaks downward.

**I/O.** `lib/io/` is NetCDF and HDF5 mechanics — file handles, parallel-IO, deflate,
chunking. It knows nothing about fields. `ocean_lib/io/` is the CF-compliant writer, the
restart layout and the gauges, which need `Loc`, the vertical coordinate and staggering
metadata. Fuse them and CF knowledge lands in `lib/`, and the layering rule dies with it.

**Time stepping.** `SspStep<N>` and Adams-Bashforth are generic over any tendency
function and live in `ocean_lib/stepping/`. The split-explicit barotropic/baroclinic
orchestration — the theta-weighted correction, the inner barotropic loop, the dual
anchoring — is MOM6-shaped and belongs to `ocean_physics/dycore/`. `CONTRACT_RUNTIME`
already separates these two types; the directories must agree.

**Numerics.** Tridiagonal solves, CG and quadrature contain no ocean knowledge and would
be equally at home in `lib/numerics/`. They sit in `ocean_lib/numerics/` anyway, because
their *callers* are all ocean code and a `lib/` with no other consumer for them is a
distinction without a difference. If a second, genuinely non-ocean consumer ever appears,
move them down — that direction is always safe.

---

## 4. The one deliberate exception

**`Loc` lives in `lib/core/`**, even though staggering looks like a domain concept.

It has to: `Array<T, Rank, Space, Loc>` carries it, `Arena::alloc` derives face extents
from it, and `Array` is infrastructure. The alternative — templating `Array` on an opaque
tag so `ocean_lib` could define `Loc` — buys purity and costs a template parameter with
exactly one user, forever.

It survives scrutiny on its own terms, too: `Loc` says **where in a cell a value sits**,
which is a finite-volume concept rather than an ocean one. Any staggered FV code has it,
ocean or not — so it passes the `lib/` test in §1.

Recorded here so it reads as a decision rather than an oversight, and so nobody "fixes" it
later. `Parity` sits beside it for the same reason, though it is consumed only by
`lib/comm/` at halo-group registration.

---

## 5. What this replaces

Earlier drafts of the plans wrote paths as `src/core/`, `src/lib/device/`, `src/lib/comm/`,
`src/ocean_physics/` — a flat tree with no layer boundary and no rule. Those paths are
superseded by §2 wholesale:

| old | new |
|---|---|
| `src/core/` | `src/lib/core/` (+ `src/lib/memory/` for the arena) |
| `src/lib/device/` | `src/lib/device/` |
| `src/lib/comm/` | `src/lib/comm/` |
| `src/ocean_physics/` | `src/ocean_physics/` |
| `src/mesh/` | `src/ocean_lib/mesh/` |
| `src/api/` | `src/api/` (unchanged) |

The two grep-bannable lints already specified elsewhere retarget accordingly: `debug_` and
`kernel_view()` are rejected outside kernel bodies under `src/ocean_physics/`,
`src/ice_physics/` and `src/lib/device/`.

---

## 6. Acceptance

- `no-upward-include` passes, and **fails** on a deliberately planted upward include.
- `lib/` builds with no `ocean_lib/` header on the include path at all — the strongest
  form of the test, and worth wiring as a separate CMake target rather than trusting grep.
- `ice_physics/` and `ocean_physics/` build independently of one another.
- No `mpi.h` outside `src/lib/comm/`; no backend header (`cuda_runtime.h`, `sycl.hpp`)
  outside `src/lib/device/`.
