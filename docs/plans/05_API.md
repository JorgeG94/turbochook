# 05 — The Python API

**Depends on:** 01 (Array), 03 (Arena). Independent of 02 and 04.

Contract: [`../CONTRACT_RUNTIME.md`](../CONTRACT_RUNTIME.md) §1–§2, §7.

**There are no input files.** A Python script *is* the configuration. That deletes an
entire subsystem before it is written — no namelist, no schema, no parser, no
migration tooling for renamed groups.

---

## Scope

```
src/api/spec.hpp        RunSpec + per-concern specs (inert data, no types)
src/api/validate.cpp    fail-loud checks; the ONLY place errors are authored
src/api/factory.cpp     make_solver(spec) — the ONLY place tags become types
src/api/isolver.hpp     the host virtual, called once per step
src/api/capi.cpp        extern "C" surface
python/turbochook/      pure-ctypes wrapper; NO build-time binding framework
tests/test_api.cpp      Fortran-style: driven from C++, not through Python
```

## The shape, stolen from Oceananigans

**Construct → validate → materialize.** Python builds inert spec objects; one factory
pass binds them to grid + architecture and returns a monomorphic engine. Their
`materialize_*`-per-concern pattern is the most portable idea in that codebase and it
survives the compile-time/runtime split intact.

```python
import turbochook as tc

mesh = tc.SphericalMesh(512, 512, lon=(-193.75, -171.25), lat=(53.6, 64.9),
                        west="periodic", east="periodic", south="wall", north="wall")

sim = tc.Simulation(mesh,
    stepper    = "split_rk2",
    dt         = 1200.0, n_inner = 0,          # 0 -> CFL-derived, latched at setup
    pgf        = tc.Pgf("gprime"),
    coriolis   = tc.Coriolis("sadourny"),
    continuity = tc.Continuity("ppm"))

sim.h[0][:] = h1                                # numpy, zero-copy into the arena
sim.diagnostics = ["mass", "KE", "speed"]
sim.output("state.nc", every=tc.days(1))
sim.run(days=580)
```

### Rules worth taking verbatim

- **The grid is the single source of truth for device and precision.** No `device=` on
  the model, no global precision switch. Oceananigans' mutable
  `defaults.FloatType` makes precision depend on *construction order* — do not copy it.
- **`None` means off, and off is the default** for everything optional. Bit-identity
  by construction.
- **Coordinate arguments accept `tuple | callable | numpy array`.** One argument,
  three representations.
- **Never silently re-type what was asked for.** Oceananigans downgrades `WENO(9)` to
  `Centered(8)` on a short axis and announces it in a log line, so `model.advection` no
  longer equals what was passed. **Throw, and say what is required.**
- **Errors name the value, the constraint, and what is actually built:**
  `pgf.form = 'xyz' is not available; built: montgomery, fv, fv_wright, gprime`.
- **`repr(sim)` prints the fully resolved post-materialization config.** With no input
  file, that *is* the provenance record — emit it plus a git SHA at startup and make it
  round-trippable to a runnable script.

## Work order

1. **`RunSpec` + specs.** Plain aggregates, trivially serialisable.
2. **`validate(spec)`.** Every check in one file: mutual exclusivity, required
   companions, ranges, and `max(halo_width)` over the chosen operators → the derived
   `ng`. Because arrays carry Fortran bounds, nothing else in the code learns `ng`.
3. **`ISolver` + `make_solver`.** One `switch` per axis returning a concrete type
   behind an interface — **N + M + K instantiations, not N × M × K**.
4. **The C ABI.** ~8 functions: create/destroy, step, run, set_field, get_field,
   diagnostic, extents, memory. Opaque `void*` handle.
5. **Zero-copy field access.** `mirror()` is already the numpy boundary: on a coherent
   machine or the host build it is the identity and numpy views the arena directly; on
   a discrete GPU it stages. **One code path.** Expose extents, strides and `DType`
   through the ABI so numpy can build the view correctly.
6. **ctypes wrapper.** No pybind11, no nanobind — a plain `.so` plus ctypes keeps the
   dependency policy intact and the build trivial.

## Decisions to record

- **Type/value line** (CONTRACT_RUNTIME §6): precision, backend, dimensionality, EOS,
  vcoord and advection *family* are compile-time; everything else — extents, spacing,
  coefficients, ICs, forcing, cadence, diagnostics, per-boundary topology — is runtime.
- **Testing stays native.** The correctness harness is doctest driven from C++; Python
  is the *driver*, not the test framework. Otherwise the test build stops being
  hermetic. (Standing preference, and it applies here.)
- **Units as module constants** — `tc.days(1)`, `tc.km(5)`. Trivial, large readability
  win in scripts.

## Tests

| test | asserts |
|---|---|
| spec round-trip | `RunSpec` → JSON → `RunSpec` is identical |
| unknown tag | error message names the value AND the built options |
| exclusivity | two mutually exclusive schemes → error naming both |
| derived halo | `ng == max(halo_width)` over the chosen operator set |
| zero-copy | on host, a numpy write is visible to a kernel with no explicit copy |
| staged copy | on a discrete device, the same script produces the same answer |
| provenance | `repr(sim)` round-trips to a script that reproduces the run |

## Acceptance gate

- **The SWE demonstrator is driven entirely from Python** — grid, ICs, run loop,
  diagnostics, output — with no C++ `main` and no config file anywhere.
- `repr(sim)` + git SHA reproduces a run from a clean checkout.
- Every factory error names what is built.

## Deferred

The runtime-interpreted diagnostic composition layer (`w*b`, `∂x(v)-∂y(u)` from Python
without a rebuild — CONTRACT_RUNTIME §9), callbacks and schedules, checkpoint/restart,
and any JIT path that compiles a new policy stack on demand.
