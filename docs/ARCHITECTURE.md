# TurboChook — Architecture Map

How the pieces fit: header dependencies, the concept↔model wiring, the `OceanCore`
composition, and the data flow through one step. Diagrams are **mermaid** (they render
on GitHub and in a Claude artifact). This is the visual companion to
[`DESIGN.md`](DESIGN.md) (the prose spec + ADRs) and [`FOUNDATIONS.md`](FOUNDATIONS.md)
(the directory layout).

**Status legend for the physics:** interfaces (concepts, types, composition) are all
real and type-checked; kernel *bodies* are at mixed maturity — `FvPgf`, the mesh, and
the diagnostics are implemented and tested (host + GPU); `ContinuityFlux`,
`SadournyEnstrophy`, the integrators, and the BCs are M2 stubs behind real signatures.

---

## 1. Header / layer dependency graph

Arrows read **"includes / depends on."** The base layer (`core/types` + `lib/*`) sits
under everything; `ocean_core.hpp` sits on top and pulls the whole stack together.

```mermaid
flowchart TD
    subgraph base["core + lib — base layer (stdlib only)"]
        types["core/types.hpp<br/>Real · Index · Field"]
        err["lib/error.hpp"]
        arena["lib/arena.hpp"]
        log["lib/log.hpp"]
        prof["lib/profiler.hpp"]
    end
    subgraph num["numerics"]
        par["parallel.hpp<br/>tc::par · for_each_*"]
        integ["integrator.hpp<br/>Integrator · SSPRK2"]
    end
    subgraph meshl["mesh"]
        meshc["mesh.hpp<br/>Mesh concept · Loc"]
        cart["cartesian_mesh.hpp<br/>CartesianMesh"]
        iter["iterate.hpp<br/>FaceView · for_each_face"]
    end
    subgraph phys["physics"]
        recon["reconstruction.hpp<br/>Wall/Face concepts · Poly"]
        state["baro_state.hpp<br/>BaroState · Params"]
        cont["continuity.hpp<br/>ContinuityFlux"]
        cor["coriolis.hpp<br/>SadournyEnstrophy"]
        pgf["pgf.hpp<br/>FvPgf"]
        ocore["ocean_core.hpp<br/>OceanCore&lt;…&gt;"]
    end
    bcp["bc/bc.hpp<br/>WallBC · PeriodicBC"]
    diag["diag/diagnostics.hpp<br/>total_mass · any_nonfinite"]

    arena --> types
    arena --> err
    par --> types
    meshc --> types
    cart --> meshc
    iter --> meshc
    iter --> par
    recon --> types
    state --> types
    state --> arena
    state --> cart
    integ --> state
    cont --> recon
    cont --> state
    cont --> cart
    pgf --> state
    pgf --> iter
    cor --> state
    cor --> cart
    bcp --> cart
    bcp --> state
    diag --> meshc
    diag --> state
    diag --> par
    ocore --> cont
    ocore --> cor
    ocore --> pgf
    ocore --> bcp
    ocore --> integ
    ocore --> log
    ocore --> prof
```

The **dependency rule** (ADR-7 / FOUNDATIONS): `core/` depends on nothing but the
stdlib; `lib/` on the stdlib + `core/`; everything above on both. Physics modules never
include each other's internals — they meet only inside `ocean_core.hpp`.

---

## 2. Concept ↔ model map

Every policy axis is a **concept** (a compile-time contract); each concrete type
**models** it (`..|>` = "models / satisfies"). Swapping a scheme = swapping a type at
the `OceanCore` instantiation — no inheritance, no vtables.

```mermaid
classDiagram
    class Mesh {
        <<concept>>
    }
    CartesianMesh ..|> Mesh

    class WallReconstruction {
        <<concept>>
    }
    class FaceReconstruction {
        <<concept>>
    }
    class Reconstructor {
        <<concept>>
    }
    Pcm ..|> WallReconstruction
    Plm ..|> WallReconstruction
    Ppm ..|> WallReconstruction
    Pqm ..|> WallReconstruction
    Weno5 ..|> FaceReconstruction
    Weno7 ..|> FaceReconstruction
    Weno9 ..|> FaceReconstruction
    WallReconstruction --> Reconstructor : union
    FaceReconstruction --> Reconstructor : union

    class ContinuityModule {
        <<concept>>
    }
    class PgfModule {
        <<concept>>
    }
    class CoriolisModule {
        <<concept>>
    }
    class Integrator {
        <<concept>>
    }
    class BoundaryCondition {
        <<concept>>
    }
    ContinuityFlux ..|> ContinuityModule
    FvPgf ..|> PgfModule
    SadournyEnstrophy ..|> CoriolisModule
    SSPRK2 ..|> Integrator
    ForwardEuler ..|> Integrator
    WallBC ..|> BoundaryCondition
    PeriodicBC ..|> BoundaryCondition
```

Notes:
- **`Reconstructor = WallReconstruction ‖ FaceReconstruction`** — a *disjoint* umbrella
  (a scheme is one kind or the other, never both). `ContinuityFlux<Scheme>` constrains on
  `WallReconstruction` (swept flux integrates a `Poly`); WENO's `FaceReconstruction` is
  reserved for the planned tracer-advection consumer.
- `PpmContinuity` is the alias `ContinuityFlux<Ppm>`.
- The module concepts (`ContinuityModule`/`PgfModule`/`CoriolisModule`) are structurally
  the same `{ init; compute(s,k,mesh,p) }` — distinct *names* document each slot's intent.

---

## 3. The composition — `OceanCore` (the god-state)

`OceanCore` is templated on the five policy axes and owns the state + one module per
slot. One line names the whole scheme.

```mermaid
classDiagram
    class OceanCore {
        <<templated>>
        -CartesianMesh mesh_
        -Arena arena_
        -Params p_
        -Cont cont_
        -Cor cor_
        -Pgf pgf_
        -Bc bc_
        -BaroState state_
        -BaroState k_
        +init()
        +step()
        +baro_rhs(s, k)
    }
    class BaroState {
        +Field2 eta
        +Field2 u
        +Field2 v
    }
    OceanCore *-- BaroState : state_, k_
    OceanCore *-- CartesianMesh : mesh_
    OceanCore ..> Integrator : drives step
```

`OceanCore<Cont, Cor, Pgf, Bc, Integ>` — five policy slots. `arena_` is borrowed (one
arena per run). `BaroState` is the staggered C-grid state: `eta` at centres, `u` on
x-faces, `v` on y-faces; `k_` is the RK scratch register. The whole PoC scheme is one
line:

```cpp
using BarotropicPoC = OceanCore<PpmContinuity, SadournyEnstrophy, FvPgf, WallBC, SSPRK2>;
```

---

## 4. Data flow through one step

`step()` hands the state + two host callables (the RHS op and the BC op) to the
integrator. The RHS is a **sum of operator tendencies** accumulated into the scratch
register `k`; the integrator combines stages into the new state. Kernels reach the grid
only through the `FaceView` seam and mesh metrics.

```mermaid
flowchart TD
    S["BaroState s {eta,u,v}"] --> ADV
    M["CartesianMesh (metrics, connectivity, wet)"] --> ADV
    P["Params (dt, g, H)"] --> ADV
    subgraph ADV["OceanCore::step → Integrator::advance"]
        direction TB
        BC["bc.fill_halos(s, mesh)<br/>(before each stage → interior branch-free)"] --> RHS
        subgraph RHS["baro_rhs(s, k): zero k, then SUM tendencies"]
            direction LR
            C1["Continuity.compute<br/>→ k.eta"]
            C2["PGF.compute<br/>→ k.u, k.v"]
            C3["Coriolis.compute<br/>→ k.u, k.v"]
        end
        RHS --> CMB["Integrator combine<br/>s ← s + dt·(stages)"]
    end
    CMB --> NS["new BaroState"]
    SEAM["for_each_face(mesh, FaceView)<br/>li/ri connectivity · span metric"] -. feeds .-> RHS
    NS -. device reduction .-> D["diagnostics<br/>total_mass · CFL · NaN → host scalar"]
```

The **boundary** the whole design protects (DESIGN §3): only cheap PODs cross into a
kernel — `Field` views + a `Params` struct, **by value**. No `virtual`, `std::function`,
or captured `this` past this line, or offload breaks. See
[`CPP_PRIMER.md`](CPP_PRIMER.md) for *why* (lambda captures, `mdspan`, concepts).
