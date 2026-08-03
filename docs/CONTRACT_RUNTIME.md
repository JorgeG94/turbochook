# How a simulation is composed and run

The companion to [`CONTRACT_MEMORY.md`](CONTRACT_MEMORY.md). That one says how arrays
and kernels work; this one says how a *run* gets built from a Python script and stepped.

---

## 0. The rule that makes this tractable

[`REDESIGN.md`](REDESIGN.md) §3 counted ~2000 instantiations if every scheme axis is a
template parameter of one monolithic core. That count assumed the wrong factoring.

> **Compile-time INSIDE an operator. Runtime AT the operator boundary.**

- The PPM stencil inside a tracer-advection kernel is **per-cell** ⇒ compile-time.
  `TracerFlux<Ppm>` and `TracerFlux<Weno5>` are distinct instantiations, fully inlined.
- *Which* advection operator runs is decided **once per stage** ⇒ runtime. One virtual
  call in front of a kernel launch that costs milliseconds.

The instantiation count collapses from **N × M × K** to **N + M + K** — from ~2000 to
a few dozen — and Python gains unrestricted freedom to combine schemes. This is the
same §3 rule ("compile-time only if the polymorphism boundary is per-cell"), applied
one level up.

The exception is fusion, and it is bounded — see §5.

---

## 1. The flow

```
  Python script  (dataclasses; the config, since there are no input files)
        │  C ABI: one JSON string, or a POD struct
        ▼
  RunSpec        plain C++ data: runtime values + scheme TAGS. No types yet.
        │  validate(spec)      fail loud; name the value, the constraint, the options
        ▼
  make_solver(spec)            THE ONE PLACE tags become types
        │
        ▼
  unique_ptr<ISolver>          erased at the coarsest boundary; step() is per-step
```

`ISolver` is a host virtual called **once per step** in front of dozens of kernel
launches. Virtual is legal here and always was — it is forbidden *inside* kernels, and
nothing about this crosses the device boundary.

---

## 2. Specs are inert data

```cpp
struct PgfSpec        { std::string form = "fv";        Real rho0 = 1035.0; };
struct CoriolisSpec   { std::string form = "sadourny";  std::string pv_scheme = "centered"; };
struct TracerAdvSpec  { std::string scheme = "ppm";     int order = 5; };
struct VmixSpec       { std::string scheme;             std::map<std::string, Real> params; };

struct RunSpec {
    GridSpec              grid;
    TimeSpec              time;          // dt, n_inner (0 = CFL-derived), stepper tag
    PgfSpec               pgf;
    CoriolisSpec          coriolis;
    TracerAdvSpec         tracer_adv;
    std::vector<VmixSpec> vmix;          // an ENSEMBLE — see §4.2
    std::vector<VmixSpec> hvisc;
};
```

No types, no templates, trivially serialisable — so this struct *is* the provenance
record. Dump it (resolved, post-validation) plus a git SHA at startup and the run is
reproducible without an input file existing.

---

## 3. One interface + one factory per axis

```cpp
struct IPgf {
    virtual ~IPgf() = default;
    // Called ONCE PER STAGE. Launches its own kernels; the tendency is accumulated.
    virtual void compute(const LayeredState&, LayeredState& tend,
                         const Mesh&, const Params&) = 0;
    virtual int         halo_width() const = 0;   // framework takes max() over all operators
    virtual const char* name()       const = 0;
};

// The ONE place a config tag becomes a type. Note the error: it names the value,
// the constraint, AND what is actually built (the Oceananigans lesson).
std::unique_ptr<IPgf> make_pgf(const PgfSpec& s) {
    if (s.form == "montgomery") return std::make_unique<Op<MontgomeryPgf>>(s);
    if (s.form == "fv")         return std::make_unique<Op<FvPgf>>(s);
    if (s.form == "fv_wright")  return std::make_unique<Op<FvWrightPgf<WrightEos>>>(s);
    if (s.form == "gprime")     return std::make_unique<Op<ReducedGravityPgf>>(s);
    throw ConfigError("pgf.form = '{}' is not available; built: "
                      "montgomery, fv, fv_wright, gprime", s.form);
}

// Tracer advection is NOT a peer slot of continuity — see the box below.
std::unique_ptr<IContinuityTracer> make_continuity_tracer(const TransportSpec& s) {
    if (s.tracer == "ppm") return std::make_unique<Op<ContinuityTracer<Ppm, Ppm>>>(s);
    if (s.tracer == "plm") return std::make_unique<Op<ContinuityTracer<Ppm, Plm>>>(s);
    if (s.tracer == "pcm") return std::make_unique<Op<ContinuityTracer<Ppm, Pcm>>>(s);
    throw ConfigError("transport.tracer = '{}'; built: ppm, plm, pcm", s.tracer);
}
```

> **Two corrections here, both from the reference codes.**
>
> **Continuity and tracer advection are ONE operator, not two slots.** MOM6 reconstructs
> the pre-advection volume *backwards* from the post-continuity thickness so the two can
> never disagree; rakali went further and fused them into a single entry point, computing
> the mass flux once and reusing it. If they are peer `unique_ptr`s, the
> thickness/tracer consistency invariant is maintained by review rather than by
> construction — and when it breaks, tracers both drift *and* overshoot local extrema.
> `ContinuityTracer<MassRecon, TracerRecon>` has two compile-time reconstruction axes and
> is legal under invariant 7 (fusion *inside* one operator).
>
> **WENO is not a drop-in `TracerRecon`, and the earlier example was wrong.** MOM6 ships
> exactly PLM, PPM and PPM-H3 for tracers — WENO appears only in the Coriolis PV
> interpolation. The reason is structural: the PPM tracer flux is **not a face value**,
> it is the CFL-weighted integral of the reconstruction over the *swept volume*. A
> standard WENO-Z reconstruction yields a point value with no swept integral and no
> positivity guarantee, so substituting it silently changes the scheme's order *and*
> loses monotonicity. rakali built a `weno5_face_swept` variant explicitly for this.
> Ship PLM and PPM; treat swept-volume WENO as a separate, separately-gated
> reconstruction.
>
> **Store concentration, not `h·Tr`.** Conservation then follows from the *flux* being
> exact rather than from never dividing; the EOS and vmix want concentration anyway; and
> it removes a `1/h` guard from every consumer. rakali pays for `hTr` with thin-layer
> floors scattered through the code.
>
> **The tracer registry locks before `Arena::seal()`.** Both reference codes arrived at
> this independently, for the same reason: the device map must be static.

`Op<Scheme>` is a thin adaptor giving any scheme class the interface. The **scheme's
own genericity stays compile-time** — `TracerFlux<Weno5>`'s stencil inlines exactly as
if there were no interface, because the virtual call happens *outside* the launch.

**Cost accounting.** 4 PGF forms + 3 Coriolis forms + 5 advection schemes = **12**
instantiations supporting **60** combinations. Every combination Python can name is
legal; none of them cost build time.

---

## 4. Dynamics owns the order; config owns the occupants

```cpp
class SplitDynamics {
    std::unique_ptr<IContinuity>          cont_;
    std::unique_ptr<ICoriolis>            cor_;
    std::unique_ptr<IPgf>                 pgf_;
    std::unique_ptr<ITracerAdv>           tadv_;
    std::vector<std::unique_ptr<IVmix>>   vmix_;     // ensemble, §4.2
    std::vector<std::unique_ptr<IHvisc>>  hvisc_;    // ensemble
    BarotropicSubcycler                   bt_;
public:
    void forward_stage(Real dt);       // Φ — see §4.1
    LayeredState& state();
};
```

**The config chooses the occupant of a slot, never the sequence.** Operator order is a
correctness property (rakali: BPG → friction → Smagorinsky → truncation → Coriolis →
recompute), not a knob. Exposing it to Python would be exposing a way to be wrong.

### 4.1 Φ — one MOM6 split stage

> **The sketch below was missing three load-bearing terms.** Each is cheap in the
> operator and pervasive to retrofit, so they are named here rather than discovered:
>
> - **Coriolis reference subtraction.** Without subtracting `Cor_ref` inside the
>   substep, the barotropic Coriolis is integrated **twice** and the Δu corrector hands
>   every layer an extra `dt·f·v̄` rotation per stage. The symptom is subtle — a
>   slightly wrong western boundary current, not a blow-up.
> - **The `uhbt0` offset.** `uhbt0 = Σₖ uhₖ − (depth-mean velocity × face area)`, added
>   as a constant to every substep's transport so the barotropic mass flux matches what
>   the layers will actually produce.
> - **`visc_rem` weighting of the depth mean.** The depth mean is **not** a plain
>   thickness average: `wt_u = frhatu · visc_rem`, renormalised to sum to 1. It is
>   weighted by how much of a barotropic acceleration each layer actually *retains*
>   after vertical friction.
>
> And the PGF double-count guard is **three-sided, not two**: the substep's PGF works on
> the anomaly `(eta_PF_BT − eta_PF)·gtot`; the accumulated `u_accel_bt` sums only
> `Cor + PF`, never `BT_force`; and the layers get `u_accel_bt − ∇[(pbceₖ − gtot)·e_anom]`.
> Miss any one and the surface pressure gradient is applied 1.5 or 2 times.

```cpp
void SplitDynamics::forward_stage(Real dt) {
    derive_bt_from_layers(state_, bt_entry_);           // 1. entry barotropic state
    save(ubt_n_, bt_entry_);

    zero(kcor_); zero(kpgf_);                           // 2. slow tendencies
    for (int l = 0; l < NL; ++l) cor_->compute(state_.layer(l), kcor_.layer(l), mesh_, p_);
    pgf_->compute(state_, kpgf_, mesh_, p_);
    for (auto& hv : hvisc_) hv->contribute(state_, kcor_, mesh_, p_);   // ensemble, +=

    depth_mean_faces(kcor_, state_, f_fast_);           // 3. BT forcing; PGF excluded —
    depth_mean_faces(kpgf_, state_, dm_pgf_);           //    the substep owns -g∇η
    add_uv(f_full_, 1, f_fast_, 1, dm_pgf_);

    apply_slow_momentum(dt);                            // 4. forward Euler on layers

    for (int m = 0; m < n_inner_; ++m)                  // 5. FB barotropic subcycle
        bt_.substep(bt_, f_fast_, mesh_, p_, dt / n_inner_);

    inject_fast_increment(dt);                          // 6. Δu = U_end − U_n − Δt·F_full
    advance_thickness_mean_anchored(dt);                // 7. dual anchor + h-rescale
}
```

Exactly the Hallberg flow, with three of the four subtleties already recorded in ADR-9
(dual anchoring, the two-sided PGF double-count guard, mass consistency). **Every
operator call here is one virtual dispatch in front of a multi-millisecond launch.**

### 4.2 Ensembles vs slots

Two shapes, and they are not interchangeable:

- **Slot** (`cont_`, `cor_`, `pgf_`) — exactly one occupant. `make_*` returns one.
- **Ensemble** (`vmix_`, `hvisc_`) — a set contributing into a shared accumulator with
  one assembly gate. But **a flat `std::vector<IVmix>` of `contribute()` calls is
  wrong**, for three independently fatal reasons, all established against MOM6 and
  rakali. The corrected design is below.

#### Why the flat vector fails

1. **It forbids fusion, and forbids it exactly where fusion is measured to pay.**
   PP81, background/Bryan-Lewis, double diffusion, convective adjustment, the
   mixed-layer `1/z²` term, the heat/salt split, and the floors/ceilings are **all
   per-interface maps over the same `(i,j,k)`**. As separate virtual `contribute()`
   calls that is **6+ round trips of `kv`/`kt`/`ks` through DRAM**; fused it is one.
   This is the identical case to the measured continuity 1.36× and ALE-remap 1.33×
   wins (REDESIGN §1.5) — and rakali currently pays the unfused cost. Our own rule
   ("compile-time inside an operator, runtime at its boundary") demanded fusion here;
   the flat vector applied it at the wrong granularity.
2. **No cadence axis.** rakali runs each scheme's `*_compute` at *thermo cadence,
   stage 1 only*, and only the cheap `*_merge` every stage. A `contribute()`-only
   interface silently makes the 2–4× more expensive schedule the default — and the
   expensive schemes here are 18%+ of a stage.
3. **Contributors are not independent.** Convective adjustment reads the boundary-layer
   depth that KPP/EPBL wrote. A flat unordered list plus "check it in `validate()`"
   cannot express "B consumes A's output".

#### The corrected design

```cpp
struct VmixContext {                     // shared, demand-driven precompute
    View<Real,3> n2, drho_int, shear2, dz, t_filled, s_filled;
    View<Real,2> u_star, b_0, bld;       // bld: written by the BL scheme, read later
    View<Real,3> tke_to_kd, max_tke;     // only if some contributor asks
};

enum class VmixPhase { Precompute, Interior, BoundaryLayer, SurfaceFlux,
                       PostBoundary, Split };

struct IVmixColumn {                     // genuinely separate launches
    virtual ~IVmixColumn() = default;
    virtual void refresh(const LayeredState&, VmixContext&, const Mesh&, const Params&) = 0;
    virtual void merge(VmixField&) = 0;              // cheap, EVERY stage
    virtual VmixPhase   phase()   const = 0;
    virtual Cadence     cadence() const = 0;         // refresh() gating
    virtual ContextMask needs()   const = 0;         // union drives the precompute
    virtual int         halo_width() const { return 0; }
};

class VmixStack {
    VmixContext                              ctx_;
    PointwiseVmix<...>                       pointwise_;   // ONE fused launch, compile-time
    std::vector<std::unique_ptr<IVmixColumn>> column_;      // KPP, EPBL, kappa-shear, tidal
};
```

Four things the flat version lacked, each earning its place:

- **`refresh()` / `merge()` + `cadence()`.** Retrofitting cadence later touches every
  operator.
- **`needs()` → `ContextMask`.** Same idiom as `halo_width()`: the framework takes the
  union and computes only what is asked for. MOM6 gates its `TKE_to_Kd` precompute
  exactly this way. Without it you either always pay, or you forget one.
- **A tendency output channel.** The KPP nonlocal γ is a **tracer tendency applied
  outside the tridiagonal**, not a diffusivity. An interface whose only outputs are
  `(kv, kt, ks)` structurally cannot express it.
- **Merge is per (operator, TARGET), not a per-operator tag.** MOM6's non-additive
  EPBL is `kv ← max(kv, Pr·Kd_ePBL)` **and** `kt ← kt + max(Kd_ePBL − Kd_shear, 0)`
  in the same operator — two different modes on two targets, one of them referencing a
  field that is not the accumulator. No `MergeMode` enum survives that.

#### The gate, and two constraints on it

```cpp
for (auto& c : column_) if (c->phase() == VmixPhase::Interior)      c->merge(f);
bl_->merge(f);                                    // writes bld + the γ channel
apply_surface_fluxes(f);                          // EPBL consumes cTKE from here
for (auto& c : column_) if (c->phase() == VmixPhase::PostBoundary)  c->merge(f);
pointwise_.run(ctx_, f);                          // ONE fused launch
split_kd_heat_salt(f, ddiff_);                    // TERMINAL — not a contributor
vmix_assemble(f, floors_, m);                     // the ONE gate
```

- **The split is a terminal derivation, not a contributor.** Anything placed after it
  silently never reaches `ks`. That is a constraint between the contributor *set* and a
  derivation *step*, which no pairwise `validate()` check can see — so it is structural.
- **A single gate makes a runaway contributor unattributable.** MOM6 clips per
  contributor at ≥5 sites; rakali's single gate is better, but add a **debug-only
  per-contributor bound assertion** so a bad closure is still diagnosable.
- **`SurfaceFlux` is a phase, not an afterthought.** MOM6 runs KPP → surface fluxes →
  EPBL, because EPBL consumes the `cTKE`/`dSV_dT`/`dSV_dS` that flux application
  produces. rakali papers over this and its EPBL is energetically weaker for it.

> **Vertical index convention — decide before writing a line.** MOM6 has `k=1` at the
> **surface**; rakali has `k=1` at the **bed**. Every formula in every scheme above
> flips. Pick one, put it in `CONTRACT_MEMORY.md`, and make the analytical tests assert
> it (rakali uses three: dense current at the bed, buoyant plume at the surface,
> positive surface heat flux warming the top layer).

Ordering constraints that remain genuinely *configurational* — a biharmonic backstop is
mandatory under MEKE backscatter; KPP and EPBL are mutually exclusive — stay in
`validate()`, which reports them naming both offenders.

#### The generalisation: the ensemble belongs to the COEFFICIENT, not the operator

Vertical mixing forced the split above. The lateral closures make the underlying rule
explicit, and it is the opposite of where the first draft put the abstraction:

**GM, Redi, MLE and horizontal viscosity are *slots* — exactly one occupant each. What
is ensemble-shaped is the diffusivity each of them consumes.** The `Kh` assembly chain
is a genuine additive ensemble with one gate:

```
Kh  = background                          (or 2D-from-file — an EXCLUSIVE alternative)
   += Visbeck  (KHTH_Slope_Cff · L² · SN)                       ADDITIVE
   += MEKE     (two exclusive forms)                            ADDITIVE
   *= resolution function                                       MULTIPLICATIVE
   *= depth function                                            MULTIPLICATIVE
    = clamp(Kh_min, Kh_max)                                     the gate
    = min(Kh, Kh_CFL)                                           ONE CFL gate
```

**Horizontal viscosity is a third shape again, and forcing it into the ensemble would
silently change the physics.** Its default composition is `max()`, not `+`; two options
(`RE_AH`, `USE_LEITHY`) are hard overrides that discard everything above them; and
**the two CFL clamps are coupled** — the Laplacian consumes part of the budget and emits
`visc_bound_rem`, and the biharmonic is clamped against only what is left. So:

```cpp
struct HviscStack {
    std::unique_ptr<IHarmonicClosure>   harmonic_;    // ONE: none|smag|leith|qg_leith
    std::unique_ptr<IBiharmonicClosure> biharmonic_;  // ONE: none|const|smag_ah|leith_bi
    std::vector<std::unique_ptr<ICoeffModifier>> mods_;  // the real ensemble:
                                                        // MEKE Ku/Au, anisotropy, resolution
    ViscAssembly assemble_;   // TWO-STAGE COUPLED clamp, emits visc_bound_rem
};
```

Note MEKE's `Ku` is added **after** the floor and **may be negative** (that is
backscatter), which is precisely why the biharmonic backstop check has to be a
predicate on the *coefficient*, not on whether a biharmonic scheme was selected — a
flow-aware closure chosen with a zero coefficient passes the naive check and provides
no backstop at all.

---

## 5. The stepper is the template parameter

Everything is **a combination rule over a forward operator Φ**. Φ differs; the
combination is the policy.

```cpp
// The Shu-Osher coefficient table IS the scheme's published definition, and it makes
// the SSP property visible: every row is a convex combination (a, b >= 0, a+b == 1).
template <int Stages> struct SspBlend;
template <> struct SspBlend<2> { static constexpr Real ab[1][2] = {{0.5, 0.5}}; };
template <> struct SspBlend<3> { static constexpr Real ab[2][2] = {{0.75, 0.25},
                                                                   {1.0/3, 2.0/3}}; };

template <int Stages>
struct SspStep {
    static constexpr int n_registers = 1;                   // s0
    template <class State, class Phi>
    static void advance(State s, std::span<State> reg, Phi phi) {
        axpby(reg[0], Real(1), s, Real(0), s);              // s0 <- s^n
        phi();                                              // stage 1: pure Φ
        for (auto& c : SspBlend<Stages>::ab) { phi(); axpby(s, c[0], reg[0], c[1], s); }
    }
};

using SspRk3 = SspStep<3>;    // imaginary-axis stable to |ωΔt| < 1.73
```

> **Correction — MOM6's outer scheme is NOT `SspStep<2>`.** An earlier draft claimed
> the two were one algorithm and two tables. They are not. MOM6's predictor advances by
> `dt·BE` with **`BE = 0.6` by default**, and the corrector restarts from `u^n` and
> applies the corrector tendency over the **full** `dt`. That is a θ-weighted
> predictor-corrector deliberately damped off the midpoint — a one-parameter family
> with **no SSP property and no convex-combination row**. MOM6's own documentation says
> "instability may occur near 0.5", i.e. near the SSP value.
>
> It survives on the imaginary axis only because the fast modes are carried by the
> forward-backward barotropic substep, not the outer scheme, and because `BE = 0.6`
> adds damping. Given we already root-caused a blow-up to SSP-RK2's imaginary-axis
> instability, this distinction is load-bearing.

```cpp
template <int BE_num, int BE_den>          // BE = 0.6 -> ThetaPredictorCorrector<3,5>
struct ThetaPredictorCorrector {
    static constexpr int n_registers = 1;  // s0
    template <class State, class Phi>
    static void advance(State s, std::span<State> reg, Phi phi) {
        axpby(reg[0], Real(1), s, Real(0), s);          // s0 <- s^n
        phi(Real(BE_num) / Real(BE_den));               // predictor over BE*dt
        axpby(s, Real(1), reg[0], Real(0), s);          // RESTART from s^n
        phi(Real(1));                                    // corrector over the FULL dt
    }
};
```

Two genuinely different stepper families, then — the SSP table family and the θ family
— sharing only the `advance(state, registers, Φ)` shape. The split stage Φ is unchanged
by the choice, which is what makes ADR-9's oracle work (the split must reproduce the
unsplit run).

```cpp
template <class Stepper, class Dyn>
class Core final : public ISolver {
    Dyn dyn_;
    std::array<typename Dyn::State, Stepper::n_registers> reg_;
public:
    void step(Real dt) override {
        Stepper::advance(dyn_.state(), reg_, [&]{ dyn_.forward_stage(dt); });
    }
};
```

**Adams–Bashforth needs one widening.** It combines tendency *history across steps*,
not stages within one, so Φ must be able to report the tendency it used:
`phi(k_out)`. RK ignores the argument; AB stores it and needs a self-starter. Design
the contract with the out-parameter now; implement AB when wanted.

The stepper tag is the one place a **runtime** string picks a **template** argument, so
it is the one place a dispatch table is unavoidable:

```cpp
std::unique_ptr<ISolver> make_solver(const RunSpec& s) {
    validate(s);
    auto dyn = make_split_dynamics(s);        // all the make_* factories above
    if (s.time.stepper == "split_rk2") return std::make_unique<Core<MomSplitRk2, Dyn>>(std::move(dyn));
    if (s.time.stepper == "ssp_rk3")   return std::make_unique<Core<SspRk3,      Dyn>>(std::move(dyn));
    throw ConfigError("time.stepper = '{}'; built: split_rk2, ssp_rk3", s.time.stepper);
}
```

Three steppers × one dynamics type = three instantiations. Not a product.

---

## 6. Where this rule breaks: fusion

Runtime operator dispatch means operators **cannot be inlined into each other**. That
conflicts with REDESIGN §4 (Oceananigans fuses the entire momentum RHS into one kernel).
The benchmark data bounds the conflict precisely:

- The two giants — Redi and kappa-shear, **58% of a stage** — are monolithic column
  solvers. Their internals are not a composition of operator policies, so nothing is
  lost by dispatching them at the boundary.
- Fusion paid where **intermediate arrays reached DRAM** (continuity 1.36×,
  ALE remap 1.33×) — and that fusion is *within* one operator, not across the slot
  boundary. It costs nothing here.
- The barotropic subcycle is launch-bound, but its composition never varies: it is
  always continuity + Coriolis + PGF on the 2D state. **Fuse it once, as a concrete
  class.** No policy dispatch required.

So: **operators dispatch at runtime; fusion happens inside an operator, or inside a
purpose-built fused component with a fixed composition.** If a measurement ever demands
fusing *across* two runtime-selected slots, that specific pair gets one registered
monomorphic fast path — a handful, chosen by a profile, not a cartesian product.

---

## 7. What a run looks like end to end

```python
import turbochook as tc

mesh = tc.SphericalMesh(512, 512, lon=(-193.75, -171.25), lat=(53.6, 64.9),
                        west="periodic", east="periodic", south="wall", north="wall")

sim = tc.Simulation(
    mesh,
    stepper   = "split_rk2",          # MOM6 outer scheme over the split stage
    dt        = 1200.0, n_inner = 0,  # 0 -> CFL-derived, latched at setup
    pgf       = tc.Pgf("fv_wright", rho0=1035.0),
    coriolis  = tc.Coriolis("sadourny", pv_scheme="weno5"),   # needs nghost >= 3
    tracer_adv= tc.TracerAdvection("weno", order=5),
    vmix      = [tc.Pp81(), tc.Kpp(), tc.TidalMixing(gamma=0.3)],   # ENSEMBLE
    hvisc     = [tc.Smagorinsky(c=0.15), tc.Biharmonic(nu4=1e10)],  # ENSEMBLE
)

sim.h[0][:], sim.h[1][:] = h1, h2         # numpy, zero-copy into the arena
sim.diagnostics = ["mass", "KE", "speed"]
sim.output("state.nc", every=tc.days(1))
sim.run(days=580)
```

What that produces, exactly once, at `make_solver`:

| slot | tag | instantiated |
|---|---|---|
| stepper | `split_rk2` | `Core<SspStep<2>, SplitDynamics>` |
| pgf | `fv_wright` | `Op<FvWrightPgf<WrightEos>>` |
| coriolis | `sadourny` + `weno5` | `Op<SadournyPv<Weno5>>` → declares `halo_width = 3` |
| tracer_adv | `weno` order 5 | `Op<TracerFlux<Weno5>>` → `halo_width = 3` |
| vmix | 3 tags | 3 objects in one vector, `contribute()` + one `vmix_assemble` |
| hvisc | 2 tags | 2 objects in one vector |

`validate()` then takes `max(halo_width)` over every operator and sizes the halo — so
`ng = 3` is *derived*, and because arrays carry Fortran bounds
(`CONTRACT_MEMORY.md` §1.1) nothing else in the code has to know.

---

## 8. MPI on day 0 — the seams, not the implementation

Nothing here builds an exchange. Every item is cheap to add now and expensive or
pervasive to retrofit, which is the only test that matters for a day-0 decision.

> The single most useful artefact here is `MOM6/config_src/infra/FMS2/` — **6,631 lines,
> 12 modules, 233 symbols. That is the entire list of what a serious ocean model needs
> from its infrastructure layer**, and it is the shopping list. Note what MOM6 never
> touches at all: the exchange grid (0 call sites, 5,797 lines), `field_manager`,
> `tracer_manager`, `mosaic`, `interpolator`.

### 8.1 Build now

**Real ghost cells — and delete the index-wrapping BC path. This is rung 0 and it is
not an MPI item.** Today periodic-x is done by *index wrapping in the mesh*
(`shift_x`/`bc_at`). MPI cannot work that way: a rank's west-neighbour data has to
physically be in memory. `CONTRACT_MEMORY.md` §1.1 promises ghosts-via-bounds, but the
wrapping path must be *removed*, not merely supplemented — otherwise "operators never
learn MPI exists" is false at every boundary-touching operator. Periodic-x becomes a
halo *fill* at a 1×1 decomposition.

**A `Region` parameter on `do_concurrent` — an arbitrary inclusive index box.**

```cpp
template <class F>
void do_concurrent(KernelTag, Region r, F f, LaunchOpts = {});   // Region = {is,ie,js,je}
```

Not "the interior inset by n". MOM6's largest exchange win runs the barotropic subcycle
on a **wider-than-interior** domain and marches *inward*, eliding exchanges until the
valid region is consumed — `ceil(n_inner/stencil)` exchanges instead of `n_inner`, in
the hottest loop in the model. A `Region` typed as an inset cannot express that.

Be blunt about what this does and doesn't buy: `std::for_each(par_unseq)` exposes **no
stream handle**, so genuine comm/compute *overlap* on the stdpar backend needs the
native escape hatch or a host thread. `Region` buys exchange **elision** for free;
overlap is a separate, later thing.

**Grouped exchange as the primitive, not per-field.** MOM6 has 155 grouped-pass sites
against 299 single-field ones, and its entire barotropic hot loop is grouped —
`eta`, `ubt/vbt`, `ubt_int/vbt_int`, `uhbt_int/vhbt_int` in **one** message set. At
global 1 km the per-rank subdomain is small and **message count dominates**. So:

```cpp
HaloGroup g;                                  // built once at setup
g.add(eta, Loc::Center, Parity::Scalar, /*width=*/1);
g.add(u, v, Loc::XFace, Loc::YFace, Parity::Vector, 1);   // vector PAIR, together
exchange(g);                                  // one message set
```

**Per-exchange width**, so a 1-ring scheme doesn't move 4 rings — 4× the bytes on every
call otherwise. `halo_width()` per operator sizes the *allocation*; the exchange takes
its own argument.

**Vector components exchange as a PAIR.** Not per-field. Across a 180° fold the pair is
sign-flipped, but any 90° rotation *swaps* the components — which a per-field tag
cannot express at all.

> **Correction to an earlier draft: `Parity` should NOT be a template parameter of
> `Array`.** `Loc` earns its place because it changes the **extents** (a face field has
> one more face than cells in its normal direction — that is exactly MOM6's symmetric
> memory), and it is genuinely consumed by output, restart layout and diagnostic
> staggering. `Parity` changes no layout, no extent, and nothing whatsoever except a
> sign in one pack loop of one function. Putting it in the type doubles the alias
> surface on the most-used type in the codebase and forces it into every signature that
> handles a field generically — `mirror`, `copy`, the numpy handoff, diagnostics,
> restart — **none of which care**. Demote it to halo-group registration, where it is
> still declared once and checked once, at a fraction of the blast radius. And if 90°
> rotations ever arrive, per-field parity is *wrong*, so the template parameter would
> have to be removed rather than extended.

**A `Decomp` on the mesh, even at 1×1** — local/global extents, this rank's global
offset, and `neighbour(edge) -> optional<rank>`. **Optional, not a dense `px×py`
index**: dropping all-land subdomains removes ~25–30% of ranks on a real global grid,
and that only works if "no neighbour" is representable. One line now.

**Root-only logging and collective-safe fatal errors.** At scale every `log()` prints
once per rank, and a `throw` on one rank deadlocks the rest at the next collective.
`fatal()` must become `MPI_Abort`; the logger must default to root-only. Pervasive to
retrofit (every log site, every throw), trivial now — and it matters at the Python
boundary too, where an exception has to become an abort rather than a return code.

**`comm::init()` is ONE call.** An earlier draft showed three ordered public calls and
then said "bake it in so it cannot be got wrong" — which is exposing a way to be wrong.
`comm::init()` reads the launcher's local-rank environment, pins
`CUDA_VISIBLE_DEVICES` / `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, calls `MPI_Init`, then
`device::initialize()`. No caller has a legitimate reason to interleave those.
- **CUDA:** visibility must be pinned **before `MPI_Init`** — UCX creates a primary
  context there, so with all GPUs visible every rank *also* lands one on device 0, while
  the binding diagnostic still looks correct.
- **Intel:** one rank per **tile**, never a COMPOSITE root device — measured **6.3×**.

**An `Efp` accumulator type, not just a `Sum::Reproducible` flag.** The reproducible sum
is a 6×`int64` fixed-point representation (Hallberg & Adcroft 2014) — integer addition
is exact and associative, so any order, any blocking and any rank count give the same
bits. But MOM6 keeps running surface-flux integrals **as `Efp` across the whole run**
and forms budget residuals by exact subtraction. If the reduction only returns a
`double` you get a reproducible *instantaneous* total but not a reproducible 580-day
*budget residual* — and that residual is the actual conservation oracle. Expose `Efp`
with `+`, `-`, `to_real`, and a **batched** `allreduce(span<Efp>)` (MOM6 batches five
budget terms into one collective). ~150 lines, ported verbatim from
`MOM6/src/framework/MOM_coms.F90` — and reimplemented, not bridged, because it must run
on-device.

**Restart checksums, from the first restart.** Sum the `int64` bit patterns of the
interior values, `Allreduce`, store as a variable attribute, verify on read. **Ten
lines, and order-independent because integer addition is exact** — so a restart written
on 4 ranks is verifiable on 6. There is no reason not to have it on day 0.

**`GlobalIndex` for the global HORIZONTAL index.** The earlier justification ("a 1 km ×
100-layer ocean exceeds 2³¹ cells") was true but irrelevant — you never form a global
3-D index and must never let that concept exist. Where the wider type actually earns its
keep: a 1 km tripolar horizontal grid is ~1.2e9 points, under 2³¹ by only 2×, and 500 m
is over. Confine it to `Decomp` and I/O.

### 8.2 Defer

Non-blocking exchange and true comm/compute overlap; GPU-aware vs host-staged buffer
selection; the I/O server and parallel HDF5 (per-rank files plus an offline merge is
proven); land-block elimination (keep the `optional` seam, don't build it); nesting and
multi-tile mosaics; `data_override` (3 sites in all of MOM6).

**The tripolar fold is only half-deferrable.** The exchange kernel can wait, but it
*constrains `Decomp`* — global `nx` must be even and the x-partition mirror-symmetric
about the midpoint — and it needs a **separate seam-row fixup pass** (the pole DOF is
duplicated; B-grid vectors get the two pole points zeroed and the halo columns mirrored).
That does not fall out of rotation-plus-sign-flip. Design `Decomp` for it now or redesign
`Decomp` later.

### 8.3 Do not reimplement

- **The exchange grid.** Zero call sites in MOM6; standalone SIS2↔MOM6 flux exchange is
  a direct array copy on a shared grid. 5,797 lines nobody should write.
- **`mpp` itself — and do not bridge to it either.** ~30k lines of overlap machinery for
  mosaics, nests, unstructured domains and adjoints. A single-tile tripolar C-grid needs
  the 8-neighbour case plus one fold: a few hundred lines. Bridging would drag in FMS's
  Fortran build, its namelist layer, and its host-resident buffer model.
- **Calendar arithmetic.** Steal the *design* — `{days, seconds, ticks}`, **integer,
  never float** — and none of the 3,000 lines; `std::chrono` gives exact Gregorian, and
  noleap/360-day are ~40 lines each.
- **Runtime horizontal interpolation** (~7k lines). Require pre-interpolated inputs and
  ship a Python tool; it is an offline problem.
- **`field_manager` / `diag_table` / YAML** (~10k lines). The config is Python.

---

## 9. Invariants

1. **Compile-time inside an operator; runtime at its boundary.** N + M + K, never N×M×K.
2. **`make_*` is the only place a tag becomes a type**, and its error names the value,
   the constraint, and what is actually built.
3. **Config chooses occupants, never the order.** Operator sequence is correctness.
4. **Slots hold one. Ensembles split in two**: the per-cell contributors fuse into ONE
   compile-time-composed launch; only the genuinely separate column solvers get runtime
   dispatch. They fold into a shared accumulator with one assembly gate **plus a
   terminal derivation step that is not a contributor** (the heat/salt split), and a
   debug-only per-contributor bound assertion so the single gate cannot hide a runaway
   closure.
5. **The stepper is a combination rule over Φ.** MOM6 split RK2 and SSP-RK3 are one
   algorithm and two coefficient tables.
6. **Every operator declares `halo_width()`; the framework takes the max.**
7. **Fusion lives inside an operator or inside a fixed-composition component** — never
   across a runtime slot boundary without a profile demanding it.
8. **`RunSpec` is the provenance record.** Dump it resolved, with a git SHA, at startup.
9. **MPI never appears outside `comm/`**, and no operator learns that it exists — the
   decomposition changes the work region, not the kernel.
