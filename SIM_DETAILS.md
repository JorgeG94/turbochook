# bc_inst — Two-Layer Baroclinic Instability: Experiment Spec

A self-contained, **language-agnostic** description of the baroclinic-instability run so the
identical experiment can be set up in another dycore (e.g. `rakali_dc`) for a head-to-head
physics *and* performance comparison. Everything here is math + numbers; our specific numerical
choices are listed separately (§7) as reference — match them or substitute your own and note the
difference.

Origin: emulates `two_layer_sw`'s `baroclinic_highlat` case; the physical setup is identical,
the IC amplitudes differ (§5).

---

## 1. Model

Two-layer, reduced-gravity, hydrostatic shallow water on a rotating sphere (isopycnal / stacked
shallow water). Prognostic per layer k∈{1,2}: thickness `h_k` and horizontal velocity `(u_k,v_k)`.
Upper layer 1 on top of lower layer 2 (ρ₂ > ρ₁).

Governing equations (vector-invariant / flux form; use whatever your core uses):

```
∂h_k/∂t + ∇·(h_k u_k) = 0
∂u_k/∂t = -(ζ_k + f) k×u_k - ∇(½|u_k|²) - ∇p_k / ρ₁ + D
```

Layer pressures (÷ρ₁), reduced-gravity form:

```
p₁/ρ₁ = g (h₁ + h₂)                    (upper feels the free surface)
p₂/ρ₁ = g h₁ + (ρ₂/ρ₁) g h₂            (lower feels surface + interface)
      = g (h₁ + h₂) + g' h₂,   g' = g (ρ₂ - ρ₁)/ρ₁
```

`D` = lateral dissipation (§7).

---

## 2. Domain & grid (Arakawa C-grid, spherical lon/lat)

| | value |
|---|---|
| longitude | `lon ∈ [-193.75°, -171.25°]`  (span 22.5°) |
| latitude  | `lat ∈ [ 53.625°,  64.875°]`  (span 11.25°) |
| central latitude | `φ₀ = 59.25°` |
| Earth radius R | `6.371 × 10⁶ m` |
| Ω | `7.2921 × 10⁻⁵ rad s⁻¹` |
| grid | `N × N`, N = 256 or 512 |

Metrics (analytic): `dy = R·Δφ` (constant), `dx = R·cosφ·Δλ` (shrinks poleward).

| N | dy | dx@φ₀ | dx@north edge (min) |
|---|---|---|---|
| 256 | 4886 m | 4997 m | 4150 m |
| 512 | 2443 m | 2498 m | 2075 m |

Coriolis: **full spherical** `f = 2Ω sinφ` (evaluated at the location where each term lives —
corners for PV). Not a β-plane. `f₀ = f(φ₀) = 1.2534 × 10⁻⁴ s⁻¹`.

---

## 3. Boundary conditions

- **x (zonal): PERIODIC** — re-entrant channel (essential; the zonal jet can't live in a closed box).
- **y (meridional): WALL** — free-slip, no normal flow.

---

## 4. Layer & fluid parameters

| | value |
|---|---|
| H₁ (upper rest thickness) | 500 m |
| H₂ (lower rest thickness) | 1500 m |
| ρ₁ | 1025 kg m⁻³ |
| ρ₂ | 1027 kg m⁻³ |
| g | 9.81 m s⁻² |
| g' = g(ρ₂−ρ₁)/ρ₁ | **0.019141 m s⁻²**  (note: ÷ρ₁, not ρ₂) |

Derived wave speeds / scales:

| | value |
|---|---|
| external (barotropic) gravity wave `c_ext = √(g(H₁+H₂))` | 140.07 m s⁻¹ |
| internal (baroclinic) wave `c_int = √(g'·H₁H₂/(H₁+H₂))` | 2.679 m s⁻¹ |
| 1st baroclinic deformation radius `R_d = c_int/f₀` | **21.4 km** |

`R_d` resolution: ~5 cells at 256², ~10 cells at 512² — the key resolution number (see §6).

---

## 5. Initial condition — balanced tanh jet + front-localized perturbation

Meridional coordinate about the jet centre: `y = R·(φ − φ₀)` (metres). Zonal coordinate for the
perturbation: `xm = R·cosφ₀·(λ − λ₁)` (metres), λ₁ = lon1.

**Interface displacement** (a tanh front), plus a front-localized meander seed:

```
ξ(x,y) = Δξ·tanh(y/L)  +  A_pert·sech²(y/L)·cos(k_x · xm)
η(y)   = -(g'/g)·ξ                        (free-surface signature)
h₁ = H₁ + (η − ξ)
h₂ = H₂ +  ξ
```

**Geostrophically-balanced zonal jet** (upper layer only; lower layer at rest):

```
u₁(y) = (g'/f)·∂ξ_base/∂y = (g'/f)·(Δξ/L)·sech²(y/L)
u₂ = 0 ,   v₁ = v₂ = 0
```
(f is the full variable Coriolis at the u-point latitude. Note: the geostrophic velocity uses the
*base* tanh only; the perturbation is an unbalanced interface displacement that the instability grows.)

**Perturbation amplitude:** `A_pert = 0.2·Δξ` (= `noise_amp·Δξ·4`, noise_amp = 0.05).

**Perturbation wavenumber** (deformation-radius scale, deterministic — not random):

```
k_mag = 1/(2 R_d)
k_y   = k_mag/√2                                  (= 1.654e-5 m⁻¹, unused by the sech² seed but defines the scale)
n_x   = round( (k_mag/√2)·L_x/(2π) )              L_x = R·cosφ₀·(lon2−lon1) = 1279.2 km  →  n_x = 3
k_x   = 2π·n_x / L_x                              = 1.4735e-5 m⁻¹  (3 wavelengths across the domain)
```

### Two configurations

| param | **Julia-faithful** (weak) | **Tuned** (what we run) |
|---|---|---|
| jet half-width `L` | 100 km | **40 km** |
| interface amplitude `Δξ` | 100 m | **200 m** |
| `A_pert = 0.2·Δξ` | 20 m | 40 m |
| jet peak `u₁ = (g'/f₀)(Δξ/L)` | 0.153 m/s | **0.764 m/s** |
| Rossby number `Ro = u₁/(f₀L)` | 0.012 | **0.152** |

**Why two.** The Julia-faithful jet is the exact reference, but in *our* PPM+Sadourny scheme at
256² (R_d ~4–5 cells) it's *marginally* resolved and the weak jet **decays instead of rolling up**;
we use the tuned (stronger, narrower) jet to get vigorous eddies. At 512² the faithful jet does
roll up. **This is the single most interesting thing to check in rakali:** does the faithful
`L=100km, U=0.15` jet go unstable at 256²/512² in *your* numerics? If so, our decay is a
scheme-diffusion artifact at marginal R_d, not physics.

---

## 6. What to compare (the science)

Run to **50 days** for the rollup, or **90–240 days** for the equilibrated eddy field (the Julia
reference runs 90–240). Compare:

- **Does it go unstable, and when?** Track a base-state-zero quantity: `max|v₁|` (v≡0 initially),
  or eddy kinetic energy. Growth is exponential in the linear phase; e-folding ~a few days for the
  tuned jet.
- **Most-unstable wavelength** ≈ `2π√2 R_d ≈ 190 km` (~6 waves across the 1279 km domain).
- **Conservation**: total mass Σ h_k·area (should be ~machine-eps over the run), total energy.
- **Fields to dump** (we write CF-1.11 NetCDF: `h, u, v, ζ, ū(y)` per layer, per time) — diff eddy
  scale, growth rate, energy vs the Julia `diag_bc_inst.nc`.
- **Resolution sensitivity**: 256² vs 512² (the R_d-resolution question above).

---

## 7. Our numerical choices (reference — match or note the difference)

These are *our* methods; rakali will use its own. Listed so differences are explicit.

| aspect | ours (TurboChook) |
|---|---|
| horizontal grid | Arakawa C-grid; η/h at centres, u on x-faces, v on y-faces, ζ at corners |
| continuity (layers) | PPM flux-form (van-Leer + Colella-Woodward + monotonic limiter) |
| continuity (barotropic subcycle) | 1st-order upwind (Pcm) — the fast mode needs no PPM |
| Coriolis + advection | Sadourny (1975) enstrophy-conserving, vector-invariant PV flux |
| pressure gradient | reduced-gravity two-layer (§1) |
| time integration | **split-explicit**: SSP-RK3 outer + Forward-Backward Euler barotropic subcycle. *Also runnable unsplit* (SSP-RK3, dt at the external-wave CFL) |
| dissipation `D` | Shapiro 5-point filter on h only, coefficient **ε = 0.006** per step (`h ← (1−ε)h + ¼ε·Σ_4 neighbours`). No explicit viscosity. |

Julia reference (`two_layer_sw`) uses instead: WENO-Z5 continuity, centred momentum, AB3-AM3
time stepping, **Leith** viscosity (ν = Δ³|∇ζ|) + Shapiro (ε=0.005), unsplit dt=6 s.

### Timestep (ours)
- **Unsplit** (SSP-RK3): `dt` limited by the external wave. `dt ≲ 0.15·dx_min/c_ext` → ~4.4 s at
  512². (RK3 imaginary-axis-stable for the gravity wave to |ωΔt|<√3.)
- **Split**: outer `dt` limited by the *internal* wave via the RK3 outer:
  `dt_max ≈ √3·dx_min/(c_int·π) ≈ 0.55·dx_min/c_int` → **~854 s @ 256², ~427 s @ 512²** (verified
  empirically). Barotropic substeps `M = ⌈dt/dt_bt⌉` with the FB CFL `dt_bt < dx/(c_ext·√2)` →
  e.g. 512²: `dt=400 s, M=64` (dt_bt=6.25 s).

### Reference performance (for the speed comparison)
Split, 512², `dt=400 s, M=64`, no I/O, 2-day timing:

| machine | s / sim-day | 50-day run |
|---|---|---|
| Tesla V100 | 15.4 | ~13 min |
| GH200 (Hopper) | 9.2 | ~7.7 min |

Our split is **launch-bound** (~18 tiny GPU kernels per barotropic substep × M × 3 RK3 stages);
that's why the GH200 is only ~1.7× the V100 rather than ~3–4×. If rakali's `do concurrent` loops
are fatter (fewer, larger kernels), it should be less launch-bound — an interesting axis to compare.

---

## 8. One-line reproduction (TurboChook side)

```
# tuned, 512², dt=400 s, M=64, 50 days
./demo_baroclinic_split  <out>  512  50  <steps/frame>  0.006  400  40  200  64
#                          grid days              eps    dt   L   Δξ   M
# Julia-faithful: ... 100000 0.006 0 100 100   (dt=0 → auto CFL, L=100, Δξ=100; needs 512²+ to roll up)
```
Source of truth: `examples/programs/demo_baroclinic_split.cpp` (IC + params), `src/physics/split_two_layer.hpp`
(split stepper), `src/physics/two_layer_pgf.hpp` (reduced-gravity PGF).
