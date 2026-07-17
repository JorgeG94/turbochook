# TurboChook — Rakali Ocean → turbochook Porting Map

Where each ocean algorithm lives in the Rakali Fortran core, so a turbochook operator can be
ported from the right place. **Scope: the `sim_type='ocean'` C-grid dyn-core only** — the
coastal / A-grid / HLL / KNP path and all `*_unstr` modules are out of scope.

All paths are relative to `../rakali_dc` (i.e. `/home/jorge/nci/cdx/rakali_dc`). **Port the
algorithm, not the source** — read the Fortran to understand the math, then write idiomatic
C++ per DESIGN. Read `../rakali_dc/src/core/ocean/README.md` first (the slot map + design
contract). Milestone tags map to `ROADMAP.md`.

> Vertical convention (load-bearing): **k=1 = bed, k=nz = surface**. `u_face_x(i,j,k)` sits
> between cells (i,j) and (i+1,j), i=1..nx+1. C-grid staggering: η/T/S at centres (T),
> u at E/W faces (Cu), v at N/S faces (Cv), PV/vorticity at corners (Cq).

## Top-level driver — split-explicit two-stage RK2  → turbochook `Integrator` + `baro_rhs`

- `src/core/ocean/dynamics/split_rk2/rki_ocean_dyn.F90` — module `rki_ocean_dyn`, type `ocean_dyn_t`.
- Procedures: **`ocean_dyn_step_split`** (the outer step to port), **`run_stage_split`** (one FE
  stage: slow computes → applies → BT substeps → correction), `rk2_average`, `save_state`,
  `apply_velocity_truncation`.
- Algorithm: SSP-RK2 (Heun) outer step; each FE stage: (1) derive (η, u_bt, v_bt) from layers,
  (2) run slow kernels into scratch, (3) sum per-face slow tendency `F_slow_u/v`,
  (4) thickness-weighted depth-mean → `F_bt_u/v`, (5) baroclinic applies → u*,v*,h*,
  (6) `n_inner` barotropic fast substeps with `F_bt` constant, (7) BT correction
  `(⟨u_bt⟩ − u_bt^n − dt·F_bt)` back into layers; then RK2-average; optional ALE remap after.
- Real driver call site: `src/driver/rki_driver.F90:1740`. **Ignore** `rki_ocean_solver.F90` (stub).

## 1. Continuity-PPM  → `Continuity` policy  · **[M2]**

- `src/core/ocean/kernels/structured/continuity_ppm/rki_continuity.F90` — `rki_continuity`, `continuity_t`.
- **M2 (2D barotropic):** `continuity_compute_fluxes_barotropic` / `continuity_apply_fluxes_barotropic`.
- Multilayer (M3): `continuity_compute_fluxes` / `continuity_apply_fluxes` / `continuity_step_split`.
- PPM helpers: `ppm_limited_slope`, `ppm_cell_limiter`, `ppm_limit_pos`, `ppm_mirror_h`, `volcfl_face`.
- Tracer drain (later): `continuity_tracer_step_split` / `continuity_tracer_drain`.
- Algorithm: ∂h/∂t = −∇·(hu) via a PPM flux integral — per-cell limited parabola of h
  (slope + monotonic + positivity limiters), integrate swept volume over u·dt per face, update
  by flux divergence.

## 2. Coriolis — Sadourny PV-conserving  → `Coriolis` policy  · **[M2]**

- `src/core/ocean/kernels/structured/coriolis_adv/rki_coriolis_adv.F90` — `rki_coriolis_adv`, `coriolis_adv_t`.
- **M2 (2D):** `coriolis_adv_compute_tendencies_barotropic` / `_apply_tendencies_barotropic`.
- Multilayer dispatcher: `coriolis_adv_compute_tendencies` → `_sadourny` (enstrophy, default),
  `_sadourny_energy` (energy form), `_hk` (Arakawa–Hsu); selector `parse_pv_variant`.
- Algorithm: vector-invariant Coriolis+advection as a PV flux. PV `q=(f+ζ)/h` at corners;
  tendency = cross product of q with mass flux (hu,hv). Enstrophy form conserves potential
  enstrophy; `sadourny_energy` conserves KE; `sadourny_hk` blends q.

## 3. Pressure gradient force  → `PGF` policy  · **[M2; gprime for M3]**

- `src/pressure_force/structured/rki_ocean_pressure_force.F90` — `rki_ocean_pressure_force`, `ocean_pressure_force_t`.
- Column reconstruction helpers: `src/pressure_force/structured/rki_ocean_pgf_reconstruct.F90`
  (`rki_ocean_pgf_reconstruct`: `plm_edges_column`, `ppm_edges_column`, `boole_dpa_intz_layer`).
- Dispatcher: `ocean_pressure_force_compute` / `ocean_pressure_force_apply`; impls
  `compute_gprime_impl`, `compute_fv_mom6_impl`, `compute_fv_mom6_reconstruct_impl`; selector
  `parse_opgf_variant`.
- Variants: `OPGF_VARIANT_MONT` (Montgomery), `_FV_LITE`, `_FV_WRIGHT`, **`_GPRIME`** (reduced-
  gravity 2-layer → M3, `gprime_gfs`/`gprime_gint`), `_FV_MOM6`.
- Algorithm: −(1/ρ₀)∇p at faces. Montgomery = layer Montgomery-potential differences; FV
  variants = finite-volume interface-pressure reconstruction (PLM/PPM edges + Boole quadrature
  of the density anomaly) for the horizontal pressure-work integral per face; gprime recovers
  ∇η from ∇(Σh)−∇b with per-interface g′.

## 4. Split-explicit barotropic loop  → `Integrator` (fast substep) + BaroState coupling  · **[M2]**

- Substeps: `src/core/ocean/kernels/structured/barotropic/rki_barotropic_substep.F90` —
  `rki_barotropic_substep`: `barotropic_substep_linear`, `_nonlinear`, `_nonlinear_interior`.
  Algorithm: forward-backward-Euler fast substeps — advance η from BT mass-flux divergence,
  then BT u/v from −g∇η + constant `F_bt`, accumulating time-mean η, u_bt, v_bt over `n_inner`.
- Coupling: `src/core/ocean/kernels/structured/barotropic/rki_barotropic_coupling.F90` —
  `rki_barotropic_coupling`: `derive_bt_from_layers`, `sum_slow_tendencies_into_F_slow`,
  `face_depth_mean_u/v`, `apply_bt_correction`, `compute_pbce`, `compute_gtot_faces`.
- BT continuity relation: `src/core/ocean/kernels/structured/barotropic/rki_bt_cont_type.F90`
  (`rki_bt_cont_type`: `find_uhbt`, `find_duhbt_du`, `uhbt_to_ubt`).
- `auto_n_inner` CFL: `src/core/ocean/state/rki_ocean_setup.F90` (`rki_ocean_setup`) —
  `bt_auto_n_inner`, `metrics_bt_cfl_length` (2D CFL length = min 1/√(1/dxT²+1/dyT²)).
- Optional wide-halo march-in: `src/core/ocean/dynamics/split_rk2/rki_ocean_bt_wide.F90`.

## 5. C-grid metrics / staggering / grid  → `Mesh` (`CartesianMesh`)  · **[M2]**

- `src/core/ocean/state/rki_ocean_metrics.F90` — `rki_ocean_metrics`, `ocean_metrics_t`.
- Procedures: `metrics_fill_cartesian` (start here), `_spherical`/`_from_supergrid`/`_tripolar`
  (later), `metrics_fill_coriolis`, `metrics_finalize`, `metrics_apply_land_mask`, `adcroft_recip`.
- Setup driver: `src/core/ocean/state/rki_ocean_setup.F90` → `configure_ocean_metrics`,
  `configure_ocean_land_mask`.
- Algorithm: Arakawa C-grid metric arrays — dx/dy + reciprocals (`idxT/idyT`, `idxCu/idyCv`),
  cell areas + reciprocals, `wet_mask`/`wet_u`/`wet_v`; Adcroft reciprocals (1/x→0 at x=0) armor
  land; Coriolis f at corners (β-plane or planetary).

## 6. Ocean state slot map  → `BaroState` (M2) / layered state (M3)  · **[M2/M3]**

- God state: `src/core/ocean/state/rki_ocean_state.F90` — `rki_ocean_state`, `ocean_state_t`
  (`ocean_state_enter_data`/`_exit_data`/`_seed_from_cfg`).
- **Barotropic prognostics (M2):** `src/core/structured/rki_barotropic_cgrid_state.F90` —
  `barotropic_cgrid_state_t` (η/`h`, `u_face_x`, `v_face_y`, `mass_flux_*`, RK saves).
- Multilayer prognostics (M3): `src/core/structured/rki_multilayer_cgrid_state.F90` —
  `multilayer_cgrid_state_t` (`h_layer`, `u/v_face_*_layer`, `hu/hv_face_*_layer`,
  `mass_flux_*_layer`, `rho_layer`, `w_interface`, `tracers(:)` with `idx_salinity`/`_temperature`/`_age`).

## 7. EOS  → `physics/eos.hpp`  · **[M4]**

- `src/equation_of_state/structured/rki_ocean_eos.F90` — `rki_ocean_eos`, `ocean_eos_t`.
- `ocean_eos_compute`, `eos_density_point`, `eos_specvol_derivs`, `eos_wright_pgf_column_sweep_impl`,
  `eos_freezing_point`, `parse_eos_variant`.
- Algorithm: Wright (1997) nonlinear ρ(S,T,p) rational fit (+ specvol derivatives for FV-PGF),
  and a linear ρ = ρ₀ − α(T−T₀) + β(S−S₀) variant.

## 8. Vertical coordinate / ALE remap  → `Vcoord` policy  · **[M4]**

- Ocean vcoord slot: `src/core/ocean/vcoord/rki_ocean_vcoord.F90` — `ocean_vcoord_t`
  (`compute_target_h`, `compute_target_h_rho`, `invert_density_targets`, `parse_ocean_vcoord_type`;
  enums `VCOORD_*`).
- Shared math: `src/ALE/rki_vcoord.F90` (`rki_vcoord`, `vcoord_target_dz_column*`, `zstar_full_build_column`).
- Remap orchestrator: `src/ALE/rki_ocean_remap.F90` (`ocean_apply_ale_remap_step`, `_centres`,
  `_faces`, `ocean_remap_tracer_column`).
- Column kernel: `src/ALE/rki_kernel_remap.F90` (`remap_column`, `remap_column_pcm/plm/ppm/ppm_h4/pqm`).
- Algorithm: compute a target grid per step for the coordinate family, then conservatively
  remap `h_layer`, tracer hTr, and face velocities from Lagrangian → target via PPM.

## 9. Vertical mixing  → `Vmix` policy  · **[M5]**

- PP81 + KPP: `src/parameterizations/vertical/structured/rki_ocean_vmix.F90` — `ocean_vmix_t`
  (`vmix_compute_pp81`, `vmix_apply_kpp_overlay`, `vmix_apply_nonlocal_tendencies`, `vmix_assemble`).
- EPBL: `src/parameterizations/vertical/structured/rki_ocean_epbl.F90` — `ocean_epbl_t`
  (`epbl_compute`, `epbl_merge_into_kv_kt`, `epbl_lf17_*`).
- **Ignore** `rki_ml_kpp.F90` (coastal A-grid KPP — wrong path).
- Later ocean vmix slots (same dir): `rki_ocean_kappa_shear.F90`, `rki_ocean_tidal_mixing.F90`,
  `rki_ocean_vdiff.F90` (backward-Euler tridiagonal + implicit stress/drag),
  `rki_ocean_bottom_drag.F90`, `rki_ocean_surface_stress.F90`.
