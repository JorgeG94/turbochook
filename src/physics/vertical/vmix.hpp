#pragma once
// =============================================================================
// physics/vertical/vmix.hpp — vertical mixing / boundary-layer parameterization.
//
// Sets the vertical viscosity/diffusivity that closes the momentum + tracer columns
// (an implicit vertical diffusion solve). Policy axis:
//   Pp81       — Pacanowski–Philander Richardson-number interior mixing.
//   Kpp        — K-profile boundary-layer overlay (the classic surface BL).
//   Epbl       — energetic PBL (MOM6-style).
//   plus kappa-shear (interior shear) and a bottom-drag term.
//
// A COLUMN kernel: serial vertical recurrence in the implicit solve → occupancy-bound
// (STATUS #6). Meaningless in a pure 2-layer isopycnal core; it lands with z*/ALE
// columns (ROADMAP M5: PP81 interior → KPP overlay).
//
// TODO(M5): Vmix concept + Pp81 (interior) then Kpp (overlay); bottom drag.
// rakali north-star: src/parameterizations/vertical/structured/rki_ocean_vmix.F90,
//                    rki_ocean_epbl.F90, rki_ml_kpp.F90, rki_ocean_kappa_shear.F90,
//                    rki_ocean_bottom_drag.F90, rki_ocean_wave_speed.F90
// =============================================================================

#include <concepts>
#include "lib/arena.hpp"
#include "physics/state/layered_state.hpp"

namespace tc {

// TODO(M5): template <int NL> class KppVmix {
//   Field2 kappa_m_, kappa_h_;  // per-interface diffusivities (column workspace)
//   void init(Arena&, mesh);
//   void compute(LayeredState<NL> s, LayeredState<NL> k, mesh, Params p) const; // implicit vdiff
// };

} // namespace tc
