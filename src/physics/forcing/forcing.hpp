#pragma once
// =============================================================================
// physics/forcing/forcing.hpp — surface & body forcing.
//
// The external drivers added as per-step SOURCES to the tendencies:
//   wind stress   → momentum (τ/ρ0h into u,v);
//   buoyancy/heat/freshwater flux → tracers (S,T);
//   tidal potential (astronomical) → a body force on the barotropic mode;
//   river inflow  → mass + tracer sources at cells.
// Idealized ANALYTIC forms first (a double-gyre wind-stress curl is the canonical
// spin-up test); time-dependent DATA forcing comes via a future data-override/IO path.
//
// TODO(Later): Forcing concept + WindStress (analytic curl) → surface flux → tides.
// rakali north-star: src/parameterizations/vertical/structured/rki_ocean_surface_stress.F90,
//                    rki_ocean_surface_flux.F90; src/core/ocean/forcing/rki_ocean_tides.F90,
//                    rki_ocean_tide_astro.F90, rki_ocean_river.F90
// =============================================================================

#include "core/types.hpp"
#include "physics/state/baro_state.hpp"

namespace tc {

// TODO(Later): struct WindStress { Real tau0; // adds -curl-forced τ/ρ0h to k.u/k.v
//   template <Mesh M> void compute(BaroState s, BaroState k, const M& mesh, Params p) const; };
//   (double-gyre: τ_x = -τ0 cos(πy/L) → the classic gyre spin-up.)

} // namespace tc
