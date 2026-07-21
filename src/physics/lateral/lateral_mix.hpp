#pragma once
// =============================================================================
// physics/lateral/lateral_mix.hpp — mesoscale / submesoscale lateral closure.
//
// The eddy PARAMETERIZATIONS for runs that don't resolve the deformation radius:
//   Gm   — Gent–McWilliams isopycnal bolus (thickness-diffusing) transport.
//   Redi — isoneutral tracer diffusion (mixing along, not across, density surfaces).
//   Meke — mesoscale eddy kinetic energy (a prognostic that sets the GM coefficient).
//   Mle  — mixed-layer restratification by submesoscale fronts.
// These are PHYSICS closures (real ocean processes), distinct from dissipation.hpp
// (grid-scale numerical hygiene). They need isopycnal slopes → EOS. Long-arc (Later);
// relevant once coarse (eddy-parameterized) configs matter.
//
// TODO(Later): Gm + Redi with slope tapering; MEKE budget; MLE front restratification.
// rakali north-star: src/parameterizations/lateral/structured/rki_ocean_gm.F90,
//                    rki_ocean_redi.F90, rki_ocean_meke.F90, rki_ocean_mle.F90,
//                    rki_ocean_isopycnal_slopes.F90, rki_ocean_lateral_mix.F90
// =============================================================================

#include "core/types.hpp"

namespace tc {

// TODO(Later): Gm / Redi / Meke / Mle policies on a lateral-closure axis, consuming
//              isopycnal slopes from the EOS. Each an {init; compute} operator.

} // namespace tc
