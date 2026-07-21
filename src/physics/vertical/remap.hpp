#pragma once
// =============================================================================
// physics/vertical/remap.hpp — the ALE remap operator (conservative regrid to the vcoord).
//
// Periodically (cadence-gated) restore the grid the Vcoord (vcoord.hpp) wants: per
// column, reconstruct the profile (PPM/PQM), then CONSERVATIVELY re-integrate every
// quantity (h and h·{S,T,u,v}) onto the new interfaces. This IS the layered core's
// "vertical advection" — an isopycnal/ALE core has NO vertical-advection CFL (STATUS);
// cross-layer transport happens here, not via an advective term with a wave speed.
//
// A COLUMN kernel: fixed-size per-thread locals + a serial vertical recurrence →
// occupancy-bound (STATUS #6, GPU-fundamental, not a bug to fix). Must tolerate
// VANISHING layers (H_VANISHED guard, safe_math.hpp) — no uniform-dz assumptions.
//
// TODO(M4): remap<NL, Vcoord>(state, vc, mesh, p) — PPM-conservative column regrid.
// rakali north-star: src/ALE/rki_ocean_remap.F90, src/ALE/rki_kernel_remap.F90
// =============================================================================

#include "core/types.hpp"
#include "physics/state/layered_state.hpp"
#include "lib/safe_math.hpp"

namespace tc {

// TODO(M4): template <int NL, class Vcoord, class M>
//   void remap(LayeredState<NL> s, const Vcoord& vc, const M& mesh, Params p);
//   per (i,j) column: target interfaces from vc → reconstruct h·q profiles →
//   re-integrate onto targets (conserving ∫), guarding H_VANISHED. One for_each_cell,
//   the inner vertical loop serial over NL.

} // namespace tc
