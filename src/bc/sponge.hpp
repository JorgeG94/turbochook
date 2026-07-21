#pragma once
// =============================================================================
// bc/sponge.hpp — sponge / relaxation layer.
//
// Near an open boundary (or as a damping frame around a limited-area domain), relax
// the state toward a reference with a spatially-ramped rate:
//   ∂φ/∂t += -γ(x)·(φ - φ_ref),   γ ramping from 0 (interior) to γ_max (edge).
// Suppresses spurious reflections off the boundary. This is a post-tendency SOURCE
// (added to k), not a halo fill — so it's a small operator, not a BoundaryCondition.
//
// TODO(Later): Sponge — γ(x) ramp field + reference state; add to the tendency.
// rakali north-star: src/core/ocean/boundary/rki_ocean_sponge.F90
// =============================================================================

#include "core/types.hpp"
#include "physics/state/baro_state.hpp"

namespace tc {

// TODO(Later): class Sponge { Field2 gamma_; BaroState ref_;
//   void init(Arena&, mesh, ramp_width);
//   template <Mesh M> void apply(BaroState s, BaroState k, const M& mesh, Params p) const; };

} // namespace tc
