#pragma once
// =============================================================================
// physics/momentum/pgf_layered.hpp — the general N-layer pressure-gradient force.
//
// Generalizes TwoLayerReducedGravityPgf (two_layer_pgf.hpp) to arbitrary NL. The
// pressure at layer k is the MONTGOMERY potential M_k = Σ_{i≤k} g_i η_i (or the FV
// Wright form once a real EOS drives density), so -∇p_k couples layer k to ALL
// overlying thicknesses. Reduced-gravity 2-layer is the N=2 special case /
// optimization of this — the actual "two-layer-ness" the SplitCore generalization
// isolates lives HERE (plus the H/rho arrays in Params).
//
// Sibling of two_layer_pgf.hpp on the Pgf axis; takes the whole LayeredState (it
// couples), like the two-layer PGF. Consumes ρ from an Eos (eos.hpp) once nonlinear.
//
// TODO(M4): LayeredMontgomeryPgf<Eos> — Montgomery from thicknesses(+density); FV recon.
// rakali north-star: src/pressure_force/structured/rki_ocean_pressure_force.F90,
//                    rki_ml_pressure.F90, rki_ocean_pgf_reconstruct.F90
// =============================================================================

#include "lib/arena.hpp"
#include "physics/state/layered_state.hpp"

namespace tc {

// TODO(M4): template <int NL, class Eos = void> class LayeredMontgomeryPgf {
//   void init(Arena&, mesh);
//   void compute(LayeredState<NL> s, LayeredState<NL> k, mesh, Params p) const;  // -∇M_k into k.u/k.v
// };
//   (Satisfies the same 2-layer-PGF slot the Core takes; select by policy. Params grows
//    std::array<Real,NL> H, rho in the NL generalization.)

} // namespace tc
