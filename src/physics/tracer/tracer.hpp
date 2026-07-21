#pragma once
// =============================================================================
// physics/tracer/tracer.hpp — passive/active tracer advection (T, S, ideal age, …).
//
// Tracers are co-located NOUNS at cell centres (a SystemView<N> in DESIGN), advected
// by the layer transport (h·u). Flux-form, conservative, monotone. The scheme is a
// policy on a FACE-reconstruction axis (PCM → PLM → PPM → WENO5/7/9) — distinct from
// continuity's WALL axis, because tracer flux reconstructs the tracer TO the face,
// then rides the SAME mass transport continuity computed (consistency: a mismatched
// transport makes S/T grow spurious extrema).
//
// TODO(Later): TracerAdvection concept + PPM/WENO models; wire S/T; ideal-age source.
// rakali north-star: src/tracer/structured/rki_ml_tracers.F90, rki_ml_tracers_weno.F90,
//                    src/tracer/structured/rki_ocean_ideal_age.F90
// =============================================================================

#include <concepts>
#include "lib/arena.hpp"
#include "physics/state/baro_state.hpp"

namespace tc {

// A tracer scheme advects a centre field `c` by the flow's transport into tendency `kc`.
template <class T>
concept TracerAdvection =
    requires(T t, Arena& a, const CartesianMesh& mesh, BaroState flow, Field2 c, Field2 kc, Params p) {
        { t.init(a, mesh) };
        { t.compute(flow, c, kc, mesh, p) };   // ∂(h c)/∂t = -∇·(c · h u), reusing flow's face transport
    };

// TODO(Later): template <FaceReconstruction Scheme> class TracerFlux {
//   Field2 flux_x_, flux_y_;  // reuse continuity's mass flux; reconstruct c to faces (upwind Scheme)
//   void init(Arena&, mesh);  void compute(BaroState flow, Field2 c, Field2 kc, mesh, Params); };
//   (SystemView<N> to carry S/T/age together; one sweep, N components.)

} // namespace tc
