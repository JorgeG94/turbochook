#pragma once
// =============================================================================
// physics/continuity.hpp — the thickness/continuity operator.
//
// Pattern (DESIGN §6b): "a physics module is a CLASS that (a) owns its arena-
// backed workspace and (b) satisfies a per-module concept." Each scheme variant
// is such a class; the dispatch picks which fills the slot.
//
//   ∂η/∂t = -∇·(H u)      — a transport equation, PPM swept thickness flux.
//
// Note it is GENERIC over the WALL reconstruction policy: the shared FV
// flux-divergence is factored from the swept-flux reconstruction, so PCM/PLM/PPM/
// PQM are drop-in via one template parameter. It constrains on `WallReconstruction`
// (not the umbrella) because the swept flux integrates a per-cell polynomial —
// `ContinuityFlux<Weno5>` is a deliberate compile error (WENO is a FACE scheme with
// different flux math; that path is tracer advection, not continuity). Build PPM
// only for now.
// =============================================================================

#include <concepts>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/reconstruction.hpp"

namespace tc {

// The per-module concept: init your workspace from the Arena, then compute the
// tendency into `k` using state `s`. (`ContinuityModule`, `CoriolisModule`, and
// `PgfModule` are structurally identical right now — same {init; compute}. The
// distinct NAMES document the slot's intent in OceanCore<…>. A subtle C++ point:
// concepts check SHAPE, not intent, so these are currently interchangeable; a
// stronger design tags each module type so a PGF can't be passed where a
// Continuity is expected. Deferred — noted so you see the gap.)
template <class M>
concept ContinuityModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, mesh, p) };
    };

template <WallReconstruction Scheme>
class ContinuityFlux {
    // Persistent workspace, arena-backed — the face-flux buffers, members
    // allocated in init().
    Field2 mass_flux_x_{};   // on x-faces
    Field2 mass_flux_y_{};   // on y-faces
public:
    // Allocating from the MANAGED arena is all the "put it on the device" you
    // need — no separate host→device transfer step.
    void init(Arena& a, const CartesianMesh& m) {
        mass_flux_x_ = a.alloc2d(m.nx() + 1, m.ny());
        mass_flux_y_ = a.alloc2d(m.nx(),     m.ny() + 1);
    }

    // compute: the GPU discipline in miniature. HOIST member views into locals,
    // then the kernel lambda captures [=] — NEVER `this` (capturing `this` passes
    // on host but hard-crashes on GPU with cudaErrorIllegalAddress — verified on
    // nvc++ 26.5 / V100). The math is the TODO; the discipline is the point.
    void compute(BaroState s, BaroState k, const CartesianMesh& mesh, Params p) const {
        Field2 fx = mass_flux_x_, fy = mass_flux_y_;   // ← hoist, capture these by value
        // TODO(M2): implement the PPM swept thickness flux (the per-cell-gather
        // FaceView flux-divergence lands here — its first consumer). Sketch:
        //     for_each_face_x: fx[i,j] = Scheme::reconstruct(...)-based swept flux
        //     for_each_face_y: fy[i,j] = ...
        //     for_each_cell  : k.eta[i,j] -= (Σ flux·edge_len)/area   (mesh metrics)
        (void)s; (void)k; (void)mesh; (void)p; (void)fx; (void)fy;
    }
};

// M2 builds PPM only; PLM/WENO are a type-swap away via the same class.
using PpmContinuity = ContinuityFlux<Ppm>;

static_assert(ContinuityModule<PpmContinuity>);

} // namespace tc
