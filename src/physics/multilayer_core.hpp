#pragma once
// =============================================================================
// physics/multilayer_core.hpp — the two-layer composed solver (M3).
//
// The god-state for a stacked two-layer C-grid. The structure that makes it
// baroclinic: continuity and Coriolis run PER LAYER (reusing the single-layer
// operators unchanged — one instance, reused for each layer since the passes are
// synchronous), and the reduced-gravity PGF COUPLES the layers (each layer's
// pressure depends on both thicknesses). The generic Integrator steps the whole
// LayeredState<2>; a per-layer BC fills halos.
//
//     ∂h_k/∂t = -∇·(h_k u_k)                 [Continuity, per layer]
//     ∂u_k/∂t = -∇p_k + (ζ_k+f)·v_k + adv    [reduced-gravity PGF + Coriolis]
// =============================================================================

#include <array>
#include <span>
#include "lib/arena.hpp"
#include "lib/log.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/layered_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/two_layer_pgf.hpp"
#include "bc/bc.hpp"
#include "numerics/integrator.hpp"

namespace tc {

template <Mesh              Msh,
          ContinuityModule  Cont,
          CoriolisModule    Cor,
          class             Pgf,       // a 2-layer (coupling) PGF over LayeredState<2>
          BoundaryCondition Bc,
          Integrator        Integ>
class TwoLayerCore {
    static constexpr int NL = 2;
    Msh    mesh_;
    Arena& arena_;
    Params p_{};

    Cont cont_{};                       // one instance, reused per layer (synchronous passes)
    Cor  cor_{};
    Pgf  pgf_{};
    Bc   bc_{};

    LayeredState<NL> state_{};
    std::array<LayeredState<NL>, Integ::n_scratch> scratch_{};

public:
    TwoLayerCore(Msh mesh, Arena& a, Params p) : mesh_(mesh), arena_(a), p_(p) {}

    void init() {
        state_ = allocate_layered_state<NL>(arena_, mesh_);
        for (LayeredState<NL>& r : scratch_) r = allocate_layered_state<NL>(arena_, mesh_);
        cont_.init(arena_, mesh_);
        cor_ .init(arena_, mesh_);
        pgf_ .init(arena_, mesh_);
        tc::logger().info("TwoLayerCore init: {}x{} grid x {} layers, arena {}/{} bytes",
                          mesh_.nx(), mesh_.ny(), NL, arena_.bytes_used(), arena_.bytes_capacity());
    }

    // The layered RHS: zero, per-layer continuity + Coriolis, then the coupling PGF.
    void layered_rhs(LayeredState<NL> s, LayeredState<NL> k) const {
        zero_layered_state<NL>(k, mesh_);
        for (int l = 0; l < NL; ++l) cont_.compute(s.layer[l], k.layer[l], mesh_, p_);  // ∂h/∂t
        for (int l = 0; l < NL; ++l) cor_ .compute(s.layer[l], k.layer[l], mesh_, p_);  // Coriolis+adv
        pgf_.compute(s, k, mesh_, p_);                                                  // -∇p (couples)
    }

    void step() {
        Integ::advance(
            state_, std::span<LayeredState<NL>>(scratch_),
            [this](LayeredState<NL> s, LayeredState<NL> k) { layered_rhs(s, k); },
            [this](LayeredState<NL> s) { for (int l = 0; l < NL; ++l) bc_.fill_halos(s.layer[l], mesh_); },
            p_);
    }

    LayeredState<NL>& state() { return state_; }
    const Msh& mesh() const { return mesh_; }
};

// The M3 two-layer instantiation: PPM continuity + Sadourny + reduced-gravity PGF
// + wall BC + SSP-RK3 (gravity waves need the imaginary-axis-stable stepper).
using TwoLayerPoC =
    TwoLayerCore<CartesianMesh, PpmContinuity, SadournyEnstrophy,
                 TwoLayerReducedGravityPgf, WallBC, SSPRK3>;

} // namespace tc
