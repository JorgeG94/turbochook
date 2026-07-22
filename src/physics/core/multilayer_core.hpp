#pragma once
// =============================================================================
// physics/core/multilayer_core.hpp — the composed multilayer (unsplit) solver (M3).
//
// The god-state for a stacked N-layer C-grid, MultilayerCore<NL>. The structure that
// makes it baroclinic: continuity and Coriolis run PER LAYER (reusing the single-layer
// operators unchanged — one instance, reused for each layer since the passes are
// synchronous), and the coupling PGF ties the layers (each layer's pressure depends on
// the overlying thicknesses). The generic Integrator steps the whole LayeredState<NL>; a
// per-layer BC fills halos. NL=2 with the reduced-gravity PGF is today's instantiation
// (spelled MultilayerCore<2, …> at the call site); NL>2 waits on the Montgomery PGF +
// array Params (deferred).
//
//     ∂h_k/∂t = -∇·(h_k u_k)                 [Continuity, per layer]
//     ∂u_k/∂t = -∇p_k + (ζ_k+f)·v_k + adv    [coupling PGF + Coriolis]
// =============================================================================

#include <array>
#include <span>
#include "lib/arena.hpp"
#include "lib/log.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/state/layered_state.hpp"
#include "physics/continuity/continuity.hpp"
#include "physics/momentum/coriolis.hpp"
#include "physics/momentum/two_layer_pgf.hpp"
#include "bc/bc.hpp"
#include "numerics/integrator.hpp"

namespace tc {

template <int               NL,
          Mesh              Msh,
          ContinuityModule  Cont,
          CoriolisModule    Cor,
          class             Pgf,       // a coupling PGF over LayeredState<NL>
          BoundaryCondition Bc,
          Integrator        Integ>
class MultilayerCore {
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
    MultilayerCore(Msh mesh, Arena& a, Params p) : mesh_(mesh), arena_(a), p_(p) {}

    void init() {
        state_ = allocate_layered_state<NL>(arena_, mesh_);
        for (LayeredState<NL>& r : scratch_) r = allocate_layered_state<NL>(arena_, mesh_);
        cont_.init(arena_, mesh_);
        cor_ .init(arena_, mesh_);
        pgf_ .init(arena_, mesh_);
        tc::logger().info("MultilayerCore init: {}x{} grid x {} layers, arena {}/{} bytes",
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

// No bespoke two-layer alias: the two-layer case is just MultilayerCore<2, …> spelled
// out at the call site (reduced-gravity PGF + 2-layer Params fix NL=2 today; NL>2 waits
// on the Montgomery PGF + array Params — deferred).

} // namespace tc
