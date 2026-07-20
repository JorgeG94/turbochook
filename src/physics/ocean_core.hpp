#pragma once
// =============================================================================
// physics/ocean_core.hpp — the composed solver (the god-state that owns every
// operator slot).
//
// This is where the compile-time policy axes come together. `OceanCore` is
// templated on FIVE policies — continuity, Coriolis, PGF, integrator, BC — each
// CONSTRAINED by its concept (`template <ContinuityModule Cont, …>`). The
// compiler stamps out one fully-inlined solver for the chosen combination; no
// virtuals, no runtime dispatch inside the step. Choosing a different scheme =
// a different template instantiation (the runtime config string → type mapping
// is the std::variant/std::visit bridge, ADR-4 — a later, host-side concern).
//
// The RHS is a SUM OF OPERATOR TENDENCIES (not a Riemann flux divergence — this
// is the ocean C-grid regime, DESIGN §6):
//     ∂η/∂t = -∇·(H u)                 [Continuity]
//     ∂u/∂t = -g ∂η/∂x + (f v)|u + adv [PGF + Coriolis]
//     ∂v/∂t = -g ∂η/∂y - (f u)|v + adv
// =============================================================================

#include <array>
#include <span>
#include "lib/arena.hpp"
#include "lib/log.hpp"
#include "lib/profiler.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/continuity.hpp"
#include "physics/coriolis.hpp"
#include "physics/pgf.hpp"
#include "bc/bc.hpp"
#include "numerics/integrator.hpp"

namespace tc {

// The template parameters are constrained by the concepts — a mis-typed slot
// (e.g. passing a BC where a Continuity goes) is a crisp compile error at the
// instantiation, not a puzzle deep in a kernel.
template <Mesh             Msh,
          ContinuityModule Cont,
          CoriolisModule   Cor,
          PgfModule        Pgf,
          BoundaryCondition Bc,
          Integrator       Integ>
class OceanCore {
    Msh           mesh_;          // the grid model — a Mesh, not welded to Cartesian
    Arena&        arena_;         // borrowed, not owned — one arena for the run
    Params        p_{};

    // The module instances — each owns its arena-backed workspace (§6b). These
    // ARE the "slots" of the god-state.
    Cont cont_{};
    Cor  cor_{};
    Pgf  pgf_{};
    Bc   bc_{};

    BaroState state_{};                             // the prognostic state
    std::array<BaroState, Integ::n_scratch> scratch_{};  // RK register-sets (integrator sizes it)

public:
    OceanCore(Msh mesh, Arena& a, Params p)
        : mesh_(mesh), arena_(a), p_(p) {}

    // Allocate everything ONCE, up front (arena discipline). Each module wires its
    // own workspace; allocating from the managed arena is all the device setup.
    void init() {
        state_ = allocate_baro_state(arena_, mesh_);
        for (BaroState& reg : scratch_) reg = allocate_baro_state(arena_, mesh_);
        cont_.init(arena_, mesh_);
        cor_ .init(arena_, mesh_);
        pgf_ .init(arena_, mesh_);
        tc::logger().info("OceanCore init: {}x{} grid, arena {} / {} bytes",
                          mesh_.nx(), mesh_.ny(), arena_.bytes_used(), arena_.bytes_capacity());
    }

    // The RHS operator: zero k, then SUM each module's tendency into it. This is
    // `baro_rhs` from DESIGN §6 — the composition of the policy slots.
    void baro_rhs(BaroState s, BaroState k) const {
        // The RHS is a SUM of operator tendencies: zero k, then each compute() += into
        // it. Without the zero, k holds the previous stage/step's tendencies (ADR
        // review: the arena starts zero, so only the FIRST call would be accidentally
        // correct).
        zero_baro_state(k, mesh_);
        cont_.compute(s, k, mesh_, p_);   // η tendency (thickness flux divergence)
        pgf_ .compute(s, k, mesh_, p_);   // -g ∇η into u, v
        cor_ .compute(s, k, mesh_, p_);   // PV-Coriolis + advection into u, v
    }

    // One outer step. The integrator drives; we hand it the RHS and BC as HOST
    // callables (these may capture `this` — they run host-side and launch the
    // kernels; only the innermost kernel lambdas are forbidden from capturing it).
    void step() {
        TC_PROFILE("step");
        Integ::advance(
            state_, std::span<BaroState>(scratch_),
            [this](BaroState s, BaroState k) { baro_rhs(s, k); },   // RhsOp
            [this](BaroState s)              { bc_.fill_halos(s, mesh_); },  // BcOp
            p_);
    }

    BaroState& state() { return state_; }
    const CartesianMesh& mesh() const { return mesh_; }
};

// The concrete PoC instantiation for M2. This one line names the whole scheme:
// PPM continuity + Sadourny-enstrophy Coriolis + FV pressure gradient + wall BCs
// + SSP-RK2. Swapping any policy is swapping one type here.
using BarotropicPoC =
    OceanCore<CartesianMesh, PpmContinuity, SadournyEnstrophy, FvPgf, WallBC, SSPRK2>;

} // namespace tc
