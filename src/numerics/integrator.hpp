#pragma once
// =============================================================================
// numerics/integrator.hpp — the time-integration policy.
//
// KEY IDEA (DESIGN decision #7.2): the integrator knows NOTHING about physics.
// It's over (state, scratch, an RHS callable, a BC callable, params) — it only
// fills ghosts, calls the RHS op, and combines stages. Swap Fwd-Euler ↔ SSP-RK2
// without touching a single operator. This clean separation is why the RHS is
// passed AS A CALLABLE (a template parameter), not hard-wired.
//
// `n_scratch` (a compile-time constant) tells the caller how many extra register
// sets (BaroStates) to allocate for the stages. That's the whole concept.
// =============================================================================

#include <concepts>
#include "physics/baro_state.hpp"

namespace tc {

template <class I>
concept Integrator = requires {
    { I::n_scratch } -> std::convertible_to<int>;
};

// Strong-Stability-Preserving RK2 (Heun). One extra stage register.
struct SSPRK2 {
    static constexpr int n_scratch = 1;

    // advance is a TEMPLATE over the RHS/BC callables → physics-agnostic and
    // fully inlined (no std::function, no virtual). `RhsOp` computes tendencies
    // into k; `BcOp` fills halos. These callables run on the HOST and internally
    // launch the for_each kernels — so THEY may capture `this` of the OceanCore;
    // only the innermost kernel lambdas must not.
    template <class RhsOp, class BcOp>
    static void advance(BaroState s, BaroState k, RhsOp rhs, BcOp bc, Params p) {
        // TODO(M2): the two Heun stages, e.g.
        //   bc(s);  rhs(s, k);                       // k = f(s)
        //   combine(s_star, s, k, 1, p.dt);          // s* = s + dt·k   (per field)
        //   bc(s_star); rhs(s_star, k2);             // k2 = f(s*)
        //   combine(s, s, k, k2, 1, dt/2, dt/2);     // s += dt/2·(k+k2)
        // `combine` is a small axpy over each staggered field (its own extent).
        (void)s; (void)k; (void)rhs; (void)bc; (void)p;
    }
};

// Forward Euler — the simplest possible, useful as an M2 warm-up / debug baseline.
struct ForwardEuler {
    static constexpr int n_scratch = 1;
    template <class RhsOp, class BcOp>
    static void advance(BaroState s, BaroState k, RhsOp rhs, BcOp bc, Params p) {
        // TODO: bc(s); rhs(s,k); combine(s, s, k, 1, p.dt);
        (void)s; (void)k; (void)rhs; (void)bc; (void)p;
    }
};

static_assert(Integrator<SSPRK2> && Integrator<ForwardEuler>);

} // namespace tc
