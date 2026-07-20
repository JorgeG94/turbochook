#pragma once
// =============================================================================
// numerics/integrator.hpp — the time-integration policy.
//
// KEY IDEA (DESIGN decision #7.2): the integrator knows NOTHING about physics.
// It's over (state, scratch registers, an RHS callable, a BC callable, params) —
// it only fills ghosts, calls the RHS op, and combines stages. Swap Fwd-Euler ↔
// SSP-RK2 without touching a single operator. This clean separation is why the RHS
// is passed AS A CALLABLE (a template parameter), not hard-wired.
//
// REGISTERS (ADR review fix): a scheme declares how many scratch register-sets it
// needs via `n_scratch`, and `advance` receives EXACTLY that many as a
// `std::span<BaroState>`. Heun (SSP-RK2) is low-storage with 2: one tendency
// register + one predictor. The caller (OceanCore) allocates `n_scratch` states
// from the arena once and hands them in — the interface can now actually hold the
// registers RK2 needs (the old single-`k` signature could not).
// =============================================================================

#include <concepts>
#include <span>
#include "physics/baro_state.hpp"

namespace tc {

namespace detail {
// Named stand-ins for the RHS/BC callables, so the concept can constrain `advance`'s
// signature (a lambda in a requires-expression is fussy; a named type is clean).
struct SampleRhs { void operator()(BaroState, BaroState) const {} };
struct SampleBc  { void operator()(BaroState) const {} };
}  // namespace detail

// A time integrator declares `n_scratch` AND provides a conforming `advance`. The
// second requirement is what makes a mis-signatured stepper fail HERE, not deep in
// OceanCore::step.
template <class I>
concept Integrator =
    requires(BaroState s, std::span<BaroState> scratch, Params p,
             detail::SampleRhs rhs, detail::SampleBc bc) {
        { I::n_scratch } -> std::convertible_to<int>;
        { I::advance(s, scratch, rhs, bc, p) };
    };

// Strong-Stability-Preserving RK2 (Heun). Low-storage: a tendency register + a
// predictor register = 2 scratch sets.
struct SSPRK2 {
    static constexpr int n_scratch = 2;

    // `advance` is a TEMPLATE over the RHS/BC callables → physics-agnostic and fully
    // inlined (no std::function, no virtual). `RhsOp` computes tendencies into a
    // register; `BcOp` fills halos. These callables run on the HOST and internally
    // launch the for_each kernels — so THEY may capture `this` of the OceanCore; only
    // the innermost kernel lambdas must not.
    template <class State, class RhsOp, class BcOp>
    static void advance(State s, std::span<State> scratch, RhsOp rhs, BcOp bc, Params p) {
        // Heun / SSP-RK2 in Shu–Osher low-storage form (2 registers): predictor s1,
        // tendency k. `s` stays the un-touched s^n until the final average, so no
        // separate save register is needed. Generic over the state type (BaroState
        // or LayeredState) — axpby is resolved by ADL.
        //   s1 = s + dt·L(s)                              (stage-1 predictor)
        //   s1 = s1 + dt·L(s1)                            (stage-2, reuse k)
        //   s  = ½·s + ½·s1                               (= s^{n+1})
        State k  = scratch[0];
        State s1 = scratch[1];
        const Real dt = p.dt;

        bc(s);   rhs(s, k);                       // k = L(s^n)
        axpby(s1, Real(1), s, dt, k);            // s1 = s^n + dt·L(s^n)   (= s^{(1)})
        bc(s1);  rhs(s1, k);                      // k = L(s^{(1)})
        axpby(s1, Real(1), s1, dt, k);           // s1 = s^{(1)} + dt·L(s^{(1)})
        axpby(s, Real(0.5), s, Real(0.5), s1);   // s  = ½·s^n + ½·s1  = s^{n+1}
    }
};

// Forward Euler — the simplest possible, useful as an M2 warm-up / debug baseline.
// One tendency register.
struct ForwardEuler {
    static constexpr int n_scratch = 1;
    template <class State, class RhsOp, class BcOp>
    static void advance(State s, std::span<State> scratch, RhsOp rhs, BcOp bc, Params p) {
        State k = scratch[0];
        bc(s); rhs(s, k);                        // k = L(s^n)
        axpby(s, Real(1), s, p.dt, k);           // s^{n+1} = s^n + dt·L(s^n)
    }
};

// Strong-Stability-Preserving RK3 (Shu–Osher). Unlike RK2/Heun, its stability
// region INCLUDES part of the imaginary axis (|ωΔt| ≲ 1.73), so it integrates
// non-dissipative gravity waves stably — RK2 amplifies every oscillatory mode
// (√(1+(ωΔt)⁴/4) > 1) and the 2Δx grid mode blows up. Same 2 registers as RK2.
struct SSPRK3 {
    static constexpr int n_scratch = 2;
    template <class State, class RhsOp, class BcOp>
    static void advance(State s, std::span<State> scratch, RhsOp rhs, BcOp bc, Params p) {
        State s0 = scratch[0];           // saved s^n (needed at every stage combine)
        State k  = scratch[1];           // tendency
        const Real dt = p.dt;

        axpby(s0, Real(1), s, Real(0), s);                       // s0 = s^n
        bc(s); rhs(s, k);                                        // stage 1
        axpby(s, Real(1), s, dt, k);                             //   s = s^n + dt·L(s^n)   (u1)
        bc(s); rhs(s, k);                                        // stage 2
        axpby(s, Real(0.25), s, Real(0.25) * dt, k);            //   s = ¼u1 + ¼dt·L(u1)
        axpby(s, Real(0.75), s0, Real(1), s);                   //   s = ¾s^n + …           (u2)
        bc(s); rhs(s, k);                                        // stage 3
        axpby(s, Real(2) / 3, s, (Real(2) / 3) * dt, k);        //   s = ⅔u2 + ⅔dt·L(u2)
        axpby(s, Real(1) / 3, s0, Real(1), s);                  //   s = ⅓s^n + …           (u^{n+1})
    }
};

static_assert(Integrator<SSPRK2> && Integrator<ForwardEuler> && Integrator<SSPRK3>);

} // namespace tc
