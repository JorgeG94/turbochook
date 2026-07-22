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
#include "physics/state/baro_state.hpp"

namespace tc {

namespace detail {
// Named stand-ins for the RHS/BC callables, so the concept can constrain `advance`'s
// signature (a lambda in a requires-expression is fussy; a named type is clean).
struct SampleRhs { void operator()(BaroState, BaroState) const {} };
struct SampleBc  { void operator()(BaroState) const {} };

// ── SspStep<N> — the SSP-RK coefficients, ONCE (Shu–Osher convex-combination form) ──
// An SSP-RK scheme is a convex combination of a FORWARD OPERATOR φ: a nullary callable
// that advances the live state one forward-Euler-like step (by dt), in place, managing
// its own internals. `s0` is the saved-sⁿ register (the only explicit scratch the combine
// needs). This is the single home for the ½/½, ¾/¼, ⅓/⅔ coefficients — the method-of-lines
// integrators below plug φ = (fill halos; RHS; s += dt·L(s)); the split stepper plugs φ = its
// whole mode-split stage. `axpby` is resolved by ADL on the (dependent) State at instantiation.
template <int Stages> struct SspStep;
template <> struct SspStep<1> {                               // forward Euler (one φ, no save)
    template <class State, class Phi> static void run(State s, State s0, Phi phi) { (void)s; (void)s0; phi(); }
};
template <> struct SspStep<2> {                               // SSP-RK2 (Heun)
    template <class State, class Phi> static void run(State s, State s0, Phi phi) {
        axpby(s0, Real(1), s, Real(0), s);                    // s0 = sⁿ
        phi(); phi();                                         // Φ(u1),  u1 = Φ(sⁿ)
        axpby(s, Real(0.5), s0, Real(0.5), s);                // sⁿ⁺¹ = ½sⁿ + ½Φ(u1)
    }
};
template <> struct SspStep<3> {                               // SSP-RK3 (imaginary-axis-stable)
    template <class State, class Phi> static void run(State s, State s0, Phi phi) {
        axpby(s0, Real(1), s, Real(0), s);                    // s0 = sⁿ
        phi(); phi(); axpby(s, Real(0.75), s0, Real(0.25), s);// u2 = ¾sⁿ + ¼Φ(u1)
        phi();        axpby(s, Real(1) / 3, s0, Real(2) / 3, s);// sⁿ⁺¹ = ⅓sⁿ + ⅔Φ(u2)
    }
};
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

// Strong-Stability-Preserving RK2 (Heun). Two registers: the saved sⁿ (s0) + the tendency k.
struct SSPRK2 {
    static constexpr int n_scratch = 2;

    // `advance` is a TEMPLATE over the RHS/BC callables → physics-agnostic and fully
    // inlined (no std::function, no virtual). `RhsOp` computes tendencies into a
    // register; `BcOp` fills halos. These callables run on the HOST and internally
    // launch the for_each kernels — so THEY may capture `this` of the OceanCore; only
    // the innermost kernel lambdas must not. The scheme is SspStep<2> over the forward
    // operator φ = (fill halos; RHS; s += dt·L(s)); generic over the state type (axpby by ADL).
    template <class State, class RhsOp, class BcOp>
    static void advance(State s, std::span<State> scratch, RhsOp rhs, BcOp bc, Params p) {
        State k = scratch[1];
        detail::SspStep<2>::run(s, scratch[0],
            [&] { bc(s); rhs(s, k); axpby(s, Real(1), s, p.dt, k); });
    }
};

// Forward Euler — the simplest possible, useful as an M2 warm-up / debug baseline.
// One tendency register (SspStep<1> ignores the save register).
struct ForwardEuler {
    static constexpr int n_scratch = 1;
    template <class State, class RhsOp, class BcOp>
    static void advance(State s, std::span<State> scratch, RhsOp rhs, BcOp bc, Params p) {
        State k = scratch[0];
        detail::SspStep<1>::run(s, s,            // s0 unused for FE
            [&] { bc(s); rhs(s, k); axpby(s, Real(1), s, p.dt, k); });
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
        State k = scratch[1];
        detail::SspStep<3>::run(s, scratch[0],
            [&] { bc(s); rhs(s, k); axpby(s, Real(1), s, p.dt, k); });
    }
};

static_assert(Integrator<SSPRK2> && Integrator<ForwardEuler> && Integrator<SSPRK3>);

} // namespace tc
