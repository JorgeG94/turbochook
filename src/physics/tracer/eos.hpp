#pragma once
// =============================================================================
// physics/tracer/eos.hpp — the equation of state  ρ = ρ(S, T, p).
//
// Turns the tracers (S, T) into density, which drives the baroclinic pressure
// gradient (pgf_layered.hpp). A policy axis of pure, per-cell functions:
//   LinearEos — ρ = ρ0(1 - α(T-T0) + β(S-S0)); the 2-layer reduced-gravity limit.
//   WrightEos — Wright (1997) nonlinear.
//   Teos10    — TEOS-10 / Roquet polynomial.
// No workspace — a pure function, so it offloads as a flat map AND is called
// pointwise inside the PGF reconstruction. Device-callable ⇒ header-inline (STATUS #7).
//
// TODO(M4): Eos concept + fill LinearEos::rho, then Wright.
// rakali north-star: src/equation_of_state/structured/rki_ocean_eos.F90, rki_ml_eos.F90
// =============================================================================

#include <concepts>
#include "core/types.hpp"

namespace tc {

template <class E>
concept Eos = requires(const E e, Real S, Real T, Real p) {
    { e.rho(S, T, p) } -> std::convertible_to<Real>;
};

struct LinearEos {
    Real rho0{1025}, alpha{2e-4}, beta{8e-4}, T0{10}, S0{35};
    // Device-callable (inline, header-visible). TODO(M4): the affine EOS body.
    Real rho(Real S, Real T, Real p) const {
        (void)S; (void)T; (void)p;
        return rho0;   // stub — TODO: ρ0(1 - α(T-T0) + β(S-S0))
    }
};

static_assert(Eos<LinearEos>);

} // namespace tc
