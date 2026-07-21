#pragma once
// =============================================================================
// diag/diagnostics.hpp — device-resident reduction diagnostics (DESIGN ADR-8).
//
// A diagnostic is a PURE REDUCTION over (state, mesh) → a single scalar
// (`std::transform_reduce` over a flat iota, offloading via `tc::par`). The state
// stays DEVICE-RESIDENT — only the scalar crosses to the host. A per-step
// full-field host copy silently reintroduces the ~100–140× migration penalty
// (STATUS #4), so we never do that; a scalar reduction is the whole diagnostic.
//
// Kernel rules apply to the reduction lambdas: capture BY VALUE (the Field views
// + a copy of the small POD mesh), never `this`, no throw. NaN/CFL detection is a
// host-side reduction here → the caller throws host-side (error.hpp discipline:
// kernels never throw).
// =============================================================================

#include <execution>
#include <numeric>
#include <ranges>
#include <cmath>
#include <algorithm>
#include <functional>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "physics/state/baro_state.hpp"
#include "physics/state/layered_state.hpp"
#include "numerics/parallel.hpp"
#include "diag/reduce.hpp"
#include "diag/registry.hpp"

namespace tc {

// ── the "totals" are all ONE reduce over different integrands (ADR-8) ────────────
// total_mass = ∫ h — the mass/continuity oracle (drift ~ eps IS conservation). It is
// `global_integral` of the thickness; total_salt would be ∫h·S, etc. (reduce.hpp).
template <Mesh M>
Real total_mass(const BaroState& s, const M& mesh) {
    const Field2 h = s.eta;
    return global_integral(mesh, [=](Index i, Index j) { return h[i, j]; });
}
template <int NL, Mesh M>
Real total_mass(const LayeredState<NL>& s, const M& mesh) {
    Real m = 0; for (int l = 0; l < NL; ++l) m += total_mass(s.layer[l], mesh); return m;
}

// total kinetic energy = ∫ ½ h |u_centre|²  (faces averaged to centres).
template <int NL, Mesh M>
Real total_ke(const LayeredState<NL>& s, const M& mesh) {
    Real ke = 0;
    for (int l = 0; l < NL; ++l) {
        const Field2 h = s.layer[l].eta, u = s.layer[l].u, v = s.layer[l].v;
        ke += global_integral(mesh, [=](Index i, Index j) {
            const Real uc = Real(0.5) * (u[i, j] + u[i + 1, j]);
            const Real vc = Real(0.5) * (v[i, j] + v[i, j + 1]);
            return Real(0.5) * h[i, j] * (uc * uc + vc * vc);
        });
    }
    return ke;
}

// max flow speed over all layers (centre |u|) — the CFL / |u|max diagnostic.
template <int NL, Mesh M>
Real max_speed(const LayeredState<NL>& s, const M& mesh) {
    Real mx = 0;
    for (int l = 0; l < NL; ++l) {
        const Field2 u = s.layer[l].u, v = s.layer[l].v;
        mx = std::max(mx, global_max(mesh, [=](Index i, Index j) {
            const Real uc = Real(0.5) * (u[i, j] + u[i + 1, j]);
            const Real vc = Real(0.5) * (v[i, j] + v[i, j + 1]);
            return std::sqrt(uc * uc + vc * vc);
        }));
    }
    return mx;
}

// Reduce one field to "does it hold any non-finite value?" — a device OR expressed
// as a count>0 (booleans don't reduce as cleanly as an int sum).
inline bool field_nonfinite(Field2 f, Index nx, Index ny) {
    auto ids = std::views::iota(Index{0}, nx * ny);
    const int bad = std::transform_reduce(par, ids.begin(), ids.end(), 0, std::plus<int>{},
        [=](Index n) { return std::isfinite(f[n % nx, n / nx]) ? 0 : 1; });
    return bad > 0;
}

// any_nonfinite: the NaN/Inf gate over the whole staggered state. Feeds the
// host-side throw (kernels never throw — error.hpp).
template <Mesh M>
bool any_nonfinite(const BaroState& s, const M& m) {
    return field_nonfinite(s.eta, m.extent_x(Loc::Center), m.extent_y(Loc::Center))
        || field_nonfinite(s.u,   m.extent_x(Loc::XFace),  m.extent_y(Loc::XFace))
        || field_nonfinite(s.v,   m.extent_x(Loc::YFace),  m.extent_y(Loc::YFace));
}

// ── the default registry (DESIGN ADR-8 rev) ──────────────────────────────────────
// Wire the catalog Quantities to the reductions above, erasing the OUTCOME (each eval
// is a host closure that launches its device reduce internally — the integrand stays
// monomorphic). This is the single source of truth for "what scalars a run reports":
// the Reporter iterates it, and adding a scalar is one `reg.scalar(...)` line here (or
// at the call site), not a new function threaded through the reporter.
template <int NL, Mesh M>
Registry<LayeredState<NL>, M> default_diagnostics() {
    using State = LayeredState<NL>;
    Registry<State, M> reg;
    reg.scalar(Q_MASS,  [](const State& s, const M& m) { return total_mass(s, m); });
    reg.scalar(Q_KE,    [](const State& s, const M& m) { return total_ke(s, m);   });
    reg.scalar(Q_SPEED, [](const State& s, const M& m) { return max_speed(s, m);  });
    return reg;
}

} // namespace tc
