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
#include <functional>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "physics/baro_state.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// total_mass = Σ η·area over wet centres. Doubles as the M2 continuity oracle:
// drift ~ machine-eps IS mass conservation (the analytical test reuses this).
template <Mesh M>
Real total_mass(const BaroState& s, const M& mesh) {
    const Field2 eta = s.eta;                          // hoist view → capture by value
    const M      m   = mesh;                           // small POD mesh, by value (never `this`)
    const Index  nx  = m.extent_x(Loc::Center);
    const Index  ny  = m.extent_y(Loc::Center);
    auto ids = std::views::iota(Index{0}, nx * ny);
    return std::transform_reduce(par, ids.begin(), ids.end(), Real(0), std::plus<Real>{},
        [=](Index n) {
            const Index i = n % nx, j = n / nx;
            return eta[i, j] * m.area(Loc::Center, i, j) * m.wet(Loc::Center, i, j);
        });
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

} // namespace tc
