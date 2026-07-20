#pragma once
// =============================================================================
// diag/reduce.hpp — the generic global reductions (DESIGN ADR-8).
//
// THE REDUCTION IS FACTORED FROM THE INTEGRAND. "total X" is not a bespoke
// function — it is the ONE area-weighted global sum applied to X's per-cell
// density. total_mass = ∫ h ; total_salt = ∫ h·S ; total_heat = ∫ ρ·cp·h·T ;
// total_energy = ∫ (½h|u|²+½gη²). One verb (`global_integral`), N nouns (a per-cell
// functor). Adding a conserved-tracer total is a lambda, not a new function.
//
// Offloads: `transform_reduce` over a flat iota computes on-device; only the scalar
// crosses to the host (the state stays device-resident — ADR-8's ~100× rule). The
// integrand is captured BY VALUE (Field views + POD mesh); kernel rules apply.
//
// `L` is the staggering the quantity lives on (Center for mass/tracers/energy,
// Corner for enstrophy) — it drives BOTH the loop extent and the area/wet weight.
// =============================================================================

#include <execution>
#include <numeric>
#include <ranges>
#include <functional>
#include <limits>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// Σ f(i,j)·area_L(i,j)·wet_L(i,j)  over the L-grid.  f gives the INTENSIVE per-cell
// value; the area + land-mask weighting is applied here so callers never repeat it.
template <Loc L = Loc::Center, Mesh M, class Cell>
Real global_integral(const M& mesh, Cell f) {
    const M m = mesh;
    const Index nx = m.extent_x(L), ny = m.extent_y(L);
    auto ids = std::views::iota(Index{0}, nx * ny);
    return std::transform_reduce(par, ids.begin(), ids.end(), Real(0), std::plus<Real>{},
        [=](Index n) { const Index i = n % nx, j = n / nx;
                       return f(i, j) * m.area(L, i, j) * m.wet(L, i, j); });
}

// max of f over the L-grid (no weighting) — max speed, max CFL, extrema.
template <Loc L = Loc::Center, Mesh M, class Cell>
Real global_max(const M& mesh, Cell f) {
    const M m = mesh;
    const Index nx = m.extent_x(L), ny = m.extent_y(L);
    auto ids = std::views::iota(Index{0}, nx * ny);
    return std::transform_reduce(par, ids.begin(), ids.end(),
        std::numeric_limits<Real>::lowest(),
        [](Real a, Real b) { return a > b ? a : b; },
        [=](Index n) { const Index i = n % nx, j = n / nx; return f(i, j); });
}

} // namespace tc
