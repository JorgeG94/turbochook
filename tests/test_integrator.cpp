// =============================================================================
// tests/test_integrator.cpp — the time-integration policies (SSP-RK2, Euler).
// Exact algebraic oracles: for the linear ODE dy/dt = -λy, one Heun step gives
// y1 = y0·(1 - λΔt + ½(λΔt)²) exactly, and Euler gives y0·(1 - λΔt). We embed
// that scalar ODE in a BaroState field via a synthetic RHS.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <array>
#include <span>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "numerics/integrator.hpp"

namespace {
// A synthetic RHS: out = -λ·in on η, zero on u/v (a decoupled decay per cell).
auto decay_rhs(tc::Real lambda) {
    return [lambda](tc::BaroState in, tc::BaroState out) {
        tc::combine_field(out.eta, -lambda, in.eta, tc::Real(0), in.eta);
        tc::combine_field(out.u,   tc::Real(0), in.u, tc::Real(0), in.u);
        tc::combine_field(out.v,   tc::Real(0), in.v, tc::Real(0), in.v);
    };
}
constexpr auto noop_bc = [](tc::BaroState) {};

tc::Real eta_uniform(tc::Arena& a, const tc::CartesianMesh& m, tc::BaroState s, tc::Real v) {
    const tc::Field2 e = s.eta;
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center),
                      [=](tc::Index i, tc::Index j) { e[i, j] = v; });
    return v;
}
}  // namespace

TEST_CASE("SSPRK2 reproduces the exact Heun step for linear decay") {
    tc::CartesianMesh m(4, 4, 1.0, 1.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    std::array<tc::BaroState, tc::SSPRK2::n_scratch> scratch{
        tc::allocate_baro_state(arena, m), tc::allocate_baro_state(arena, m)};

    const tc::Real eta0 = 5.0, lambda = 0.3, dt = 0.1;
    eta_uniform(arena, m, s, eta0);

    tc::Params p{ .nx = 4, .ny = 4, .dx = 1, .dy = 1, .dt = dt, .g = 9.81, .H = 1000 };
    tc::SSPRK2::advance(s, std::span<tc::BaroState>(scratch), decay_rhs(lambda), noop_bc, p);

    const tc::Real x = lambda * dt;
    const tc::Real expect = eta0 * (tc::Real(1) - x + tc::Real(0.5) * x * x);
    const tc::Field2 e = s.eta;
    for (tc::Index j = 0; j < m.ny(); ++j)
        for (tc::Index i = 0; i < m.nx(); ++i)
            CHECK(e[i, j] == doctest::Approx(expect));
    // untouched fields stay zero (multi-field wiring)
    const tc::Field2 u = s.u;
    CHECK(u[2, 2] == doctest::Approx(0.0));
}

TEST_CASE("ForwardEuler reproduces the exact Euler step for linear decay") {
    tc::CartesianMesh m(4, 4, 1.0, 1.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    std::array<tc::BaroState, tc::ForwardEuler::n_scratch> scratch{
        tc::allocate_baro_state(arena, m)};

    const tc::Real eta0 = 5.0, lambda = 0.3, dt = 0.1;
    eta_uniform(arena, m, s, eta0);

    tc::Params p{ .nx = 4, .ny = 4, .dx = 1, .dy = 1, .dt = dt, .g = 9.81, .H = 1000 };
    tc::ForwardEuler::advance(s, std::span<tc::BaroState>(scratch), decay_rhs(lambda), noop_bc, p);

    const tc::Real expect = eta0 * (tc::Real(1) - lambda * dt);
    const tc::Field2 e = s.eta;
    for (tc::Index j = 0; j < m.ny(); ++j)
        for (tc::Index i = 0; i < m.nx(); ++i)
            CHECK(e[i, j] == doctest::Approx(expect));
}

TEST_CASE("SSPRK3 reproduces the exact 3rd-order step for linear decay") {
    tc::CartesianMesh m(4, 4, 1.0, 1.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    std::array<tc::BaroState, tc::SSPRK3::n_scratch> scratch{
        tc::allocate_baro_state(arena, m), tc::allocate_baro_state(arena, m)};

    const tc::Real eta0 = 5.0, lambda = 0.3, dt = 0.1;
    eta_uniform(arena, m, s, eta0);

    tc::Params p{ .nx = 4, .ny = 4, .dx = 1, .dy = 1, .dt = dt, .g = 9.81, .H = 1000 };
    tc::SSPRK3::advance(s, std::span<tc::BaroState>(scratch), decay_rhs(lambda), noop_bc, p);

    const tc::Real x = lambda * dt;                              // 3rd-order Taylor of e^{-x}
    const tc::Real expect = eta0 * (tc::Real(1) - x + x * x / 2 - x * x * x / 6);
    const tc::Field2 e = s.eta;
    for (tc::Index j = 0; j < m.ny(); ++j)
        for (tc::Index i = 0; i < m.nx(); ++i)
            CHECK(e[i, j] == doctest::Approx(expect));
}

TEST_CASE("SSPRK2 integrates a constant tendency exactly (linear in t)") {
    tc::CartesianMesh m(4, 4, 1.0, 1.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    std::array<tc::BaroState, tc::SSPRK2::n_scratch> scratch{
        tc::allocate_baro_state(arena, m), tc::allocate_baro_state(arena, m)};

    const tc::Real eta0 = 2.0, c = 1.5, dt = 0.25;
    eta_uniform(arena, m, s, eta0);

    // constant tendency: out.eta = c everywhere (independent of in)
    auto const_rhs = [c](tc::BaroState in, tc::BaroState out) {
        const tc::Field2 e = out.eta;
        tc::for_each_cell(e.extent(0), e.extent(1), [=](tc::Index i, tc::Index j) { e[i, j] = c; });
        tc::combine_field(out.u, tc::Real(0), in.u, tc::Real(0), in.u);
        tc::combine_field(out.v, tc::Real(0), in.v, tc::Real(0), in.v);
    };
    tc::Params p{ .nx = 4, .ny = 4, .dx = 1, .dy = 1, .dt = dt, .g = 9.81, .H = 1000 };
    tc::SSPRK2::advance(s, std::span<tc::BaroState>(scratch), const_rhs, noop_bc, p);
    tc::SSPRK2::advance(s, std::span<tc::BaroState>(scratch), const_rhs, noop_bc, p);

    const tc::Real expect = eta0 + c * (2 * dt);   // exact for a t-linear solution
    const tc::Field2 e = s.eta;
    for (tc::Index j = 0; j < m.ny(); ++j)
        for (tc::Index i = 0; i < m.nx(); ++i)
            CHECK(e[i, j] == doctest::Approx(expect));
}
