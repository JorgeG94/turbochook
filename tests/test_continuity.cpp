// =============================================================================
// tests/test_continuity.cpp — the PPM thickness-flux continuity operator.
// Oracles:
//   • lake-at-rest: u=v=0 ⇒ zero tendency;
//   • linearized divergence: h=H, u=a·x, v=0 ⇒ ∂h/∂t = -H·a exactly (interior);
//   • mass conservation: Σ area·∂h/∂t = 0 to machine-eps (flux-form telescoping),
//     the M2 continuity oracle (ADR-8: total_mass drift ~ eps).
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/continuity.hpp"

using tc::Real;
using tc::Index;

namespace {
void fill_center(tc::Field2 f, const tc::CartesianMesh& m, auto g) {
    tc::for_each_cell(m.extent_x(tc::Loc::Center), m.extent_y(tc::Loc::Center),
                      [=](Index i, Index j) { f[i, j] = g(i, j); });
}
void fill_xface(tc::Field2 f, const tc::CartesianMesh& m, auto g) {
    tc::for_each_cell(m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace),
                      [=](Index i, Index j) { f[i, j] = g(i, j); });
}
void fill_yface(tc::Field2 f, const tc::CartesianMesh& m, auto g) {
    tc::for_each_cell(m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace),
                      [=](Index i, Index j) { f[i, j] = g(i, j); });
}
}  // namespace

TEST_CASE("PpmContinuity lake-at-rest: u=v=0 ⇒ zero tendency") {
    const Index nx = 12, ny = 10;
    tc::CartesianMesh m(nx, ny, 100.0, 200.0);
    tc::Arena arena(4u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);

    fill_center(s.eta, m, [](Index, Index) { return Real(1000); });   // flat thickness
    fill_xface(s.u, m, [](Index, Index) { return Real(0); });
    fill_yface(s.v, m, [](Index, Index) { return Real(0); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 200, .dt = 1, .g = 9.81, .H = 1000 };
    cont.compute(s, k, m, p);

    const tc::Field2 ke = k.eta;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i)
            CHECK(ke[i, j] == doctest::Approx(0.0));
}

TEST_CASE("PpmContinuity linearized divergence: h=H, u=a·x ⇒ ∂h/∂t = -H·a") {
    const Index nx = 16, ny = 8;
    const Real dx = 100.0, dy = 100.0, H = 1000.0, a = 1.0e-4;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(4u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);

    fill_center(s.eta, m, [=](Index, Index) { return H; });                 // flat H
    fill_xface(s.u, m, [=](Index i, Index) { return a * (Real(i) * dx); });  // u = a·x_face
    fill_yface(s.v, m, [](Index, Index) { return Real(0); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = 1, .g = 9.81, .H = H };
    cont.compute(s, k, m, p);

    const tc::Field2 ke = k.eta;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx - 1; ++i)          // interior (away from wall convergence)
            CHECK(ke[i, j] == doctest::Approx(-H * a));   // -H ∂u/∂x
}

TEST_CASE("PpmContinuity conserves mass: Σ area·∂h/∂t ≈ 0 (telescoping)") {
    const Index nx = 20, ny = 16;
    const Real dx = 100.0, dy = 100.0;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(8u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::PpmContinuity cont; cont.init(arena, m);

    // arbitrary non-trivial state (walls block outflow ⇒ total mass must be conserved)
    fill_center(s.eta, m, [](Index i, Index j) { return Real(1000) + Real(0.1) * Real(i) - Real(0.05) * Real(j); });
    fill_xface(s.u, m, [](Index i, Index j) { return Real(0.3) + Real(0.01) * Real(j); });
    fill_yface(s.v, m, [](Index i, Index j) { return Real(0.2) - Real(0.005) * Real(i); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = 1, .g = 9.81, .H = 1000 };
    cont.compute(s, k, m, p);

    // net mass rate vs. total activity — drift must be machine-eps relative.
    const tc::Field2 ke = k.eta;
    Real net = 0, activity = 0;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 0; i < nx; ++i) {
            const Real rate = m.area(tc::Loc::Center, i, j) * ke[i, j];
            net += rate;
            activity += std::abs(rate);
        }
    CHECK(activity > 0.0);                          // the test is actually exercising flux
    CHECK(std::abs(net) / activity < 1e-12);        // conserved to FP64 machine-eps
}
