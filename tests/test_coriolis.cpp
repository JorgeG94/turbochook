// =============================================================================
// tests/test_coriolis.cpp — the vector-invariant Sadourny Coriolis+advection.
// Oracles:
//   • uniform flow on an f-plane ⇒ ζ=0, ∇KE=0, so ∂u/∂t = f·v, ∂v/∂t = -f·u exactly;
//   • GEOSTROPHIC BALANCE: a linear-η zonal jet with u = -(g/f)∂η/∂y makes PGF and
//     Coriolis cancel exactly (k.u = k.v = 0) — the balance every ocean lives near
//     and the seed state for baroclinic instability. On a linear η + constant u this
//     discretises exactly, so the residual is machine-eps.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <cmath>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/state/baro_state.hpp"
#include "physics/momentum/coriolis.hpp"
#include "physics/momentum/pgf.hpp"

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

TEST_CASE("Sadourny: uniform flow on an f-plane ⇒ ∂u/∂t=f·v, ∂v/∂t=-f·u") {
    const Index nx = 12, ny = 10;
    const Real U = 0.4, V = -0.3, H = 1000.0, f0 = 1.0e-4;
    tc::CartesianMesh m(nx, ny, 100.0, 100.0, f0, /*beta*/0.0);
    tc::Arena arena(4u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::SadournyEnstrophy cor; cor.init(arena, m);

    fill_center(s.eta, m, [=](Index, Index) { return H; });
    fill_xface(s.u, m, [=](Index, Index) { return U; });
    fill_yface(s.v, m, [=](Index, Index) { return V; });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = 100, .dy = 100, .dt = 1, .g = 9.81, .H = H };
    cor.compute(s, k, m, p);

    const tc::Field2 ku = k.u, kv = k.v;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx; ++i)              // interior u-faces
            CHECK(ku[i, j] == doctest::Approx(f0 * V));
    for (Index j = 1; j < ny; ++j)                  // interior v-faces
        for (Index i = 0; i < nx; ++i)
            CHECK(kv[i, j] == doctest::Approx(-f0 * U));
}

TEST_CASE("Sadourny + PGF: a geostrophic zonal jet is in exact balance (k≈0)") {
    const Index nx = 16, ny = 12;
    const Real dx = 1.0e4, dy = 1.0e4, g = 9.81, H = 1000.0, f0 = 1.0e-4;
    const Real deta_dy = 1.0e-6;                     // η slope (linear in y)
    const Real u_geo = -(g / f0) * deta_dy;          // geostrophic zonal velocity
    tc::CartesianMesh m(nx, ny, dx, dy, f0, /*beta*/0.0);
    tc::Arena arena(8u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);
    tc::SadournyEnstrophy cor; cor.init(arena, m);
    tc::FvPgf pgf;

    fill_center(s.eta, m, [=](Index, Index j) { return H + deta_dy * m.y(tc::Loc::Center, 0, j); });
    fill_xface(s.u, m, [=](Index, Index) { return u_geo; });
    fill_yface(s.v, m, [=](Index, Index) { return Real(0); });
    tc::zero_baro_state(k, m);

    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = 1, .g = g, .H = H };
    pgf.compute(s, k, m, p);      // -g∇η
    cor.compute(s, k, m, p);      // (ζ+f)×v etc.  → must cancel the PGF

    const Real scale = g * deta_dy;                  // the magnitude each term carries
    const tc::Field2 ku = k.u, kv = k.v;
    for (Index j = 0; j < ny; ++j)
        for (Index i = 1; i < nx; ++i)
            CHECK(std::abs(ku[i, j]) < 1e-12 * scale);          // x-momentum: nothing to balance
    for (Index j = 1; j < ny; ++j)
        for (Index i = 0; i < nx; ++i)
            CHECK(std::abs(kv[i, j]) < 1e-10 * scale);          // PGF + Coriolis cancel to eps
}
