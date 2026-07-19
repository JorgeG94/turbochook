// =============================================================================
// tests/test_pgf.cpp — the pressure-gradient operator (FvPgf), the first real
// kernel. Two analytical oracles (DESIGN §10):
//   • lake-at-rest — a flat surface stays flat (well-balancedness);
//   • constant gradient — a linear η yields the exact -g∇η (proves it computes
//     the gradient, not merely that zero stays zero).
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/pgf.hpp"

namespace {
void zero_field(tc::Field2 f, tc::Index nx, tc::Index ny) {
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) { f[i, j] = tc::Real(0); });
}
}  // namespace

TEST_CASE("FvPgf lake-at-rest: flat η ⇒ zero tendency (well-balanced)") {
    const tc::Index nx = 10, ny = 8;
    tc::CartesianMesh m(nx, ny, /*dx*/500.0, /*dy*/300.0);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);

    const tc::Field2 eta = s.eta;
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) { eta[i, j] = tc::Real(1.7); });
    zero_field(k.u, m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace));
    zero_field(k.v, m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace));

    tc::Params p{ .nx = nx, .ny = ny, .dx = m.dx(), .dy = m.dy(),
                  .dt = 1.0, .g = 9.81, .H = 1000.0 };
    tc::FvPgf pgf;
    pgf.compute(s, k, m, p);

    const tc::Field2 ku = k.u, kv = k.v;
    for (tc::Index j = 0; j < ny; ++j)
        for (tc::Index i = 1; i < nx; ++i)          // interior u-faces i∈[1,nx-1]
            CHECK(ku[i, j] == doctest::Approx(0.0));
    for (tc::Index j = 1; j < ny; ++j)              // interior v-faces j∈[1,ny-1]
        for (tc::Index i = 0; i < nx; ++i)
            CHECK(kv[i, j] == doctest::Approx(0.0));
}

TEST_CASE("FvPgf recovers a constant gradient exactly") {
    const tc::Index nx = 12, ny = 6;
    const tc::Real dx = 250.0, dy = 400.0, g = 9.81, sx = 1.0e-4, sy = -2.0e-4;
    tc::CartesianMesh m(nx, ny, dx, dy);
    tc::Arena arena(1u << 20);
    tc::BaroState s = tc::allocate_baro_state(arena, m);
    tc::BaroState k = tc::allocate_baro_state(arena, m);

    // η = sx·x_centre + sy·y_centre ⇒ ∂η/∂x = sx, ∂η/∂y = sy (exact on the C-grid;
    // the cross term cancels because a u-face differences same-j centres).
    const tc::Field2 eta = s.eta;
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) {
        eta[i, j] = sx * m.x(tc::Loc::Center, i, j) + sy * m.y(tc::Loc::Center, i, j);
    });
    zero_field(k.u, m.extent_x(tc::Loc::XFace), m.extent_y(tc::Loc::XFace));
    zero_field(k.v, m.extent_x(tc::Loc::YFace), m.extent_y(tc::Loc::YFace));

    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = 1.0, .g = g, .H = 1000.0 };
    tc::FvPgf pgf;
    pgf.compute(s, k, m, p);

    const tc::Field2 ku = k.u, kv = k.v;
    for (tc::Index j = 0; j < ny; ++j)
        for (tc::Index i = 1; i < nx; ++i)
            CHECK(ku[i, j] == doctest::Approx(-g * sx));    // -g ∂η/∂x
    for (tc::Index j = 1; j < ny; ++j)
        for (tc::Index i = 0; i < nx; ++i)
            CHECK(kv[i, j] == doctest::Approx(-g * sy));    // -g ∂η/∂y
}
