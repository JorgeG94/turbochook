#pragma once
// =============================================================================
// physics/coriolis.hpp — the Coriolis operator (a compile-time policy slot).
//
//   ∂u/∂t += (f v)|u ,   ∂v/∂t += -(f u)|v      (plus the momentum advection)
//
// The C-grid ocean way is a POTENTIAL-VORTICITY-conserving flux form (Sadourny
// 1975), evaluated at cell CORNERS: q = (f + ζ)/h, then a PV-flux into the u/v
// tendencies. `SadournyEnstrophy` is the enstrophy-conserving variant. An
// energy-conserving sibling would be a second class filling the same slot —
// that's the whole point of the policy axis.
// =============================================================================

#include <concepts>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "numerics/parallel.hpp"

namespace tc {

template <class M>
concept CoriolisModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, mesh, p) };
    };

class SadournyEnstrophy {
    Field2 q_{};       // relative vorticity ζ at corners (Bu) — persistent workspace
    Field2 ke_{};      // kinetic energy at centres (T)
public:
    template <Mesh M> void init(Arena& a, const M& m) {
        q_  = a.alloc2d(m.extent_x(Loc::Corner), m.extent_y(Loc::Corner));
        ke_ = a.alloc2d(m.extent_x(Loc::Center), m.extent_y(Loc::Center));
    }

    // Vector-invariant Coriolis + advection (Sadourny 1975, enstrophy form), from
    // rki_coriolis_adv.F90:
    //   ∂u/∂t += (ζ+f)·v|u − ∂KE/∂x ,   ∂v/∂t += −(ζ+f)·u|v − ∂KE/∂y
    // Sphere curvature is IMPLICIT in the varying metrics — no explicit tanφ/R term.
    // Three synchronised passes: ζ at corners, KE at centres, then the PV-flux into
    // k.u/k.v (accumulate). Free-slip / all-wet (slip=1). `h == s.eta`.
    template <Mesh M> void compute(BaroState s, BaroState k, const M& mesh, Params p) const {
        const M m = mesh;
        const Field2 h = s.eta, u = s.u, v = s.v;
        const Field2 ku = k.u, kv = k.v;
        const Field2 q = q_, ke = ke_;
        const Index nx = m.nx(), ny = m.ny();
        const bool per_x = (m.edge(Edge::West)  == EdgeConn::Periodic);
        const bool per_y = (m.edge(Edge::South) == EdgeConn::Periodic);
        (void)p;

        // ── Pass 1: ζ at corners (circulation / area). A WALL edge zeroes its outer
        //    corners; a PERIODIC edge makes them interior wrap corners (shift_x reads
        //    the opposite side). ──
        for_each_cell(m.extent_x(Loc::Corner), m.extent_y(Loc::Corner), [=](Index i, Index j) {
            const bool bx = !per_x && (i < 1 || i > nx - 1);
            const bool by = !per_y && (j < 1 || j > ny - 1);
            if (bx || by) { q[i, j] = Real(0); return; }
            const Real circ =
                (v[shift_x(m, i, 0), j] * m.dy(Loc::YFace, i, j)
               - v[shift_x(m, i - 1, 0), j] * m.dy(Loc::YFace, i - 1, j))                         // ∂v/∂x
              - (u[shift_x(m, i, 0), j] * m.dx(Loc::XFace, i, j)
               - u[shift_x(m, i, 0), j - 1] * m.dx(Loc::XFace, i, j - 1));                         // ∂u/∂y
            q[i, j] = circ / m.area(Loc::Corner, i, j);       // Adcroft reciprocal on land
        });

        // ── Pass 2: kinetic energy at centres (area-weighted). ──
        for_each_cell(nx, ny, [=](Index i, Index j) {
            const Real e = m.area(Loc::XFace, i,     j) * u[i,     j] * u[i,     j]
                         + m.area(Loc::XFace, i + 1, j) * u[i + 1, j] * u[i + 1, j]
                         + m.area(Loc::YFace, i, j    ) * v[i, j    ] * v[i, j    ]
                         + m.area(Loc::YFace, i, j + 1) * v[i, j + 1] * v[i, j + 1];
            ke[i, j] = Real(0.25) * e / m.area(Loc::Center, i, j);
        });

        // ── Pass 3a: du/dt at u-faces. wall-x untouched (→0); periodic-x computes
        //    faces 0,nx identically (wrap), keeping the duplicated storage in sync. ──
        for_each_cell(nx + 1, ny, [=](Index i, Index j) {
            if (!per_x && (i == 0 || i == nx)) return;
            // thickness-weighted v from the 4 v-faces around the u-point (topology-aware)
            const Real hSW = Real(0.5) * (bc_at(m, h, i - 1, j - 1) + bc_at(m, h, i - 1, j));
            const Real hNW = Real(0.5) * (bc_at(m, h, i - 1, j)     + bc_at(m, h, i - 1, j + 1));
            const Real hSE = Real(0.5) * (bc_at(m, h, i, j - 1)     + bc_at(m, h, i, j));
            const Real hNE = Real(0.5) * (bc_at(m, h, i, j)         + bc_at(m, h, i, j + 1));
            const Real vh   = v[shift_x(m, i - 1, 0), j] * hSW + v[shift_x(m, i - 1, 0), j + 1] * hNW
                            + v[shift_x(m, i, 0),     j] * hSE + v[shift_x(m, i, 0),     j + 1] * hNE;
            const Real heff = hSW + hNW + hSE + hNE;
            const Real v_at_u = heff > Real(0) ? vh / heff : Real(0);
            const Real zeta = Real(0.5) * (q[i, j] + q[i, j + 1]);                              // S+N corners
            const Real fc   = Real(0.5) * (m.coriolis(Loc::Corner, i, j) + m.coriolis(Loc::Corner, i, j + 1));
            const Real ke_gx = (bc_at(m, ke, i, j) - bc_at(m, ke, i - 1, j)) / m.dx(Loc::XFace, i, j);
            ku[i, j] += ((zeta + fc) * v_at_u - ke_gx) * m.wet(Loc::XFace, i, j);   // 0 across a coast face
        });

        // ── Pass 3b: dv/dt at v-faces. wall-y untouched (→0); periodic-y computes
        //    faces 0,ny identically. ──
        for_each_cell(nx, ny + 1, [=](Index i, Index j) {
            if (!per_y && (j == 0 || j == ny)) return;
            const Real hSW = Real(0.5) * (bc_at(m, h, i - 1, j - 1) + bc_at(m, h, i, j - 1));
            const Real hSE = Real(0.5) * (bc_at(m, h, i, j - 1)     + bc_at(m, h, i + 1, j - 1));
            const Real hNW = Real(0.5) * (bc_at(m, h, i - 1, j)     + bc_at(m, h, i, j));
            const Real hNE = Real(0.5) * (bc_at(m, h, i, j)         + bc_at(m, h, i + 1, j));
            const Real uh   = u[i, j - 1] * hSW + u[i + 1, j - 1] * hSE + u[i, j] * hNW + u[i + 1, j] * hNE;
            const Real heff = hSW + hSE + hNW + hNE;
            const Real u_at_v = heff > Real(0) ? uh / heff : Real(0);
            const Real zeta = Real(0.5) * (q[i, j] + q[i + 1, j]);                              // W+E corners
            const Real fc   = Real(0.5) * (m.coriolis(Loc::Corner, i, j) + m.coriolis(Loc::Corner, i + 1, j));
            const Real ke_gy = (bc_at(m, ke, i, j) - bc_at(m, ke, i, j - 1)) / m.dy(Loc::YFace, i, j);
            kv[i, j] += (-(zeta + fc) * u_at_v - ke_gy) * m.wet(Loc::YFace, i, j);
        });
    }
};

static_assert(CoriolisModule<SadournyEnstrophy>);

} // namespace tc
