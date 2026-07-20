#pragma once
// =============================================================================
// physics/continuity.hpp — the thickness/continuity operator.
//
// Pattern (DESIGN §6b): "a physics module is a CLASS that (a) owns its arena-
// backed workspace and (b) satisfies a per-module concept." Each scheme variant
// is such a class; the dispatch picks which fills the slot.
//
//   ∂η/∂t = -∇·(H u)      — a transport equation, PPM swept thickness flux.
//
// Note it is GENERIC over the WALL reconstruction policy: the shared FV
// flux-divergence is factored from the swept-flux reconstruction, so PCM/PLM/PPM/
// PQM are drop-in via one template parameter. It constrains on `WallReconstruction`
// (not the umbrella) because the swept flux integrates a per-cell polynomial —
// `ContinuityFlux<Weno5>` is a deliberate compile error (WENO is a FACE scheme with
// different flux math; that path is tracer advection, not continuity). Build PPM
// only for now.
// =============================================================================

#include <array>
#include <concepts>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/baro_state.hpp"
#include "physics/reconstruction.hpp"
#include "numerics/parallel.hpp"

namespace tc {

// The per-module concept: init your workspace from the Arena, then compute the
// tendency into `k` using state `s`. (`ContinuityModule`, `CoriolisModule`, and
// `PgfModule` are structurally identical right now — same {init; compute}. The
// distinct NAMES document the slot's intent in OceanCore<…>. A subtle C++ point:
// concepts check SHAPE, not intent, so these are currently interchangeable; a
// stronger design tags each module type so a PGF can't be passed where a
// Continuity is expected. Deferred — noted so you see the gap.)
template <class M>
concept ContinuityModule =
    requires(M m, Arena& a, const CartesianMesh& mesh, BaroState s, BaroState k, Params p) {
        { m.init(a, mesh) };
        { m.compute(s, k, mesh, p) };
    };

template <WallReconstruction Scheme>
class ContinuityFlux {
    // Persistent workspace, arena-backed — the face-flux buffers, members
    // allocated in init().
    Field2 mass_flux_x_{};   // on x-faces
    Field2 mass_flux_y_{};   // on y-faces
public:
    // Allocating from the MANAGED arena is all the "put it on the device" you
    // need — no separate host→device transfer step.
    template <Mesh M> void init(Arena& a, const M& m) {
        mass_flux_x_ = a.alloc2d(m.extent_x(Loc::XFace), m.extent_y(Loc::XFace));
        mass_flux_y_ = a.alloc2d(m.extent_x(Loc::YFace), m.extent_y(Loc::YFace));
    }

    // compute: the GPU discipline in miniature. HOIST member views into locals,
    // then the kernel lambda captures [=] — NEVER `this` (capturing `this` passes
    // on host but hard-crashes on GPU with cudaErrorIllegalAddress — verified on
    // nvc++ 26.5 / V100). The math is the TODO; the discipline is the point.
    // ∂h/∂t = -∇·(h u). Three synchronised passes (stdpar has no async):
    //   1. reconstruct h to each face (Scheme, generic over its radius), pick the
    //      UPWIND cell's edge, form the width-weighted transport flux;
    //   2. (y-faces, same);
    //   3. per cell, divergence of its 4 face fluxes / area → accumulate into k.eta.
    // Mass is conserved to machine-eps by the flux-form telescoping (single-valued
    // face flux + zero wall flux + area·iarea≡1), independent of the scheme.
    // `h == s.eta` is the (total) layer thickness. (rki_continuity.F90.)
    template <Mesh M> void compute(BaroState s, BaroState k, const M& mesh, Params p) const {
        const M      m  = mesh;                        // POD copy, capture by value
        const Field2 h  = s.eta, u = s.u, v = s.v;
        const Field2 fx = mass_flux_x_, fy = mass_flux_y_;
        const Field2 ke = k.eta;
        const Index  nx = m.nx(), ny = m.ny();
        constexpr int R = Scheme::radius;
        (void)p;

        // ── Pass 1: x-face mass flux. Interior faces i∈[1,nx-1]; walls i=0,nx → 0. ──
        for_each_cell(nx + 1, ny, [=](Index i, Index j) {
            if (i == 0 || i == nx) { fx[i, j] = Real(0); return; }        // wall = no flux
            std::array<Real, 2 * R + 1> west{}, east{};                   // cells i-1, i
            for (int s2 = -R; s2 <= R; ++s2) {
                west[s2 + R] = clamp_at(h, (i - 1) + s2, j, nx, ny);
                east[s2 + R] = clamp_at(h,  i      + s2, j, nx, ny);
            }
            const Real hR_west = Scheme::reconstruct(west).at_right();    // west cell's east edge
            const Real hL_east = Scheme::reconstruct(east).at_left();     // east cell's west edge
            const Real uu = u[i, j];
            const Real hf = (uu >= Real(0)) ? hR_west : hL_east;          // upwind donor edge
            // × wet: no mass crosses a wet–dry (coast) face. ×1 (compiles away) all-wet.
            fx[i, j] = uu * hf * m.dy(Loc::XFace, i, j) * m.wet(Loc::XFace, i, j);
        });

        // ── Pass 2: y-face mass flux. Interior faces j∈[1,ny-1]; walls j=0,ny → 0. ──
        for_each_cell(nx, ny + 1, [=](Index i, Index j) {
            if (j == 0 || j == ny) { fy[i, j] = Real(0); return; }
            std::array<Real, 2 * R + 1> south{}, north{};                 // cells j-1, j
            for (int s2 = -R; s2 <= R; ++s2) {
                south[s2 + R] = clamp_at(h, i, (j - 1) + s2, nx, ny);
                north[s2 + R] = clamp_at(h, i,  j      + s2, nx, ny);
            }
            const Real hN_south = Scheme::reconstruct(south).at_right();  // south cell's north edge
            const Real hS_north = Scheme::reconstruct(north).at_left();   // north cell's south edge
            const Real vv = v[i, j];
            const Real hf = (vv >= Real(0)) ? hN_south : hS_north;
            fy[i, j] = vv * hf * m.dx(Loc::YFace, i, j) * m.wet(Loc::YFace, i, j);
        });

        // ── Pass 3: divergence → tendency. k.eta += -(Δfx + Δfy)/area (accumulate). ──
        for_each_cell(nx, ny, [=](Index i, Index j) {
            const Real div = (fx[i + 1, j] - fx[i, j]) + (fy[i, j + 1] - fy[i, j]);
            ke[i, j] += -div / m.area(Loc::Center, i, j);                 // ∂h/∂t = -∇·(hu)
        });
    }
};

// M2 builds PPM only; PLM/WENO are a type-swap away via the same class.
using PpmContinuity = ContinuityFlux<Ppm>;

static_assert(ContinuityModule<PpmContinuity>);

} // namespace tc
