// =============================================================================
// examples/programs/demo_baroclinic_split.cpp — TWO-LAYER BAROCLINIC INSTABILITY (the north star).
// A high-latitude zonal jet in geostrophic balance goes unstable and rolls up into
// eddies — emulating two_layer_sw's bc_inst case. Spherical band at 59.25°N,
// periodic-x / wall-y, reduced-gravity two-layer, SSP-RK3, with a Shapiro filter to
// bleed the enstrophy cascade at the grid scale.
//
//   build-{host,gpu}/demo_baroclinic <out_dir> [grid] [days] [steps_per_frame]
//
// IC (initial_condition.jl / tls_initial_condition.F90):
//   ξ(y)=Δξ tanh(y/L),  η=−(g'/g)ξ,  h1=H1+η−ξ, h2=H2+ξ,  u1=(g'/f)∂ξ/∂y, u2=0
//   + deterministic sinusoid perturbation (h1+=δ, h2−=δ) at the deformation scale.
// Renders the interface displacement ξ = h2−H2 (the meandering front IS the
// instability). Assemble frames with ffmpeg.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/spherical_mesh.hpp"
#include "physics/state/layered_state.hpp"
#include "diag/report.hpp"
#ifdef TC_HAVE_NETCDF
#include "io/ocean_output.hpp"
#endif
#include "physics/core/multilayer_core.hpp"
#include "physics/core/split_two_layer.hpp"

using tc::Real;
using tc::Index;
using tc::Loc;

namespace {
// diverging blue–white–red, input clamped to [-1,1]
void colormap(Real v, unsigned char& r, unsigned char& g, unsigned char& b) {
    v = v < -1 ? -1 : (v > 1 ? 1 : v);
    if (v >= 0) { r = 255; g = (unsigned char)(255 * (1 - v)); b = (unsigned char)(255 * (1 - v)); }
    else        { b = 255; g = (unsigned char)(255 * (1 + v)); r = (unsigned char)(255 * (1 + v)); }
}

// One Shapiro (5-point) smoothing sweep of a CENTRE field, topology-aware, out of
// place: f ← (1−ε)f + ¼ε Σ neighbours. Bleeds the 2Δx noise the enstrophy cascade
// piles up; only h is filtered (m,n untouched), as in two_layer_sw.
template <class M>
void shapiro_center(tc::Field2 f, tc::Field2 tmp, const M& mesh, Real eps) {
    const M m = mesh; const Index nx = m.nx(), ny = m.ny();
    tc::for_each_cell(nx, ny, [=](Index i, Index j) {
        const Real nb = bc_at(m, f, i - 1, j) + bc_at(m, f, i + 1, j)
                      + bc_at(m, f, i, j - 1) + bc_at(m, f, i, j + 1);
        tmp[i, j] = (Real(1) - eps) * f[i, j] + Real(0.25) * eps * nb;
    });
    tc::for_each_cell(nx, ny, [=](Index i, Index j) { f[i, j] = tmp[i, j]; });
}
}  // namespace

int main(int argc, char** argv) {
    const std::string outdir = argc > 1 ? argv[1] : "tmp/bci";
    const Index N     = argc > 2 ? Index(std::atoi(argv[2])) : 128;
    const Real  days  = argc > 3 ? Real(std::atof(argv[3])) : 40.0;
    const int   every = argc > 4 ? std::atoi(argv[4]) : 30;
    const Real  eps_cli = argc > 5 ? Real(std::atof(argv[5])) : -1.0;   // Shapiro strength (h)
    const Real  dt_cli  = argc > 6 ? Real(std::atof(argv[6])) : -1.0;   // fixed dt (0/neg = CFL)
    const Real  Ljet_km = argc > 7 ? Real(std::atof(argv[7])) : 100.0;  // jet half-width (km)
    const Real  Dxi_cli = argc > 8 ? Real(std::atof(argv[8])) : 100.0;  // interface amplitude (m)

    // ── bc_inst high-lat parameters (baroclinic_highlat.nml) ────────────────────
    const Real lon1 = -193.75, lon2 = -171.25, lat1 = 53.625, lat2 = 64.875;
    const Real g = 9.81, H1 = 500.0, H2 = 1500.0, rho1 = 1025.0, rho2 = 1027.0;
    const Real gp = g * (rho2 - rho1) / rho1;                       // reduced gravity
    const Real Dxi = Dxi_cli, Ljet = Ljet_km * 1.0e3, noise_amp = 0.05;   // jet + perturbation (args 7,8)
    const Real eps_shapiro = eps_cli >= 0 ? eps_cli : 0.006;        // grid-scale filter (arg5)

    const Real R = tc::EARTH_RADIUS, Om = tc::EARTH_OMEGA, d2r = tc::DEG2RAD;
    const Real phi0 = 0.5 * (lat1 + lat2) * d2r;
    tc::SphericalMesh mesh(N, N, lon1, lat1, lon2, lat2, R, Om,
                           tc::EdgeConn::Periodic, tc::EdgeConn::Periodic,
                           tc::EdgeConn::Wall, tc::EdgeConn::Wall);

    // CFL from the fastest cell (smallest dx, at the northern edge) and the external
    // gravity-wave speed √(g(H1+H2)).
    const Real cext = std::sqrt(g * (H1 + H2));
    Real dxmin = 1e30;
    for (Index j = 0; j < N; ++j) dxmin = std::min(dxmin, mesh.dx(Loc::Center, 0, j));
    const Real dt = dt_cli > 0 ? dt_cli : 0.15 * dxmin / cext;      // arg6, else CFL-safe
    const Real cfl = dt * cext / dxmin;
    const int nsteps = int(days * 86400.0 / dt);

    tc::Arena arena(std::size_t(N) * N * 3 * 8 * 24 + (64u << 20));
    tc::Params p{ .nx = N, .ny = N, .dx = dxmin, .dy = mesh.dy(Loc::Center, 0, 0), .dt = dt,
                  .g = g, .H = H1 + H2, .H1 = H1, .H2 = H2, .rho1 = rho1, .rho2 = rho2 };
    constexpr int Msub = 24;                                  // template default (compile-time)
    const int Msub_cli = argc > 9 ? std::atoi(argv[9]) : Msub; // arg9: runtime barotropic substeps
    tc::SplitTwoLayerCore<tc::SphericalMesh, tc::PpmContinuity, tc::SadournyEnstrophy,
                          tc::TwoLayerReducedGravityPgf, tc::WallBC, Msub> core(mesh, arena, p, Msub_cli);
    core.init();
    tc::Field2 tmp = arena.alloc2d(N, N);                            // Shapiro scratch

    // perturbation wavenumbers (initial_condition.jl): deformation-radius scale
    const Real cint = std::sqrt(gp * H1 * H2 / (H1 + H2));
    const Real f0 = Real(2) * Om * std::sin(phi0);
    const Real Ld = std::abs(cint / f0), kmag = Real(1) / (Real(2) * Ld);
    const Real Lx = R * std::cos(phi0) * (lon2 - lon1) * d2r;
    const Real ky = kmag / std::sqrt(Real(2));
    const int  nmx = std::max(1, int(std::lround(kmag / std::sqrt(Real(2)) * Lx / (Real(2) * std::acos(Real(-1))))));
    const Real kx = Real(2) * std::acos(Real(-1)) * Real(nmx) / Lx;

    // ── IC: balanced tanh jet (layer 1 flows, layer 2 at rest) + perturbation ──────
    tc::LayeredState<2>& st = core.state();
    const tc::Field2 h1 = st.layer[0].eta, h2 = st.layer[1].eta;
    const tc::Field2 u1 = st.layer[0].u,  u2 = st.layer[1].u;
    const tc::Field2 v1 = st.layer[0].v,  v2 = st.layer[1].v;
    const Real PI = std::acos(Real(-1));
    // Front-LOCALISED meander: perturb the interface by a sech²(y/L)-enveloped wave
    // in x, so the seed sits ON the jet (where the shear is) and projects onto the
    // growing baroclinic mode — a whole-domain checkerboard mostly excites far-field
    // gravity waves and barely grows.
    const Real pert = noise_amp * Dxi * Real(4);                    // ~20 m front displacement
    tc::for_each_cell(mesh.extent_x(Loc::Center), mesh.extent_y(Loc::Center), [=](Index i, Index j) {
        const Real y     = R * (mesh.y(Loc::Center, i, j) * d2r - phi0);
        const Real xm    = R * std::cos(phi0) * ((mesh.x(Loc::Center, i, j) - lon1) * d2r);
        const Real sech0 = Real(1) / std::cosh(y / Ljet);
        const Real xi    = Dxi * std::tanh(y / Ljet) + pert * sech0 * sech0 * std::cos(kx * xm);
        const Real eta   = -(gp / g) * xi;
        h1[i, j] = H1 + (eta - xi);
        h2[i, j] = H2 + xi;
    });
    tc::for_each_cell(mesh.extent_x(Loc::XFace), mesh.extent_y(Loc::XFace), [=](Index i, Index j) {
        const Real y    = R * (mesh.y(Loc::XFace, i, j) * d2r - phi0);
        const Real f    = Real(2) * Om * std::sin(mesh.y(Loc::XFace, i, j) * d2r);
        const Real sech = Real(1) / std::cosh(y / Ljet);
        u1[i, j] = (gp / f) * (Dxi / Ljet) * sech * sech;           // geostrophic jet
        u2[i, j] = 0;
    });
    tc::for_each_cell(mesh.extent_x(Loc::YFace), mesh.extent_y(Loc::YFace), [=](Index i, Index j) { v1[i, j] = 0; v2[i, j] = 0; });

    std::fprintf(stderr, "SPLIT bc_inst: %dx%d  dt=%.2fs (CFL=%.2f)  %d steps (%.0f days)  Ld=%.1fkm nmx=%d cint=%.2f eps=%.3f\n",
                 int(N), int(N), double(dt), double(cfl), nsteps, double(days), double(Ld / 1000), nmx, double(cint), double(eps_shapiro));

    const int ppx = std::max(1, 512 / int(N));
    const int W = int(N) * ppx, Hh = int(N) * ppx;
    std::vector<unsigned char> img(std::size_t(W) * Hh * 3);
    const Real vis = 1.2 * Dxi;                                     // colour range on ξ

    tc::Reporter rep;
#ifdef TC_HAVE_NETCDF
    tc::OceanOutput<2, float> ncout(outdir + "/state.nc", mesh, "degrees_east", "degrees_north");
#endif
    const auto t_start = std::chrono::steady_clock::now();
    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % every == 0) {
            const tc::Field2 e = st.layer[1].eta;
            for (int py = 0; py < Hh; ++py)
                for (int qx = 0; qx < W; ++qx) {
                    const Index i = qx / ppx, j = py / ppx;
                    unsigned char r, gg, b;
                    colormap((e[i, j] - H2) / vis, r, gg, b);       // interface ξ = h2−H2
                    const std::size_t idx = (std::size_t(Hh - 1 - py) * W + qx) * 3;
                    img[idx] = r; img[idx + 1] = gg; img[idx + 2] = b;
                }
            char path[512];
            std::snprintf(path, sizeof(path), "%s/frame_%04d.ppm", outdir.c_str(), frame);
            if (FILE* fp = std::fopen(path, "wb")) {
                std::fprintf(fp, "P6\n%d %d\n255\n", W, Hh);
                std::fwrite(img.data(), 1, img.size(), fp);
                std::fclose(fp);
            }
            ++frame;
        }
        if (n % (every * 10) == 0) {
            rep.report(core.state(), mesh, Real(n) * dt / 86400.0, dt, long(n), days);
#ifdef TC_HAVE_NETCDF
            ncout.write(core.state(), mesh, Real(n) * dt);
#endif
        }
        core.step();
        shapiro_center(h1, tmp, mesh, eps_shapiro);
        shapiro_center(h2, tmp, mesh, eps_shapiro);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr, "\n  done: %.0f sim-days in %.1f s wall  (%.2f s/sim-day, %.0f steps/s)\n",
                 double(days), wall, wall / (days > 0 ? days : 1.0), nsteps / (wall > 0 ? wall : 1.0));
    std::printf("bc_inst: wrote %d frames (%dx%d) to %s\n", frame, W, Hh, outdir.c_str());
    return 0;
}
