// =============================================================================
// src/demo_geostrophic.cpp — a visual demo of the barotropic core: GEOSTROPHIC
// ADJUSTMENT on an f-plane. An initial Gaussian sea-surface bump collapses; the
// unbalanced part radiates away as inertia-gravity waves while the rest settles
// into a rotating, geostrophically balanced vortex. Renders the η anomaly to PPM
// frames (diverging blue–white–red colormap) — assemble with ffmpeg/ImageMagick.
//
//   build-host/demo_geostrophic <out_dir>     (writes frame_XXXX.ppm)
//
// Host-build demo (reads the state per-pixel on the host); the physics is the same
// GPU-validated stack driving the tests.
// =============================================================================

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/ocean_core.hpp"

namespace {
// Diverging colormap: v∈[-1,1] → blue (−) · white (0) · red (+).
void colormap(tc::Real v, unsigned char& r, unsigned char& g, unsigned char& b) {
    v = v < -1 ? -1 : (v > 1 ? 1 : v);
    if (v >= 0) { r = 255; g = (unsigned char)(255 * (1 - v)); b = (unsigned char)(255 * (1 - v)); }
    else        { b = 255; g = (unsigned char)(255 * (1 + v)); r = (unsigned char)(255 * (1 + v)); }
}
}  // namespace

int main(int argc, char** argv) {
    const std::string outdir = argc > 1 ? argv[1] : "tmp/frames";

    const tc::Index nx = 140, ny = 140;
    const tc::Real dx = 18000.0, dy = 18000.0;          // 18 km → ~2520 km box
    const tc::Real g = 9.81, H = 1000.0, f0 = 1.0e-4;   // f-plane, Rd = √(gH)/f ≈ 990 km
    const tc::Real A = 1.0, Lbump = 260000.0;           // 1 m bump, 260 km wide

    tc::CartesianMesh mesh(nx, ny, dx, dy, f0, /*beta*/0.0);
    const tc::Real c  = std::sqrt(g * H);
    const tc::Real dt = 0.4 * dx / c;                   // CFL ≈ 0.4

    tc::Arena arena(96u << 20);
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    tc::BarotropicPoC core(mesh, arena, p);
    core.init();

    // IC: η = H + A·exp(−r²/L²), at rest.
    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    const tc::Real x0 = 0.5 * nx * dx, y0 = 0.5 * ny * dy;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
        [=](tc::Index i, tc::Index j) {
            const tc::Real x = mesh.x(tc::Loc::Center, i, j), y = mesh.y(tc::Loc::Center, i, j);
            const tc::Real r2 = ((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (Lbump * Lbump);
            eta[i, j] = H + A * std::exp(-r2);
        });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](tc::Index i, tc::Index j) { u[i, j] = 0; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](tc::Index i, tc::Index j) { v[i, j] = 0; });

    const int nsteps = 900, every = 6, px = 4;          // ~150 frames, 4 px/cell → 560²
    const tc::Real vis = 0.5 * A;                        // colormap saturates at ±0.5 m
    const int W = int(nx) * px, Hh = int(ny) * px;
    std::vector<unsigned char> img(std::size_t(W) * Hh * 3);

    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % every == 0) {
            const tc::Field2 e = core.state().eta;
            for (int py = 0; py < Hh; ++py)
                for (int qx = 0; qx < W; ++qx) {
                    const tc::Index i = qx / px, j = py / px;
                    unsigned char r, gg, b; colormap((e[i, j] - H) / vis, r, gg, b);
                    const int oy = Hh - 1 - py;          // north up
                    const std::size_t idx = (std::size_t(oy) * W + qx) * 3;
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
        core.step();
    }
    std::printf("geostrophic adjustment: wrote %d frames (%dx%d) to %s\n", frame, W, Hh, outdir.c_str());
    return 0;
}
