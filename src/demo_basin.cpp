// =============================================================================
// src/demo_basin.cpp — geostrophic adjustment in a CIRCULAR BASIN with the
// perturbation at the coast. Exercises land/sea masking (MaskedMesh): the ocean is
// a disc, land outside; a Gaussian bump sits near the edge, collapses, and the
// waves radiate and curve around the circular coastline while rotation traps a
// vortex. Renders ocean (diverging colormap) + land (grey). Assemble with ffmpeg.
//
//   build-host/demo_basin <out_dir> [grid]
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "mesh/masked_mesh.hpp"
#include "physics/ocean_core.hpp"

namespace {
void colormap(tc::Real v, unsigned char& r, unsigned char& g, unsigned char& b) {
    v = v < -1 ? -1 : (v > 1 ? 1 : v);
    if (v >= 0) { r = 255; g = (unsigned char)(255 * (1 - v)); b = (unsigned char)(255 * (1 - v)); }
    else        { b = 255; g = (unsigned char)(255 * (1 + v)); r = (unsigned char)(255 * (1 + v)); }
}
}  // namespace

int main(int argc, char** argv) {
    const std::string outdir = argc > 1 ? argv[1] : "tmp/frames";
    const tc::Index nx = argc > 2 ? tc::Index(std::atoi(argv[2])) : 160, ny = nx;
    const tc::Real dx = 15000.0, dy = 15000.0;          // 15 km cells
    const tc::Real g = 9.81, H = 1000.0, f0 = 1.0e-4;
    const tc::Real A = 1.0, Lbump = 180000.0;

    tc::Arena arena(160u << 20);

    // circular wet mask (a disc of radius Rbasin), allocated from the arena FIRST
    tc::Field2 mask = arena.alloc2d(nx, ny);
    const tc::Real cx = 0.5 * nx * dx, cy = 0.5 * ny * dy;
    const tc::Real Rbasin = 0.46 * nx * dx;
    tc::CartesianMesh geom(nx, ny, dx, dy, f0, /*beta*/0.0);
    tc::for_each_cell(nx, ny, [=](tc::Index i, tc::Index j) {
        const tc::Real x = geom.x(tc::Loc::Center, i, j), y = geom.y(tc::Loc::Center, i, j);
        const tc::Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        mask[i, j] = r < Rbasin ? tc::Real(1) : tc::Real(0);
    });
    tc::MaskedMesh mesh(geom, mask);

    const tc::Real c  = std::sqrt(g * H);
    const tc::Real dt = 0.35 * dx / c;
    tc::Params p{ .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .g = g, .H = H };
    tc::OceanCore<tc::MaskedMesh, tc::PpmContinuity, tc::SadournyEnstrophy,
                  tc::FvPgf, tc::WallBC, tc::SSPRK3> core(mesh, arena, p);
    core.init();

    // IC: bump near the EAST coast, at rest
    const tc::Field2 eta = core.state().eta, u = core.state().u, v = core.state().v;
    const tc::Real xb = cx + 0.62 * Rbasin, yb = cy;
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
        [=](tc::Index i, tc::Index j) {
            const tc::Real x = geom.x(tc::Loc::Center, i, j), y = geom.y(tc::Loc::Center, i, j);
            const tc::Real r2 = ((x - xb) * (x - xb) + (y - yb) * (y - yb)) / (Lbump * Lbump);
            eta[i, j] = H + A * std::exp(-r2);
        });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace), [=](tc::Index i, tc::Index j) { u[i, j] = 0; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace), [=](tc::Index i, tc::Index j) { v[i, j] = 0; });

    const int nsteps = 1100, every = 7, ppx = 4;
    const tc::Real vis = 0.5 * A;
    const int W = int(nx) * ppx, Hh = int(ny) * ppx;
    std::vector<unsigned char> img(std::size_t(W) * Hh * 3);

    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % every == 0) {
            const tc::Field2 e = core.state().eta;
            for (int py = 0; py < Hh; ++py)
                for (int qx = 0; qx < W; ++qx) {
                    const tc::Index i = qx / ppx, j = py / ppx;
                    unsigned char r, gg, b;
                    if (mask[i, j] < tc::Real(0.5)) { r = gg = b = 48; }         // land
                    else colormap((e[i, j] - H) / vis, r, gg, b);
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
        if (n % (every * 12) == 0) {
            const tc::Field2 e = core.state().eta;
            tc::Real mass = 0, energy = 0, amax = 0;
            for (tc::Index j = 0; j < ny; ++j)
                for (tc::Index i = 0; i < nx; ++i)
                    if (mask[i, j] > tc::Real(0.5)) {
                        const tc::Real ar = mesh.area(tc::Loc::Center, i, j), an = e[i, j] - H;
                        mass += e[i, j] * ar; energy += tc::Real(0.5) * g * an * an * ar;
                        if (std::abs(an) > amax) amax = std::abs(an);
                    }
            std::fprintf(stderr, "t=%6.0fs  mass=%.9e  PE=%.4e  |anom|max=%.4f m\n",
                double(n) * dt, double(mass), double(energy), double(amax));
        }
        core.step();
    }
    std::printf("circular basin: wrote %d frames (%dx%d) to %s\n", frame, W, Hh, outdir.c_str());
    return 0;
}
