// src/reco_demo.cpp — dummy-but-real usage of the reconstruction policy axis.
// A top-level src/*.cpp, so it builds as its own executable in every config; run
// the host build with `./build-host/reco_demo`.
//
// A 1D constant-advection flux, GENERIC over the reconstruction scheme. The whole
// point of the axis: the assembly code is written ONCE; the scheme is a template
// parameter. Swapping Pcm -> Plm -> Ppm changes one token, not the algorithm.

#include <array>
#include <span>
#include <vector>
#include <cstdio>

#include "physics/reconstruction.hpp"   // brings in core/types.hpp (tc::Real) too

// ── A generic flux assembly over a WALL reconstruction ───────────────────────────
// Constant advection with u > 0, so the upwind cell of each face is its LEFT
// neighbour. Face `face` sits between cell (face-1) and cell (face).
//
// The design payoff in one place: we reconstruct the upwind cell's sub-cell profile
// ONCE (a `Poly`), then SAMPLE it two different ways —
//    • at_right()      → the pointwise edge value at the face          (ξ = 1)
//    • mean_over(a,b)  → the mean over the swept fraction of the cell  (PPM advection)
// One reconstruction, many samplings — that's why the wall family returns a Poly,
// not a bare edge value.
template <tc::WallReconstruction Scheme>
void advect_faces(std::span<const tc::Real> q, tc::Real cfl,
                  std::span<tc::Real> flux_point, std::span<tc::Real> flux_swept) {
    constexpr int rad = Scheme::radius;
    for (std::size_t face = rad + 1; face + rad < q.size(); ++face) {
        const std::size_t c = face - 1;                    // upwind cell (u > 0)

        std::array<tc::Real, 2 * rad + 1> w{};             // gather the stencil window
        for (int s = -rad; s <= rad; ++s) w[s + rad] = q[c + s];

        const tc::Poly<Scheme::order> prof = Scheme::reconstruct(w); // reconstruct ONCE …
        flux_point[face] = prof.at_right();                // … sample as a pointwise edge …
        flux_swept[face] = prof.mean_over(tc::Real(1) - cfl, tc::Real(1));  // … and as a swept mean
    }
}

template <tc::WallReconstruction Scheme>
void report_wall(const char* name, std::span<const tc::Real> q, tc::Real cfl) {
    std::vector<tc::Real> fp(q.size(), 0), fs(q.size(), 0);
    advect_faces<Scheme>(q, cfl, fp, fs);
    // Report face #4 (between cells 3 and 4) — it lies in the interior of ALL these
    // schemes (even PQM, whose radius=3 needs 3 ghosts each side). For q[i]=i the
    // exact edge value there is 3.5. Note fp/fs are 0 outside a scheme's interior:
    // `radius` is exactly what sets those bounds (and the field's ghost width).
    std::printf("  %-4s (radius=%d order=%d)  edge@face4 = %.3f   swept@face4 = %.3f\n",
                name, Scheme::radius, Scheme::order, fp[4], fs[4]);
}

// ── A generic assembly over a FACE reconstruction (WENO) ─────────────────────────
// Different natural shape: no polynomial, just a biased pointwise value per side.
template <tc::FaceReconstruction Scheme>
void report_face(const char* name, std::span<const tc::Real> q) {
    constexpr int rad = Scheme::radius;
    const std::size_t face = 4;
    std::array<tc::Real, 2 * rad + 1> w{};
    for (int s = -rad; s <= rad; ++s) w[s + rad] = q[face + s];

    const tc::Real qL = Scheme::reconstruct(w, tc::Bias::Left);   // left-biased state
    const tc::Real qR = Scheme::reconstruct(w, tc::Bias::Right);  // right-biased state
    std::printf("  %-6s (radius=%d)  qL@face4 = %.3f   qR@face4 = %.3f\n",
                name, Scheme::radius, qL, qR);
}

int main() {
    // A linear field q[i] = i. Exact edge value at face #4 (between cells 3,4) = 3.5.
    // PLM recovers a linear profile EXACTLY (edge = 3.5); PCM is piecewise-flat so it
    // returns the upwind cell mean (3.0) — the first-order error is visible.
    std::vector<tc::Real> q(8);
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = tc::Real(i);
    const tc::Real cfl = 0.4;

    std::puts("WALL family — same assembly, scheme swapped by one template argument:");
    report_wall<tc::Pcm>("PCM", q, cfl);
    report_wall<tc::Plm>("PLM", q, cfl);
    report_wall<tc::Ppm>("PPM", q, cfl);   // seam: PCM-fallback until the M2 body lands
    report_wall<tc::Pqm>("PQM", q, cfl);   // seam

    // report_wall<tc::Weno5>(...);  // ← would NOT compile: Weno5 models FaceReconstruction,
    //                                  not WallReconstruction. The concept rejects it.

    std::puts("\nFACE family — biased pointwise states (WENO), a separate concept:");
    report_face<tc::Weno5>("WENO5", q);
    report_face<tc::Weno7>("WENO7", q);    // seams: currently return the cell centre (=4.000)

    return 0;
}
