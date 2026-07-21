// =============================================================================
// tests/test_reconstruction.cpp — the reconstruction policy axis, focused on the
// real PPM body (van-Leer slopes + CW84 edges + monotonicity limiter). Oracles:
// constant→flat, linear→exact edges, local extremum→flattened, and the parabola's
// cell mean is exactly hc (mass-preserving) for every case.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <array>
#include <vector>
#include <cmath>
#include <cstddef>
#include "core/types.hpp"
#include "physics/continuity/reconstruction.hpp"

using tc::Real;

namespace {
// exact cell-average of a polynomial (monomial coeffs `c`) over [a, a+1], + point value.
double cellavg(const std::vector<double>& c, double a) {
    double s = 0;
    for (std::size_t k = 0; k < c.size(); ++k)
        s += c[k] * (std::pow(a + 1, double(k + 1)) - std::pow(a, double(k + 1))) / double(k + 1);
    return s;
}
double polyval(const std::vector<double>& c, double x) {
    double s = 0, xp = 1;
    for (double ck : c) { s += ck * xp; xp *= x; }
    return s;
}
// an N-cell window of cell-averages of the polynomial `c`, with the CENTRE cell at [0,1].
template <std::size_t N>
std::array<Real, N> poly_window(const std::vector<double>& c) {
    std::array<Real, N> w{};
    const int R = int(N) / 2;
    for (int k = 0; k < int(N); ++k) w[k] = Real(cellavg(c, double(k - R)));
    return w;
}
} // namespace

TEST_CASE("Poly<2> primitives: eval / edges / mean") {
    tc::Poly<2> p{{ Real(1), Real(2), Real(3) }};   // 1 + 2ξ + 3ξ²
    CHECK(p.at_left()  == doctest::Approx(1.0));                // ξ=0
    CHECK(p.at_right() == doctest::Approx(6.0));                // 1+2+3
    CHECK(p.eval(0.5)  == doctest::Approx(1 + 1 + 0.75));       // 2.75
    // ∫₀¹ (1+2ξ+3ξ²) = 1 + 1 + 1 = 3
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(3.0));
}

TEST_CASE("Ppm: constant field reconstructs flat") {
    auto p = tc::Ppm::reconstruct({ Real(3), Real(3), Real(3), Real(3), Real(3) });
    CHECK(p.at_left()  == doctest::Approx(3.0));
    CHECK(p.at_right() == doctest::Approx(3.0));
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(3.0));
}

TEST_CASE("Ppm: linear field recovered exactly (edges hc∓b/2, mean hc)") {
    // h[i] = i  → slope b=1, cell mean hc=2 at the centre.
    auto p = tc::Ppm::reconstruct({ Real(0), Real(1), Real(2), Real(3), Real(4) });
    CHECK(p.at_left()  == doctest::Approx(1.5));   // hc - b/2
    CHECK(p.at_right() == doctest::Approx(2.5));   // hc + b/2
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(2.0));
}

TEST_CASE("Ppm: local maximum is flattened (monotonicity)") {
    auto p = tc::Ppm::reconstruct({ Real(0), Real(0), Real(1), Real(0), Real(0) });
    CHECK(p.at_left()  == doctest::Approx(1.0));   // extremum → h_L=h_R=hc
    CHECK(p.at_right() == doctest::Approx(1.0));
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(1.0));
}

TEST_CASE("Ppm: local minimum is flattened") {
    auto p = tc::Ppm::reconstruct({ Real(2), Real(2), Real(1), Real(2), Real(2) });
    CHECK(p.at_left()  == doctest::Approx(1.0));
    CHECK(p.at_right() == doctest::Approx(1.0));
}

TEST_CASE("Ppm: smooth monotone profile stays monotone + mean-preserving") {
    // strictly increasing → edges bracket the centre, no new extrema, mean = hc
    const Real hm1 = 2, hc = 4, hp1 = 7;
    auto p = tc::Ppm::reconstruct({ Real(1), hm1, hc, hp1, Real(11) });
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(hc));       // mass-preserving
    CHECK(p.at_left()  <= doctest::Approx(hc));                // increasing ⇒ hL ≤ hc ≤ hR
    CHECK(p.at_right() >= doctest::Approx(hc));
    CHECK(p.at_left()  >= doctest::Approx(hm1));               // no overshoot below neighbour
    CHECK(p.at_right() <= doctest::Approx(hp1));               // no overshoot above neighbour
}

TEST_CASE("Pcm / Plm sanity") {
    // PCM: the cell value is the whole profile.
    auto pc = tc::Pcm::reconstruct({ Real(5) });
    CHECK(pc.at_left()  == doctest::Approx(5.0));
    CHECK(pc.at_right() == doctest::Approx(5.0));
    // PLM on a linear field → exact edges, mean preserved.
    auto pl = tc::Plm::reconstruct({ Real(1), Real(2), Real(3) });
    CHECK(pl.at_left()  == doctest::Approx(1.5));
    CHECK(pl.at_right() == doctest::Approx(2.5));
    CHECK(pl.mean_over(0.0, 1.0) == doctest::Approx(2.0));
}

// ── PQM (piecewise quartic) ──────────────────────────────────────────────────────
TEST_CASE("Pqm: exact for a quartic (monotone centre) — edges + mean-preserving") {
    // f = 10x + x⁴ ; the centre cell [0,1] is monotone (not a local extremum) → the
    // unlimited exact quartic is returned, so it must reproduce f to machine precision.
    const std::vector<double> c{ 0, 10, 0, 0, 1 };
    auto p = tc::Pqm::reconstruct(poly_window<7>(c));
    CHECK(p.at_left()  == doctest::Approx(polyval(c, 0.0)));            // f(0) = 0
    CHECK(p.at_right() == doctest::Approx(polyval(c, 1.0)));            // f(1) = 11
    CHECK(p.mean_over(0.0, 1.0) == doctest::Approx(cellavg(c, 0.0)));   // mass-preserving
}
TEST_CASE("Pqm: constant flat, and a local extremum is flattened to PCM") {
    auto pc = tc::Pqm::reconstruct({ Real(4),Real(4),Real(4),Real(4),Real(4),Real(4),Real(4) });
    CHECK(pc.at_left()  == doctest::Approx(4.0));
    CHECK(pc.at_right() == doctest::Approx(4.0));
    auto pe = tc::Pqm::reconstruct({ Real(0),Real(0),Real(0),Real(1),Real(0),Real(0),Real(0) });
    CHECK(pe.at_left()  == doctest::Approx(1.0));                       // extremum → flat
    CHECK(pe.at_right() == doctest::Approx(1.0));
    CHECK(pe.mean_over(0.0, 1.0) == doctest::Approx(1.0));
}

// ── WENO5/7/9 (face family) ──────────────────────────────────────────────────────
// Each WENO(2k-1) is EXACT for degree k-1 (all candidates reproduce the polynomial),
// which validates the candidate reconstruction coefficients end to end.
TEST_CASE("Weno5 exact for a quadratic (both biases)") {
    const std::vector<double> q{ 2, -1, 0.5 };
    auto w = poly_window<5>(q);
    CHECK(tc::Weno5::reconstruct(w, tc::Bias::Right) == doctest::Approx(polyval(q, 1.0)));  // cell centre's right edge
    CHECK(tc::Weno5::reconstruct(w, tc::Bias::Left)  == doctest::Approx(polyval(q, 0.0)));  // left edge (mirror)
}
TEST_CASE("Weno7 exact for a cubic (both biases)") {
    const std::vector<double> c{ 1, 2, -0.5, 0.3 };
    auto w = poly_window<7>(c);
    CHECK(tc::Weno7::reconstruct(w, tc::Bias::Right) == doctest::Approx(polyval(c, 1.0)));
    CHECK(tc::Weno7::reconstruct(w, tc::Bias::Left)  == doctest::Approx(polyval(c, 0.0)));
}
TEST_CASE("Weno9 exact for a quartic (both biases)") {
    const std::vector<double> c{ 0.5, 1, -0.2, 0.1, 0.05 };
    auto w = poly_window<9>(c);
    CHECK(tc::Weno9::reconstruct(w, tc::Bias::Right) == doctest::Approx(polyval(c, 1.0)));
    CHECK(tc::Weno9::reconstruct(w, tc::Bias::Left)  == doctest::Approx(polyval(c, 0.0)));
}
TEST_CASE("Weno: constant → constant, and a monotone step stays in-bounds (non-oscillatory)") {
    CHECK(tc::Weno5::reconstruct({ Real(3),Real(3),Real(3),Real(3),Real(3) }, tc::Bias::Right) == doctest::Approx(3.0));
    CHECK(tc::Weno9::reconstruct({ Real(2),Real(2),Real(2),Real(2),Real(2),Real(2),Real(2),Real(2),Real(2) },
                                 tc::Bias::Right) == doctest::Approx(2.0));
    const Real r = tc::Weno5::reconstruct({ Real(0),Real(0),Real(0),Real(1),Real(1) }, tc::Bias::Right);
    CHECK(r >= Real(-1e-9));            // no undershoot below the stencil min
    CHECK(r <= Real(1) + Real(1e-9));   // no overshoot above the stencil max
}
