// =============================================================================
// tests/test_reconstruction.cpp — the reconstruction policy axis, focused on the
// real PPM body (van-Leer slopes + CW84 edges + monotonicity limiter). Oracles:
// constant→flat, linear→exact edges, local extremum→flattened, and the parabola's
// cell mean is exactly hc (mass-preserving) for every case.
// Host-serial; no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <array>
#include "core/types.hpp"
#include "physics/continuity/reconstruction.hpp"

using tc::Real;

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
