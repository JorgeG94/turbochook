#pragma once
// =============================================================================
// physics/reconstruction.hpp — the reconstruction policy axis.
//
// The payoff of compile-time policies: ONE generic flux/advection kernel,
// templated over HOW it reconstructs the sub-cell profile — each scheme a policy
// the compiler inlines with zero cost. Without them you'd duplicate the kernel per
// scheme or eat a runtime branch; here it's a single template parameter.
//
// TWO KINDS, ONE FAMILY (settled in design chat). The schemes do not share a
// single interface, because they have different natural loci:
//
//   • WALL reconstruction (PCM/PLM/PPM/PQM) lives ON THE CELL: it builds a
//     per-cell sub-profile and hands back a `Poly<order>` (coeffs on ξ∈[0,1]).
//     The CONSUMER samples it however it needs — `at_left()`/`at_right()` for an
//     edge state, `mean_over(a,b)` for a swept-flux integral. One reconstruction,
//     many samplings.
//   • FACE reconstruction (WENO5/7/9) lives ON THE FACE: a nonlinear convex
//     combination of smoothness-weighted candidate stencils yields ONE biased
//     pointwise value per `Bias`. There is no single polynomial to return.
//
// So there are two sibling concepts — `WallReconstruction` and `FaceReconstruction`
// — each with its OWN algorithm signature, plus a disjoint umbrella `Reconstructor`
// (a tagged union: Wall OR Face, never a fat interface requiring both). Each scheme
// models exactly one kind and carries a `kind` tag (for the config-string → type
// bridge, ADR-4) and `radius` (stencil half-width → drives `nghost`).
//
// Consumers constrain on the KIND they actually consume: continuity's swept flux
// takes `WallReconstruction`; tracer advection (planned, WENO) will take
// `FaceReconstruction`. Code that must accept EITHER kind uses the `Reconstructor`
// umbrella and branches once on `Scheme::kind` — that both-kinds consumer arrives with
// tracer advection, not before.
//
// STATUS: this is the STRUCTURE (concepts + types + tagged seams). PCM/PLM are real
// bodies; PPM/PQM/WENO bodies are TODO(M2/later) but return correctly-shaped values
// so the concepts and static_asserts are load-bearing today. A slope/monotonicity
// LIMITER becomes an orthogonal policy on the wall family later (e.g. Plm<MinMod>).
// =============================================================================

#include <array>
#include <concepts>
#include "core/types.hpp"

namespace tc {

// Which locus a scheme reconstructs at — the tag that makes `Reconstructor` a
// tagged union rather than a fat interface, and feeds the config → type registry.
enum class ReconKind { Wall, Face };

// For a face reconstruction: which side the biased stencil leans (WENO does one
// per side to get the two states at a face).
enum class Bias { Left, Right };

// ── Poly<Order> — the wall family's return: the reconstructed sub-cell profile as
// monomial coefficients on the reference cell ξ∈[0,1]. Decouples RECONSTRUCTION
// (build the profile once) from SAMPLING (the consumer reads edges or a swept
// integral). All-constexpr → header-visible, device-inlinable (kernel rules).
template <int Order>
struct Poly {
    std::array<Real, Order + 1> c{};                    // c[0] + c[1]ξ + … + c[Order]ξ^Order

    constexpr Real eval(Real xi) const {                // Horner
        Real acc = Real(0);
        for (int k = Order; k >= 0; --k) acc = acc * xi + c[k];
        return acc;
    }
    constexpr Real at_left()  const { return c[0]; }    // ξ=0 (the cell's left wall)
    constexpr Real at_right() const { return eval(Real(1)); }  // ξ=1 (right wall)

    // Mean of the profile over ξ∈[a,b] — the swept-flux primitive (PPM advection
    // integrates the upwind cell's parabola over the swept fraction). Caller
    // guarantees b>a.
    constexpr Real mean_over(Real a, Real b) const {
        Real integ = Real(0), apow = a, bpow = b;       // a^{k+1}, b^{k+1}
        for (int k = 0; k <= Order; ++k) {
            integ += c[k] / Real(k + 1) * (bpow - apow);
            apow *= a; bpow *= b;
        }
        return integ / (b - a);
    }
};

// ── The two sibling concepts. Each requires its OWN signature; the nested `kind`
// requirement keeps the split honest (a wall scheme does NOT satisfy the face
// concept and vice-versa). Missing `radius` → substitution failure → not a model.
template <class Scheme>
concept WallReconstruction = requires(std::array<Real, 2 * Scheme::radius + 1> w) {
    requires Scheme::kind == ReconKind::Wall;
    { Scheme::radius } -> std::convertible_to<int>;
    { Scheme::order  } -> std::convertible_to<int>;
    { Scheme::reconstruct(w) } -> std::same_as<Poly<Scheme::order>>;
};

template <class Scheme>
concept FaceReconstruction = requires(std::array<Real, 2 * Scheme::radius + 1> w, Bias b) {
    requires Scheme::kind == ReconKind::Face;
    { Scheme::radius } -> std::convertible_to<int>;
    { Scheme::reconstruct(w, b) } -> std::same_as<Real>;
};

// The umbrella: "is a reconstruction scheme." A disjoint tagged union — satisfied
// by EITHER kind, forcing NEITHER interface on the other. Used only by code that
// is deliberately blind to the kind (the future tracer-advection consumer).
template <class Scheme>
concept Reconstructor = WallReconstruction<Scheme> || FaceReconstruction<Scheme>;

// ── The wall family (PCM/PLM real; PPM/PQM correctly-shaped seams) ───────────────

struct Pcm {                                   // piecewise constant, 1st order
    static constexpr ReconKind kind   = ReconKind::Wall;
    static constexpr int       radius = 0;
    static constexpr int       order  = 0;
    // The cell average IS the whole profile.
    static constexpr Poly<0> reconstruct(std::array<Real, 1> w) { return {{w[0]}}; }
};

struct Plm {                                   // piecewise linear, 2nd order
    static constexpr ReconKind kind   = ReconKind::Wall;
    static constexpr int       radius = 1;
    static constexpr int       order  = 1;
    // minmod-limited slope; profile preserves the cell mean w[1] over ξ∈[0,1]:
    //   a(ξ) = (w1 - s/2) + s·ξ   ⇒   walls w1 ∓ s/2, mean w1.
    static constexpr Poly<1> reconstruct(std::array<Real, 3> w) {
        const Real s = minmod(w[1] - w[0], w[2] - w[1]);
        return {{w[1] - Real(0.5) * s, s}};
    }
    static constexpr Real minmod(Real a, Real b) {
        if (a > Real(0) && b > Real(0)) return a < b ? a : b;
        if (a < Real(0) && b < Real(0)) return a > b ? a : b;
        return Real(0);
    }
};

struct Ppm {                                   // piecewise parabolic, 3rd order (M2 target)
    static constexpr ReconKind kind   = ReconKind::Wall;
    static constexpr int       radius = 2;
    static constexpr int       order  = 2;
    // TODO(M2): the real body — 4-point edge interpolation
    //   a_{i+1/2} = (7(a_i+a_{i+1}) − (a_{i-1}+a_{i+2}))/12, then the
    //   Colella–Woodward monotonicity limiter + a positivity limiter → (aL,aR,a6),
    //   which map to monomial coeffs {aL, (aR−aL)+a6, −a6}. Shape is settled; the
    //   coefficient body + its lake-at-rest / mass-conservation tests are the M2
    //   work. Until then, PCM-fallback so the concept stays satisfied.
    static constexpr Poly<2> reconstruct(std::array<Real, 5> w) {
        return {{w[2], Real(0), Real(0)}};
    }
};

struct Pqm {                                   // piecewise quartic, 5th order — seam
    static constexpr ReconKind kind   = ReconKind::Wall;
    static constexpr int       radius = 3;
    static constexpr int       order  = 4;
    // TODO(later): White–Adcroft edge values + edge slopes → quartic (two-pass).
    static constexpr Poly<4> reconstruct(std::array<Real, 7> w) {
        return {{w[3], Real(0), Real(0), Real(0), Real(0)}};
    }
};

// ── The face family (WENO — nonlinear pointwise; bodies are seams) ───────────────

struct Weno5 {                                 // 5-point, 5th order — seam
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 2;
    // TODO(later): 3 candidate stencils, smoothness indicators βk, nonlinear
    // weights → the biased face value. Positivity-preserving flavour (DESIGN §6).
    static constexpr Real reconstruct(std::array<Real, 5> w, Bias b) {
        (void)b; return w[radius];
    }
};

struct Weno7 {                                 // 7-point, 7th order — seam
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 3;
    static constexpr Real reconstruct(std::array<Real, 7> w, Bias b) {
        (void)b; return w[radius];
    }
};

struct Weno9 {                                 // 9-point, 9th order — seam
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 4;
    static constexpr Real reconstruct(std::array<Real, 9> w, Bias b) {
        (void)b; return w[radius];
    }
};

// The split is real and honest: wall schemes model ONLY the wall concept, face
// schemes model ONLY the face concept, and both satisfy the umbrella.
static_assert(WallReconstruction<Pcm> && WallReconstruction<Plm> &&
              WallReconstruction<Ppm> && WallReconstruction<Pqm>);
static_assert(FaceReconstruction<Weno5> && FaceReconstruction<Weno7> &&
              FaceReconstruction<Weno9>);
static_assert(Reconstructor<Ppm> && Reconstructor<Weno5>);
static_assert(!FaceReconstruction<Ppm> && !WallReconstruction<Weno5>);

} // namespace tc
