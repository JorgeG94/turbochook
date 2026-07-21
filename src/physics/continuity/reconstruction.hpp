#pragma once
// =============================================================================
// physics/continuity/reconstruction.hpp — the reconstruction policy axis.
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

// van-Leer / monotonized-central limited slope over (a,b,c) centred on b:
// sign(Δc)·min(|Δc|, 2|Δl|, 2|Δr|), zero at a local extremum. (rki_continuity.F90
// `ppm_limited_slope`.) Shared by the wall family.
constexpr Real vanleer_slope(Real a, Real b, Real c) {
    const Real dl = b - a, dr = c - b;
    if (dl * dr <= Real(0)) return Real(0);              // extremum → flatten
    const Real dc  = Real(0.5) * (dl + dr);
    const Real adl = dl < Real(0) ? -dl : dl;
    const Real adr = dr < Real(0) ? -dr : dr;
    Real mag = dc < Real(0) ? -dc : dc;                  // |Δc|
    if (Real(2) * adl < mag) mag = Real(2) * adl;
    if (Real(2) * adr < mag) mag = Real(2) * adr;
    return dc < Real(0) ? -mag : mag;                    // sign(mag, Δc)
}

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
    // The real PPM body (translated from rki_continuity.F90): van-Leer edge slopes,
    // Colella–Woodward edge values (CW84 eq 1.6), then the CW monotonicity limiter
    // (eq 1.10). Returns a parabola whose cell mean is exactly `hc` (mass-preserving).
    // Window w = {h[i-2..i+2]}, cell at w[2]. (Positivity limiter is a later opt-in.)
    static constexpr Poly<2> reconstruct(std::array<Real, 5> w) {
        const Real hc    = w[2];
        const Real dh_m1 = vanleer_slope(w[0], w[1], w[2]);
        const Real dh_0  = vanleer_slope(w[1], w[2], w[3]);
        const Real dh_p1 = vanleer_slope(w[2], w[3], w[4]);
        Real hL = Real(0.5) * (w[1] + hc) - (dh_0  - dh_m1) / Real(6);   // west edge
        Real hR = Real(0.5) * (hc + w[3]) - (dh_p1 - dh_0 ) / Real(6);   // east edge

        // CW84 monotonicity limiter (uses the pre-limit edges).
        const Real dlr = hR - hL;
        const Real h6  = Real(6) * (hc - Real(0.5) * (hL + hR));
        if ((hR - hc) * (hc - hL) <= Real(0)) { hL = hc; hR = hc; }      // extremum → flat
        else if (dlr * h6 >  dlr * dlr) hL = Real(3) * hc - Real(2) * hR;  // left overshoot
        else if (dlr * h6 < -dlr * dlr) hR = Real(3) * hc - Real(2) * hL;  // right overshoot

        // Parabola from (hc, hL, hR); mean over ξ∈[0,1] is hc by construction.
        const Real a6 = Real(6) * (hc - Real(0.5) * (hL + hR));
        return {{ hL, (hR - hL) + a6, -a6 }};
    }
};

// ── exact polynomial reconstruction from cell averages (shared by PQM + WENO) ─────
// The unique degree-(K-1) polynomial whose average over cell j — the unit interval
// [off0+j, off0+j+1] — equals s[j]. KEY POINT: the moment matrix A depends only on the
// stencil GEOMETRY (off0), not the data — so the reconstruction is a LINEAR map a=A⁻¹·s.
// We solve A (Gauss–Jordan) at COMPILE TIME to bake A⁻¹ / the face-value weight vectors —
// literally the constants rakali hand-tunes — and leave only a dot product at runtime.
// (A compile-time zero pivot would be a constexpr divide-by-zero = a hard build error, not
// UB, so the solve is self-checking. rki_kernel_remap.F90 / rki_ml_tracers_weno.F90.)
namespace detail {
constexpr Real sq(Real x)   { return x * x; }
constexpr Real aabs(Real x) { return x < Real(0) ? -x : x; }
constexpr Real ipow(Real b, int e) { Real r = Real(1); for (int k = 0; k < e; ++k) r *= b; return r; }

// A[r][c] = ∫_{off0+r}^{off0+r+1} ξ^c dξ — geometry only, no data.
template <int K>
constexpr std::array<std::array<Real, K>, K> moment_matrix(int off0) {
    std::array<std::array<Real, K>, K> A{};
    for (int r = 0; r < K; ++r) { const int o = off0 + r;
        for (int c = 0; c < K; ++c) A[r][c] = (ipow(Real(o + 1), c + 1) - ipow(Real(o), c + 1)) / Real(c + 1);
    }
    return A;
}
// Gauss–Jordan solve M·x = rhs (no pivot; M is a well-conditioned moment matrix).
template <int K>
constexpr std::array<Real, K> gj_solve(std::array<std::array<Real, K>, K> M, std::array<Real, K> rhs) {
    for (int k = 0; k < K; ++k) {
        const Real piv = M[k][k];
        for (int j = k; j < K; ++j) M[k][j] /= piv;
        rhs[k] /= piv;
        for (int r = 0; r < K; ++r) if (r != k) { const Real f = M[r][k];
            for (int j = k; j < K; ++j) M[r][j] -= f * M[k][j];
            rhs[r] -= f * rhs[k];
        }
    }
    return rhs;
}
// The K weights c with cᵀ·s = the exact reconstruction evaluated at x. Since a value
// v(x)=xvecᵀ·a=xvecᵀ·A⁻¹·s=(A⁻ᵀ·xvec)ᵀ·s, we have c=A⁻ᵀ·xvec ⇒ solve Aᵀ c = xvec.
template <int K>
constexpr std::array<Real, K> face_weights(int off0, Real x) {
    const auto A = moment_matrix<K>(off0);
    std::array<std::array<Real, K>, K> At{};
    for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) At[i][j] = A[j][i];
    std::array<Real, K> xv{}; Real xp = Real(1);
    for (int c = 0; c < K; ++c) { xv[c] = xp; xp *= x; }
    return gj_solve<K>(At, xv);
}
// The K WENO candidate weight-rows: candidate r reconstructs the centre cell's right
// edge (x=1) from its K-cell sub-stencil (off0 = r-(K-1)). A K×K table — geometry only.
template <int K>
constexpr std::array<std::array<Real, K>, K> weno_weight_table() {
    std::array<std::array<Real, K>, K> C{};
    for (int r = 0; r < K; ++r) C[r] = face_weights<K>(r - (K - 1), Real(1));
    return C;
}
// A⁻¹ (geometry only) — the full-poly reconstruction PQM needs: coeffs a = A⁻¹·s.
template <int K>
constexpr std::array<std::array<Real, K>, K> inv_moment(int off0) {
    const auto A = moment_matrix<K>(off0);
    std::array<std::array<Real, K>, K> inv{};
    for (int col = 0; col < K; ++col) {
        std::array<Real, K> e{}; e[col] = Real(1);
        const auto x = gj_solve<K>(A, e);                // A·x = e_col ⇒ x = col of A⁻¹
        for (int r = 0; r < K; ++r) inv[r][col] = x[r];
    }
    return inv;
}
// WENO-Z convex combine: αr = dr·(1 + (τ/(ε+βr))²), ω = α/Σα, value = Σ ωr·pr.
template <int NC>
constexpr Real weno_combine(const Real (&p)[NC], const Real (&beta)[NC], const Real (&d)[NC], Real tau) {
    constexpr Real EPS = Real(1e-36);
    Real a[NC]{}; Real asum = Real(0);
    for (int r = 0; r < NC; ++r) { const Real t = tau / (EPS + beta[r]); a[r] = d[r] * (Real(1) + t * t); asum += a[r]; }
    Real f = Real(0);
    for (int r = 0; r < NC; ++r) f += a[r] / asum * p[r];
    return f;
}
} // namespace detail

struct Pqm {                                   // piecewise quartic, 5th order
    static constexpr ReconKind kind   = ReconKind::Wall;
    static constexpr int       radius = 3;
    static constexpr int       order  = 4;
    // The exact degree-4 reconstruction from the 5 central cells (w[1..5], centre w[3]),
    // in the centre cell's ξ∈[0,1] frame — mass-preserving (mean ≡ w[3]) by construction.
    // A light extremum flatten (→ PCM at a local max/min) is the safety limiter; rakali's
    // full column-coupled White–Adcroft IH4IH3 limiter is the REMAP-path variant (it
    // solves a tridiagonal over the whole column — belongs in physics/vertical/remap.hpp,
    // not this per-cell horizontal interface). (rki_kernel_remap.F90 remap_column_pqm.)
    static constexpr Poly<4> reconstruct(std::array<Real, 7> w) {
        const Real hc = w[3];
        if ((hc - w[2]) * (w[4] - hc) <= Real(0)) return {{ hc, 0, 0, 0, 0 }};   // extremum → flat
        constexpr auto Ainv = detail::inv_moment<5>(-2);         // A⁻¹ baked at COMPILE time
        const std::array<Real, 5> s{ w[1], w[2], w[3], w[4], w[5] };
        std::array<Real, 5> a{};                                 // runtime: a = A⁻¹·s (a matrix-vector, no solve)
        for (int i = 0; i < 5; ++i) { Real acc = Real(0);
            for (int j = 0; j < 5; ++j) acc += Ainv[i][j] * s[j]; a[i] = acc; }
        return {{ a[0], a[1], a[2], a[3], a[4] }};
    }
};

// ── The face family (WENO — nonlinear pointwise; bodies are seams) ───────────────

// WENO-Z (Borges 2008 τ-weights) over the classic Jiang–Shu (WENO5) / Balsara–Shu
// (WENO7) smoothness indicators, with rakali's simplified 3-term β for WENO9. Each
// candidate value is the EXACT reconstruction of the centre cell's downwind edge from
// its k-cell sub-stencil (detail::exact_poly at ξ=1); the σ→0 point value of rakali's
// swept-average face state. `Bias::Left` = the mirror (reverse the window, reconstruct
// the right edge). (rki_ml_tracers_weno.F90 weno{5,7,9}_face_swept.)

struct Weno5 {                                 // 5-point, 5th order — WENO-Z
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 2;
    static constexpr Real reconstruct(std::array<Real, 5> w, Bias bias) {
        if (bias == Bias::Left) w = { w[4], w[3], w[2], w[1], w[0] };
        constexpr auto C = detail::weno_weight_table<3>();       // candidate coeffs, baked at COMPILE time
        Real p[3];
        for (int r = 0; r < 3; ++r) p[r] = C[r][0]*w[r] + C[r][1]*w[r+1] + C[r][2]*w[r+2];
        const Real b[3] = {
            Real(13)/12 * detail::sq(w[0] - 2*w[1] + w[2]) + Real(1)/4 * detail::sq(w[0] - 4*w[1] + 3*w[2]),
            Real(13)/12 * detail::sq(w[1] - 2*w[2] + w[3]) + Real(1)/4 * detail::sq(w[1] - w[3]),
            Real(13)/12 * detail::sq(w[2] - 2*w[3] + w[4]) + Real(1)/4 * detail::sq(3*w[2] - 4*w[3] + w[4]),
        };
        const Real d[3] = { Real(1)/10, Real(6)/10, Real(3)/10 };
        return detail::weno_combine<3>(p, b, d, detail::aabs(b[0] - b[2]));
    }
};

struct Weno7 {                                 // 7-point, 7th order — WENO-Z
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 3;
    static constexpr Real reconstruct(std::array<Real, 7> w, Bias bias) {
        if (bias == Bias::Left) w = { w[6], w[5], w[4], w[3], w[2], w[1], w[0] };
        constexpr auto C = detail::weno_weight_table<4>();       // candidate coeffs, baked at COMPILE time
        Real p[4];
        for (int r = 0; r < 4; ++r) { Real acc = Real(0); for (int j = 0; j < 4; ++j) acc += C[r][j]*w[r+j]; p[r] = acc; }
        const Real b[4] = {   // Balsara–Shu (2000) smoothness indicators
            w[0]*(547*w[0] - 3882*w[1] + 4642*w[2] - 1854*w[3]) + w[1]*(7043*w[1] - 17246*w[2] + 7042*w[3]) + w[2]*(11003*w[2] - 9402*w[3]) + 2107*detail::sq(w[3]),
            w[1]*(267*w[1] - 1642*w[2] + 1602*w[3] - 494*w[4]) + w[2]*(2843*w[2] - 5966*w[3] + 1922*w[4]) + w[3]*(3443*w[3] - 2522*w[4]) + 547*detail::sq(w[4]),
            w[2]*(547*w[2] - 2522*w[3] + 1922*w[4] - 494*w[5]) + w[3]*(3443*w[3] - 5966*w[4] + 1602*w[5]) + w[4]*(2843*w[4] - 1642*w[5]) + 267*detail::sq(w[5]),
            w[3]*(2107*w[3] - 9402*w[4] + 7042*w[5] - 1854*w[6]) + w[4]*(11003*w[4] - 17246*w[5] + 4642*w[6]) + w[5]*(7043*w[5] - 3882*w[6]) + 547*detail::sq(w[6]),
        };
        const Real d[4] = { Real(1)/35, Real(12)/35, Real(18)/35, Real(4)/35 };
        return detail::weno_combine<4>(p, b, d, detail::aabs(b[0] - b[3]));
    }
};

struct Weno9 {                                 // 9-point, 9th order — WENO-Z
    static constexpr ReconKind kind   = ReconKind::Face;
    static constexpr int       radius = 4;
    static constexpr Real reconstruct(std::array<Real, 9> w, Bias bias) {
        if (bias == Bias::Left) { std::array<Real, 9> r{}; for (int i = 0; i < 9; ++i) r[i] = w[8 - i]; w = r; }
        constexpr auto C = detail::weno_weight_table<5>();       // candidate coeffs, baked at COMPILE time
        Real p[5];
        for (int r = 0; r < 5; ++r) { Real acc = Real(0); for (int j = 0; j < 5; ++j) acc += C[r][j]*w[r+j]; p[r] = acc; }
        Real b[5];            // rakali's simplified 3-term β, centred on w[r+2]
        for (int r = 0; r < 5; ++r) {
            const int c = r + 2;
            b[r] = Real(13)/12 * detail::sq(w[c-1] - 2*w[c] + w[c+1])
                 + Real(1)/4  * detail::sq(w[c-1] - w[c+1])
                 + Real(1)/80 * detail::sq(w[c-2] - 4*w[c-1] + 6*w[c] - 4*w[c+1] + w[c+2]);
        }
        const Real d[5] = { Real(1)/126, Real(20)/126, Real(60)/126, Real(40)/126, Real(10)/126 };
        return detail::weno_combine<5>(p, b, d, detail::aabs(b[0] - b[4]));
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

// The candidate weights ARE computed at compile time and match the textbook constants:
// WENO5's centred candidate (cells {i,i+1,i+2}, right edge) is (2,5,-1)/6. If this line
// compiles, the Gauss–Jordan ran in the compiler and produced rakali's hand-tuned numbers.
namespace detail_test {
inline constexpr auto C5 = tc::detail::weno_weight_table<3>();
static_assert(tc::detail::aabs(C5[2][0] - tc::Real(1)/3) < tc::Real(1e-12) &&   //  2/6
              tc::detail::aabs(C5[2][1] - tc::Real(5)/6) < tc::Real(1e-12) &&   //  5/6
              tc::detail::aabs(C5[2][2] + tc::Real(1)/6) < tc::Real(1e-12),     // -1/6
              "WENO5 centred candidate weights must be (2,5,-1)/6");
}

} // namespace tc
