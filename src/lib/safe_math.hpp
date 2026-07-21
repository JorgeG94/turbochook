#pragma once
// =============================================================================
// lib/safe_math.hpp — guarded numeric helpers (host AND device).
//
// Small, device-callable helpers used across the operators: NaN-safe division
// (0/0 → 0, the free-slip / dry-cell idiom), vanishing-layer guards (ALE tolerates
// h→0 from the start — DESIGN ADR-6), clamped roots. These are called INSIDE kernels,
// so they are pure `inline` and header-visible (STATUS #7 — a device helper hidden in
// a .cpp fails to link under nvc++).
//
// TODO(M4): grow as remap / layered-PGF need them (pos_sqrt, clamped ratios).
// rakali north-star: src/framework/rki_safe_math.F90
// =============================================================================

#include "core/types.hpp"

namespace tc {

// device-callable ⇒ inline, header-visible.
inline Real safe_div(Real a, Real b) { return b != Real(0) ? a / b : Real(0); }

// Minimum layer thickness — a layer thinner than this is treated as vanished (ALE).
inline constexpr Real H_VANISHED = Real(1e-10);

// TODO(M4): inline Real pos_sqrt(Real x) { return x > 0 ? std::sqrt(x) : 0; }  (+ <cmath>)

} // namespace tc
