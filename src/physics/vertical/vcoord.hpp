#pragma once
// =============================================================================
// physics/vertical/vcoord.hpp — the generalized vertical coordinate (the GVC policy seam).
//
// The DUAL-NATURE axis (the one place noun/verb genuinely blurs): the coordinate
// CHOICE is a policy (this file); the coordinate SURFACES are prognostic STATE (the
// layer thicknesses in LayeredState). A Vcoord defines the TARGET grid the ALE remap
// (remap.hpp) restores each cadence, plus the z-from-h diagnostic. Siblings:
//   Isopycnal — today's pure-Lagrangian layered core (density-class layers, no remap).
//   Zstar     — z* (η-following), Sigma (terrain-following), HybridALE (blend).
//
// Named-but-empty until GVC lands (ROADMAP M4). The seam exists now so z*/ALE drop in
// as a policy on SplitCore<NL, Vcoord, …> without restructuring — DESIGN ADR-6.
//
// TODO(M4): Vcoord concept (target thickness per column) + Isopycnal; then Zstar/ALE.
// rakali north-star: src/ALE/rki_vcoord.F90, src/core/ocean/vcoord/rki_ocean_vcoord.F90
// =============================================================================

#include <concepts>
#include "core/types.hpp"

namespace tc {

template <class V>
concept Vcoord = requires(const V v) {
    { v.is_lagrangian() } -> std::convertible_to<bool>;
    // TODO(M4): { v.target_thickness(col_state, k) } -> Real;  // the grid remap restores
};

struct Isopycnal {
    // Pure-Lagrangian: layers ARE the coordinate, no remap between them.
    bool is_lagrangian() const { return true; }
};

// TODO(M4): struct Zstar { bool is_lagrangian(){return false;} Real target_thickness(...); };
//           struct HybridALE { ... };

static_assert(Vcoord<Isopycnal>);

} // namespace tc
