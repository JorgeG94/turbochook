#pragma once
// =============================================================================
// diag/quantity.hpp — canonical metadata for a physical quantity (host-side).
//
// A Quantity is the typo-proof IDENTITY a diagnostic/variable carries into output:
// the symbol (its NetCDF var name = the cross-language id), CF units + standard_name,
// the staggering Loc it lives on, and its Dim. A small, EXTENSIBLE catalog covers the
// common quantities (add a line, no central enum to edit); ad-hoc quantities construct
// inline. This catalog is the vocabulary a (future) Python config selects diagnostics
// by — `diagnostics=["mass","KE","zeta"]` are catalog symbols.
//
// For OUTPUT there is no variable/diagnostic distinction: h/u/v are Quantities with an
// identity/interp integrand; zeta is a Quantity with a curl integrand. Same metadata,
// different closure (see registry.hpp).
//
// TODO(M4): grow the catalog as operators land (tracers S/T, EOS ρ, vmix outputs).
// rakali north-star: src/core/ocean/diag/rki_ocean_diag.F90, src/framework/rki_units.F90
// =============================================================================

#include <string_view>
#include "core/types.hpp"
#include "mesh/mesh.hpp"     // Loc
#include "lib/units.hpp"     // Dim

namespace tc {

struct Quantity {
    const char* symbol;         // NetCDF var name — the stable cross-language id
    const char* cf_units;       // canonical CF/UDUNITS string, e.g. "m s-1"
    const char* standard_name;  // CF standard_name (or "")
    Loc         loc;            // staggering it lives on (drives extent + weight)
    Dim         dim;
};

// ── the catalog — extensible: a new user adds one line, no closed enum ──
inline constexpr Quantity Q_MASS  {"mass",  "kg",     "sea_water_mass",               Loc::Center, Dim::Mass};
inline constexpr Quantity Q_KE    {"KE",    "J",      "",                             Loc::Center, Dim::Energy};
inline constexpr Quantity Q_SPEED {"speed", "m s-1",  "sea_water_speed",              Loc::Center, Dim::Speed};
inline constexpr Quantity Q_H     {"h",     "m",      "cell_thickness",               Loc::Center, Dim::Length};
inline constexpr Quantity Q_U     {"u",     "m s-1",  "eastward_sea_water_velocity",  Loc::XFace,  Dim::Speed};
inline constexpr Quantity Q_V     {"v",     "m s-1",  "northward_sea_water_velocity", Loc::YFace,  Dim::Speed};
inline constexpr Quantity Q_ZETA  {"zeta",  "s-1",    "ocean_relative_vorticity",     Loc::Corner, Dim::Vorticity};

// host-side lookup by symbol (defined in quantity.cpp — the .cpp seam). The (future)
// Python name path resolves "KE" → this Quantity. Returns nullptr for ad-hoc/unknown.
const Quantity* find_quantity(std::string_view symbol);

} // namespace tc
