#pragma once
// =============================================================================
// lib/config.hpp — the run configuration (code-first) + named presets.
//
// One struct describing a run: grid, domain, physical params, scheme selections (as
// tags mapping to the compile-time policies), time control, output cadence, and which
// diagnostics to emit (catalog symbols). Presets (bc_inst, double_gyre,
// geostrophic_adjustment) are functions returning a filled RunConfig — the demos build
// one in code today; a (future) Python/JSON front-end would populate the same struct,
// then the factory (api/handle.hpp) maps its scheme tags → a Core specialization.
//
// TODO(M4): flesh out fields + presets; define the scheme-tag enums the factory reads.
// rakali north-star: src/core/rki_config.F90, rki_config_schema.F90, rki_nml_schema.F90
// =============================================================================

#include <string>
#include <vector>
#include "core/types.hpp"

namespace tc {

struct RunConfig {
    Index nx{128}, ny{128};
    Real  days{40};
    Real  dt{0};                 // 0 → CFL-derived from the external gravity-wave speed
    int   msub{24};              // barotropic substeps (SplitCore)
    // TODO(M4): domain (lon/lat box OR L in metres), g + std::array H/rho per layer,
    //           scheme tags (continuity/coriolis/pgf/vcoord/bc/integrator),
    //           output cadence, std::vector<std::string> diagnostics (catalog symbols).
    std::vector<std::string> diagnostics{"mass", "KE", "speed"};
};

// TODO(M4): RunConfig preset_bc_inst();  RunConfig preset_double_gyre();
//           RunConfig preset_geostrophic_adjustment();

} // namespace tc
