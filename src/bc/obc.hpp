#pragma once
// =============================================================================
// bc/obc.hpp — open boundary conditions.
//
// Let waves and flow LEAVE the domain (radiation: Flather for the barotropic mode,
// Orlanski/characteristic for the baroclinic) and prescribe INCOMING data (tides,
// nesting from a parent model). Distinct from wall/periodic, which are CLOSED
// topology — an OBC is an open edge coupled to external boundary data. Needed for
// limited-area / coastal / nested configs, not the idealized channel runs.
//
// TODO(Later): Flather barotropic OBC + baroclinic radiation; a boundary-data feed.
// rakali north-star: src/core/ocean/boundary/rki_ocean_obc.F90, rki_ocean_obc_baroclinic.F90,
//                    rki_ocean_boundary_data.F90, rki_ocean_boundary_types.F90
// =============================================================================

#include "core/types.hpp"

namespace tc {

// TODO(Later): FlatherOBC / RadiationOBC on the boundary axis, consuming a
//              BoundaryData feed (prescribed η, transports, tracers at the edge).

} // namespace tc
