#pragma once
// =============================================================================
// io/restart.hpp — checkpoint / restart.
//
// Write the FULL prognostic state — every layer's (h, u, v), tracers, and coordinate
// surfaces — to NetCDF and read it back EXACTLY, so a long integration resumes without
// drift. Distinct from io/ocean_output.hpp, which is LOSSY (faces averaged to centres,
// optional float) for visualization — restart is exact, on the native STAGGERED grids.
// Iterates the field registry (every prognostic Quantity), like the other registry
// consumers.
//
// TODO(Later): write_restart / read_restart over the registry; native staggered layout;
//              round-trip test (write → read → bitwise-equal state).
// rakali north-star: src/core/ocean/io/rki_ocean_restart.F90, rki_ocean_restart_io.F90
// =============================================================================

#include "core/types.hpp"

namespace tc {

// TODO(Later): template <int NL, Mesh M>
//   void write_restart(std::string_view path, const LayeredState<NL>& s, const M& mesh);
//   void read_restart (std::string_view path, LayeredState<NL>& s, const M& mesh);
//   (exact staggered layout — no centre interpolation, unlike OceanOutput.)

} // namespace tc
