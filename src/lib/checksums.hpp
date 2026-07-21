#pragma once
// =============================================================================
// lib/checksums.hpp — field checksums (host-side, debugging / reproducibility).
//
// Reduce a field to a stable digest so two runs can be compared: host vs GPU, or
// before vs after a refactor (did this change the answer bit-for-bit?). An
// order-INDEPENDENT reduce (so the parallel and serial digests agree) — e.g. a sum of
// per-cell bit-casts, or a commutative hash. Host-side, called at cadence (like the
// diagnostics), never per step.
//
// TODO(Later): checksum(Field2) — order-independent digest; a diff harness across builds.
// rakali north-star: src/framework/rki_checksums.F90
// =============================================================================

#include <cstdint>
#include "core/types.hpp"

namespace tc {

// TODO(Later): std::uint64_t checksum(Field2 f, Index nx, Index ny);
//              (order-independent so gpu/host/multicore digests match.)

} // namespace tc
