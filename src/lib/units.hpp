#pragma once
// =============================================================================
// lib/units.hpp — dimensional units + conversion (host-side metadata).
//
// The model computes in SI; units are a PRESENTATION concern applied at the
// boundary (a NetCDF attribute, a Python display), NEVER threaded through kernels
// (no unit-carrying arithmetic — that's the mp-units road the dep policy forbids).
// This provides the canonical vocabulary so a Quantity emits a valid CF/UDUNITS
// string, plus an optional Dim-checked convert() for the terminal reporter. If the
// consumer is Python, conversion is free there (UDUNITS/pint) — this just guarantees
// the canonical string is right.
//
// TODO(M4): flesh out the catalog of Units + convert(); today the Dim enum + shape.
// rakali north-star: src/framework/rki_units.F90
// =============================================================================

#include "core/types.hpp"

namespace tc {

// The physical dimension a quantity carries — the axis unit conversion is valid
// within (converting across dimensions is an error, not a silent rescale).
enum class Dim {
    Dimensionless, Length, Speed, Mass, Time,
    Temperature, Salinity, Density, Energy, Vorticity, Pressure
};

// A unit as an affine map to SI: x_si = x * to_si + offset (offset for °C↔K etc.).
struct Unit { const char* symbol; Real to_si; Real offset; Dim dim; };

// TODO(M4): inline Real convert(Real x, Unit from, Unit to) — assert from.dim==to.dim
//           (throw tc::Error on mismatch), then (x*from.to_si + from.offset) mapped
//           back through `to`. Plus a small catalog (M_PER_S, KNOT, METRE, KM, ...).

} // namespace tc
