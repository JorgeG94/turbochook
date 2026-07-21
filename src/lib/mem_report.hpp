#pragma once
// =============================================================================
// lib/mem_report.hpp — arena / allocation accounting.
//
// Walk the Arena (and, later, the field registry) to report bytes used vs capacity,
// the high-water mark, and a per-field breakdown — the memory tracking the god-state /
// handle exists to enable (answers a future tc_memory(handle)). The Arena already
// tracks total bytes; the per-Quantity breakdown comes from iterating the registry
// (the same list that drives I/O). Host-side.
//
// TODO(Later): report_memory(const Arena&) → a formatted block; registry-driven
//              per-field sizes.
// rakali north-star: src/framework/rki_mem_report.F90
// =============================================================================

#include "lib/arena.hpp"

namespace tc {

// TODO(Later): void report_memory(const Arena& a);  // bytes used/capacity/high-water
//              + per-Quantity breakdown via the diagnostics Registry.

} // namespace tc
