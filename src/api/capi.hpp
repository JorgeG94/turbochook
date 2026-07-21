#pragma once
// =============================================================================
// api/capi.hpp — the extern "C" ABI (the PySCF-style handle boundary).
//
// A thin C surface over ISolver: an OPAQUE handle Python (ctypes) holds, plus named
// queries. Opaque outside, monomorphic inside — the outer wall of the host/device
// firewall. The shared MANAGED Arena means numpy in/out is a plain copy, no
// marshalling and no data clauses. Deliberately a plain extern "C" .so from nvc++ (NOT
// OpenACC, NOT a binding framework — the "thin C-ABI bridge" the dep policy prefers,
// same as the NetCDF layer). ctypes wraps it into `import turbochook`.
//
// TODO(Later): implement in capi.cpp over make_solver(); a CMake .so target + a pure-
//              Python ctypes wrapper. Only build once the runtime factory exists.
// rakali north-star: src/api/rki_api.F90, src/api/rki_api_unstr.F90
// =============================================================================

extern "C" {

// The opaque handle Python holds:  typedef void* tc_handle;
//
// TODO(Later):
//   void*  tc_create(const char* config_json);   // build a solver from a RunConfig
//   void   tc_step(void* h);                      // advance one outer step
//   double tc_diagnostic(void* h, const char* name);        // registry scalar
//   void   tc_snapshot (void* h, const char* name, double* out);  // registry field → caller buffer
//   void   tc_set_ic   (void* h, const char* field, const double* in);
//   double tc_memory   (void* h);                 // arena high-water bytes
//   void   tc_destroy  (void* h);

} // extern "C"
