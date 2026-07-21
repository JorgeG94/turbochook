#pragma once
// =============================================================================
// api/handle.hpp — the host-side solver interface behind the opaque handle.
//
// ISolver is a HOST virtual base. Virtual is legal HERE (unlike in a kernel): it is
// called once per STEP from the driver, never per cell — one indirection in front of a
// step that launches dozens of kernels. Concrete templated Cores (SplitCore<NL,…>) sit
// behind it via SolverImpl<Core>; a factory maps a RunConfig → the right compile-time
// specialization. This is the ERASED middle tier of the handle:
//     void* (ABI, Python)  →  ISolver* (host)  →  SplitCore<…> (monomorphic, device).
// The middle tier is only needed when schemes are chosen at RUNTIME (e.g. from Python);
// a single build config can cast void*→Core directly.
//
// TODO(Later): flesh ISolver + SolverImpl<Core> + make_solver(RunConfig).
// rakali north-star: src/api/rki_handle.F90, src/api/rki_api.F90
// =============================================================================

#include <string_view>
#include "core/types.hpp"

namespace tc {

struct ISolver {
    virtual ~ISolver() = default;
    virtual void step() = 0;
    virtual Real diagnostic(std::string_view name) const = 0;               // registry scalar by symbol
    virtual void snapshot(std::string_view name, Real* host_out) const = 0; // registry field → host buffer
    // TODO(Later): set_ic(field, host_in); run(days, cadence); memory();
    //              extents(name) so Python can size numpy buffers.
};

// TODO(Later): template <class Core> struct SolverImpl final : ISolver { Core core; ... };
//              std::unique_ptr<ISolver> make_solver(const RunConfig& cfg);   // the factory (switch over tags)

} // namespace tc
