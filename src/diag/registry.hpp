#pragma once
// =============================================================================
// diag/registry.hpp — the diagnostics registry (DESIGN ADR-8, the revision).
//
// A diagnostic = a Quantity + a host-erased OUTCOME closure. THE RULE: erase the
// host-side result, NEVER the integrand. Inside `eval`/`fill`, a CONCRETE monomorphic
// integrand meets the device reduce (reduce.hpp) — the std::function is called on the
// host and never crosses to a kernel, so the prime directive holds. Runtime (not a
// compile-time tuple) because name lookup — `diagnostic("KE")`, config-driven output,
// the Python path — needs a name-indexed list. `State` and `M` are fixed per run.
//
// The Registry is a HOST-side list the Core owns; the sinks (Reporter, OceanOutput)
// iterate it generically. ONE list, three consumers: sum bytes (memory accounting),
// write NetCDF (I/O), lookup-by-name (Python). Variables ≡ diagnostics here (a
// prognostic is a FieldDiag whose fill is identity/interp).
//
// TODO(M4): populate default_diagnostics(); refactor Reporter + OceanOutput to iterate
//           this instead of hardcoding total_mass/… and h/u/v/zeta/ubar.
// rakali north-star: src/core/ocean/diag/rki_ocean_diag.F90, rki_ocean_diag_derived.F90,
//                    rki_ocean_budgets.F90 (budgets = a set of ScalarDiags)
// =============================================================================

#include <functional>
#include <string_view>
#include <utility>
#include <vector>
#include "core/types.hpp"
#include "diag/quantity.hpp"

namespace tc {

// 0-D → Reporter / log. eval launches a device reduce internally and returns one scalar.
template <class State, class M>
struct ScalarDiag {
    Quantity q;
    std::function<Real(const State&, const M&)> eval;
};

// ≥1-D → NetCDF. fill writes the (rank-shaped) result into a host buffer the sink owns.
// rank 2 = field (nx·ny), rank 1 = profile (ny) — the sink sizes the buffer from q.loc.
template <class State, class M>
struct FieldDiag {
    Quantity q;
    int      rank;
    std::function<void(const State&, const M&, Real*)> fill;
};

template <class State, class M>
struct Registry {
    std::vector<ScalarDiag<State, M>> scalars;
    std::vector<FieldDiag<State, M>>  fields;

    void scalar(Quantity q, std::function<Real(const State&, const M&)> f) {
        scalars.push_back({q, std::move(f)});
    }
    void field(Quantity q, int rank, std::function<void(const State&, const M&, Real*)> f) {
        fields.push_back({q, rank, std::move(f)});
    }
    // TODO(M4): Real value(std::string_view sym, const State&, const M&) — name lookup
    //           over `scalars` for the diagnostic("KE") / Python path.
};

// TODO(M4): template <int NL, class M> Registry<LayeredState<NL>, M> default_diagnostics();
//   scalars: Q_MASS → global_integral(m,[=]{ h[i,j] }); Q_KE → ∫½h|u|²; Q_SPEED → global_max.
//   fields:  Q_H (identity), Q_U/Q_V (face→centre), Q_ZETA (corner curl → centre).
//   Each authored with the reduce verb + integrand bound together (erase the outcome).

} // namespace tc
