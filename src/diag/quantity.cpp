// =============================================================================
// diag/quantity.cpp — host-side Quantity lookup (the .cpp seam in miniature).
//
// This is pure HOST code — it never enters a kernel — so it lives in a .cpp,
// compiled ONCE (not re-instantiated in every device TU) and skipping the nvc++
// device pipeline. It's the flagship of the header/.cpp split: device integrands +
// reduce verbs stay header-visible (STATUS #7), host metadata/lookup comes here.
//
// TODO(M4): add Dim-checked convert() (units.hpp) for the terminal reporter.
// rakali north-star: src/framework/rki_units.F90
// =============================================================================

#include "diag/quantity.hpp"
#include <array>
#include <string_view>

namespace tc {

namespace {
// The registered catalog, in one place. Adding a Quantity above → add it here.
constexpr std::array CATALOG{
    &Q_MASS, &Q_KE, &Q_SPEED, &Q_H, &Q_U, &Q_V, &Q_ZETA,
};
} // namespace

const Quantity* find_quantity(std::string_view symbol) {
    for (const Quantity* q : CATALOG)
        if (symbol == q->symbol) return q;
    return nullptr;   // ad-hoc / unknown symbol — caller carries inline metadata
}

} // namespace tc
