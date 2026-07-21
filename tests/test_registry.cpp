// =============================================================================
// tests/test_registry.cpp — the diagnostics Registry (DESIGN ADR-8 rev).
// Proves: default_diagnostics wraps the reductions faithfully (behaviour-preserving),
// name lookup + fail-loud, runtime extensibility, and the FieldDiag fill mechanism.
// Host-serial (par → seq); no doctest main (test_m0.cpp owns it).
// =============================================================================

#include <doctest/doctest.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
#include "core/types.hpp"
#include "lib/arena.hpp"
#include "numerics/parallel.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/state/layered_state.hpp"
#include "diag/diagnostics.hpp"     // total_mass/ke/speed + default_diagnostics
#include "diag/registry.hpp"
#include "diag/quantity.hpp"

namespace {
using tc::Index; using tc::Loc; using tc::Field2; using tc::Real;

// A 2-layer state: layer 0 = (η=5, u=1, v=0), layer 1 = (η=3, u=0, v=0).
void fill_two_layer(tc::LayeredState<2> s, const tc::CartesianMesh& m) {
    auto set = [&](Field2 f, Loc l, Real val) {
        tc::for_each_cell(m.extent_x(l), m.extent_y(l), [=](Index i, Index j) { f[i, j] = val; });
    };
    set(s.layer[0].eta, Loc::Center, 5.0); set(s.layer[0].u, Loc::XFace, 1.0); set(s.layer[0].v, Loc::YFace, 0.0);
    set(s.layer[1].eta, Loc::Center, 3.0); set(s.layer[1].u, Loc::XFace, 0.0); set(s.layer[1].v, Loc::YFace, 0.0);
}
} // namespace

TEST_CASE("Registry: default_diagnostics wraps the reductions faithfully") {
    tc::CartesianMesh m(4, 3, 2.0, 2.0);
    tc::Arena arena(1u << 20);
    auto s = tc::allocate_layered_state<2>(arena, m);
    fill_two_layer(s, m);

    const auto reg = tc::default_diagnostics<2, tc::CartesianMesh>();

    // behaviour-preserving: the registry value == the direct reduction it wraps
    CHECK(reg.value("mass",  s, m) == doctest::Approx(tc::total_mass(s, m)));
    CHECK(reg.value("KE",    s, m) == doctest::Approx(tc::total_ke(s, m)));
    CHECK(reg.value("speed", s, m) == doctest::Approx(tc::max_speed(s, m)));

    // anchor to a hand value: mass = Σ_layers Σ_cells η·area = (5+3)·(4·3)·(2·2) = 384
    CHECK(reg.value("mass", s, m) == doctest::Approx((5.0 + 3.0) * 4 * 3 * (2.0 * 2.0)));
    // |u|max: layer 0 centre speed = 1 (u=1,v=0); layer 1 = 0  ⇒  max = 1
    CHECK(reg.value("speed", s, m) == doctest::Approx(1.0));
}

TEST_CASE("Registry: unknown scalar name fails loud") {
    tc::CartesianMesh m(4, 3, 2.0, 2.0);
    tc::Arena arena(1u << 20);
    auto s = tc::allocate_layered_state<2>(arena, m);
    fill_two_layer(s, m);
    const auto reg = tc::default_diagnostics<2, tc::CartesianMesh>();
    CHECK_THROWS_AS(reg.value("not_a_diagnostic", s, m), std::runtime_error);
}

TEST_CASE("Registry: extensible — add a scalar by lambda, retrieve it by name") {
    tc::CartesianMesh m(4, 3, 2.0, 2.0);
    tc::Arena arena(1u << 20);
    auto s = tc::allocate_layered_state<2>(arena, m);
    fill_two_layer(s, m);

    auto reg = tc::default_diagnostics<2, tc::CartesianMesh>();
    reg.scalar(tc::Quantity{"answer", "1", "", Loc::Center, tc::Dim::Dimensionless},
               [](const tc::LayeredState<2>& st, const tc::CartesianMesh& mm) {
                   (void)st; (void)mm; return Real(42);
               });
    CHECK(reg.value("answer", s, m) == doctest::Approx(42.0));
    CHECK(reg.value("mass",   s, m) == doctest::Approx(tc::total_mass(s, m)));  // originals still there
}

TEST_CASE("Quantity catalog: find_quantity resolves symbols, nullptr for unknown") {
    CHECK(tc::find_quantity("KE")   != nullptr);
    CHECK(tc::find_quantity("KE")->dim == tc::Dim::Energy);
    CHECK(tc::find_quantity("zeta")->loc == Loc::Corner);
    CHECK(tc::find_quantity("nope") == nullptr);
}

TEST_CASE("Registry: FieldDiag fill writes the expected centre buffer") {
    tc::CartesianMesh m(4, 3, 2.0, 2.0);
    tc::Arena arena(1u << 20);
    auto s = tc::allocate_layered_state<2>(arena, m);
    fill_two_layer(s, m);

    tc::Registry<tc::LayeredState<2>, tc::CartesianMesh> reg;
    reg.field(tc::Q_H, 2, [](const tc::LayeredState<2>& st, const tc::CartesianMesh& mm, Real* out) {
        const Field2 h = st.layer[0].eta;                     // layer-0 thickness → centre buffer
        for (Index j = 0; j < mm.ny(); ++j)
            for (Index i = 0; i < mm.nx(); ++i)
                out[std::size_t(i) + std::size_t(j) * mm.nx()] = h[i, j];
    });

    std::vector<Real> buf(std::size_t(m.nx()) * m.ny(), Real(-1));
    reg.fields[0].fill(s, m, buf.data());
    CHECK(buf.front() == doctest::Approx(5.0));
    CHECK(buf.back()  == doctest::Approx(5.0));   // η=5 everywhere in layer 0
    CHECK(reg.fields[0].q.symbol == std::string("h"));
}
