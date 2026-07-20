// =============================================================================
// tests/test_netcdf.cpp — the RAII NetCDF wrapper (io/netcdf.hpp). Compiled to a
// no-op unless TC_HAVE_NETCDF is set (CMake -DTC_NETCDF=ON). Round-trips a small
// file: define dims (incl. UNLIMITED time) + vars + CF attrs → write two time
// records of a column-major (nx,ny) field into a C-order (time,y,x) var → read
// back and verify the layout maps with NO transpose (the whole point).
// =============================================================================

#include <doctest/doctest.h>

#ifdef TC_HAVE_NETCDF
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "io/netcdf.hpp"
#include "io/ocean_output.hpp"
#include "lib/arena.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "physics/layered_state.hpp"
#include "numerics/parallel.hpp"

TEST_CASE("netcdf: RAII define/write/read round-trip, layout-preserving") {
    std::system("mkdir -p tmp");
    const char* path = "tmp/test_nc_roundtrip.nc";
    const int nx = 8, ny = 5, nrec = 2;
    auto val = [](int i, int j, int rec) { return double(i) + 10.0 * j + 100.0 * rec; };

    // ── write ──────────────────────────────────────────────────────────────────
    {
        auto f = tc::nc::File::create(path);
        const int dt = f.def_unlimited("time");          // C-order: time, y, x
        const int dy = f.def_dim("y", ny);
        const int dx = f.def_dim("x", nx);
        const int vx = f.def_var<double>("x", {dx});
        const int vt = f.def_var<double>("time", {dt});
        const int ve = f.def_var<double>("eta", {dt, dy, dx});
        f.att(ve, "units", "m");
        f.att(ve, "long_name", "surface height");
        f.att(ve, "coordinates", "x y");
        f.global_att("Conventions", "CF-1.11");
        f.global_att("source", "TurboChook test");
        f.enddef();

        std::vector<double> x(nx);
        for (int i = 0; i < nx; ++i) x[i] = 100.0 * i;
        f.put_all(vx, x.data());

        std::vector<double> eta(std::size_t(nx) * ny);    // column-major (x fastest), our Field layout
        for (int rec = 0; rec < nrec; ++rec) {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) eta[std::size_t(i) + std::size_t(j) * nx] = val(i, j, rec);
            f.put_time(vt, std::size_t(rec), 3600.0 * rec);
            f.put<double>(ve, {std::size_t(rec), 0, 0}, {1, std::size_t(ny), std::size_t(nx)}, eta.data());
        }
    }  // RAII closes the file here

    // ── read back ────────────────────────────────────────────────────────────────
    {
        auto f = tc::nc::File::open(path);
        CHECK(f.dimlen("x") == std::size_t(nx));
        CHECK(f.dimlen("y") == std::size_t(ny));
        CHECK(f.dimlen("time") == std::size_t(nrec));

        std::vector<double> eta(std::size_t(nrec) * ny * nx);
        f.get_all(f.varid("eta"), eta.data());            // stored (time,y,x) row-major
        for (int rec = 0; rec < nrec; ++rec)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    CHECK(eta[std::size_t(rec) * ny * nx + std::size_t(j) * nx + i] == doctest::Approx(val(i, j, rec)));
    }
    std::remove(path);
}

TEST_CASE("netcdf: a failed op throws tc::Error (fail-loud)") {
    CHECK_THROWS_AS(tc::nc::File::open("tmp/does_not_exist_12345.nc"), tc::Error);
}

TEST_CASE("netcdf: OceanOutput writes a two-layer state (faces→centres, CF, per-layer)") {
    using tc::Index;
    const Index nx = 10, ny = 6;
    tc::CartesianMesh mesh(nx, ny, 1000.0, 1000.0);
    tc::Arena arena(8u << 20);
    auto s = tc::allocate_layered_state<2>(arena, mesh);
    // h_l = 100·(l+1)+i ;  u_l = 0.1·(l+1)+0.01·i ;  v = 0
    tc::for_each_cell(mesh.extent_x(tc::Loc::Center), mesh.extent_y(tc::Loc::Center),
        [=](Index i, Index j) { s.layer[0].eta[i, j] = 100.0 + i; s.layer[1].eta[i, j] = 200.0 + i; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::XFace), mesh.extent_y(tc::Loc::XFace),
        [=](Index i, Index j) { s.layer[0].u[i, j] = 0.1 + 0.01 * i; s.layer[1].u[i, j] = 0.2 + 0.01 * i; });
    tc::for_each_cell(mesh.extent_x(tc::Loc::YFace), mesh.extent_y(tc::Loc::YFace),
        [=](Index i, Index j) { s.layer[0].v[i, j] = 0; s.layer[1].v[i, j] = 0; });

    std::system("mkdir -p tmp");
    const char* path = "tmp/test_ocean.nc";
    { tc::OceanOutput<2> out(path, mesh, "m", "m");
      out.write(s, 0.0);
      out.write(s, 3600.0);
      CHECK(out.records() == 2); }                             // RAII closes here

    auto f = tc::nc::File::open(path);
    CHECK(f.dimlen("time") == 2); CHECK(f.dimlen("layer") == 2);
    CHECK(f.dimlen("x") == std::size_t(nx)); CHECK(f.dimlen("y") == std::size_t(ny));

    std::vector<double> h(std::size_t(2) * 2 * ny * nx), u(std::size_t(2) * 2 * ny * nx);
    f.get_all(f.varid("h"), h.data());
    f.get_all(f.varid("u"), u.data());
    auto at = [&](int rec, int l, int j, int i) { return ((std::size_t(rec) * 2 + l) * ny + j) * nx + i; };
    for (int l = 0; l < 2; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                CHECK(h[at(0, l, j, i)] == doctest::Approx(100.0 * (l + 1) + i));            // thickness stored as-is
                CHECK(u[at(0, l, j, i)] == doctest::Approx(0.1 * (l + 1) + 0.01 * i + 0.005)); // face→centre avg
            }
    std::remove(path);
}
#endif  // TC_HAVE_NETCDF
