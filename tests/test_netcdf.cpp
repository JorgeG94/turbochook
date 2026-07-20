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
#endif  // TC_HAVE_NETCDF
