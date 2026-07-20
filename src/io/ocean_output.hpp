#pragma once
// =============================================================================
// io/ocean_output.hpp — a CF-1.11 NetCDF writer for the layered ocean state, on the
// RAII nc::File. Follows rakali's conventions:
//   • faces → CENTRES before writing (u,v averaged to cell centres) — every field on
//     the single (x,y) centre grid, no stagger dims;
//   • dims (time-UNLIMITED, layer, y, x); records appended per write();
//   • CF-1.11 global + per-var attrs (units / long_name / coordinates);
//   • layout_left (nx,ny) buffer → C-order (…,y,x) slab with NO transpose.
// Precision `Prec` is the on-disk dtype (default double = model precision, à la
// rakali; use float to halve the file — future FP32-tracer story). Host-side, called
// at the OUTPUT cadence (never per step); loops the managed state on the host.
// Gated by the includer under TC_HAVE_NETCDF.
// =============================================================================

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "physics/layered_state.hpp"
#include "io/netcdf.hpp"

namespace tc {

template <int NL, class Prec = double>
class OceanOutput {
    nc::File    f_;
    int         vtime_ = -1, vh_ = -1, vu_ = -1, vv_ = -1, vzeta_ = -1;
    std::size_t rec_ = 0;
    Index       nx_ = 0, ny_ = 0;
    std::vector<Prec> buf_;          // (nx·ny) centre-grid scratch (interp + precision cast)
    std::vector<Real> corner_;       // (nx+1)·(ny+1) corner ζ scratch (a derived-field diagnostic)

public:
    template <Mesh M>
    OceanOutput(std::string_view path, const M& mesh,
                std::string_view xunits = "m", std::string_view yunits = "m")
        : nx_(mesh.nx()), ny_(mesh.ny()), buf_(std::size_t(mesh.nx()) * mesh.ny()),
          corner_(std::size_t(mesh.nx() + 1) * (mesh.ny() + 1)) {
        f_ = nc::File::create(path);
        const int dt = f_.def_unlimited("time");             // C-order dims: time, layer, y, x
        const int dl = f_.def_dim("layer", NL);
        const int dy = f_.def_dim("y", std::size_t(ny_));
        const int dx = f_.def_dim("x", std::size_t(nx_));

        const int vx = f_.def_var<double>("x", {dx});
        const int vy = f_.def_var<double>("y", {dy});
        vtime_ = f_.def_var<double>("time", {dt});
        f_.att(vx, "units", xunits); f_.att(vx, "axis", "X"); f_.att(vx, "long_name", "x coordinate");
        f_.att(vy, "units", yunits); f_.att(vy, "axis", "Y"); f_.att(vy, "long_name", "y coordinate");
        f_.att(vtime_, "units", "seconds since simulation start"); f_.att(vtime_, "axis", "T");

        vh_    = f_.def_var<Prec>("h", {dt, dl, dy, dx});
        vu_    = f_.def_var<Prec>("u", {dt, dl, dy, dx});
        vv_    = f_.def_var<Prec>("v", {dt, dl, dy, dx});
        vzeta_ = f_.def_var<Prec>("zeta", {dt, dl, dy, dx});
        f_.att(vh_,    "units", "m");     f_.att(vh_,    "long_name", "layer thickness");    f_.att(vh_,    "coordinates", "x y");
        f_.att(vu_,    "units", "m s-1"); f_.att(vu_,    "long_name", "eastward velocity");  f_.att(vu_,    "coordinates", "x y");
        f_.att(vv_,    "units", "m s-1"); f_.att(vv_,    "long_name", "northward velocity"); f_.att(vv_,    "coordinates", "x y");
        f_.att(vzeta_, "units", "s-1");   f_.att(vzeta_, "long_name", "relative vorticity"); f_.att(vzeta_, "coordinates", "x y");
        f_.att(vzeta_, "standard_name", "ocean_relative_vorticity");
        f_.global_att("Conventions", "CF-1.11");
        f_.global_att("source", "TurboChook");
        f_.global_att("title", "Layered shallow-water ocean state");
        f_.enddef();

        std::vector<double> x(nx_), y(ny_);                  // 1-D coords from the centre grid
        for (Index i = 0; i < nx_; ++i) x[i] = mesh.x(Loc::Center, i, 0);
        for (Index j = 0; j < ny_; ++j) y[j] = mesh.y(Loc::Center, 0, j);
        f_.put_all(vx, x.data());
        f_.put_all(vy, y.data());
    }

    // Append one time record: h, u/v (faces→centres), and relative vorticity — a DERIVED
    // diagnostic FIELD (the {}-reduce case: computed, not reduced), for every layer.
    template <Mesh M>
    void write(const LayeredState<NL>& s, const M& mesh, double time_s) {
        f_.put_time(vtime_, rec_, time_s);
        for (int l = 0; l < NL; ++l) {
            fill_center   (s.layer[l].eta);     put_layer(vh_,    l);
            fill_u_center (s.layer[l].u);       put_layer(vu_,    l);
            fill_v_center (s.layer[l].v);       put_layer(vv_,    l);
            fill_vorticity(s.layer[l], mesh);   put_layer(vzeta_, l);
        }
        f_.sync();                                            // durable per record (cheap at cadence)
        ++rec_;
    }
    std::size_t records() const { return rec_; }

private:
    void put_layer(int var, int l) {
        f_.template put<Prec>(var, {rec_, std::size_t(l), 0, 0},
                              {1, 1, std::size_t(ny_), std::size_t(nx_)}, buf_.data());
    }
    void fill_center(Field2 f)  { for (Index j=0;j<ny_;++j) for (Index i=0;i<nx_;++i) buf_[std::size_t(i)+std::size_t(j)*nx_] = Prec(f[i,j]); }
    void fill_u_center(Field2 u){ for (Index j=0;j<ny_;++j) for (Index i=0;i<nx_;++i) buf_[std::size_t(i)+std::size_t(j)*nx_] = Prec(Real(0.5)*(u[i,j]+u[i+1,j])); }
    void fill_v_center(Field2 v){ for (Index j=0;j<ny_;++j) for (Index i=0;i<nx_;++i) buf_[std::size_t(i)+std::size_t(j)*nx_] = Prec(Real(0.5)*(v[i,j]+v[i,j+1])); }

    // ζ = ∂v/∂x − ∂u/∂y  (circulation / area) at CORNERS, then averaged to centres — the
    // same Sadourny circulation the Coriolis operator uses. Boundary corners → 0 (a viz
    // diagnostic, not a prognostic; a minor seam artefact under periodicity).
    template <Mesh M>
    void fill_vorticity(const BaroState& layer, const M& mesh) {
        const Field2 u = layer.u, v = layer.v;
        const std::size_t nxc = std::size_t(nx_) + 1;
        for (Index j = 0; j <= ny_; ++j)
            for (Index i = 0; i <= nx_; ++i) {
                Real z = 0;
                if (i >= 1 && i <= nx_ - 1 && j >= 1 && j <= ny_ - 1) {
                    const Real circ = (v[i, j] * mesh.dy(Loc::YFace, i, j) - v[i - 1, j] * mesh.dy(Loc::YFace, i - 1, j))
                                    - (u[i, j] * mesh.dx(Loc::XFace, i, j) - u[i, j - 1] * mesh.dx(Loc::XFace, i, j - 1));
                    z = circ / mesh.area(Loc::Corner, i, j);
                }
                corner_[std::size_t(i) + std::size_t(j) * nxc] = z;
            }
        for (Index j = 0; j < ny_; ++j)
            for (Index i = 0; i < nx_; ++i) {
                const std::size_t c = std::size_t(i) + std::size_t(j) * nxc;
                buf_[std::size_t(i) + std::size_t(j) * nx_] =
                    Prec(Real(0.25) * (corner_[c] + corner_[c + 1] + corner_[c + nxc] + corner_[c + nxc + 1]));
            }
    }
};

} // namespace tc
