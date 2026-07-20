#pragma once
// =============================================================================
// mesh/spherical_mesh.hpp — a lon/lat spherical C-grid (the real-run geometry).
//
// Sibling of CartesianMesh (ADR-7): a SECOND model of the `Mesh` concept. Because
// the physics operators are generic over `Mesh`, this drops in with ZERO operator
// changes — the payoff of the genericity refactor. Metrics are closed-form
// (analytic-derivative, not great-circle), translated from rki_ocean_metrics.F90
// `metrics_fill_spherical`:
//   • meridional length  dy = R·Δφ            — latitude-INDEPENDENT
//   • zonal length       dx = R·cosφ·Δλ        — shrinks toward the poles
//   • Coriolis           f  = 2Ω·sinφ
//   • curvature is IMPLICIT in the varying metrics (no explicit tanφ/R term).
// Each location is evaluated at ITS OWN latitude (the C-grid consistency trick):
// cell rows (Center/XFace) at (j+½)·Δφ, face rows (YFace/Corner) at j·Δφ. The
// β-plane / Cartesian limit is just CartesianMesh(f0, β) — a separate model.
// =============================================================================

#include <array>
#include <cmath>
#include "core/types.hpp"
#include "mesh/mesh.hpp"

namespace tc {

inline constexpr Real EARTH_RADIUS = 6.371e6;                 // m
inline constexpr Real EARTH_OMEGA  = 7.2921e-5;               // rad s⁻¹
inline constexpr Real DEG2RAD      = 0.0174532925199432958;   // π/180

class SphericalMesh {
    Index nx_, ny_;
    Real  lon1_, lat1_;                 // SW corner, degrees
    Real  dlon_, dlat_;                 // cell size, degrees
    Real  R_, Omega_;
    std::array<EdgeConn, 4> edges_;

    // latitude (radians) of a location's row j — face rows at integer j, cell rows at j+½.
    Real lat_rad(Loc l, Index j) const {
        const Real jj = y_staggered(l) ? Real(j) : Real(j) + Real(0.5);
        return (lat1_ + jj * dlat_) * DEG2RAD;
    }

public:
    SphericalMesh(Index nx, Index ny, Real lon1, Real lat1, Real lon2, Real lat2,
                  Real earth_R = EARTH_RADIUS, Real omega = EARTH_OMEGA,
                  EdgeConn west = EdgeConn::Wall, EdgeConn east  = EdgeConn::Wall,
                  EdgeConn south = EdgeConn::Wall, EdgeConn north = EdgeConn::Wall)
        : nx_(nx), ny_(ny), lon1_(lon1), lat1_(lat1),
          dlon_((lon2 - lon1) / Real(nx)), dlat_((lat2 - lat1) / Real(ny)),
          R_(earth_R), Omega_(omega), edges_{west, east, south, north} {}

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Index extent_x(Loc l) const { return nx_ + (x_staggered(l) ? 1 : 0); }
    Index extent_y(Loc l) const { return ny_ + (y_staggered(l) ? 1 : 0); }

    // geographic coordinates (degrees)
    Real x(Loc l, Index i, Index /*j*/) const { return lon1_ + (x_staggered(l) ? Real(i) : Real(i) + Real(0.5)) * dlon_; }
    Real y(Loc l, Index /*i*/, Index j) const { return lat1_ + (y_staggered(l) ? Real(j) : Real(j) + Real(0.5)) * dlat_; }

    // metrics (metres) — dx at the location's latitude, dy constant
    Real dx(Loc l, Index /*i*/, Index j) const { return R_ * std::cos(lat_rad(l, j)) * dlon_ * DEG2RAD; }
    Real dy(Loc /*l*/, Index /*i*/, Index /*j*/) const { return R_ * dlat_ * DEG2RAD; }
    Real area(Loc l, Index i, Index j) const { return dx(l, i, j) * dy(l, i, j); }

    Real coriolis(Loc l, Index /*i*/, Index j) const { return Real(2) * Omega_ * std::sin(lat_rad(l, j)); }
    static constexpr Real wet(Loc, Index, Index) { return Real(1); }
    EdgeConn edge(Edge e) const { return edges_[static_cast<int>(e)]; }
};

static_assert(Mesh<SphericalMesh>);   // must satisfy the concept (incl. trivially copyable)

} // namespace tc
