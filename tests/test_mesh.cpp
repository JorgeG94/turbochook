// =============================================================================
// tests/test_mesh.cpp — the Loc-aware Mesh concept + CartesianMesh (ADR-7).
// Host-serial; no doctest main here (test_m0.cpp owns it), just add cases.
// =============================================================================

#include <doctest/doctest.h>
#include "core/types.hpp"
#include "mesh/cartesian_mesh.hpp"

TEST_CASE("CartesianMesh: staggered extents per location") {
    const tc::Index nx = 8, ny = 4;
    tc::CartesianMesh m(nx, ny, /*dx*/2.0, /*dy*/5.0);

    // η/centres at nx×ny; faces add one along their staggered axis; corners both.
    CHECK(m.extent_x(tc::Loc::Center) == nx);
    CHECK(m.extent_y(tc::Loc::Center) == ny);
    CHECK(m.extent_x(tc::Loc::XFace)  == nx + 1);
    CHECK(m.extent_y(tc::Loc::XFace)  == ny);
    CHECK(m.extent_x(tc::Loc::YFace)  == nx);
    CHECK(m.extent_y(tc::Loc::YFace)  == ny + 1);
    CHECK(m.extent_x(tc::Loc::Corner) == nx + 1);
    CHECK(m.extent_y(tc::Loc::Corner) == ny + 1);
}

TEST_CASE("CartesianMesh: coordinates — centres at (i+½)·d, faces on the line") {
    const tc::Real dx = 2.0, dy = 5.0;
    tc::CartesianMesh m(8, 4, dx, dy);

    CHECK(m.x(tc::Loc::Center, 0, 0) == doctest::Approx(0.5 * dx));   // centre offset
    CHECK(m.x(tc::Loc::XFace,  0, 0) == doctest::Approx(0.0));        // face on the line
    CHECK(m.x(tc::Loc::XFace,  3, 0) == doctest::Approx(3 * dx));
    CHECK(m.y(tc::Loc::Center, 0, 0) == doctest::Approx(0.5 * dy));
    CHECK(m.y(tc::Loc::YFace,  0, 2) == doctest::Approx(2 * dy));
    // corner is on both face lines
    CHECK(m.x(tc::Loc::Corner, 2, 0) == doctest::Approx(2 * dx));
    CHECK(m.y(tc::Loc::Corner, 0, 3) == doctest::Approx(3 * dy));
}

TEST_CASE("CartesianMesh: metrics, area, all-wet mask") {
    const tc::Real dx = 3.0, dy = 7.0;
    tc::CartesianMesh m(5, 5, dx, dy);

    CHECK(m.dx(tc::Loc::Center, 1, 1) == doctest::Approx(dx));
    CHECK(m.dy(tc::Loc::XFace,  2, 2) == doctest::Approx(dy));
    CHECK(m.area(tc::Loc::Corner, 0, 0) == doctest::Approx(dx * dy));
    CHECK(m.wet(tc::Loc::Center, 3, 3) == doctest::Approx(1.0));   // dense ⇒ all ocean
}

TEST_CASE("CartesianMesh: Coriolis is evaluated at the term's location") {
    const tc::Real dy = 10.0, f0 = 1.0, beta = 0.5;
    tc::CartesianMesh m(4, 4, /*dx*/1.0, dy, f0, beta);

    // Corner sits on the face line → y = j·dy ; Center → y = (j+½)·dy.
    CHECK(m.coriolis(tc::Loc::Corner, 0, 2) == doctest::Approx(f0 + beta * (2 * dy)));
    CHECK(m.coriolis(tc::Loc::Center, 0, 2) == doctest::Approx(f0 + beta * (2.5 * dy)));
}

TEST_CASE("CartesianMesh: topology defaults to closed walls, is configurable") {
    tc::CartesianMesh closed(4, 4, 1.0, 1.0);
    CHECK(closed.edge(tc::Edge::West)  == tc::EdgeConn::Wall);
    CHECK(closed.edge(tc::Edge::North) == tc::EdgeConn::Wall);

    tc::CartesianMesh channel(4, 4, 1.0, 1.0, 1e-4, 0.0,
                              tc::EdgeConn::Periodic, tc::EdgeConn::Periodic,
                              tc::EdgeConn::Wall, tc::EdgeConn::Wall);
    CHECK(channel.edge(tc::Edge::West) == tc::EdgeConn::Periodic);
    CHECK(channel.edge(tc::Edge::East) == tc::EdgeConn::Periodic);
    CHECK(channel.edge(tc::Edge::South) == tc::EdgeConn::Wall);
}
