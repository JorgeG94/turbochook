#pragma once
// =============================================================================
// physics/momentum/two_layer_pgf.hpp — reduced-gravity pressure gradient (2 layers). THE
// baroclinic coupling — the interface between the layers drives the vertical
// shear that feeds baroclinic instability. A model on the PGF axis (the header of
// pgf.hpp always listed "gprime 2-layer reduced gravity, arrives M3").
//
// Layer geopotentials (pressure / ρ₁), from two_layer_sw:
//     p₁ = g (h₁ + h₂)                       (upper feels the free surface)
//     p₂ = g h₁ + (ρ₂/ρ₁) g h₂                (lower feels surface + interface)
// Velocity-form acceleration of layer k is −∇pₖ on its faces. ∇ sees only the
// anomalies (constant rest thickness drops out) ⇒ flat h₁=H₁, h₂=H₂ gives zero
// (lake-at-rest). Face metric + wet mask exactly as the single-layer FvPgf.
// =============================================================================

#include "core/types.hpp"
#include "mesh/cartesian_mesh.hpp"
#include "mesh/iterate.hpp"
#include "physics/state/layered_state.hpp"

namespace tc {

class TwoLayerReducedGravityPgf {
public:
    template <Mesh M> void init(Arena& a, const M& m) { (void)a; (void)m; }

    template <Mesh M>
    void compute(LayeredState<2> s, LayeredState<2> k, const M& mesh, Params p) const {
        const Field2 h1 = s.layer[0].eta, h2 = s.layer[1].eta;
        const Field2 ku1 = k.layer[0].u, kv1 = k.layer[0].v;
        const Field2 ku2 = k.layer[1].u, kv2 = k.layer[1].v;
        const M    m = mesh;
        const Real g = p.g, r = p.rho2 / p.rho1;

        for_each_x_face(mesh, [=](FaceView f) {
            const Real inv = m.wet(Loc::XFace, f.i, f.j) / f.span;
            const Real d1  = h1[f.ri, f.rj] - h1[f.li, f.lj];      // Δh₁ across the face
            const Real d2  = h2[f.ri, f.rj] - h2[f.li, f.lj];      // Δh₂
            ku1[f.i, f.j] += -g * (d1 + d2)     * inv;             // −∂p₁/∂x, p₁ = g(h₁+h₂)
            ku2[f.i, f.j] += -g * (d1 + r * d2) * inv;             // −∂p₂/∂x, p₂ = g h₁ + r g h₂
        });
        for_each_y_face(mesh, [=](FaceView f) {
            const Real inv = m.wet(Loc::YFace, f.i, f.j) / f.span;
            const Real d1  = h1[f.ri, f.rj] - h1[f.li, f.lj];
            const Real d2  = h2[f.ri, f.rj] - h2[f.li, f.lj];
            kv1[f.i, f.j] += -g * (d1 + d2)     * inv;
            kv2[f.i, f.j] += -g * (d1 + r * d2) * inv;
        });
    }
};

} // namespace tc
