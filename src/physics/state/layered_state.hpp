#pragma once
// =============================================================================
// physics/state/layered_state.hpp — a stacked N-layer state (M3).
//
// A LayeredState<NL> is just NL BaroStates (one C-grid (h,u,v) per layer). This
// deliberately REUSES the single-layer machinery: the per-layer continuity and
// Coriolis operators run on each layer's BaroState unchanged; only the pressure
// (which couples the layers) is new. allocate/zero/axpby loop the layers, calling
// the BaroState versions — so the integrator, being generic over the state type,
// steps a LayeredState with no changes.
// =============================================================================

#include <array>
#include "physics/state/baro_state.hpp"

namespace tc {

template <int NL>
struct LayeredState {
    std::array<BaroState, NL> layer{};
};

template <int NL, Mesh M>
inline LayeredState<NL> allocate_layered_state(Arena& a, const M& m) {
    LayeredState<NL> s;
    for (BaroState& l : s.layer) l = allocate_baro_state(a, m);
    return s;
}

template <int NL, Mesh M>
inline void zero_layered_state(LayeredState<NL> s, const M& m) {
    for (BaroState& l : s.layer) zero_baro_state(l, m);
}

// dst = a·x + b·y per layer — the combine the integrator needs (found by ADL when
// SSPRK*::advance is instantiated on LayeredState).
template <int NL>
inline void axpby(LayeredState<NL> dst, Real a, LayeredState<NL> x, Real b, LayeredState<NL> y) {
    for (int l = 0; l < NL; ++l) axpby(dst.layer[l], a, x.layer[l], b, y.layer[l]);
}

} // namespace tc
