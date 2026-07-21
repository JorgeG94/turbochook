#pragma once
// =============================================================================
// bc/fold.hpp — the tripolar north-fold boundary (TOPOLOGY, ADR-7).
//
// The bipolar/tripolar cap that global ocean grids use to avoid the North Pole
// singularity: the north row folds onto ITSELF with an index reversal AND a vector
// SIGN FLIP — scalars (h, tracers) copy across the seam, vector components (u, v)
// negate (Parity::Vector). A halo-fill BC that fills ghost rows before each stage,
// exactly like WallBC / PeriodicBC. mesh.hpp already carries EdgeConn::Fold + Parity
// for this; the fold is treated as a wall until this lands (documented gap in mesh.hpp).
//
// TODO(Later): FoldBC::fill_halos — reversed-index copy of the north row + sign flip
//              on vector components (parity-aware).
// rakali north-star: src/core/ocean/boundary/rki_ocean_fold.F90, rki_ocean_fold_apply.F90,
//                    src/core/ocean/state/rki_ocean_bipolar.F90
// =============================================================================

#include "bc/bc.hpp"

namespace tc {

class FoldBC {
public:
    template <Mesh M> void fill_halos(BaroState s, const M& m) const {
        // TODO(Later): mirror the north row with index reversal; negate u,v across the
        // seam (Parity::Vector), copy h/tracers (Parity::Scalar).
        (void)s; (void)m;
    }
};

static_assert(BoundaryCondition<FoldBC>);

} // namespace tc
