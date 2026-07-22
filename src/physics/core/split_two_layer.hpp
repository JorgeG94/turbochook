#pragma once
// =============================================================================
// physics/core/split_two_layer.hpp — DEPRECATED shim.
//
// The split-explicit core is now the NL-general SplitMultilayerCore<NL>
// (physics/core/split_multilayer_core.hpp). `SplitTwoLayerCore` / `SplitTwoLayerPoC`
// live on there as NL=2 aliases. This shim keeps old includes compiling; it is removed
// once every includer points at split_multilayer_core.hpp.
// =============================================================================

#include "physics/core/split_multilayer_core.hpp"
