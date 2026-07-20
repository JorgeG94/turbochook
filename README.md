# TurboChook 🐔⚡

_It's like a normal chook, just very fast_ (Our tour guide through the Coal River Valley when asked about TurboChooks)

A from-scratch, **GPU-native C++23 ocean dynamical core**, built on **ISO C++ standard
parallelism** (`std::execution::par_unseq` + `nvc++ -stdpar=gpu`). I am only doing this 
to refresh my knowledge of C++ and do some stupid things. THe code is very heavily 
commented for my sake, don't judge too much. 

A vehicle for learning very modern C++ *and* GPU compute at once, rendering the
architecture of a hydrostatic C-grid ocean model in idiomatic modern C++.

**North star:** an Arakawa C-grid, continuity-PPM, PV-conserving,
split-explicit hydrostatic ocean model. 
**Proof-of-concept on-ramp:** a 2D C-grid *barotropic* shallow-water solver — the ocean core's fast mode, so the PoC *is* the foundation. 

**Start here:** [`docs/DESIGN.md`](docs/DESIGN.md) is the architecture spec (read it
before writing code). [`docs/ROADMAP.md`](docs/ROADMAP.md) is the milestone ladder.
[`CLAUDE.md`](CLAUDE.md) is the operational contract (build/test/conventions).

