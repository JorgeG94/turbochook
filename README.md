# TurboChook 🐔⚡

A from-scratch, **GPU-native C++23 ocean dynamical core**, built on **ISO C++ standard
parallelism** (`std::execution::par_unseq` + `nvc++ -stdpar=gpu`).

It is a spiritual re-implementation of the [Rakali](../rakali_dc) Fortran ocean core's
architecture in idiomatic modern C++ — a vehicle for learning very modern C++ *and*
GPU compute at once. **North star:** an Arakawa C-grid, continuity-PPM, PV-conserving,
split-explicit hydrostatic ocean model. **Proof-of-concept on-ramp:** a 2D C-grid
*barotropic* shallow-water solver — the ocean core's fast mode, so the PoC *is* the
foundation. (Deliberately the C-grid/split-explicit **ocean** regime, not the coastal
HLL/HLLC Godunov one.) Layers, EOS, vertical coords, and mixing slot in on top.

**Start here:** [`docs/DESIGN.md`](docs/DESIGN.md) is the architecture spec (read it
before writing code). [`docs/ROADMAP.md`](docs/ROADMAP.md) is the milestone ladder.
[`CLAUDE.md`](CLAUDE.md) is the operational contract (build/test/conventions).

Status: **spec only** — no code yet. Milestone 0 (walking skeleton) is next.
