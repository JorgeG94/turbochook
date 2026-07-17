# TurboChook 🐔⚡

A from-scratch, **GPU-native C++23** finite-volume solver for coastal & ocean flows,
built on **ISO C++ standard parallelism** (`std::execution::par_unseq` + `nvc++ -stdpar=gpu`).

It is a spiritual re-implementation of the [Rakali](../rakali_dc) Fortran solver's
architecture in idiomatic modern C++ — a vehicle for learning very modern C++ *and*
GPU compute at once. It starts with 2D shallow-water and is designed so 3D and other
equation sets slot in without touching the compute kernels.

**Start here:** [`docs/DESIGN.md`](docs/DESIGN.md) is the architecture spec (read it
before writing code). [`docs/ROADMAP.md`](docs/ROADMAP.md) is the milestone ladder.
[`CLAUDE.md`](CLAUDE.md) is the operational contract (build/test/conventions).

Status: **spec only** — no code yet. Milestone 0 (walking skeleton) is next.
