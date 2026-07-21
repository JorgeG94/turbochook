#pragma once
// =============================================================================
// diag/report.hpp — a rakali-style run printout (DESIGN ADR-8, the host side).
//
// A `Reporter` scans the (managed) state on the HOST at the OUTPUT CADENCE — never
// per step — and prints a compact block: day / step / timing (+ ETA), |u|max, the
// advective CFL, total mass with its drift from t=0, and kinetic energy with its
// growth rate (the baroclinic-instability tell). "Just print" — std::format to
// stdout, no logger level. Diagnostics are area-weighted reductions over the mesh
// (ADR-8); this is their first, simplest consumer.
// =============================================================================

#include <chrono>
#include <cstdio>
#include <cmath>
#include <format>
#include <string>
#include <initializer_list>
#include <utility>
#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "physics/state/layered_state.hpp"
#include "diag/diagnostics.hpp"

namespace tc {

class Reporter {
    Real mass0_ = 0, ke_prev_ = 0, day_prev_ = 0;
    std::chrono::steady_clock::time_point t0_{};
    bool started_ = false, first_ = true;

    // min dx (metrics only — no state, no migration)
    template <class M> static Real min_dx(const M& mesh) {
        Real d = Real(1e30);
        for (Index j = 0; j < mesh.ny(); ++j) d = std::min(d, mesh.dx(Loc::Center, 0, j));
        return d;
    }

    static void header() {
        std::fputs(
            "\n  ┌───────────────────────────────────────────────────────────────────┐\n"
            "  │  TurboChook  —  day · step · timing · |u|max · CFL · mass · energy  │\n"
            "  └───────────────────────────────────────────────────────────────────┘\n",
            stdout);
    }

public:
    void start() { t0_ = std::chrono::steady_clock::now(); started_ = true; }

    // Print one cadence block. `total_days` drives the ETA; pass 0 to omit it.
    // `tracers` is a list of {name, total} — printed ONLY when non-empty, so the
    // block gains a salt/heat line once tracers exist and stays lean until then
    // (rakali's "print it if you have it" behaviour).
    template <int NL, class M>
    void report(const LayeredState<NL>& s, const M& mesh, Real day, Real dt, long step, Real total_days,
                std::initializer_list<std::pair<const char*, Real>> tracers = {}) {
        if (!started_) start();
        // device-offloading reductions — only the scalars cross to host (ADR-8), no
        // full-state migration. Each is the SAME global reduce over a different integrand.
        const Real mass  = total_mass(s, mesh);
        const Real umax  = max_speed(s, mesh);
        const Real ke    = total_ke(s, mesh);
        const Real dxmin = min_dx(mesh);
        if (first_) { mass0_ = mass; header(); }

        const Real wall  = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0_).count();
        const Real frac  = total_days > 0 ? day / total_days : Real(0);
        const Real eta   = frac > Real(1e-6) ? wall * (Real(1) - frac) / frac : Real(0);
        const Real cfl   = dxmin > 0 ? umax * dt / dxmin : Real(0);           // advective Courant
        const Real drift = mass0_ != 0 ? (mass - mass0_) / mass0_ : Real(0);
        Real growth = Real(0);
        if (!first_ && ke_prev_ > 0 && day > day_prev_)
            growth = (ke - ke_prev_) / (ke_prev_ * (day - day_prev_)) * Real(100);
        const bool finite = std::isfinite(mass) && std::isfinite(ke);

        std::string blk = std::format(
            "\n  Day {:>9.3f} │ step {:>8d} │ dt {:>7.1f}s │ wall {:>6.0f}s │ ETA {:>6.0f}s\n"
            "     |u|max {:>8.4f} m/s     adv-CFL {:>8.5f}{}\n"
            "     mass   {:>14.6E}     drift  {:>+11.3E}\n"
            "     KE     {:>14.6E}     growth {:>+8.2f} %/day\n",
            day, step, dt, wall, eta, umax, cfl, finite ? "" : "   ⚠ NON-FINITE",
            mass, drift, ke, growth);
        std::fputs(blk.c_str(), stdout);
        if (tracers.size() > 0) {                                    // salt/heat line — only if present
            std::string tl = "    ";
            for (const auto& [name, total] : tracers) tl += std::format(" {:<4} {:>13.6E}  ", name, total);
            tl += "\n";
            std::fputs(tl.c_str(), stdout);
        }
        std::fflush(stdout);

        ke_prev_ = ke; day_prev_ = day; first_ = false;
    }
};

} // namespace tc
