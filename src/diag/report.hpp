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
#include "physics/layered_state.hpp"

namespace tc {

class Reporter {
    Real mass0_ = 0, ke_prev_ = 0, day_prev_ = 0;
    std::chrono::steady_clock::time_point t0_{};
    bool started_ = false, first_ = true;

    // area-weighted host reductions: total mass, max speed, kinetic energy, min dx
    template <int NL, class M>
    static void scan(const LayeredState<NL>& s, const M& mesh,
                     Real& mass, Real& umax, Real& ke, Real& dxmin) {
        const Index nx = mesh.nx(), ny = mesh.ny();
        mass = 0; ke = 0; Real umax2 = 0; dxmin = Real(1e30);
        for (Index j = 0; j < ny; ++j) dxmin = std::min(dxmin, mesh.dx(Loc::Center, 0, j));
        for (int l = 0; l < NL; ++l) {
            const Field2 h = s.layer[l].eta, u = s.layer[l].u, v = s.layer[l].v;
            for (Index j = 0; j < ny; ++j)
                for (Index i = 0; i < nx; ++i) {
                    const Real ar = mesh.area(Loc::Center, i, j);
                    const Real uc = Real(0.5) * (u[i, j] + u[i + 1, j]);   // faces → centre
                    const Real vc = Real(0.5) * (v[i, j] + v[i, j + 1]);
                    const Real sp2 = uc * uc + vc * vc;
                    mass += h[i, j] * ar;
                    ke   += Real(0.5) * h[i, j] * sp2 * ar;
                    if (sp2 > umax2) umax2 = sp2;
                }
        }
        umax = std::sqrt(umax2);
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
        Real mass, umax, ke, dxmin;
        scan(s, mesh, mass, umax, ke, dxmin);
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
