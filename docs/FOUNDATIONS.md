# TurboChook — Foundations (`src/core/` + `src/lib/`) & Directory Layout

The two foundational, physics-free layers — `core/` (the numeric substrate) and
`lib/` (physics-agnostic plumbing) — a thin base, far smaller because the C++23
stdlib gives us most of it (`std::format`/`std::print`, `std::expected`,
`std::source_location`, `std::mdspan`, `std::chrono`, parallel algorithms).

## 1. Directory layout — *where things go matters*

`src/` holds all the sources, header-first (templated device code must be visible at the call
site → mostly `.hpp`). Host-only pieces with global state or heavy includes may have a
paired `.cpp`. The **library** is `src/` (the subdirectories); the thin program `main`s live
in `examples/programs/` and link the library. No package manager, so no separate `app/`.

```
turbochook/
├── src/                          # the LIBRARY (header-first; physics-free base + ocean modules)
│   ├── core/                     # NUMERIC SUBSTRATE — zero physics; stdlib only.
│   │   ├── types.hpp             # Real, Index, Field<Rank> = mdspan<layout_left>, aliases
│   │   └── vec.hpp               # tc::Vec<N> — fixed-size numeric vector (glm-style; NOT std::vector/Eigen)
│   ├── lib/                      # PLUMBING — physics-agnostic support; stdlib + core/.
│   │   ├── log.hpp / error.hpp / arena.hpp / profiler.hpp   # logging, errors, memory arena, timing
│   │   ├── units.hpp / config.hpp / safe_math.hpp           # units/Dim, RunConfig, guarded math
│   │   ├── checksums.hpp / mem_report.hpp                   # field digests, arena accounting  [stubs]
│   │   └── (later)                                          # assert.hpp, mpi.hpp — assertions, MPI wrapper
│   ├── mesh/                     # grid extents, metrics; ghost/halo is the TARGET (interior = [nghost, nghost+n)) — M2 is no-ghost interim, see DESIGN #3
│   ├── physics/                  # ocean operator policies, grouped BY ROLE:
│   │   ├── state/                #   baro_state, layered_state            (the nouns / values)
│   │   ├── continuity/           #   continuity (PPM flux), reconstruction (PCM…WENO)
│   │   ├── momentum/             #   coriolis (Sadourny PV), pgf, two_layer_pgf, pgf_layered
│   │   ├── tracer/               #   tracer advection, eos                [stubs → M4/Later]
│   │   ├── vertical/             #   vcoord, remap (ALE), vmix            [stubs → M4/M5]
│   │   ├── lateral/              #   dissipation (Shapiro/Leith), lateral_mix (GM/Redi)
│   │   ├── forcing/              #   surface stress/flux, tides           [stub → Later]
│   │   └── core/                 #   ocean_core, multilayer_core, split_two_layer, barotropic
│   ├── numerics/                 # parallel.hpp (for_each_cell / for_each_face), integrator.hpp
│   ├── diag/                     # reduce.hpp, quantity + registry (the Registry), report, diagnostics
│   ├── bc/                       # boundary-condition policies (wall, periodic; fold/sponge/obc stubs)
│   ├── io/                       # ocean_output (NetCDF), restart, netcdf wrapper
│   └── api/                      # handle (ISolver), capi (extern "C")   [stubs → Later]
├── examples/
│   └── programs/                 # thin program mains — the executables (each links the library)
├── tests/                        # host-serial analytical tests (test_*.cpp)
├── docs/
└── CMakeLists.txt
```

- **Namespace:** a single flat `tc` namespace (short call sites — `tc::Continuity`, not
  `tc::physics::Continuity`). Directories organise files, not namespaces.
- **Include path:** `src/` is on the include path → `#include "core/types.hpp"`,
  `#include "lib/log.hpp"`, `#include "physics/momentum/coriolis.hpp"`. No deep `<turbochook/...>`
  prefix for a standalone repo.
- **Dependency rule:** `core/` (numeric types) depends on nothing but the stdlib; `lib/`
  (plumbing) depends on the stdlib and `core/`. Everything else depends on `core/` + `lib/`.
  Physics/numerics/bc never include each other's internals — they compose through the
  concepts in DESIGN §5.

## 2. `core/types.hpp`

```cpp
#pragma once
#include <mdspan>
#include <cstdint>

namespace tc {
using Real  = double;
using Index = int;                                    // grid index type (signed, 0-based)

// The grid field == the Fortran array (column-major, contiguous), but 0-based.
template <int Rank>
using Field = std::mdspan<Real, std::dextents<Index, Rank>, std::layout_left>;
using Field2 = Field<2>;
using Field3 = Field<3>;
} // namespace tc
```

## 2b. `core/vec.hpp` — the fixed-size numeric vector (`tc::Vec<N>`)

The per-cell value type (`Cons`/`Flux` = N conserved variables). **Home-baked, and NOT
`std::vector`** (which is heap/dynamic and illegal in a kernel). Model it on **glm** (eager,
GPU/kernel-safe fixed vec/mat), **not Eigen** — we deliberately omit Eigen's machinery:

- **No expression templates.** For N=3–4 the compiler unrolls trivially; expression templates
  bring gnarly device codegen + the `auto x = a + b;` dangling-reference footgun. Eager,
  by-value evaluation only.
- **Trivially copyable by construction** (a single `std::array<Real,N>` member) — the whole
  point, so it captures by value into a `par_unseq` lambda with zero fuss. (Eigen's alignment
  attributes / `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` baggage is exactly what we avoid.)
- **Fixed-N only** — no dynamic sizing, no heap, ever.
- **Scope discipline:** `Vec<N>` stays minimal (`[]`, `+`, `-`, `scalar*`, `dot`, structured
  bindings). A `Mat<R,C>` is **deferred** until physics forces it (flux Jacobians for a Roe/
  eigen-decomposition solver, or rotation matrices) — the barotropic/PV-Coriolis operators
  need none. Do **not** grow this
  into a linear-algebra library.

```cpp
#pragma once
#include <array>
#include <cstddef>
#include "core/types.hpp"      // Real

namespace tc {

template <int N>
struct Vec {                   // aggregate; trivially copyable; brace-elision init: Vec<3>{a,b,c}
    std::array<Real, N> d{};
    constexpr Real  operator[](int i) const { return d[i]; }
    constexpr Real& operator[](int i)       { return d[i]; }
    static constexpr int size() { return N; }
};

template <int N> constexpr Vec<N> operator+(Vec<N> a, Vec<N> b) {
    Vec<N> r; for (int i=0;i<N;++i) r[i]=a[i]+b[i]; return r; }
template <int N> constexpr Vec<N> operator-(Vec<N> a, Vec<N> b) {
    Vec<N> r; for (int i=0;i<N;++i) r[i]=a[i]-b[i]; return r; }
template <int N> constexpr Vec<N> operator*(Real s, Vec<N> a) {
    Vec<N> r; for (int i=0;i<N;++i) r[i]=s*a[i]; return r; }
template <int N> constexpr Real dot(Vec<N> a, Vec<N> b) {
    Real s=0; for (int i=0;i<N;++i) s+=a[i]*b[i]; return s; }

// structured-binding support (tuple protocol) → `auto [h,hu,hv] = q;`
template <std::size_t I, int N> constexpr Real get(const Vec<N>& v) { return v.d[I]; }

} // namespace tc

template <int N> struct std::tuple_size<tc::Vec<N>>
    : std::integral_constant<std::size_t, N> {};
template <std::size_t I, int N> struct std::tuple_element<I, tc::Vec<N>>
    { using type = tc::Real; };
```

Usage: a per-layer state aliases it (`using Cons = tc::Vec<3>;`); physics destructures
(`auto [h, s, t] = q;`), generic code indexes (`q[v]`), and vector math reads like the formula
(`a*x + b*y`). Introduced at **M3** (layers) — the M2 barotropic PoC is scalar-per-face and
doesn't need it. Operators resolve by ADL (never write `tc::operator+`).

## 3. `lib/log.hpp` — the logger

Design: C++23 `std::format`/`std::print` (no dependency, no iostream), runtime level
threshold, compile-time-checked format strings, a process-global accessor. **Host-only** —
never called from a kernel.

```cpp
#pragma once
#include <format>
#include <string_view>
#include <mutex>
#include <cstdio>
#if __has_include(<print>)
  #include <print>
  #define TC_HAS_PRINT 1
#endif

namespace tc {

enum class LogLevel { trace, debug, info, warn, error, off };

constexpr std::string_view to_string(LogLevel l) {
    switch (l) {
        case LogLevel::trace: return "TRACE";
        case LogLevel::debug: return "DEBUG";
        case LogLevel::info:  return "INFO";
        case LogLevel::warn:  return "WARN";
        case LogLevel::error: return "ERROR";
        default:              return "OFF";
    }
}

class Logger {
    LogLevel level_ = LogLevel::info;
    std::mutex mtx_;
    template <class... Args>
    void emit(LogLevel l, std::format_string<Args...> fmt, Args&&... args) {
        if (l < level_) return;                       // cheap early-out below threshold
        std::lock_guard lock(mtx_);
        std::string msg = std::format(fmt, std::forward<Args>(args)...);
#if TC_HAS_PRINT
        std::println(stderr, "[{:<5}] {}", to_string(l), msg);
#else                                                 // fallback if <print> is unavailable
        std::fputs(std::format("[{:<5}] {}\n", to_string(l), msg).c_str(), stderr);
#endif
    }
public:
    void set_level(LogLevel l) { level_ = l; }
    LogLevel level() const { return level_; }
    template <class... A> void trace(std::format_string<A...> f, A&&... a){ emit(LogLevel::trace,f,std::forward<A>(a)...);} 
    template <class... A> void debug(std::format_string<A...> f, A&&... a){ emit(LogLevel::debug,f,std::forward<A>(a)...);} 
    template <class... A> void info (std::format_string<A...> f, A&&... a){ emit(LogLevel::info, f,std::forward<A>(a)...);} 
    template <class... A> void warn (std::format_string<A...> f, A&&... a){ emit(LogLevel::warn, f,std::forward<A>(a)...);} 
    template <class... A> void error(std::format_string<A...> f, A&&... a){ emit(LogLevel::error,f,std::forward<A>(a)...);} 
};

inline Logger& logger() { static Logger inst; return inst; }   // process-global

} // namespace tc
```

Usage: `tc::logger().info("grid {}x{}, dt={:.3f}", nx, ny, dt);` — the format string is
checked against the args **at compile time** (`std::format_string`), so `"{}"`/arg
mismatches are build errors, not runtime surprises.

Notes:
- **Host-only.** Kernels never log — device code has no I/O. Errors from a kernel are
  signalled via a flag/NaN buffer (see §4) that the host inspects and logs.
- The mutex only matters if you log from concurrent host threads; the solver loop is
  single-threaded orchestration around parallel algorithms, so it's cheap insurance.
- Portability: `<print>` is C++23 (libstdc++ 14+, libc++ 17+). The `__has_include` guard
  falls back to `std::format` + `std::fputs`. `std::format` itself is C++20 and widely
  available.

## 4. `lib/error.hpp` — error handling (exceptions on the host)

Decision: **exceptions for the host path** (not mission-critical; setup failures are
genuinely exceptional; RAII unwinds the arena/files cleanly on throw; catch once at `main`).
`Error` derives from `std::runtime_error` and bakes a `std::source_location` into its
message so `what()` self-reports code + file:line. **No `std::expected` ceremony.**

The one hard constraint (see the split below): **exceptions cannot cross into a kernel** —
`std::for_each(par_unseq, …)` with a throwing callable is `std::terminate`, and device code
cannot throw. So kernels never throw; device failures are detected on the host and *then*
thrown.

```cpp
#pragma once
#include <stdexcept>
#include <string>
#include <string_view>
#include <source_location>
#include <format>

namespace tc {

enum class Errc {
    invalid_config, unknown_scheme, out_of_memory,
    io_failure, nan_detected, cfl_violation, not_implemented,
};

constexpr std::string_view to_string(Errc c) {
    switch (c) {
        case Errc::invalid_config:  return "invalid_config";
        case Errc::unknown_scheme:  return "unknown_scheme";
        case Errc::out_of_memory:   return "out_of_memory";
        case Errc::io_failure:      return "io_failure";
        case Errc::nan_detected:    return "nan_detected";
        case Errc::cfl_violation:   return "cfl_violation";
        case Errc::not_implemented: return "not_implemented";
        default:                    return "unknown";
    }
}

class Error : public std::runtime_error {
    Errc code_;
    std::source_location where_;
public:
    Error(Errc c, std::string_view msg,
          std::source_location loc = std::source_location::current())
        : std::runtime_error(std::format("[{}] {} ({}:{})",
                             to_string(c), msg, loc.file_name(), loc.line())),
          code_(c), where_(loc) {}
    Errc code() const noexcept { return code_; }
    const std::source_location& where() const noexcept { return where_; }
};

// throw helper — reads like a statement at the call site
[[noreturn]] inline void fail(Errc c, std::string_view msg,
        std::source_location loc = std::source_location::current()) {
    throw Error(c, msg, loc);
}

} // namespace tc
```

Usage — host ops just throw; one handler at the driver:

```cpp
CoriolisKind parse_coriolis(std::string_view s) {
    if (s == "enstrophy") return CoriolisKind::sadourny_enstrophy;
    if (s == "energy")    return CoriolisKind::sadourny_energy;
    fail(Errc::unknown_scheme, std::format("unknown coriolis scheme '{}'", s));
}

int main() try {
    run();                                  // any tc::Error unwinds to here
    return 0;
} catch (const tc::Error& e) {
    tc::logger().error("{}", e.what());     // already carries code + file:line
    return 1;
} catch (const std::exception& e) {
    tc::logger().error("unhandled: {}", e.what());
    return 1;
}
```

### Error philosophy — the host/device split (fail-loud)

- **Host** (config, setup, I/O, dispatch, step boundary): `throw tc::Error`; catch once at
  `main`; RAII cleans up on unwind. Bad config / unknown scheme fail **at setup**, not mid-run.
- **Kernels never throw.** `par_unseq` + an escaping exception = `std::terminate`, and device
  code can't throw anyway. Kernels are plain math.
- **Device failures** (NaN, CFL blow-up) are signalled through *data*: a flag/NaN buffer the
  host reduces after each step (`std::transform_reduce` for `any(isnan)` / max CFL). A bad
  reduction → `fail(Errc::nan_detected, …)` **on the host** → same one handler. So device
  errors surface as exceptions at the step boundary, never from inside the kernel.
  **CPU-green ≠ device-correct** — these post-step guards are the net.
- Don't throw per-cell in the hot loop — detect via the reduction, throw once at the boundary.

## 5. `lib/profiler.hpp` — nested-region timing (RAII, `steady_clock`)

A hierarchical profiler (active-region stack + per-region `child_time` →
**self = total − child**; flat report by default, opt-in tree with inclusive/self columns).
C++ makes it cleaner: **RAII scopes** replace manual start/stop — a guard starts on
construction, stops on destruction, so nesting is LIFO-by-construction and exception-safe.

Use **`std::chrono::steady_clock`** (monotonic) — *not* `high_resolution_clock`, which is
often `system_clock` and can jump backwards under clock adjustment. Host-side wall-clock
around a `for_each(par_unseq, …)` measures the kernel because stdpar syncs per call; use
NVTX/nsys for the fine GPU timeline later.

```cpp
#pragma once
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

namespace tc {

class Profiler {
    using clock = std::chrono::steady_clock;
    struct Region { std::string name; double total=0, child=0; long calls=0; int parent=-1, depth=0; };
    std::vector<Region> regions_;
    std::vector<int> stack_;                 // active-region indices (the nesting stack)
    bool enabled_ = true;

    int find_or_create(std::string_view n) {
        for (int i = 0; i < (int)regions_.size(); ++i) if (regions_[i].name == n) return i;
        regions_.push_back(Region{std::string(n)}); return (int)regions_.size() - 1;
    }
    void close(int idx, clock::time_point t0) {
        double dt = std::chrono::duration<double>(clock::now() - t0).count();
        regions_[idx].total += dt; regions_[idx].calls += 1;
        if (!stack_.empty() && stack_.back() == idx) stack_.pop_back();
        if (!stack_.empty()) regions_[stack_.back()].child += dt;   // attribute to enclosing
    }
public:
    void set_enabled(bool e) { enabled_ = e; }
    void reset() { regions_.clear(); stack_.clear(); }
    void report(bool tree = false) const;    // flat (default) / tree: inclusive vs self, %

    class Scope {                            // RAII guard — start on ctor, stop on dtor
        Profiler* p_; int idx_; clock::time_point t0_;
        friend class Profiler;
        Scope(Profiler* p, int idx) : p_(p), idx_(idx), t0_(clock::now()) {}
    public:
        Scope(Scope&& o) noexcept : p_(o.p_), idx_(o.idx_), t0_(o.t0_) { o.p_ = nullptr; }
        Scope(const Scope&) = delete; Scope& operator=(const Scope&) = delete;
        ~Scope() { if (p_) p_->close(idx_, t0_); }
    };
    [[nodiscard]] Scope scope(std::string_view name) {
        if (!enabled_) return Scope{nullptr, -1};
        int idx = find_or_create(name);
        int parent = stack_.empty() ? -1 : stack_.back();
        if (regions_[idx].calls == 0) {      // first-seen parent/depth (display only)
            regions_[idx].parent = parent;
            regions_[idx].depth  = parent < 0 ? 0 : regions_[parent].depth + 1;
        }
        stack_.push_back(idx);
        return Scope{this, idx};
    }
};

inline Profiler& profiler() { static Profiler inst; return inst; }

#define TC_CONCAT_(a,b) a##b
#define TC_CONCAT(a,b)  TC_CONCAT_(a,b)
#define TC_PROFILE(name) auto TC_CONCAT(tc_prof_, __LINE__) = ::tc::profiler().scope(name)

} // namespace tc
```

Usage — nesting is automatic:

```cpp
void step() {
    TC_PROFILE("step");
    { TC_PROFILE("rhs");     baro_rhs<PPM, Sadourny, FvPgf>(s, k, p); }  // for_each(par_unseq) inside
    { TC_PROFILE("combine"); combine(view(U1), view(U), view(U), view(K), 1,0,1, p); }
}
// "step".child = rhs + combine; self("step") = total - child. Tree report shows both.
```

## 6. Compiler/stdlib support notes

- `std::print`, `std::mdspan` are C++23; `std::format`, `std::source_location` are C++20;
  `std::chrono::steady_clock` is C++11. On a stdlib lacking `<print>`, the logger falls back
  (§3). On a stdlib lacking `std::mdspan`, hand-roll a minimal `layout_left` `tc::mdview`
  behind a `__has_include(<mdspan>)` seam (DESIGN §9) — **never Kokkos**. (`std::expected` is
  C++23 too but we chose exceptions over it — §4.)
- No third-party dependencies in `core/` or `lib/`. That is the whole point of the base — the stdlib is
  the dependency.
