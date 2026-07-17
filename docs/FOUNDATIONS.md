# TurboChook — Foundations (`lib/core/`) & Directory Layout

The foundational, physics-free layer — turbochook's equivalent of Rakali's `pic`, but far
smaller because the C++23 stdlib gives us most of it (`std::format`/`std::print`,
`std::expected`, `std::source_location`, `std::mdspan`, `std::chrono`, parallel algorithms).

## 1. Directory layout — *where things go matters*

`lib/` **is** the library, header-first (templated device code must be visible at the call
site → mostly `.hpp`). Host-only pieces with global state or heavy includes may have a
paired `.cpp`.

```
turbochook/
├── lib/                          # the library
│   ├── core/                     # FOUNDATION — zero physics. The "pic" of turbochook.
│   │   ├── types.hpp             # Real, Index, Field<Rank> = mdspan<layout_left>, aliases
│   │   ├── log.hpp               # Logger (std::format/std::print), levels, global accessor
│   │   ├── error.hpp             # Error, Result<T> = std::expected<T,Error>, source_location
│   │   ├── arena.hpp             # the memory arena (DESIGN ADR-3)
│   │   └── timer.hpp             # scoped wall-clock timing (profiler seed)
│   ├── mesh/                     # grid extents, metrics, ghost/halo (0-based; interior = [nghost, nghost+n))
│   ├── physics/                  # EquationSet policies (swe.hpp, …)
│   ├── numerics/                 # parallel.hpp (for_each_cell), riemann.hpp, integrator.hpp
│   ├── bc/                       # boundary-condition policies
│   └── io/                       # field dump, config parsing
├── app/                          # thin executables (swe2d.cpp wires config → solver → output)
├── tests/                        # host-serial analytical tests (test_*.cpp)
├── docs/
└── CMakeLists.txt
```

- **Namespace:** a single flat `tc` namespace (short call sites — `tc::HLL`, not
  `tc::numerics::HLL`). Directories organise files, not namespaces.
- **Include path:** `lib/` is on the include path → `#include "core/log.hpp"`,
  `#include "numerics/riemann.hpp"`. No deep `<turbochook/...>` prefix for a standalone repo.
- **Dependency rule:** `core/` depends on nothing but the stdlib. Everything depends on
  `core/`. Physics/numerics/bc never include each other's internals — they compose through
  the concepts in DESIGN §5.

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

## 3. `core/log.hpp` — the logger

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

## 4. `core/error.hpp` — error handling that's actually nice

Design: `std::expected<T, Error>` (C++23) as `Result<T>`; every `Error` carries a
`std::source_location` so failures self-report their origin; monadic composition
(`.and_then`/`.transform`/`.or_else`); fail-loud, fail-early.

```cpp
#pragma once
#include <expected>
#include <string>
#include <source_location>
#include <format>
#include <utility>

namespace tc {

enum class Errc {
    ok, invalid_config, unknown_scheme, out_of_memory,
    io_failure, nan_detected, cfl_violation, not_implemented,
};

constexpr std::string_view to_string(Errc c) {
    switch (c) {
        case Errc::ok:              return "ok";
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

struct Error {
    Errc code;
    std::string message;
    std::source_location where;
    Error(Errc c, std::string msg,
          std::source_location loc = std::source_location::current())
        : code(c), message(std::move(msg)), where(loc) {}
};

template <class T>
using Result = std::expected<T, Error>;

// Build an error to `return` from a Result<T> function.
inline std::unexpected<Error> fail(Errc c, std::string msg,
        std::source_location loc = std::source_location::current()) {
    return std::unexpected(Error{c, std::move(msg), loc});
}

// Top-level handler: log an Error with its origin (host, at the driver boundary).
inline void report(const Error& e) {
    logger().error("[{}] {}  ({}:{})",
                   to_string(e.code), e.message, e.where.file_name(), e.where.line());
}

} // namespace tc
```

Usage — fallible host ops return `Result<T>`:

```cpp
Result<FluxKind> parse_flux(std::string_view s) {
    if (s == "hll")  return FluxKind::hll;
    if (s == "hllc") return FluxKind::hllc;
    return fail(Errc::unknown_scheme, std::format("unknown flux scheme '{}'", s));
}

// compose monadically; handle once at the top
auto cfg = load_config(path)
              .and_then(validate)
              .and_then(build_solver);
if (!cfg) { report(cfg.error()); return 1; }         // fail-loud, one place
```

### Error philosophy (the host/device split — mirrors Rakali fail-loud)

- **Host orchestration** (config, setup, I/O, dispatch): return `Result<T>`; propagate
  monadically; `report()` + non-zero exit at the driver boundary. Invalid config / unknown
  scheme fail **at setup**, not mid-run.
- **Device kernels** cannot use `expected`/exceptions. They signal failure through data: a
  device error-flag buffer or a NaN sentinel. The host runs a reduction after the step
  (e.g. `std::transform_reduce` for `any(isnan)` / max CFL), and converts a bad result into
  an `Error` (`Errc::nan_detected`, `Errc::cfl_violation`) → `report()` → abort. **CPU-green
  ≠ device-correct**; these post-step guards are the fail-loud net.
- **Exceptions** are reserved for genuinely exceptional host-side conditions
  (`std::bad_alloc`, unrecoverable I/O). The hot loop neither throws nor catches.

## 5. `core/timer.hpp` (seed only)

A scoped `std::chrono::steady_clock` RAII timer that accumulates into a named registry — the
minimal seed of a profiler. Keep it host-side; expand later if needed. (Not required before
M2.)

## 6. Compiler/stdlib support notes

- `std::expected`, `std::print`, `std::mdspan` are C++23; `std::format`, `std::source_location`
  are C++20. On a stdlib lacking `<print>`, the logger falls back (§3). On a stdlib lacking
  `std::mdspan`, vendor `kokkos/mdspan` behind a `std::` shim (DESIGN §9).
- No third-party dependencies in `core/`. That is the whole point of `core/` — the stdlib is
  the dependency.
