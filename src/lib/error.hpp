#pragma once
// =============================================================================
// lib/error.hpp — error handling. EXCEPTIONS, on the HOST path only.
//
// Why exceptions (DESIGN/FOUNDATIONS §4): setup/config/I/O failures are genuinely
// exceptional, RAII unwinds the arena+files cleanly on throw, and one catch at
// main() keeps every call site terminal-clean (`throw` reads like a statement,
// no error-code plumbing threaded through every return). We deliberately did NOT
// use std::expected here — the host path isn't hot and exceptions are simpler.
//
// THE HARD CONSTRAINT: exceptions may NOT cross into a kernel.
//   std::for_each(par_unseq, …) with a callable that throws → std::terminate,
//   and device code can't throw at all. So: kernels are pure math and never
//   throw; device failures (NaN, CFL blow-up) are detected by a POST-STEP host
//   reduction over a flag/NaN buffer, and THEN thrown host-side. Device errors
//   surface as exceptions at the step boundary — never from inside the loop.
//
// Modern-C++ things to notice:
//   • std::source_location (C++20): a default argument `= current()` captures the
//     CALLER's file/line automatically — no __FILE__/__LINE__ macros.
//   • Deriving from std::runtime_error and baking file:line into what() means the
//     message is self-describing; the one handler just prints what().
// =============================================================================

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

// `[[noreturn]]` tells the compiler control never comes back — so a function that
// ends in `fail(...)` needs no dummy return, and the optimizer knows the path is
// dead. Reads like a statement at the call site: `fail(Errc::unknown_scheme, …)`.
[[noreturn]] inline void fail(Errc c, std::string_view msg,
        std::source_location loc = std::source_location::current()) {
    throw Error(c, msg, loc);
}

} // namespace tc

// Pattern at the driver:
//   int main() try { run(); return 0; }
//   catch (const tc::Error& e)     { tc::logger().error("{}", e.what()); return 1; }
//   catch (const std::exception& e){ tc::logger().error("unhandled: {}", e.what()); return 1; }
