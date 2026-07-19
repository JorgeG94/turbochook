#pragma once
// =============================================================================
// lib/log.hpp — the logger. HOST-ONLY (a kernel has no I/O).
//
// Modern-C++ things to notice here:
//   • std::format / std::print (C++20/23) replace iostream AND printf. The
//     format string is checked *at compile time* against the argument types
//     (via std::format_string), so `logger().info("{} {}", x)` — too few args —
//     is a BUILD error, not a runtime surprise.
//   • A process-global accessor via a function-local `static` (Meyers singleton):
//     `logger()` returns the one instance, constructed on first use, thread-safe
//     initialisation guaranteed by the standard. No global-init-order headaches.
//   • `__has_include(<print>)` — feature-detect the header and fall back to
//     std::format + fputs if the stdlib lacks <print> (older libstdc++).
// =============================================================================

#include <format>
#include <string_view>
#include <mutex>
#include <cstdio>
#include <utility>       // std::forward
#if __has_include(<print>)
  #include <print>
  #define TC_HAS_PRINT 1
#else
  #define TC_HAS_PRINT 0
#endif

namespace tc {

enum class LogLevel { trace, debug, info, warn, error, off };

// A free function, not a member — `constexpr` so it can fold at compile time.
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

    // The core emit. `std::format_string<Args...>` is the compile-time-checked
    // format-string type: the parameter pack `Args...` must match the `{}`s.
    template <class... Args>
    void emit(LogLevel l, std::format_string<Args...> fmt, Args&&... args) {
        if (l < level_) return;                       // cheap early-out below threshold
        std::lock_guard lock(mtx_);                   // only matters if logging from threads
        std::string msg = std::format(fmt, std::forward<Args>(args)...);
#if TC_HAS_PRINT
        std::println(stderr, "[{:<5}] {}", to_string(l), msg);
#else
        std::fputs(std::format("[{:<5}] {}\n", to_string(l), msg).c_str(), stderr);
#endif
    }

public:
    void set_level(LogLevel l) { level_ = l; }
    LogLevel level() const { return level_; }

    // One thin wrapper per level. `A&&... a` + `std::forward` is perfect
    // forwarding — pass the args through untouched (no copies, preserve
    // value category). Standard modern-C++ variadic-forwarding boilerplate.
    template <class... A> void trace(std::format_string<A...> f, A&&... a){ emit(LogLevel::trace, f, std::forward<A>(a)...); }
    template <class... A> void debug(std::format_string<A...> f, A&&... a){ emit(LogLevel::debug, f, std::forward<A>(a)...); }
    template <class... A> void info (std::format_string<A...> f, A&&... a){ emit(LogLevel::info,  f, std::forward<A>(a)...); }
    template <class... A> void warn (std::format_string<A...> f, A&&... a){ emit(LogLevel::warn,  f, std::forward<A>(a)...); }
    template <class... A> void error(std::format_string<A...> f, A&&... a){ emit(LogLevel::error, f, std::forward<A>(a)...); }
};

// Process-global logger. `static` local = constructed once, on first call.
inline Logger& logger() { static Logger inst; return inst; }

} // namespace tc

// Usage:  tc::logger().info("grid {}x{}, dt={:.3f}", nx, ny, dt);
// Never call from inside a kernel — device code cannot do I/O. A kernel signals
// failure through data (a NaN/flag buffer) that the host reduces and logs.
