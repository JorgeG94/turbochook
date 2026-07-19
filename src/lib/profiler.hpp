#pragma once
// =============================================================================
// lib/profiler.hpp — nested-region wall-clock timing (a hierarchical
// profiler). HOST-side.
//
// The one idiom to learn here is RAII (Resource Acquisition Is Initialisation):
// instead of manual start()/stop() pairs you can forget to match, a `Scope`
// guard starts timing in its constructor and stops in its destructor. Because
// C++ destroys locals in reverse order of construction, nesting is LIFO
// automatically, and it's exception-safe (the dtor runs even if the scope exits
// via a throw). `TC_PROFILE("name")` drops such a guard into the current block.
//
// Self vs inclusive time: each region tracks total (inclusive) and how much of
// that was spent in CHILDREN → self = total − child. That's what tells you where
// the time actually goes vs. where it merely passes through.
//
// Clock choice: std::chrono::steady_clock — MONOTONIC. Not high_resolution_clock
// (often an alias of system_clock, which can jump backwards on NTP adjustment).
// Host wall-clock around a for_each(par_unseq,…) measures the kernel because
// stdpar synchronises per call.
// =============================================================================

#include <chrono>
#include <string>
#include <string_view>
#include <vector>
#include <cstdio>
#include <algorithm>

namespace tc {

class Profiler {
    using clock = std::chrono::steady_clock;
    struct Region {
        std::string name;
        double total = 0, child = 0;      // inclusive time, and time spent in children
        long calls = 0;
        int parent = -1, depth = 0;       // for the tree display
    };
    std::vector<Region> regions_;
    std::vector<int> stack_;              // indices of currently-open regions
    bool enabled_ = true;

    int find_or_create(std::string_view n) {
        for (int i = 0; i < (int)regions_.size(); ++i)
            if (regions_[i].name == n) return i;
        regions_.push_back(Region{std::string(n)});
        return (int)regions_.size() - 1;
    }
    // Called from ~Scope: accumulate elapsed, pop the stack, and attribute this
    // region's elapsed time to its PARENT's `child` bucket (so parent.self
    // subtracts it out).
    void close(int idx, clock::time_point t0) {
        double dt = std::chrono::duration<double>(clock::now() - t0).count();
        regions_[idx].total += dt;
        regions_[idx].calls += 1;
        if (!stack_.empty() && stack_.back() == idx) stack_.pop_back();
        if (!stack_.empty()) regions_[stack_.back()].child += dt;
    }

public:
    void set_enabled(bool e) { enabled_ = e; }
    void reset() { regions_.clear(); stack_.clear(); }

    // RAII guard. Move-only (you can return it), non-copyable (copying would
    // double-count). A null guard (p_ == nullptr) is the "profiling disabled" no-op.
    class Scope {
        Profiler* p_; int idx_; clock::time_point t0_;
        friend class Profiler;
        Scope(Profiler* p, int idx) : p_(p), idx_(idx), t0_(clock::now()) {}
    public:
        Scope(Scope&& o) noexcept : p_(o.p_), idx_(o.idx_), t0_(o.t0_) { o.p_ = nullptr; }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { if (p_) p_->close(idx_, t0_); }
    };

    // `[[nodiscard]]` — the returned guard MUST be bound to a named local, else it
    // dies at the end of the full expression and times nothing. The attribute
    // makes `profiler().scope("x");` (discarded) a compiler warning.
    [[nodiscard]] Scope scope(std::string_view name) {
        if (!enabled_) return Scope{nullptr, -1};
        int idx = find_or_create(name);
        int parent = stack_.empty() ? -1 : stack_.back();
        if (regions_[idx].calls == 0) {   // first sighting → record parent/depth for display
            regions_[idx].parent = parent;
            regions_[idx].depth  = parent < 0 ? 0 : regions_[parent].depth + 1;
        }
        stack_.push_back(idx);
        return Scope{this, idx};
    }

    // Flat report (default) or an indented tree with inclusive vs self columns.
    void report(bool tree = false) const {
        std::fputs("\n-- profiler --------------------------------------------------\n", stderr);
        std::fputs("  region                         calls    incl(s)    self(s)\n", stderr);
        for (int i = 0; i < (int)regions_.size(); ++i) {
            const Region& r = regions_[i];
            const double self = r.total - r.child;
            std::string label = r.name;
            if (tree) label = std::string(2 * r.depth, ' ') + label;    // indent by depth
            std::fprintf(stderr, "  %-28s %6ld  %9.4f  %9.4f\n",
                         label.c_str(), r.calls, r.total, self);
        }
        std::fputs("--------------------------------------------------------------\n", stderr);
    }
};

inline Profiler& profiler() { static Profiler inst; return inst; }

// Token-paste a unique guard name (`tc_prof_<line>`) so two TC_PROFILE calls in
// one scope don't collide. This is why the macro dance exists.
#define TC_CONCAT_(a, b) a##b
#define TC_CONCAT(a, b)  TC_CONCAT_(a, b)
#define TC_PROFILE(name) auto TC_CONCAT(tc_prof_, __LINE__) = ::tc::profiler().scope(name)

} // namespace tc
