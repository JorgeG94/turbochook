// =============================================================================
// spike 04 — compile-time memory-space tags + the mirror-is-a-no-op idiom.
//
// HOST ONLY. No GPU, no CUDA, no nvc++ — plain g++/clang++. Run this one first;
// it costs nothing and it settles the API shape before any hardware is involved.
//
// THE QUESTION (two of them):
//   1. Can "host code touched device memory" be a COMPILE error rather than a
//      fault? Kokkos catches most of this at runtime with a clear message; we can
//      be stricter, because we never select a memory space dynamically.
//   2. Does the mirror idiom collapse to nothing when the data is already
//      host-accessible? That is the property that lets ONE diagnostic / NetCDF /
//      restart / unit-test path serve the host build, the coherent-memory build,
//      and the discrete-GPU build with no #ifdef at the call site.
//
// PASS = this file compiles and runs. The NEGATIVE half must FAIL to compile:
//     make s04_negative      # expected: a static_assert about host access
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

namespace tc {

enum class Space { Host, Device };

// On a discrete GPU this is the truth. On a coherent machine Device memory is in
// fact host-readable, but the TYPE stays conservative on purpose: code written to
// the strict contract is correct everywhere, and the mirror below makes the
// coherent case cost nothing anyway.
constexpr bool host_accessible(Space s) { return s == Space::Host; }

// ── the owning handle. Kernels never see this; they see view(). ──────────────
template <class T, int Rank, Space S>
class Tensor {
    T*   p_  = nullptr;
    int  n0_ = 0, n1_ = 0;
    static_assert(Rank == 2, "spike keeps Rank=2");
public:
    static constexpr Space space = S;

    Tensor() = default;
    Tensor(T* p, int n0, int n1) : p_(p), n0_(n0), n1_(n1) {}

    T*  data()  const { return p_; }
    int n0()    const { return n0_; }
    int n1()    const { return n1_; }
    std::size_t bytes() const { return std::size_t(n0_) * n1_ * sizeof(T); }

    // THE GATE. Host-side subscripting of Device storage does not compile.
    // (A kernel gets a raw view instead — see view() — so the gate costs the
    // device path nothing.)
    T& operator()(int i, int j) const {
        // ASCII ONLY in a static_assert message: nvc++'s EDG frontend renders
        // non-ASCII as '???' (verified, nvc++ 26.5). Comments may use whatever;
        // text that has to survive a COMPILER DIAGNOSTIC must not.
        static_assert(host_accessible(S),
                      "host subscript of Space::Device storage - take a mirror() first");
        return p_[i + n0_ * j];
    }

    // What crosses into a kernel: a bare pointer + extents, trivially copyable,
    // no space tag, no ownership. Exactly today's Field2 role.
    struct RawView {
        T*  p; int n0;
        T&  operator()(int i, int j) const { return p[i + n0 * j]; }
    };
    RawView view() const { return RawView{p_, n0_}; }
};

// ── the mirror idiom ─────────────────────────────────────────────────────────
// If the source is already host-accessible, mirror() RETURNS THE SOURCE — no
// allocation, and the subsequent copy() is a no-op. That is the whole trick: the
// call site is written once and pays only where it must.
template <class T, int Rank, Space S>
auto mirror(const Tensor<T, Rank, S>& d, std::vector<T>& backing) {
    if constexpr (host_accessible(S)) {
        (void)backing;
        return d;                                            // identity: zero cost
    } else {
        backing.resize(std::size_t(d.n0()) * d.n1());
        return Tensor<T, Rank, Space::Host>(backing.data(), d.n0(), d.n1());
    }
}

template <class T, int Rank, Space Sd, Space Ss>
void copy(const Tensor<T, Rank, Sd>& dst, const Tensor<T, Rank, Ss>& src) {
    if (dst.data() == src.data()) return;                    // the no-op case
    // A real implementation dispatches to cudaMemcpy on the space pair; the spike
    // only needs to prove the SHAPE, so host->host is enough here.
    std::memcpy(dst.data(), src.data(), src.bytes());
}

} // namespace tc

int main() {
    std::printf("\n=== spike 04: compile-time space tags + mirror no-op ===\n");
    int fails = 0;

    // A "device" tensor — in this host-only spike its bytes are ordinary memory,
    // but its TYPE says Device, so the gate is live.
    std::vector<double> storage(16 * 16, 3.5);
    tc::Tensor<double, 2, tc::Space::Device> d(storage.data(), 16, 16);

    // Kernels take view(): no space tag, no gate, nothing to get wrong.
    auto v = d.view();
    v(3, 4) = 7.25;

    // Host reads go through a mirror. On a Host/coherent build this allocates
    // nothing and copies nothing; on a discrete build it stages.
    std::vector<double> backing;
    auto h = tc::mirror(d, backing);
    tc::copy(h, d);
    const bool ok_val = (h(3, 4) == 7.25);
    std::printf("  [%s] mirror + copy round-trips the value\n", ok_val ? "PASS" : "FAIL");
    fails += !ok_val;

    // The no-op property, checked as a TYPE fact rather than a runtime one.
    tc::Tensor<double, 2, tc::Space::Host> hh(storage.data(), 16, 16);
    auto h2 = tc::mirror(hh, backing);
    constexpr bool identity =
        std::is_same_v<decltype(h2), tc::Tensor<double, 2, tc::Space::Host>>;
    const bool no_alloc = (h2.data() == hh.data());
    std::printf("  [%s] mirror of host-accessible storage is the identity (no alloc)\n",
                (identity && no_alloc) ? "PASS" : "FAIL");
    fails += !(identity && no_alloc);

#ifdef TC_NEGATIVE
    // MUST NOT COMPILE. `make s04_negative` succeeding is the failure.
    std::printf("  negative: expecting a static_assert below...\n");
    const double bad = d(3, 4);
    std::printf("  !!! NEGATIVE TEST COMPILED — the gate is not live (%f)\n", bad);
    return 1;
#endif

    std::printf("  ---- spike 04: %s ----\n\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
