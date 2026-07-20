#pragma once
// =============================================================================
// io/netcdf.hpp — a small RAII / exception-based C++ wrapper over the NetCDF-C API.
//
// The "more C++ way" vs rakali's procedural `nc_check(nf90_...)` at every call site:
//   • errors → `tc::Error` exceptions, checked ONCE in `check()` (fail-loud, RAII
//     unwinds) — no per-call error-code plumbing;
//   • the file handle is RAII (`File` closes itself, move-only) — no leaked ncid;
//   • def_var / put are type-DISPATCHED templates (double↔float↔int) — no
//     `_real`/`_double` twins.
//
// Thin C-ABI bridge over libnetcdf — policy-clean (NetCDF is a C library, not a C++
// framework; CLAUDE.md endorses a thin C-ABI bridge). Gated on TC_HAVE_NETCDF, set by
// CMake when -DTC_NETCDF=ON locates the library; the core builds fine without it.
//
// Layout note (matches rakali's reasoning, inverted for the C API): our grid fields
// are `layout_left` (nx,ny) with x contiguous, so a C-order NetCDF var declared
// `[..., y, x]` (x fastest) consumes the buffer byte-for-byte — NO transpose.
// =============================================================================

#include <netcdf.h>
#include <string>
#include <string_view>
#include <initializer_list>
#include <cstddef>
#include <source_location>
#include <type_traits>
#include "lib/error.hpp"

namespace tc::nc {

// Every NetCDF call funnels through here: non-zero → throw. This is the whole error
// story (rakali wraps every single call; we wrap the wrapper).
inline void check(int rc, std::source_location loc = std::source_location::current()) {
    if (rc != NC_NOERR) throw Error(Errc::io_failure, nc_strerror(rc), loc);
}

template <class T> struct nctype;
template <> struct nctype<double> { static constexpr int id = NC_DOUBLE; };
template <> struct nctype<float>  { static constexpr int id = NC_FLOAT;  };
template <> struct nctype<int>    { static constexpr int id = NC_INT;    };

// RAII NetCDF-4 file. Move-only; closes on destruction (define/write phases are just
// methods, exactly as the netCDF workflow expects: create → def_* → enddef → put_*).
class File {
    int id_ = -1;
    explicit File(int id) : id_(id) {}

public:
    File() = default;
    ~File() { close(); }
    File(File&& o) noexcept : id_(o.id_) { o.id_ = -1; }
    File& operator=(File&& o) noexcept { if (this != &o) { close(); id_ = o.id_; o.id_ = -1; } return *this; }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    static File create(std::string_view path) {
        int id; check(nc_create(std::string(path).c_str(), NC_CLOBBER | NC_NETCDF4, &id));
        return File(id);
    }
    static File open(std::string_view path) {                 // read-back (tests / restart)
        int id; check(nc_open(std::string(path).c_str(), NC_NOWRITE, &id));
        return File(id);
    }
    void close() { if (id_ >= 0) { nc_close(id_); id_ = -1; } }
    int id() const { return id_; }

    // ── define mode ───────────────────────────────────────────────────────────────
    int def_dim(std::string_view name, std::size_t len) {
        int d; check(nc_def_dim(id_, std::string(name).c_str(), len, &d)); return d;
    }
    int def_unlimited(std::string_view name) {
        int d; check(nc_def_dim(id_, std::string(name).c_str(), NC_UNLIMITED, &d)); return d;
    }
    template <class T>
    int def_var(std::string_view name, std::initializer_list<int> dims) {
        int v; check(nc_def_var(id_, std::string(name).c_str(), nctype<T>::id,
                                int(dims.size()), std::data(dims), &v));
        return v;
    }
    void deflate(int var, int level) { if (level > 0) check(nc_def_var_deflate(id_, var, 1, 1, level)); }

    void att(int var, std::string_view name, std::string_view val) {
        check(nc_put_att_text(id_, var, std::string(name).c_str(), val.size(), val.data()));
    }
    void att(int var, std::string_view name, double val) {
        check(nc_put_att_double(id_, var, std::string(name).c_str(), NC_DOUBLE, 1, &val));
    }
    void global_att(std::string_view name, std::string_view val) { att(NC_GLOBAL, name, val); }

    void enddef() { check(nc_enddef(id_)); }
    void sync()   { check(nc_sync(id_)); }

    // ── data mode ─────────────────────────────────────────────────────────────────
    // typed hyperslab write (a (y,x) or (layer,y,x) slab at a time record, etc.)
    template <class T>
    void put(int var, std::initializer_list<std::size_t> start,
             std::initializer_list<std::size_t> count, const T* data) {
        if constexpr (std::is_same_v<T, double>) check(nc_put_vara_double(id_, var, std::data(start), std::data(count), data));
        else if constexpr (std::is_same_v<T, float>) check(nc_put_vara_float(id_, var, std::data(start), std::data(count), data));
        else check(nc_put_vara_int(id_, var, std::data(start), std::data(count), data));
    }
    template <class T> void put_all(int var, const T* data) {          // whole static var (coords)
        if constexpr (std::is_same_v<T, double>) check(nc_put_var_double(id_, var, data));
        else if constexpr (std::is_same_v<T, float>) check(nc_put_var_float(id_, var, data));
        else check(nc_put_var_int(id_, var, data));
    }
    void put_time(int var, std::size_t rec, double t) {               // one scalar at a record
        const std::size_t s = rec, c = 1; check(nc_put_vara_double(id_, var, &s, &c, &t));
    }

    // ── read-back (minimal — for tests / restarts) ─────────────────────────────────
    int varid(std::string_view name) const { int v; check(nc_inq_varid(id_, std::string(name).c_str(), &v)); return v; }
    std::size_t dimlen(std::string_view name) const {
        int d; std::size_t n; check(nc_inq_dimid(id_, std::string(name).c_str(), &d));
        check(nc_inq_dimlen(id_, d, &n)); return n;
    }
    template <class T> void get_all(int var, T* out) const {
        if constexpr (std::is_same_v<T, double>) check(nc_get_var_double(id_, var, out));
        else if constexpr (std::is_same_v<T, float>) check(nc_get_var_float(id_, var, out));
        else check(nc_get_var_int(id_, var, out));
    }
};

} // namespace tc::nc
