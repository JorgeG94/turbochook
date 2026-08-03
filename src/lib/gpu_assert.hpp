#pragma once
// =============================================================================
// lib/gpu_assert.hpp — every CUDA / HIP / SYCL call funnels through here.
//
// Backend runtimes report failure by RETURN CODE, and an unchecked return code
// is a bug that surfaces hundreds of lines later as a wrong number. Today's port
// spent an afternoon on exactly that shape: a failed allocation became a
// `nullptr`, a failed free was ignored entirely, and neither said anything.
//
// TWO checkers, because a destructor may not throw:
//
//   TC_GPU_CHECK(expr)          throws tc::Error naming the call and the code
//   TC_GPU_CHECK_NOTHROW(expr)  logs and continues -- for destructors ONLY
//
// Destructors are `noexcept(true)`. An asserting free in one calls
// std::terminate with no diagnostic, which is strictly worse than the unchecked
// free it replaced. This is not hypothetical: it is the defect the
// DeviceAllocation design in plans/02_DEVICE.md calls out by name.
//
// `nullopt` means SUCCESS. That shape (rather than a bool) lets each backend
// decide what counts as success -- see cudaErrorCudartUnloading below.
// =============================================================================

#include <optional>
#include <string>
#include <source_location>
#include "lib/error.hpp"
#include "lib/log.hpp"

#if   defined(TC_STDPAR_CUDA)
#  include <cuda_runtime.h>
#elif defined(TC_STDPAR_HIP)
#  include <hip/hip_runtime.h>
#elif defined(TC_STDPAR_SYCL)
#  include <sycl/sycl.hpp>
#endif

namespace tc::detail {

#if defined(TC_STDPAR_CUDA)
inline std::optional<std::string> gpu_error_string(cudaError_t rc) {
    // cudaErrorCudartUnloading is NOT a failure. It is what a free returns when
    // the runtime has already torn down during static destruction -- i.e. exactly
    // when an arena at namespace scope releases its pool. Treating it as an error
    // makes every clean exit print a spurious complaint.
    if (rc == cudaSuccess || rc == cudaErrorCudartUnloading) return std::nullopt;
    return std::to_string(int(rc)) + " " + cudaGetErrorString(rc);
}
#elif defined(TC_STDPAR_HIP)
inline std::optional<std::string> gpu_error_string(hipError_t rc) {
    if (rc == hipSuccess || rc == hipErrorDeinitialized) return std::nullopt;
    return std::to_string(int(rc)) + " " + hipGetErrorString(rc);
}
#endif

// Throwing check. Names the CALL, not just the code -- "cudaMallocManaged(&p,
// bytes) failed: 2 out of memory" is actionable; "error 2" is not.
inline void gpu_check_throw(std::optional<std::string> err, const char* expr,
                            std::source_location loc = std::source_location::current()) {
    if (!err) return;
    throw Error(Errc::out_of_memory,
                std::string(expr) + " failed: " + *err, loc);
}

// Non-throwing check, for destructors. Logs at error level and returns.
inline void gpu_check_log(std::optional<std::string> err, const char* expr,
                          std::source_location loc = std::source_location::current()) noexcept {
    if (!err) return;
    try {
        logger().error("{} failed: {} ({}:{})", expr, *err, loc.file_name(), loc.line());
    } catch (...) {
        // A logger that throws during teardown must not take the process with it.
    }
}

} // namespace tc::detail

// TC_DISABLE_GPU_ASSERTS keeps the call, drops the check -- for a measured hot
// path only. The call must still be EVALUATED, so this is `expr`, never nothing.
#if defined(TC_DISABLE_GPU_ASSERTS)
#  define TC_GPU_CHECK(expr)         (expr)
#  define TC_GPU_CHECK_NOTHROW(expr) (expr)
#else
#  define TC_GPU_CHECK(expr)         ::tc::detail::gpu_check_throw(::tc::detail::gpu_error_string(expr), #expr)
#  define TC_GPU_CHECK_NOTHROW(expr) ::tc::detail::gpu_check_log  (::tc::detail::gpu_error_string(expr), #expr)
#endif
