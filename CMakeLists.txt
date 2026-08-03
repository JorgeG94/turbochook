cmake_minimum_required(VERSION 3.25)
project(turbochook LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
# TurboChook — GPU-native C++23 solver on ISO standard parallelism.
# SKETCH (M0). Validate against a real nvc++/g++ toolchain; C++23 mdspan/print
# availability depends on the underlying stdlib, not just the compiler (M0 checks it).
# ─────────────────────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ── stdpar backend selection ─────────────────────────────────────────────────
#   gpu       : nvc++  -stdpar=gpu       (NVIDIA offload)
#   multicore : nvc++  -stdpar=multicore (CPU threads)
#   hip       : hipcc  --hipstdpar       (AMD offload)
#   sycl      : icpx   -fsycl             (Intel offload, raw SYCL)
#   off       : g++/clang, host-serial   (par_unseq → seq via TC_STDPAR_OFF; no TBB)
#
# All five run the SAME source: every parallel loop goes through
# tc::do_concurrent (src/numerics/parallel.hpp), which is the only place a
# backend dialect appears. Verified end to end by spikes/alpha on cc70, cc90,
# hipstdpar and oneDPL -- see docs/CONTRACT_MEMORY.md §0.0.
set(TC_STDPAR "off" CACHE STRING "stdpar backend: gpu | multicore | hip | sycl | off")
set_property(CACHE TC_STDPAR PROPERTY STRINGS gpu multicore hip sycl off)
set(TC_GPU_ARCH "cc70" CACHE STRING "GPU arch for nvc++ (cc70 | cc80 | cc90)")
set(TC_AMD_ARCH "gfx90a" CACHE STRING "GPU arch for hipcc (gfx90a | gfx942 | ...)")

# One INTERFACE target carries flags + include path + defines to everything.
add_library(tc_flags INTERFACE)
target_include_directories(tc_flags INTERFACE ${CMAKE_SOURCE_DIR}/src)

if(TC_STDPAR STREQUAL "gpu")
  # Explicit gpu with a non-NVHPC compiler is a HARD ERROR — never a silent CPU
  # downgrade.
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "TC_STDPAR=gpu requires nvc++ (NVHPC); got ${CMAKE_CXX_COMPILER_ID}.")
  endif()
  target_compile_options(tc_flags INTERFACE -stdpar=gpu -gpu=${TC_GPU_ARCH})
  target_link_options   (tc_flags INTERFACE -stdpar=gpu -gpu=${TC_GPU_ARCH})
  target_compile_definitions(tc_flags INTERFACE TC_STDPAR_CUDA)
elseif(TC_STDPAR STREQUAL "multicore")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "TC_STDPAR=multicore requires nvc++ (NVHPC).")
  endif()
  target_compile_options(tc_flags INTERFACE -stdpar=multicore)
  target_link_options   (tc_flags INTERFACE -stdpar=multicore)

elseif(TC_STDPAR STREQUAL "hip")
  # AMD. --hipstdpar routes std::execution::par_unseq through rocThrust. Verified
  # by spikes/alpha to accept explicitly hipMalloc'd device pointers -- which was
  # NOT a safe assumption, since hipstdpar's design leans on HMM/XNACK to make
  # ORDINARY HOST allocations device-reachable. Some configurations still want
  # HSA_XNACK=1 in the environment.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "TC_STDPAR=hip needs hipcc (a Clang derivative); got ${CMAKE_CXX_COMPILER_ID}.\n"
      "  cmake -B build_hip -DCMAKE_CXX_COMPILER=hipcc -DTC_STDPAR=hip -DTC_AMD_ARCH=gfx90a")
  endif()
  target_compile_options(tc_flags INTERFACE --hipstdpar --offload-arch=${TC_AMD_ARCH})
  target_link_options   (tc_flags INTERFACE --hipstdpar --offload-arch=${TC_AMD_ARCH})
  target_compile_definitions(tc_flags INTERFACE TC_STDPAR_HIP)

elseif(TC_STDPAR STREQUAL "sycl")
  # Intel. RAW SYCL -- no oneDPL. Note the deliberate absence of
  # -fsycl-pstl-offload, measured to double binary size for no gain (spike 05).
  #
  # TC_STDPAR_SYCL makes parallel.hpp take the oneDPL branch: its own algorithms,
  # its own device policy (which needs a queue), and oneapi::dpl::counting_iterator
  # instead of std::views::iota -- oneDPL REJECTS iota_view as a read-only proxy
  # iterator. Every other backend keeps iota_view, which is the shape verified to
  # offload on nvc++.
  #
  # NOT SUFFICIENT ON ITS OWN: src/lib/arena.hpp is a std::vector<std::byte>, i.e.
  # ordinary host heap. nvc++ -stdpar promotes the heap to MANAGED memory, which is
  # why that works today; oneDPL has no such mechanism, so a device-policy kernel
  # over arena memory fails at RUNTIME. Intel needs the arena on USM
  # (sycl::malloc_shared / malloc_device) before it can actually run.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR
      "TC_STDPAR=sycl requires icpx (IntelLLVM); got ${CMAKE_CXX_COMPILER_ID}.\n"
      "  cmake -B build_sycl -DCMAKE_CXX_COMPILER=icpx -DTC_STDPAR=sycl")
  endif()
  # -fsycl-unnamed-lambda: default-on in recent icpx, spelled out because the
  # oneDPL policy path is sensitive to it and a silent default is a bad thing to
  # depend on twice in one day.
  target_compile_options(tc_flags INTERFACE -fsycl -fsycl-unnamed-lambda)
  target_link_options   (tc_flags INTERFACE -fsycl)
  target_compile_definitions(tc_flags INTERFACE TC_STDPAR_SYCL)

else() # off → host serial; src/numerics/parallel.hpp picks std::execution::seq (no TBB dep)
  target_compile_definitions(tc_flags INTERFACE TC_STDPAR_OFF)
  # libc++ (Apple Clang / LLVM Clang) gates the <execution> PSTL policies behind
  # its experimental library — without this flag std::execution::seq is simply
  # absent ("no member named 'seq'"). GNU libstdc++ ships them unconditionally
  # and g++ rejects the flag, so scope it to Clang. Compile + link (PSTL runtime).
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(tc_flags INTERFACE -fexperimental-library)
    target_link_options   (tc_flags INTERFACE -fexperimental-library)
  endif()
endif()

# ── NetCDF output (optional; thin C-ABI bridge over libnetcdf-c, per the dep policy) ─
# Config-FIRST, like rakali: netcdf-c ships netCDFConfig.cmake → the netCDF::netcdf
# imported target (carries include dirs + rpath). Fall back to pkg-config, then a manual
# find. Point it with -DCMAKE_PREFIX_PATH=<prefix> (standard) or -DNETCDF_ROOT=<prefix>.
option(TC_NETCDF "Build with NetCDF output (needs libnetcdf-c)" OFF)
if(TC_NETCDF)
  set(NETCDF_ROOT "" CACHE PATH "NetCDF-C install prefix (optional hint)")
  find_package(netCDF QUIET HINTS "${NETCDF_ROOT}" "$ENV{NETCDF_DIR}")
  if(netCDF_FOUND)
    set(_tc_nc netCDF::netcdf)
    message(STATUS "TurboChook NetCDF         : find_package (${netCDF_DIR})")
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(NETCDF QUIET IMPORTED_TARGET netcdf)
    endif()
    if(TARGET PkgConfig::NETCDF)
      set(_tc_nc PkgConfig::NETCDF)
      message(STATUS "TurboChook NetCDF         : pkg-config")
    else()
      find_path   (NETCDF_INCLUDE_DIR netcdf.h HINTS ${NETCDF_ROOT}/include ENV NETCDF_DIR)
      find_library(NETCDF_LIBRARY NAMES netcdf HINTS ${NETCDF_ROOT}/lib     ENV NETCDF_DIR)
      if(NOT NETCDF_INCLUDE_DIR OR NOT NETCDF_LIBRARY)
        message(FATAL_ERROR "TC_NETCDF=ON: netcdf not found via find_package / pkg-config / "
                            "NETCDF_ROOT. Set -DCMAKE_PREFIX_PATH=<prefix> or -DNETCDF_ROOT=<prefix>.")
      endif()
      add_library(tc_netcdf_manual INTERFACE)
      target_include_directories(tc_netcdf_manual INTERFACE ${NETCDF_INCLUDE_DIR})
      target_link_libraries     (tc_netcdf_manual INTERFACE ${NETCDF_LIBRARY})
      get_filename_component(_nc_libdir ${NETCDF_LIBRARY} DIRECTORY)
      target_link_options       (tc_netcdf_manual INTERFACE "-Wl,-rpath,${_nc_libdir}")
      set(_tc_nc tc_netcdf_manual)
      message(STATUS "TurboChook NetCDF         : manual (${NETCDF_LIBRARY})")
    endif()
  endif()
  target_compile_definitions(tc_flags INTERFACE TC_HAVE_NETCDF)
  target_link_libraries     (tc_flags INTERFACE ${_tc_nc})
endif()

message(STATUS "TurboChook stdpar backend : ${TC_STDPAR}")
message(STATUS "TurboChook compiler       : ${CMAKE_CXX_COMPILER_ID}")

# ── the library: header-only until subdir .cpp TUs appear, then STATIC ────────
# Lib sources are src/<subdir>/*.cpp (the thin program mains live in examples/programs/).
file(GLOB TC_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*/*.cpp)
if(TC_LIB_SRC)
  add_library(turbochook STATIC ${TC_LIB_SRC})
  target_link_libraries(turbochook PUBLIC tc_flags)
else()
  add_library(turbochook INTERFACE)
  target_link_libraries(turbochook INTERFACE tc_flags)
endif()

# ── example programs (examples/programs/*.cpp are the thin mains; each links the lib)
file(GLOB TC_APPS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/examples/programs/*.cpp)
foreach(src ${TC_APPS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE turbochook)
endforeach()

# ── tests (doctest via FetchContent; one runner via CTest) ───────────────────
include(CTest)  # defines BUILD_TESTING (default ON)
if(BUILD_TESTING)
  # Prefer an already-installed doctest. FetchContent's git clone runs at CONFIGURE
  # time, so on a compute node with no outbound network it BLOCKS until the TCP
  # timeout -- silently, with no output, looking exactly like a hung cmake. That is
  # the commonest "cmake is stuck" on a cluster and it has nothing to do with the
  # compiler being configured.
  find_package(doctest QUIET)
  if(NOT doctest_FOUND)
    message(STATUS "doctest not installed -> cloning from GitHub (needs OUTBOUND NETWORK).")
    message(STATUS "  If this hangs, you are on a node without network access.")
    message(STATUS "  Re-run with -DBUILD_TESTING=OFF, or install doctest and set")
    message(STATUS "  -DCMAKE_PREFIX_PATH=<prefix>.")
    set(FETCHCONTENT_QUIET OFF)          # show git progress rather than blocking mutely
    include(FetchContent)
    FetchContent_Declare(doctest
      GIT_REPOSITORY https://github.com/doctest/doctest.git
      GIT_TAG        v2.5.3)
    FetchContent_MakeAvailable(doctest)
  endif()

  file(GLOB TC_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
  if(TC_TEST_SRC)
    add_executable(tc_tests ${TC_TEST_SRC})
    target_link_libraries(tc_tests PRIVATE turbochook doctest::doctest)
    # doctest_SOURCE_DIR only exists on the FetchContent path; an installed
    # doctest ships the module on CMAKE_MODULE_PATH instead.
    if(DEFINED doctest_SOURCE_DIR)
      include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
    else()
      include(doctest)
    endif()
    doctest_discover_tests(tc_tests)   # registers each TEST_CASE with CTest

    # A SOURCE-level guard for a GPU-only bug. SYCL names an unnamed-lambda kernel
    # by its mangled closure type, mangling does not encode the TU, and doctest's
    # DOCTEST_ANON_FUNC_<n> counter restarts per file -- so two test files collide
    # on one kernel name. That cost 24 of 67 tests on Aurora, half of them as
    # WRONG NUMBERS rather than errors, and no compiler flag prevents it.
    # Runs with no GPU and no SYCL toolchain, which is the point: the bug is
    # invisible on the machine most work happens on.
    add_test(NAME tu_namespaces
             COMMAND ${CMAKE_SOURCE_DIR}/tools/check_tu_namespaces.sh)
  endif()
endif()
