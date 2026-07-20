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
#   gpu       : nvc++  -stdpar=gpu       (offload; managed memory by default)
#   multicore : nvc++  -stdpar=multicore (CPU threads)
#   off       : g++/clang, host-serial   (par_unseq → seq via TC_STDPAR_OFF; no TBB)
set(TC_STDPAR "off" CACHE STRING "stdpar backend: gpu | multicore | off")
set_property(CACHE TC_STDPAR PROPERTY STRINGS gpu multicore off)
set(TC_GPU_ARCH "cc70" CACHE STRING "GPU arch for nvc++ (cc70 | cc80 | cc90)")

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
elseif(TC_STDPAR STREQUAL "multicore")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "TC_STDPAR=multicore requires nvc++ (NVHPC).")
  endif()
  target_compile_options(tc_flags INTERFACE -stdpar=multicore)
  target_link_options   (tc_flags INTERFACE -stdpar=multicore)
else() # off → host serial; src/numerics/parallel.hpp picks std::execution::seq (no TBB dep)
  target_compile_definitions(tc_flags INTERFACE TC_STDPAR_OFF)
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

# ── the library (header-only for now; becomes STATIC when .cpp TUs appear) ────
add_library(turbochook INTERFACE)
target_link_libraries(turbochook INTERFACE tc_flags)

# ── executables (top-level src/*.cpp are the thin mains; subdir *.hpp is the lib)
file(GLOB TC_APPS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
foreach(src ${TC_APPS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE turbochook)
endforeach()

# ── tests (doctest via FetchContent; one runner via CTest) ───────────────────
include(CTest)  # defines BUILD_TESTING (default ON)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.5.3)
  FetchContent_MakeAvailable(doctest)

  file(GLOB TC_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
  if(TC_TEST_SRC)
    add_executable(tc_tests ${TC_TEST_SRC})
    target_link_libraries(tc_tests PRIVATE turbochook doctest::doctest)
    include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
    doctest_discover_tests(tc_tests)   # registers each TEST_CASE with CTest
  endif()
endif()
