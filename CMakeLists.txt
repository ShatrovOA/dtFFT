#==============================================================================#
#                                                                              #
#   dtFFT - DataTyped Fast Fourier Transform                                   #
#   https://github.com/ShatrovOA/dtFFT                                         #
#                                                                              #
#==============================================================================#
#
#  Layout of this file:
#     1. Project definition
#     2. Build type & language standards
#     3. User options
#     4. MPI detection
#     5. Main library target
#     6. Core features (Fortran/MPI checks, persistent comm, RMA)
#     7. OpenMP
#     8. GPU support (CUDA, NCCL, NVSHMEM)
#     9. MPI linkage & profiler
#    10. Compression libraries
#    11. Compiler flags & coverage
#    12. Configuration header, include directories & public headers
#    13. Installation & package export
#    14. Subdirectories
#    15. Configuration summary
#
#==============================================================================#

cmake_minimum_required(VERSION 3.25)

#------------------------------------------------------------------------------#
# 1. Project definition
#------------------------------------------------------------------------------#
project(dtfft
  VERSION       3.2.0
  DESCRIPTION   "Library to perform FFT on a distributed memory cluster."
  HOMEPAGE_URL  "https://github.com/ShatrovOA/dtFFT"
  LANGUAGES     Fortran C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#------------------------------------------------------------------------------#
# 2. Build type & language standards
#------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release." FORCE)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD          14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------------------------#
# 3. User options
#------------------------------------------------------------------------------#

# --- Build & interfaces -------------------------------------------------------
option(DTFFT_BUILD_SHARED           "Build shared library"                                 ON)
option(DTFFT_BUILD_C_CXX_API        "Create C API"                                         ON)
option(DTFFT_BUILD_PYTHON_API       "Create Python API"                                    OFF)
option(DTFFT_BUILD_TESTS            "Create dtFFT tests"                                   OFF)
option(DTFFT_BUILD_EXAMPLES         "Build dtFFT examples"                                 ON)
option(DTFFT_ENABLE_COVERAGE        "Create coverage with gcov utility"                    OFF)

if(DTFFT_BUILD_PYTHON_API)
  set(DTFFT_BUILD_C_CXX_API ON)
endif()

# --- Core / MPI ---------------------------------------------------------------
option(DTFFT_USE_MPI                "Use Fortran `mpi` module instead of `mpi_f08`"        OFF)
option(DTFFT_ENABLE_PERSISTENT_COMM "Enable MPI persistent calls"                          OFF)
option(DTFFT_WITH_RMA               "Enable support of MPI RMA backends"                   OFF)
option(DTFFT_ENABLE_INPUT_CHECK     "Enable checks for user input for all dtFFT functions" ON)
option(DTFFT_WITH_PROFILER          "Build dtFFT with profiler support"                    OFF)

# --- FFT libraries ------------------------------------------------------------
option(DTFFT_WITH_FFTW              "Build dtFFT with FFTW support"                        OFF)
option(DTFFT_WITH_MKL               "Build dtFFT with MKL DFTI support"                    OFF)
option(DTFFT_WITH_CUFFT             "Build dtFFT with cuFFT support"                       OFF)
option(DTFFT_WITH_VKFFT             "Build dtFFT with VkFFT support"                       OFF)

# --- OpenMP -------------------------------------------------------------------
option(DTFFT_WITH_OPENMP            "Enable OpenMP parallelization for generic approach"   OFF)
option(DTFFT_WITH_FFTW_THREADS      "Enable threads support of FFTW library instead of OpenMP" OFF)

# --- GPU ----------------------------------------------------------------------
option(DTFFT_WITH_CUDA              "Build dtFFT with CUDA support"                        OFF)
option(DTFFT_WITH_MOCK_ENABLED      "Mock GPU libraries calls with CPU implementations for testing and debugging without GPU hardware" OFF)
option(DTFFT_ENABLE_DEVICE_CHECKS   "Enable error checking for all GPU libraries calls"    ON)
option(DTFFT_WITH_NCCL              "Build dtFFT with NCCL support. With nvfortran and DTFFT_WITH_CUDA=ON requires NCCL_ROOT environment variable pointing to a custom NCCL installation" OFF)
option(DTFFT_WITHOUT_NCCL           "Build dtFFT without NCCL support. Disables automatic NCCL usage when nvfortran is detected and DTFFT_WITH_CUDA=ON" OFF)
option(DTFFT_WITHOUT_NVSHMEM        "Build dtFFT without NVSHMEM-enabled backends. Disables automatic NVSHMEM usage when nvfortran is detected and DTFFT_WITH_CUDA=ON" OFF)

# --- Compression libraries ----------------------------------------------------
option(DTFFT_WITH_ZFP               "Enable ZFP support for data compression"              OFF)
option(DTFFT_WITH_SZ3               "Enable SZ3 support for data compression"              OFF)
option(DTFFT_WITH_SZP               "Enable SZp support for data compression"              OFF)
# option(DTFFT_WITH_CUSZ              "Enable cuSZ CUDA support for data compression"        OFF)
option(DTFFT_WITH_CUSZP             "Enable cuSZp support for data compression"            OFF)
option(DTFFT_WITH_BITCOMP           "Enable bitcomp support for data compression"          OFF)

# --- Derived settings ---------------------------------------------------------
# Transpose-only build when no FFT executor is requested
if(NOT DTFFT_WITH_FFTW AND NOT DTFFT_WITH_MKL AND NOT DTFFT_WITH_CUFFT AND NOT DTFFT_WITH_VKFFT)
  set(DTFFT_TRANSPOSE_ONLY ON)
endif()

# GPU FFT executors imply CUDA support
if(DTFFT_WITH_VKFFT OR DTFFT_WITH_CUFFT)
  set(DTFFT_WITH_CUDA ON)
endif()

#------------------------------------------------------------------------------#
# 4. MPI detection
#------------------------------------------------------------------------------#
find_package(MPI REQUIRED)

if(MPI_Fortran_HAVE_F08_MODULE)
  if(DTFFT_USE_MPI)
    add_compile_definitions(DTFFT_USE_MPI)
  endif()
else()
  if(NOT DTFFT_USE_MPI)
    message(FATAL_ERROR "Requested `mpi_f08` module is missing. Try configure with -DDTFFT_USE_MPI=ON")
  endif()
  if(NOT MPI_Fortran_HAVE_F90_MODULE)
    message(FATAL_ERROR "Neither `mpi_f08` nor `mpi` modules were found..")
  endif()
  add_compile_definitions(DTFFT_USE_MPI)
endif()

if(DTFFT_USE_MPI)
  set(mpi_mod "mpi")
else()
  set(mpi_mod "mpi_f08")
endif()

#------------------------------------------------------------------------------#
# 5. Main library target
#------------------------------------------------------------------------------#
# Link-time optimization for release builds
if(uppercase_CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

if(DTFFT_BUILD_SHARED)
  add_library(dtfft SHARED)
else()
  add_library(dtfft STATIC)
endif()
set_target_properties(dtfft PROPERTIES LINKER_LANGUAGE Fortran)
if(DTFFT_BUILD_PYTHON_API)
  set_target_properties(dtfft PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#------------------------------------------------------------------------------#
# 6. Core features
#------------------------------------------------------------------------------#
include(CheckMPIFeatures)
include(CheckFortranFeatures)

# MPI count / address kinds.
# 64-bit MPI indexing (MPI_COUNT_KIND / MPI_ADDRESS_KIND) is currently disabled.
target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:CNT_KIND=int32;ADDR_KIND=int32>)

if(DTFFT_ENABLE_INPUT_CHECK)
  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:ENABLE_INPUT_CHECK>)
endif()

# --- Required Fortran features ------------------------------------------------
check_assumed_rank_and_type()
check_block_statement()
if(NOT HAVE_ASSUMED_RANK_AND_TYPE OR NOT HAVE_BLOCK_STATEMENT)
  message(FATAL_ERROR "Both assumed-rank arrays and Block statement are required")
endif()

# --- MPI implementation workarounds -------------------------------------------
check_ompi_fix_required("${MPI_C_INCLUDE_DIRS}")
check_mpich_fix_required("${MPI_C_INCLUDE_DIRS}")

if(OMPI_FIX_REQUIRED)
  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:OMPI_FIX_REQUIRED>)
  message(STATUS "OpenMPI < 5 detected. Applying workaround for Collective non-blocking calls")
endif()
if(MPICH_FIX_REQUIRED AND NOT DTFFT_USE_MPI)
  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:MPICH_FIX_REQUIRED>)
  message(STATUS "MPICH < 4.1 detected. Applying workaround for MPI_Waitsome 0 indexes returned")
endif()

# --- Persistent communications ------------------------------------------------
if(DTFFT_ENABLE_PERSISTENT_COMM)
  check_persistent_collectives("${MPI_Fortran_MODULE_DIR}" "${MPI_Fortran_LIBRARIES}" ${mpi_mod})

  # Point-to-point persistent calls, e.g. MPI_Send_init (used by GPU p2p algorithms)
  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:ENABLE_PERSISTENT_COMM>)
  if(HAVE_PERSISTENT_COLLECTIVES)
    message(STATUS "Persistent communications ENABLED")
    # Collective persistent calls, e.g. MPI_Alltoall_init
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:ENABLE_PERSISTENT_COLLECTIVES>)
  else()
    message(WARNING "MPI implementation do not support collective persistent functions")
  endif()
endif()

# --- MPI RMA backends ---------------------------------------------------------
if(DTFFT_WITH_RMA)
  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_WITH_RMA>)
endif()

#------------------------------------------------------------------------------#
# 7. OpenMP
#------------------------------------------------------------------------------#
if(DTFFT_WITH_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS Fortran)
  target_compile_definitions(dtfft PRIVATE DTFFT_WITH_OPENMP)
  target_link_libraries(dtfft PRIVATE OpenMP::OpenMP_Fortran)
endif()

#------------------------------------------------------------------------------#
# 8. GPU support (CUDA, NCCL, NVSHMEM)
#------------------------------------------------------------------------------#
set(NCCL_FOUND  FALSE)
set(USE_NCCL    OFF)
set(USE_NVSHMEM OFF)

# Mock mode emulates NCCL on CPU
if(DTFFT_WITH_MOCK_ENABLED)
  set(NCCL_FOUND TRUE)
  set(USE_NCCL   ON)
endif()

if(DTFFT_WITH_CUDA AND NOT DTFFT_WITH_MOCK_ENABLED)
  # --- CUDA Toolkit -----------------------------------------------------------
  find_package(CUDAToolkit 11.0 REQUIRED)
  # CUDA::nvrtc is not linked explicitly: cmake implicitly links cuda_driver
  target_link_libraries(dtfft PUBLIC  CUDA::cudart)
  target_link_libraries(dtfft PRIVATE ${CMAKE_DL_LIBS})

  if(NOT DTFFT_ENABLE_DEVICE_CHECKS)
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DEVICE_NO_ERROR_CHECK>)
  endif()

  # --- NVIDIA HPC SDK ---------------------------------------------------------
  if(CMAKE_Fortran_COMPILER_ID MATCHES NVHPC)
    # <sdk>/compilers/bin/nvfortran -> <sdk>/cmake
    get_filename_component(NVHPC_ROOT_DIR "${CMAKE_Fortran_COMPILER}" DIRECTORY)  # <sdk>/compilers/bin
    get_filename_component(NVHPC_ROOT_DIR "${NVHPC_ROOT_DIR}" DIRECTORY)          # <sdk>/compilers
    get_filename_component(NVHPC_ROOT_DIR "${NVHPC_ROOT_DIR}" DIRECTORY)          # <sdk>
    list(APPEND CMAKE_PREFIX_PATH "${NVHPC_ROOT_DIR}/cmake")

    set(WITH_HPC_SDK ON)
    set(NVHPC_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  endif()

  # --- NCCL -------------------------------------------------------------------
  if(WITH_HPC_SDK)
    if(DTFFT_WITHOUT_NCCL)
      message(STATUS "NCCL support disabled via DTFFT_WITHOUT_NCCL")
    else()
      if(DTFFT_WITH_NCCL)
        message(STATUS "DTFFT_WITH_NCCL overrides NVHPC default NCCL search")
        find_package(NCCL REQUIRED) # FindNCCL.cmake
      endif()

      # DTFFT_WITH_NCCL is OFF or manual search failed: fall back to NVHPC SDK
      if(NOT NCCL_FOUND)
        find_package(NVHPC REQUIRED COMPONENTS NCCL)
        if(NVHPC_FOUND AND TARGET NVHPC::NCCL)
          message(STATUS "Using NCCL from NVHPC SDK")
          set(NCCL_FOUND TRUE)
          get_target_property(NCCL_INCLUDE_DIRS NVHPC::NCCL INTERFACE_INCLUDE_DIRECTORIES)
          get_target_property(NCCL_LIBRARIES    NVHPC::NCCL INTERFACE_LINK_LIBRARIES)
        else()
          message(WARNING "NVHPC compiler detected, but NCCL not found. Consider DTFFT_WITH_NCCL=ON with NCCL_ROOT or DTFFT_WITHOUT_NCCL=ON")
        endif()
      endif()
    endif()
  elseif(DTFFT_WITH_NCCL)
    # Non-NVHPC compiler: DTFFT_WITHOUT_NCCL is ignored
    find_package(NCCL REQUIRED) # FindNCCL.cmake
  endif()

  if(NCCL_FOUND)
    include(CheckNCCLFeatures)
    check_nccl_features("${NCCL_INCLUDE_DIRS}" "${NCCL_LIBRARIES}"
                        "${CUDAToolkit_INCLUDE_DIRS}" "${CUDAToolkit_LIBRARY_DIR}")

    set(USE_NCCL ON)
    if(NCCL_HAVE_MEMALLOC)
      target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:NCCL_HAVE_MEMALLOC>)
    endif()
    if(NCCL_HAVE_COMMREGISTER)
      target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:NCCL_HAVE_COMMREGISTER>)
    endif()
    target_link_libraries(dtfft PRIVATE ${NCCL_LIBRARIES})
  endif()

  # --- NVSHMEM / cuFFTMp ------------------------------------------------------
  if(WITH_HPC_SDK AND NOT DTFFT_WITHOUT_NVSHMEM)
    find_package(NVHPC REQUIRED COMPONENTS MATH)

    get_target_property(_dtfft_cufftmp_include_dirs NVHPC::CUFFTMP INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _dtfft_cufftmp_include_dirs OR _dtfft_cufftmp_include_dirs MATCHES "NOTFOUND$")
      set(_dtfft_cufftmp_include_dirs "")
    endif()
    include(CheckCuFFTMpFeatures)
    check_cufftmp_features("${_dtfft_cufftmp_include_dirs}" "${CUDAToolkit_INCLUDE_DIRS}")
    if(CUFFTMP_API_HAS_RESHAPE_COMM)
      target_compile_definitions(dtfft PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_CUFFTMP_API_HAS_RESHAPE_COMM>
      )
    endif()

    set(USE_NVSHMEM ON)
    target_link_libraries(dtfft PUBLIC NVHPC::CUFFTMP)
    if(CMAKE_Fortran_COMPILER_VERSION LESS "24.7")
      target_link_libraries(dtfft PRIVATE CUDA::nvml)
    endif()
  endif()
endif()

#------------------------------------------------------------------------------#
# 9. MPI linkage & profiler
#------------------------------------------------------------------------------#
target_link_libraries(dtfft PUBLIC MPI::MPI_Fortran)

if(DTFFT_WITH_PROFILER)
  target_compile_definitions(dtfft PRIVATE DTFFT_WITH_PROFILER)
  if(DTFFT_WITH_CUDA AND NOT DTFFT_WITH_MOCK_ENABLED)
    target_link_libraries(dtfft PRIVATE CUDA::nvtx3)
  else()
    find_package(caliper REQUIRED)
    message(STATUS "Found caliper: ${caliper_DIR}")
    target_link_libraries(dtfft PUBLIC caliper)
  endif()
endif()

#------------------------------------------------------------------------------#
# 10. Compression libraries
#------------------------------------------------------------------------------#
set(DTFFT_WITH_COMPRESSION OFF)

# --- ZFP ----------------------------------------------------------------------
if(DTFFT_WITH_ZFP)
  find_package(zfp REQUIRED)
  message(STATUS "Found zfp: ${zfp_DIR}")
  target_link_libraries(dtfft PRIVATE zfp::zfp)
  set(DTFFT_WITH_COMPRESSION ON)
  if(ZFP_WITH_CUDA)
    target_compile_definitions(dtfft PRIVATE ZFP_WITH_CUDA)
  endif()
  if(ZFP_WITH_OPENMP AND DTFFT_WITH_OPENMP)
    target_compile_definitions(dtfft PRIVATE ZFP_WITH_OPENMP)
  endif()
endif()

# --- SZ3 ----------------------------------------------------------------------
if(DTFFT_WITH_SZ3)
  find_package(SZ3 REQUIRED)
  if(NOT TARGET PkgConfig::ZSTD)
    find_package(PkgConfig)
    pkg_search_module(ZSTD IMPORTED_TARGET libzstd)
  endif()
  message(STATUS "Found SZ3: ${SZ3_DIR}")
  target_link_libraries(dtfft PRIVATE SZ3::SZ3)
  set(DTFFT_WITH_COMPRESSION ON)
endif()

# --- SZp ----------------------------------------------------------------------
if(DTFFT_WITH_SZP)
  find_package(SZp REQUIRED)
  message(STATUS "Found SZp: ${SZp_DIR}")
  target_link_libraries(dtfft PRIVATE SZp::SZp)
  set(DTFFT_WITH_COMPRESSION ON)
endif()

# --- cuSZ (CUDA) --------------------------------------------------------------
# if(DTFFT_WITH_CUSZ)
#   if(NOT DTFFT_WITH_CUDA OR DTFFT_WITH_MOCK_ENABLED)
#     message(FATAL_ERROR "DTFFT_WITH_CUSZ requires real CUDA support (DTFFT_WITH_CUDA=ON and DTFFT_WITH_MOCK_ENABLED=OFF)")
#   endif()
#   find_package(CUSZ CONFIG REQUIRED)
#   if(TARGET CUSZ::cusz)
#     target_link_libraries(dtfft PRIVATE CUSZ::cusz)
#   elseif(TARGET cusz)
#     target_link_libraries(dtfft PRIVATE cusz)
#   else()
#     message(FATAL_ERROR "CUSZ package does not provide CUSZ::cusz or cusz target")
#   endif()
#   set(DTFFT_WITH_COMPRESSION ON)
# endif()

# --- cuSZp (CUDA) -------------------------------------------------------------
if(DTFFT_WITH_CUSZP)
  if(NOT DTFFT_WITH_CUDA OR DTFFT_WITH_MOCK_ENABLED)
    message(FATAL_ERROR "DTFFT_WITH_CUSZP requires real CUDA support (DTFFT_WITH_CUDA=ON and DTFFT_WITH_MOCK_ENABLED=OFF)")
  endif()
  find_package(cuSZp CONFIG REQUIRED)
  if(DTFFT_BUILD_SHARED)
    target_link_libraries(dtfft PRIVATE cuSZp::cuSZp_shared)
  else()
    target_link_libraries(dtfft PRIVATE cuSZp::cuSZp_static)
  endif()
  set(DTFFT_WITH_COMPRESSION ON)
endif()

if(DTFFT_WITH_BITCOMP)
  if(NOT DTFFT_WITH_CUDA OR DTFFT_WITH_MOCK_ENABLED)
    message(FATAL_ERROR "DTFFT_WITH_BITCOMP requires real CUDA support (DTFFT_WITH_CUDA=ON and DTFFT_WITH_MOCK_ENABLED=OFF)")
  endif()
  find_package(nvcomp REQUIRED)
  if(DTFFT_BUILD_SHARED)
    target_link_libraries(dtfft PRIVATE nvcomp::nvcomp nvcomp::nvcomp_cpu)
  else()
    target_link_libraries(dtfft PRIVATE nvcomp::nvcomp_static nvcomp::nvcomp_cpu_static)
  endif()
  set(DTFFT_WITH_COMPRESSION ON)
endif()

if(DTFFT_ENABLE_PERSISTENT_COMM AND DTFFT_WITH_COMPRESSION)
  message(WARNING
    "DTFFT_ENABLE_PERSISTENT_COMM does not support compressed fused MPI exchanges; "
    "compressed fused plans will use non-persistent point-to-point communication.")
endif()
#------------------------------------------------------------------------------#
# 11. Compiler flags & coverage
#------------------------------------------------------------------------------#
set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")

# --- Compiler-specific flags --------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CompilerFlags.cmake")

# --- Coverage (GNU only) ------------------------------------------------------
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  if(DTFFT_RUNNING_CICD)
    set(DTFFT_ENABLE_COVERAGE ON)
  endif()

  if(DTFFT_ENABLE_COVERAGE)
    if(uppercase_CMAKE_BUILD_TYPE MATCHES RELEASE)
      message(FATAL_ERROR "Coverage can only be performed with debug build")
    endif()
    set(DTFFT_BUILD_TESTS ON)

    find_program(LCOV_PATH lcov)
    if(NOT LCOV_PATH)
      message(FATAL_ERROR "Unable to find lcov utility")
    endif()
    message(STATUS "Found lcov: ${LCOV_PATH}")

    string(REPLACE "." ";" GCOV_VERSION ${CMAKE_C_COMPILER_VERSION})
    list(GET GCOV_VERSION 0 GCOV_VERSION_MAJOR)
    find_program(GCOV_PATH gcov-${GCOV_VERSION_MAJOR})
    if(NOT GCOV_PATH)
      message(FATAL_ERROR "Unable to find gcov utility")
    endif()
    message(STATUS "Found gcov: ${GCOV_PATH}")

    if(NOT DTFFT_RUNNING_CICD)
      find_program(GENHTML_PATH genhtml)
      if(GENHTML_PATH)
        set(GENHTML_COMMAND ${GENHTML_PATH} coverage.info --output-directory coverage_html)
      endif()
    endif()

    # Run tests for every available platform before collecting coverage
    if(DTFFT_WITH_CUDA AND NOT SKIP_CUDA_COVERAGE)
      set(_dtfft_coverage_test_commands
        COMMAND DTFFT_PLATFORM=cuda ${CMAKE_MAKE_PROGRAM} test
        COMMAND DTFFT_PLATFORM=host ${CMAKE_MAKE_PROGRAM} test
      )
    else()
      set(_dtfft_coverage_test_commands
        COMMAND ${CMAKE_MAKE_PROGRAM} test
      )
    endif()

    add_custom_target(coverage
      ${_dtfft_coverage_test_commands}
      COMMAND ${LCOV_PATH} --ignore-errors negative --capture --directory "CMakeFiles" --output-file coverage.info --gcov-tool ${GCOV_PATH}
      COMMAND ${LCOV_PATH} --ignore-errors unused --remove coverage.info '/usr/*' '/opt/*' 'src/*/interfaces/' --output-file coverage.info
      COMMAND ${GENHTML_COMMAND}
    )

    target_link_libraries(dtfft PRIVATE gcov)
    set_target_properties(dtfft PROPERTIES
      COMPILE_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
      LINK_FLAGS    "--coverage"
    )
  endif()
endif()

#------------------------------------------------------------------------------#
# 12. Configuration header, include directories & public headers
#------------------------------------------------------------------------------#
# --- Values consumed by dtfft_config.h.in -------------------------------------
math(EXPR DTFFT_VERSION_CODE
  "${PROJECT_VERSION_MAJOR} * 100000 + ${PROJECT_VERSION_MINOR} * 1000 + ${PROJECT_VERSION_PATCH}"
)
set(DTFFT_VERSION "(X) * 100000 + (Y) * 1000 + (Z)")

if(NCCL_FOUND)
  set(DTFFT_WITH_NCCL ON)
else()
  set(DTFFT_WITH_NCCL OFF)
endif()

if(USE_NVSHMEM)
  set(DTFFT_WITH_NVSHMEM ON)
else()
  set(DTFFT_WITH_NVSHMEM OFF)
endif()

if(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(DTFFT_DEBUG ON)
else()
  set(DTFFT_DEBUG OFF)
endif()

set(DTFFT_CONF_HEADER ${PROJECT_BINARY_DIR}/dtfft_config.h CACHE STRING "DTFFT Config header")
configure_file(${PROJECT_SOURCE_DIR}/include/dtfft_config.h.in "${DTFFT_CONF_HEADER}")

# --- Target properties --------------------------------------------------------
set(DTFFT_MODULE_DIR ${PROJECT_BINARY_DIR}/modules)
set(DTFFT_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")
set(DTFFT_HEADERS    "${DTFFT_HEADER_DIR}/dtfft.f03")

if(DTFFT_BUILD_C_CXX_API)
  list(APPEND DTFFT_HEADERS
    "${DTFFT_HEADER_DIR}/dtfft.h"
    "${DTFFT_HEADER_DIR}/dtfft.hpp"
    "${DTFFT_CONF_HEADER}"
  )
  target_link_libraries(dtfft PUBLIC MPI::MPI_C MPI::MPI_CXX)
endif()

set_target_properties(dtfft PROPERTIES
  Fortran_MODULE_DIRECTORY ${DTFFT_MODULE_DIR}
  VERSION                  ${CMAKE_PROJECT_VERSION}
  PUBLIC_HEADER            "${DTFFT_HEADERS}"
)

target_include_directories(dtfft PUBLIC
  $<$<COMPILE_LANGUAGE:Fortran>:$<BUILD_INTERFACE:${DTFFT_MODULE_DIR}>>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/core/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<$<COMPILE_LANGUAGE:Fortran>:$<INSTALL_INTERFACE:include/modules>>
  $<INSTALL_INTERFACE:include>
)

#------------------------------------------------------------------------------#
# 13. Installation & package export
#------------------------------------------------------------------------------#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library installation directory")
set(DTFFT_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/dtfft"
    CACHE STRING "Path to dtFFT CMake files")

install(TARGETS dtfft
  EXPORT        dtfftTargets
  LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME       DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION include
)

install(EXPORT dtfftTargets
  DESTINATION    "${DTFFT_INSTALL_CMAKEDIR}"
  CONFIGURATIONS Debug Release
  FILE           dtfftTargets.cmake
)

write_basic_package_version_file(dtfftConfigVersion.cmake
  VERSION       ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/dtfftConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfig.cmake"
  INSTALL_DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfigVersion.cmake"
  DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
)

install(DIRECTORY ${DTFFT_MODULE_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.mod"
)

#------------------------------------------------------------------------------#
# 14. Subdirectories
#------------------------------------------------------------------------------#
if(DTFFT_RUNNING_CICD)
  set(DTFFT_TEST_MAX_PROCS 8)
  if(DTFFT_MPI_VENDOR STREQUAL "openmpi")
    set(DTFFT_TEST_MPIEXEC_FLAGS "--oversubscribe")
  endif()
else()
  set(DTFFT_TEST_MAX_PROCS ${MPIEXEC_MAX_NUMPROCS})
endif()

function(dtfft_add_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE dtfft)
endfunction()

function(dtfft_add_test target source)
  cmake_parse_arguments(PARSE_ARGV 2 TEST
    "TEST_UTILS;WITH_RMA;NAME_NPROCS"
    "MIN_PROCS;MAX_PROCS;LINKER_LANGUAGE;SKIP_RETURN_CODE"
    "LIBRARIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS"
  )

  dtfft_add_executable(${target} ${source})
  if(TEST_TEST_UTILS)
    target_link_libraries(${target} PRIVATE test_suite)
  endif()
  if(TEST_LIBRARIES)
    target_link_libraries(${target} PRIVATE ${TEST_LIBRARIES})
  endif()
  if(TEST_INCLUDE_DIRECTORIES)
    target_include_directories(${target} PRIVATE ${TEST_INCLUDE_DIRECTORIES})
  endif()
  if(TEST_COMPILE_DEFINITIONS)
    target_compile_definitions(${target} PRIVATE ${TEST_COMPILE_DEFINITIONS})
  endif()
  if(TEST_WITH_RMA AND DTFFT_WITH_RMA)
    target_compile_definitions(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_WITH_RMA>
    )
  endif()
  if(DTFFT_RUNNING_CICD)
    target_compile_definitions(${target} PRIVATE DTFFT_RUNNING_CICD)
  endif()

  if(TEST_LINKER_LANGUAGE)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE ${TEST_LINKER_LANGUAGE})
  else()
    get_filename_component(_source_extension "${source}" EXT)
    if(_source_extension STREQUAL ".F90")
      set_target_properties(${target} PROPERTIES LINKER_LANGUAGE Fortran)
    elseif(_source_extension STREQUAL ".cpp" OR _source_extension STREQUAL ".cxx")
      set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
    elseif(_source_extension STREQUAL ".c")
      set_target_properties(${target} PROPERTIES LINKER_LANGUAGE C)
    endif()
  endif()

  if(NOT DEFINED TEST_MIN_PROCS)
    set(TEST_MIN_PROCS 1)
  endif()
  if(NOT DEFINED TEST_MAX_PROCS)
    set(TEST_MAX_PROCS ${DTFFT_TEST_MAX_PROCS})
  endif()

  if(TEST_MAX_PROCS LESS TEST_MIN_PROCS)
    message(WARNING "${target} needs at least ${TEST_MIN_PROCS} MPI ranks, but only ${TEST_MAX_PROCS} are available. Test is not registered")
    return()
  endif()

  foreach(_nproc RANGE ${TEST_MIN_PROCS} ${TEST_MAX_PROCS})
    if(TEST_NAME_NPROCS)
      set(_test_name "${target}_nproc_${_nproc}")
    else()
      set(_test_name "${target}")
    endif()
    add_test(
      NAME "${_test_name}"
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_nproc} ${DTFFT_TEST_MPIEXEC_FLAGS} ./${target}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(DEFINED TEST_SKIP_RETURN_CODE)
      set_tests_properties("${_test_name}" PROPERTIES SKIP_RETURN_CODE ${TEST_SKIP_RETURN_CODE})
    endif()
  endforeach()
endfunction()

if(DTFFT_BUILD_PYTHON_API)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 3.0.0 REQUIRED)
endif()

if(DTFFT_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src)

if(DTFFT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#------------------------------------------------------------------------------#
# 15. Configuration summary
#------------------------------------------------------------------------------#
function(dtfft_summary_line label value)
  string(LENGTH "${label}" _len)
  math(EXPR _pad "27 - ${_len}")
  if(_pad LESS 1)
    set(_pad 1)
  endif()
  string(REPEAT " " ${_pad} _spaces)
  message(STATUS "  ${label}${_spaces}: ${value}")
endfunction()

message(STATUS "")
message(STATUS "==================== dtFFT configuration summary ====================")
message(STATUS "General")
dtfft_summary_line("Version"            "${dtfft_VERSION}")
dtfft_summary_line("Build type"         "${uppercase_CMAKE_BUILD_TYPE}")
dtfft_summary_line("Fortran compiler"   "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
dtfft_summary_line("Install dir"        "${CMAKE_INSTALL_PREFIX}")
dtfft_summary_line("Build shared libs"  "${DTFFT_BUILD_SHARED}")
dtfft_summary_line("Build C/C++ API"    "${DTFFT_BUILD_C_CXX_API}")
dtfft_summary_line("Build Python API"   "${DTFFT_BUILD_PYTHON_API}")
dtfft_summary_line("Build tests"        "${DTFFT_BUILD_TESTS}")
dtfft_summary_line("Coverage"           "${DTFFT_ENABLE_COVERAGE}")
dtfft_summary_line("Profiler"           "${DTFFT_WITH_PROFILER}")

message(STATUS "Parallelism")
dtfft_summary_line("Use `mpi` module"   "${DTFFT_USE_MPI}")
dtfft_summary_line("OpenMP enabled"     "${DTFFT_WITH_OPENMP}")
dtfft_summary_line("CUDA enabled"       "${DTFFT_WITH_CUDA}")
if(DTFFT_WITH_CUDA)
  if(DTFFT_WITH_MOCK_ENABLED)
    dtfft_summary_line("Mock CUDA"      "ON")
  endif()
  dtfft_summary_line("NCCL enabled"     "${DTFFT_WITH_NCCL}")
  if(WITH_HPC_SDK)
    dtfft_summary_line("NVSHMEM enabled" "${DTFFT_WITH_NVSHMEM}")
  endif()
endif()

message(STATUS "FFT executors")
if(DTFFT_TRANSPOSE_ONLY)
  message(STATUS "  Building transpose only interface")
else()
    foreach(_lib IN ITEMS FFTW MKL CUFFT VKFFT)
        dtfft_summary_line("${_lib}" "${DTFFT_WITH_${_lib}}")
    endforeach()
endif()

message(STATUS "Compression")
dtfft_summary_line("Enabled" "${DTFFT_WITH_COMPRESSION}")
if(DTFFT_WITH_COMPRESSION)
    foreach(_lib IN ITEMS ZFP SZ3 SZP CUSZP BITCOMP)
        dtfft_summary_line("${_lib}" "${DTFFT_WITH_${_lib}}")
    endforeach()
endif()
message(STATUS "=====================================================================")
message(STATUS "")
