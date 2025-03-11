cmake_minimum_required(VERSION 3.20)
project(dtfft VERSION 2.0.0
              DESCRIPTION "Library to perform FFT on a distributed memory cluster."
              HOMEPAGE_URL "https://github.com/ShatrovOA/dtFFT"
              LANGUAGES Fortran C CXX)

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

set(CMAKE_C_STANDARD 11)
set(C_STANDARD_REQUIRED ON)

# FFT Libraries
option(DTFFT_WITH_FFTW "Build dtFFT with FFTW support" OFF)
option(DTFFT_WITH_MKL "Build dtFFT with MKL DFTI support" OFF)
option(DTFFT_WITH_CUFFT "Build dtFFT with cuFFT support" OFF)
option(DTFFT_WITH_VKFFT "Build dtFFT with VkFFT support" OFF)

option(DTFFT_BUILD_TESTS "Create dtFFT tests" OFF)
option(DTFFT_BUILD_SHARED "Build shared library" ON)
option(DTFFT_BUILD_C_CXX_API "Create C API" ON)

option(DTFFT_ENABLE_COVERAGE "Create coverage with gcov utility" OFF)

option(DTFFT_USE_MPI "Use Fortran `mpi` module instead of `mpi_f08`" OFF)

option(DTFFT_ENABLE_PERSISTENT_COMM "Enable MPI persistent calls" OFF)

option(DTFFT_WITH_PROFILER "Build dtFFT with profiler support" OFF)

# Device libraries
option(DTFFT_WITH_CUDA "Build dtFFT with CUDA support" OFF)
# option(DTFFT_WITH_HPC_SDK "Build dtFFT with Nvidia HPC SDK Tools.
#                            This means we are using nvfortran, nvc, nvc++ compiler set and CUDA aware-OpenMPI shipped with HPC SDK.
#                            When DTFFT_WITH_CUDA=ON, NCCL and NVSHMEM-enabled backends will be available automatically" OFF)

option(DTFFT_WITH_NCCL "Build dtFFT with NCCL support.
                        Enabling this option when DTFFT_WITH_HPC_SDK=ON and DTFFT_WITH_CUDA=ON
                        requires to set NCCL_ROOT enviroment variable to find custom installation of NCCL" OFF)

option(DTFFT_WITHOUT_NCCL "Build dtFFT without NCCL support.
                           Disables automatic NCCL usage when DTFFT_WITH_HPC_SDK=ON AND DTFFT_WITH_CUDA=ON" OFF)
option(DTFFT_WITHOUT_NVSHMEM "Build dtFFT without NVSHMEM-enabled backends.
                              Disables automatic NVSHMEM usage when DTFFT_WITH_HPC_SDK=ON AND DTFFT_WITH_CUDA=ON" OFF)



# set(DTFFT_CUDA_CC_LIST "70;80;90" CACHE STRING "List of CUDA compute capabilities to build for")

set(DTFFT_RUNNING_CICD OFF CACHE STRING "Running tests in GitHub actions")


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

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
    message(FATAL_ERROR "Neither `mpi_f08` nor `mpi` modules were found.." )
  endif()
  add_compile_definitions(DTFFT_USE_MPI)
endif()


if(DTFFT_BUILD_SHARED)
  add_library(dtfft SHARED)
else()
  add_library(dtfft STATIC)
endif()

include(CheckFortranFeatures)

check_assumed_rank_and_type()
check_block_statement()

if (NOT HAVE_ASSUMED_RANK_AND_TYPE OR NOT HAVE_BLOCK_STATEMENT)
  message(FATAL_ERROR "Both assumed-rank arrays and Block statement are required")
endif()

include(CheckMPIFeatures)

if ( DTFFT_ENABLE_PERSISTENT_COMM )

  check_persistent_collectives(${MPI_Fortran_MODULE_DIR} ${MPI_Fortran_LIBRARIES})

  if ( HAVE_PERSISTENT_COLLECTIVES )
    message(STATUS "Persistent communications ENABLED")
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_HAVE_PERSISTENT_COLLECTIVES>)
  else()
    message(WARNING "MPI implementation do not support collective persistent functions")
  endif()
endif()

if( NOT DTFFT_WITH_FFTW
    AND NOT DTFFT_WITH_MKL
    AND NOT DTFFT_WITH_CUFFT
    AND NOT DTFFT_WITH_VKFFT
)
  set ( DTFFT_TRANSPOSE_ONLY ON )
endif()
if ( DTFFT_WITH_VKFFT OR DTFFT_WITH_CUFFT )
  set( DTFFT_WITH_CUDA ON )
endif()


math(EXPR DTFFT_VERSION_CODE
  "${PROJECT_VERSION_MAJOR} * 100000 + ${PROJECT_VERSION_MINOR} * 1000 + ${PROJECT_VERSION_PATCH}"
)

set ( DTFFT_VERSION "(X) * 100000 + (Y) * 1000 + (Z)" )
set ( DTFFT_CONF_HEADER ${PROJECT_BINARY_DIR}/dtfft_config.h CACHE STRING "DTFFT Config header")
configure_file( ${PROJECT_SOURCE_DIR}/include/dtfft_config.h.in "${DTFFT_CONF_HEADER}" )


if(DTFFT_WITH_CUDA)
  find_package(CUDAToolkit 11.0 REQUIRED)
  target_link_libraries(dtfft PRIVATE CUDA::cuda_driver)
  target_link_libraries(dtfft PRIVATE CUDA::nvrtc)
  target_link_libraries(dtfft PUBLIC CUDA::cudart)

  if( CMAKE_Fortran_COMPILER_ID MATCHES NVHPC )
    # Get the directory two levels up
    get_filename_component(PARENT_DIR "${CMAKE_Fortran_COMPILER}" DIRECTORY)  # /compilers/bin/
    get_filename_component(PARENT_DIR "${PARENT_DIR}" DIRECTORY)  # /compilers
    get_filename_component(PARENT_DIR "${PARENT_DIR}" DIRECTORY)  # /
    list(APPEND CMAKE_PREFIX_PATH "${PARENT_DIR}/cmake")

    set(WITH_HPC_SDK ON)
    set(NVHPC_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  endif()

  set(NCCL_FOUND FALSE)
  if (WITH_HPC_SDK)
    if (DTFFT_WITHOUT_NCCL)
      message(STATUS "NCCL support disabled via DTFFT_WITHOUT_NCCL")
    else()
      if (DTFFT_WITH_NCCL)
        message(STATUS "DTFFT_WITH_NCCL overrides NVHPC default NCCL search")
        find_package(NCCL REQUIRED) # Use FindNCCL.cmake
      endif()

      # If DTFFT_WITH_NCCL is OFF or manual search failed, try NVHPC
      if (NOT NCCL_FOUND)

        find_package(NVHPC REQUIRED COMPONENTS NCCL)
        if (NVHPC_FOUND AND TARGET NVHPC::NCCL)
          message(STATUS "Using NCCL from NVHPC SDK")
          set(NCCL_FOUND TRUE)
          get_target_property(NCCL_INCLUDE_DIRS NVHPC::NCCL INTERFACE_INCLUDE_DIRECTORIES)
          get_target_property(NCCL_LIBRARIES NVHPC::NCCL INTERFACE_LINK_LIBRARIES)
        else()
          message(WARNING "NVHPC compiler detected, but NCCL not found. Consider DTFFT_WITH_NCCL=ON with NCCL_ROOT or DTFFT_WITHOUT_NCCL=ON")
        endif()
      endif()
    endif()
  else()
    # Non-NVHPC compiler: DTFFT_WITHOUT_NCCL ignored
    if (DTFFT_WITH_NCCL)
      find_package(NCCL REQUIRED) # Use FindNCCL.cmake
    endif()
  endif()

  # If NCCL found, check features
  if (NCCL_FOUND)
    # Find CUDA Toolkit for NCCL feature checks
    include(CheckNCCLFeatures)
    check_nccl_features("${NCCL_INCLUDE_DIRS}" "${NCCL_LIBRARIES}"
                        "${CUDAToolkit_INCLUDE_DIRS}" "${CUDAToolkit_LIBRARY_DIR}")

    set(USE_NCCL ON)
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_WITH_NCCL>)
    if ( NCCL_HAVE_MEMALLOC )
      target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:NCCL_HAVE_MEMALLOC>)
    endif()
    if ( NCCL_HAVE_COMMREGISTER )
      target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:NCCL_HAVE_COMM_REGISTER>)
    endif()
    target_include_directories(dtfft PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(dtfft PRIVATE ${NCCL_LIBRARIES})
  endif()

  if ( DTFFT_USE_MPI )
    set(HAVE_MPI_INT64 OFF)
  else()
    check_int64_supported(${MPI_Fortran_MODULE_DIR} ${MPI_Fortran_LIBRARIES})
  endif()
  if ( HAVE_MPI_INT64 )
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:CNT_KIND=MPI_COUNT_KIND;ADDR_KIND=MPI_ADDRESS_KIND>)
  else()
    message(WARNING "MPI 64bit integer indexing is disabled. Overflows might occur if MPI Backends will be used")
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:CNT_KIND=int32;ADDR_KIND=int32>)
  endif()

  if( WITH_HPC_SDK AND NOT DTFFT_WITHOUT_NVSHMEM)
    find_package(NVHPC REQUIRED COMPONENTS MATH)

    set(USE_NVSHMEM ON)
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran,C>:DTFFT_WITH_NVSHMEM>)
    target_link_libraries(dtfft PRIVATE NVHPC::CUFFTMP)
    target_link_libraries(dtfft PRIVATE CUDA::nvml)
  endif()
endif()


target_link_libraries(dtfft PUBLIC MPI::MPI_Fortran)

set(DTFFT_MODULE_DIR ${PROJECT_BINARY_DIR}/modules)
set_target_properties(dtfft PROPERTIES Fortran_MODULE_DIRECTORY ${DTFFT_MODULE_DIR})
set_target_properties(dtfft PROPERTIES VERSION ${CMAKE_PROJECT_VERSION})
set(DTFFT_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")
SET(DTFFT_HEADERS "${DTFFT_HEADER_DIR}/dtfft.f03")

if(DTFFT_BUILD_C_CXX_API)
  set(DTFFT_HEADERS "${DTFFT_HEADERS};${DTFFT_HEADER_DIR}/dtfft.h;${DTFFT_HEADER_DIR}/dtfft.hpp;${DTFFT_CONF_HEADER}")
  target_link_libraries(dtfft PUBLIC MPI::MPI_CXX)
endif()
set_target_properties(dtfft PROPERTIES PUBLIC_HEADER "${DTFFT_HEADERS}")
target_include_directories(dtfft PUBLIC
  $<BUILD_INTERFACE:${DTFFT_MODULE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<$<COMPILE_LANGUAGE:Fortran>:$<INSTALL_INTERFACE:include/modules>>
  $<INSTALL_INTERFACE:include>
)


if ( DTFFT_WITH_PROFILER )
  target_compile_definitions(dtfft PRIVATE DTFFT_WITH_PROFILER)
  if ( DTFFT_WITH_CUDA )
    target_link_libraries(dtfft PRIVATE CUDA::nvtx3)
  else()
    find_package( caliper REQUIRED )
    message(STATUS "Found caliper: ${caliper_DIR}")
    target_link_libraries(dtfft PRIVATE caliper)
  endif()
endif()

set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -D__DEBUG -g")
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -std=f2018 -pedantic-errors -fbacktrace -Wall")
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -Og -g -Waliasing -Wampersand -Wconversion -Wsurprising \
  -Wintrinsics-std -Wtabs -Wintrinsic-shadow -Wline-truncation -Wtarget-lifetime -Winteger-division -Wreal-q-constant -Wunused  \
  -Wrealloc-lhs-all -Wdo-subscript -Wundefined-do-loop -Wextra -ggdb -fopt-info -fopt-info-optall-optimized -fbacktrace -fdump-fortran-optimized\
  -ftree-vectorize -Wimplicit-interface -Wunused-parameter -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -ffpe-summary=none \
  -Warray-bounds -Wimplicit-procedure -Wunderflow -Wuninitialized -fimplicit-none -fdump-core -finit-real=nan")

  if(DTFFT_RUNNING_CICD)
    set(DTFFT_ENABLE_COVERAGE ON)
  endif()
  if(DTFFT_ENABLE_COVERAGE)
    set(DTFFT_BUILD_TESTS ON)

    string(REPLACE "." ";" GCOV_VERSION ${CMAKE_C_COMPILER_VERSION})
    list(LENGTH GCOV_VERSION len)
    list(GET GCOV_VERSION 0 GCOV_VERSION_MAJOR)
    find_program( GCOV_PATH gcov-${GCOV_VERSION_MAJOR} )
    if ( NOT GCOV_PATH )
      message(FATAL_ERROR "Unable to find gcov utility")
    endif()
    message(STATUS "Found gcov: ${GCOV_PATH}")

    add_custom_target(coverage
      COMMAND ${CMAKE_MAKE_PROGRAM} test
      COMMAND ${GCOV_PATH} "${CMAKE_BINARY_DIR}/CMakeFiles/dtfft.dir/src/*.gcno"
        "${CMAKE_BINARY_DIR}/CMakeFiles/dtfft.dir/src/interfaces/api/*.gcno"
        "${CMAKE_BINARY_DIR}/CMakeFiles/dtfft.dir/src/interfaces/api/c/*.gcno"
        "${CMAKE_BINARY_DIR}/CMakeFiles/dtfft.dir/src/interfaces/fft/fftw/*.gcno"
      COMMAND mv "${CMAKE_BINARY_DIR}/*.gcov" "${PROJECT_SOURCE_DIR}/"
    )

    target_link_libraries(dtfft PRIVATE gcov)
    set_target_properties(dtfft
      PROPERTIES
      COMPILE_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
      LINK_FLAGS "-lgcov --coverage"
    )
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES Intel OR CMAKE_Fortran_COMPILER_ID MATCHES IntelLLVM)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -free -warn all -traceback")
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG}  -check all -ftrapuv -fpe3 -assume ieee_fpe_flags -debug extended -assume realloc_lhs  -fstack-protector -assume protect_parens -implicitnone")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast -march=native")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES NVHPC)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Mbackslash -traceback -Mnobounds")
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -C -Mchkptr")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast")
endif()

# if ( DTFFT_WITH_CUDA )
#   # foreach(CUDA_CC ${DTFFT_CUDA_CC_LIST})
#   #   list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
#   # endforeach()
#   # set(cuda_ver "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
#   # list(APPEND CUF_GPU_ARG "cuda${cuda_ver}")
#   # list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

#   # target_compile_options(dtfft
#   #   PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-gpu=${CUF_GPU_ARG}>
#   #   PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
#   # target_link_options(dtfft
#   #   PUBLIC $<$<LINK_LANGUAGE:Fortran>:-gpu=${CUF_GPU_ARG}>
#   #   PUBLIC $<$<LINK_LANGUAGE:Fortran>:-cuda>)
#   #   # PUBLIC $<$<LINK_LANGUAGE:C,CXX>:-cudaforlibs>)

#   # SET( NVHPC_CUDA_VERSION ${cuda_ver} )
#   # set(CMAKE_PREFIX_PATH "${CUDAToolkit_LIBRARY_ROOT}/../../cmake")
#   # find_package(NVHPC REQUIRED COMPONENTS MATH)


#   # if ( CMAKE_Fortran_COMPILER_ID MATCHES NVHPC )
#   #   set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} ${COMMON_NVHPC_FLAGS}")
#   #   set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -Mnobounds")
#   # endif()
# endif()

if( CMAKE_C_COMPILER_ID MATCHES GNU )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic -Wall -Wextra")
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES GNU )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -Wextra")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


set(DTFFT_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/dtfft"
    CACHE STRING "Path to dtFFT CMake files")


install(TARGETS dtfft
  EXPORT dtfftTargets)

install(
    EXPORT dtfftTargets
    DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
    CONFIGURATIONS Debug Release
    FILE dtfftTargets.cmake)

write_basic_package_version_file(dtfftConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfigVersion.cmake"
    DESTINATION
    "${DTFFT_INSTALL_CMAKEDIR}" )

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/dtfftConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfig.cmake"
  INSTALL_DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfig.cmake"
    DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
)

install(DIRECTORY ${DTFFT_MODULE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.mod")

add_subdirectory(src)
if(DTFFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "dtFFT configuration summary:")
message(STATUS "Version                   : ${dtfft_VERSION}")
message(STATUS "Build type                : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler                  : ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
message(STATUS "Install dir               : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build shared libs         : ${DTFFT_BUILD_SHARED}")
message(STATUS "Use `mpi` module          : ${DTFFT_USE_MPI}")
message(STATUS "Build C/C++ API:          : ${DTFFT_BUILD_C_CXX_API}")
message(STATUS "Build tests               : ${DTFFT_BUILD_TESTS}")
message(STATUS "Coverage                  : ${DTFFT_ENABLE_COVERAGE}")
message(STATUS "Profiler                  : ${DTFFT_WITH_PROFILER}")
message(STATUS "CUDA enabled              : ${DTFFT_WITH_CUDA}")
if ( DTFFT_WITH_CUDA )
message(STATUS "Custom NCCL enabled       : ${DTFFT_WITH_NCCL}")

endif()
if ( DTFFT_TRANSPOSE_ONLY )
  message(STATUS "Building transpose only interface")
else()
  if ( DTFFT_WITH_CUDA )
    message(STATUS "cuFFT enabled             : ${DTFFT_WITH_CUFFT}")
    message(STATUS "VkFFT enabled             : ${DTFFT_WITH_VKFFT}")
  else()
    message(STATUS "FFTW3 enabled             : ${DTFFT_WITH_FFTW}")
    message(STATUS "MKL enabled               : ${DTFFT_WITH_MKL}")
  endif()
endif()
