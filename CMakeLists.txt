cmake_minimum_required(VERSION 3.20)
project(dtfft VERSION 0.3.0
              DESCRIPTION "Library to perform FFT on a distibuted memory cluster."
              HOMEPAGE_URL "https://github.com/ShatrovOA/dtFFT"
              LANGUAGES Fortran C CXX)

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

option(DTFFT_WITH_FFTW "Build dtFFT with FFTW support" OFF)
option(DTFFT_WITH_MKL "Build dtFFT with MKL DFTI support" OFF)
option(DTFFT_WITH_CUFFT "Build dtFFT with cuFFT support" OFF)
option(DTFFT_WITH_VKFFT "Build dtFFT with VkFFT support" OFF)
option(DTFFT_WITH_CUDA "Build dtFFT with CUDA support" OFF)

option(DTFFT_BUILD_TESTS "Create dtFFT tests" OFF)
option(DTFFT_ENABLE_COVERAGE "Create coverage with gcov utility" OFF)
option(DTFFT_BUILD_SHARED "Build shared library" ON)
option(DTFFT_USE_MPI "Use Fortran `mpi` module instead of `mpi_f08`" OFF)
option(DTFFT_BUILD_C_CXX_API "Create C API" ON)
option(DTFFT_ENABLE_PERSISTENT_COMM "Enable MPI persistent calls" OFF)
option(DTFFT_WITH_PROFILER "Build dtFFT with profiler support" OFF)

set(DTFFT_CUDA_CC_LIST "70;80;90" CACHE STRING "List of CUDA compute capabilities to build for")

set(DTFFT_RUNNING_CICD OFF CACHE STRING "Running tests in GitHub actions")


find_package(MPI REQUIRED)
if(MPI_Fortran_HAVE_F08_MODULE)
  if(DTFFT_USE_MPI)
    add_compile_definitions(DTFFT_USE_MPI)
  endif()
else()
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

include(CheckFortranSourceCompiles)
check_fortran_source_compiles("program test
contains
subroutine test_sub(a)
type(*),  intent(inout) :: a(..)
end subroutine
end program"
HAVE_ASSUMED_RANK_AND_TYPE
SRC_EXT .F90)
check_fortran_source_compiles("program test
integer :: i = 2
print *, mod(i,i)
block
  integer :: j
  j = i + 1
  print *, mod(j,i)
endblock
end program"
HAVE_BLOCK_STATEMENT
SRC_EXT .F90)


if ( DTFFT_ENABLE_PERSISTENT_COMM )

  target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_ENABLE_PERSISTENT_COMM>)
  set(CMAKE_REQUIRED_INCLUDES ${MPI_Fortran_MODULE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_Fortran_LIBRARIES})
  check_fortran_source_compiles("
  program test
  use mpi
  implicit none
  integer :: send, recv, ierr, request

  call MPI_Alltoall_init(send, 1, MPI_INTEGER, recv, 1, MPI_INTEGER, MPI_COMM_WORLD, MPI_INFO_NULL, request, ierr)
  end program"
  HAVE_PERSISTENT_COLLECTIVES
  SRC_EXT .F90)

  if ( HAVE_PERSISTENT_COLLECTIVES )
    message(STATUS "Persistent communications ENABLED")
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:DTFFT_HAVE_PERSISTENT_COLLECTIVES>)
  else()
    message(WARNING "MPI implementation do not support collective persistent functions")
  endif()
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
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

set ( DTFFT_CONF_HEADER ${PROJECT_BINARY_DIR}/dtfft_config.h CACHE STRING "DTFFT Config header")
configure_file( ${PROJECT_SOURCE_DIR}/include/dtfft_config.h.in "${DTFFT_CONF_HEADER}" )


if(DTFFT_WITH_CUDA)
  if ( NOT CMAKE_Fortran_COMPILER_ID MATCHES NVHPC
    OR NOT CMAKE_C_COMPILER_ID MATCHES NVHPC
    OR NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC
  )
    message(FATAL_ERROR "CUDA build requires that Fortran, C and C++ compilers are NVHPC")
  endif()
  find_package(CUDAToolkit 9.0 REQUIRED)
  target_include_directories(dtfft PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:${CUDAToolkit_INCLUDE_DIRS}>)
  target_link_libraries(dtfft PUBLIC CUDA::cuda_driver)
  target_link_libraries(dtfft PUBLIC CUDA::nvrtc)
  target_link_libraries(dtfft PUBLIC CUDA::cudart)

  set(CMAKE_REQUIRED_INCLUDES ${MPI_Fortran_MODULE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_Fortran_LIBRARIES})
  if ( DTFFT_USE_MPI )
    set(HAVE_MPI_INT64 OFF)
  else()
    check_fortran_source_compiles("
      program test
      use mpi_f08
      implicit none
        integer(MPI_COUNT_KIND) :: send_count
        real, allocatable :: send_buf(:)
        integer :: ierr
        type(MPI_Request) :: request

        call MPI_Isend(send_buf, send_count, MPI_REAL, 0, 0, MPI_COMM_WORLD, request, ierr)"
      HAVE_MPI_INT64
    SRC_EXT .F90)
  endif()
  if ( HAVE_MPI_INT64 )
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:CNT_KIND=MPI_COUNT_KIND;ADDR_KIND=MPI_ADDRESS_KIND>)
  else()
    message(WARNING "MPI 64bit integer indexing is disabled. Overflows might occur if MPI Backends will be used")
    target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:CNT_KIND=int32;ADDR_KIND=int32>)
  endif()

  set(CMAKE_REQUIRED_FLAGS "-cuda")

  if ( DTFFT_USE_MPI )
    check_fortran_source_compiles("
        program main
        use mpi
        implicit none
        integer :: request
        real, device, allocatable :: buf(:)
        integer :: mpi_ierr

        call MPI_Isend(buf( 1 ), 1, MPI_REAL, 1, 0, MPI_COMM_WORLD, request, mpi_ierr)

      end program main"
      HAVE_MPI_DEVICE_SUPPORT
      SRC_EXT .F90)
    if (NOT HAVE_MPI_DEVICE_SUPPORT)
      message(FATAL_ERROR "MPI Module does not support passing buffers with `device` attribute.")
    endif()
  else()
    check_fortran_source_compiles("
      program main
      use mpi_f08
      implicit none
      type(MPI_Request) :: request
      real, device, allocatable :: buf(:)
      integer :: mpi_ierr

      call MPI_Isend(buf( 1 ), 1, MPI_REAL, 1, 0, MPI_COMM_WORLD, request, mpi_ierr)

      end program main"
      HAVE_MPI_DEVICE_SUPPORT
      SRC_EXT .F90)
    if (NOT HAVE_MPI_DEVICE_SUPPORT)
      message(FATAL_ERROR "MPI Module `mpi_f08` does not support passing buffers with `device` attribute. 
                           Consider using `-DDTFFT_USE_MPI=on` while building dtFFT")
    endif()
  endif()

  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
endif()


# target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${PROJECT_SOURCE_DIR}/src/include>)
# target_include_directories(dtfft PRIVATE ${PROJECT_BINARY_DIR})

target_include_directories(dtfft PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${MPI_Fortran_MODULE_DIR}>)
target_link_libraries(dtfft PUBLIC ${MPI_Fortran_LIBRARIES})


set(DTFFT_MODULE_DIR ${PROJECT_BINARY_DIR}/modules)
set_target_properties(dtfft PROPERTIES Fortran_MODULE_DIRECTORY ${DTFFT_MODULE_DIR})
set_target_properties(dtfft PROPERTIES VERSION ${CMAKE_PROJECT_VERSION})
# target_include_directories(dtfft PUBLIC )
set(DTFFT_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")
# SET(DTFFT_MODULE "${DTFFT_MODULE_DIR}/dtfft.mod")
SET(DTFFT_HEADERS "${DTFFT_HEADER_DIR}/dtfft.f03")

if(DTFFT_BUILD_C_CXX_API)
  set(DTFFT_HEADERS "${DTFFT_HEADERS};${DTFFT_HEADER_DIR}/dtfft.h;${DTFFT_HEADER_DIR}/dtfft.hpp;${DTFFT_CONF_HEADER}")
  target_link_libraries(dtfft PUBLIC ${MPI_CXX_LIBRARIES})
  target_include_directories(dtfft PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:${MPI_C_INCLUDE_DIRS}>)
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
  if ( NOT DTFFT_WITH_CUDA )
    find_package( caliper REQUIRED )
    message(STATUS "Found caliper: ${caliper_DIR}")
    target_link_libraries(dtfft PUBLIC caliper)
  endif()
endif()

set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -D__DEBUG -g")
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -std=f2018 -pedantic-errors -fbacktrace -Wall -Werror")
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
  set(COMMON_NVHPC_FLAGS "-traceback")
  if ( DTFFT_WITH_CUDA )
    foreach(CUDA_CC ${DTFFT_CUDA_CC_LIST})
      list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
    endforeach()
    list(APPEND CUF_GPU_ARG "cuda${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
    list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

    if ( DTFFT_WITH_PROFILER )
      set(COMMON_NVHPC_FLAGS        "${COMMON_NVHPC_FLAGS} -cudalib=nvtx3")
    endif()

    set(COMMON_NVHPC_FLAGS          "${COMMON_NVHPC_FLAGS} -cuda -cudalib=nccl -gpu=${CUF_GPU_ARG}")
    set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} ${COMMON_NVHPC_FLAGS} -Mbackslash")
    set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -C -Mchkptr -Mnobounds")
    set(CMAKE_C_FLAGS               "${CMAKE_C_FLAGS} ${COMMON_NVHPC_FLAGS}")
    set(CMAKE_CXX_FLAGS             "${CMAKE_CXX_FLAGS} ${COMMON_NVHPC_FLAGS}")
  endif()
endif()

if( CMAKE_C_COMPILER_ID MATCHES GNU )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic -Wall -Wextra")
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES GNU )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -Wextra")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/dtfftConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/dtfft"
    NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

set(DTFFT_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/dtfft"
    CACHE STRING "Path to dtFFT CMake files")


install(TARGETS dtfft EXPORT dtfft_export)

install(
    EXPORT dtfft_export
    DESTINATION "${DTFFT_INSTALL_CMAKEDIR}"
    CONFIGURATIONS Debug Release
    FILE dtfftConfig.cmake)

write_basic_package_version_file(dtfftConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION} 
    COMPATIBILITY SameMinorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dtfftConfigVersion.cmake"
    DESTINATION
    "${DTFFT_INSTALL_CMAKEDIR}" )

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
