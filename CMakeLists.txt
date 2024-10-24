cmake_minimum_required(VERSION 3.20)
project(dtFFT VERSION 0.2.0
              DESCRIPTION "Library to perform FFT on a distibuted memory cluster."
              HOMEPAGE_URL "https://github.com/ShatrovOA/dtFFT"
              LANGUAGES Fortran C CXX)

option(DTFFT_WITH_FFTW "Build dtFFT with FFTW support" OFF)
option(DTFFT_WITH_MKL "Build dtFFT with MKL DFTI support" OFF)
option(DTFFT_WITH_CUFFT "Build dtFFT with cufft support" OFF)
# option(DTFFT_WITH_KFR "Build dtFFT with KFR support" OFF)
option(DTFFT_WITH_VKFFT "Build dtFFT with VkFFT support" OFF)
# option(DTFFT_MEASURE_ITERS "Number of iterations to run in order to find best grid decomposition when passing DTFFT_MEASURE and DTFFT_PATIENT to effort_flag during plan creation" 2)

option(DTFFT_BUILD_TESTS "Create dtFFT tests" OFF)
option(DTFFT_ENABLE_COVERAGE "Create coverage with gcov utility" OFF)
option(DTFFT_BUILD_SHARED "Build shared library" ON)
option(DTFFT_USE_MPI "Use Fortran `mpi` module instead of `mpi_f08`" OFF)
option(DTFFT_BUILD_C_CXX_API "Create C API" ON)
option(DTFFT_ENABLE_PERSISTENT_COMM "Enable MPI persistent collective calls" OFF)
option(DTFFT_WITH_CALIPER "Build dtFFT with Caliper tool" OFF)
option(DTFFT_RUNNING_CICD "Running tests in GitHub actions" OFF)
# option(DTFFT_BUILD_JAVA_API "Create Java API" OFF)
# option(DTFFT_BUILD_PYTHON_API "Create python API" OFF)

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

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

if( NOT DTFFT_WITH_FFTW
    AND NOT DTFFT_WITH_MKL
    # AND NOT DTFFT_WITH_KFR
    # AND NOT DTFFT_WITH_CUFFT
    # AND NOT DTFFT_WITH_VKFFT
)
  set ( DTFFT_TRANSPOSE_ONLY ON )
endif()
set ( DTFFT_CONF_HEADER ${PROJECT_BINARY_DIR}/dtfft_config.h CACHE STRING "DTFFT Config header")
configure_file( ${PROJECT_SOURCE_DIR}/include/dtfft_config.h.in "${DTFFT_CONF_HEADER}" )


# if ( DTFFT_WITH_VKFFT OR DTFFT_WITH_CUFFT )
#   target_compile_definitions(dtfft PRIVATE DTFFT_WITH_CUDA)
# endif()
if(DTFFT_BUILD_SHARED)
  add_library(dtfft SHARED)
else()
  add_library(dtfft STATIC)
endif()
target_include_directories(dtfft PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_include_directories(dtfft PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${MPI_Fortran_MODULE_DIR}>)
target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${MPI_C_INCLUDE_DIRS}>)
target_link_libraries(dtfft PRIVATE ${MPI_Fortran_LIBRARIES})

set(DTFFT_MODULE_DIR ${PROJECT_BINARY_DIR}/mod)
set_target_properties(dtfft PROPERTIES Fortran_MODULE_DIRECTORY ${DTFFT_MODULE_DIR})
set_target_properties(dtfft PROPERTIES VERSION ${CMAKE_PROJECT_VERSION})
set(DTFFT_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")
SET(DTFFT_MODULE "${DTFFT_MODULE_DIR}/dtfft.mod")
SET(DTFFT_HEADERS "${DTFFT_MODULE};${DTFFT_HEADER_DIR}/dtfft.f03")

if(DTFFT_BUILD_C_CXX_API)
  set(DTFFT_HEADERS "${DTFFT_HEADERS};${DTFFT_HEADER_DIR}/dtfft.h;${DTFFT_HEADER_DIR}/dtfft.hpp;${DTFFT_CONF_HEADER}")
  target_link_libraries(dtfft PRIVATE ${MPI_CXX_LIBRARIES})
endif()
set_target_properties(dtfft PROPERTIES PUBLIC_HEADER "${DTFFT_HEADERS}")
target_include_directories(dtfft INTERFACE $<INSTALL_INTERFACE:include>)

set(DTFFT_MEASURE_ITERS "2" CACHE STRING "Number of iterations to run in order to find best grid decomposition
                                          when passing DTFFT_MEASURE or DTFFT_PATIENT to effort_flag parameter
                                          during plan creation")
target_compile_definitions(dtfft PRIVATE DTFFT_MEASURE_ITERS=${DTFFT_MEASURE_ITERS})

set(DTFFT_FORWARD_X_Y "2" CACHE STRING "Default id of transposition plan for X -> Y transpose")
set(DTFFT_BACKWARD_X_Y "2" CACHE STRING "Default id of transposition plan for Y -> X transpose")
set(DTFFT_FORWARD_Y_Z "2" CACHE STRING "Default id of transposition plan for Y -> Z transpose")
set(DTFFT_BACKWARD_Y_Z "2" CACHE STRING "Default id of transposition plan for Z -> Y transpose")
set(DTFFT_FORWARD_X_Z "2" CACHE STRING "Default id of transposition plan for X -> Z transpose")
set(DTFFT_BACKWARD_X_Z "2" CACHE STRING "Default id of transposition plan for Z -> X transpose")
target_compile_definitions(dtfft PRIVATE DTFFT_FORWARD_X_Y=${DTFFT_FORWARD_X_Y}
                                 PRIVATE DTFFT_BACKWARD_X_Y=${DTFFT_BACKWARD_X_Y}
                                 PRIVATE DTFFT_FORWARD_Y_Z=${DTFFT_FORWARD_Y_Z}
                                 PRIVATE DTFFT_BACKWARD_Y_Z=${DTFFT_BACKWARD_Y_Z}
                                 PRIVATE DTFFT_FORWARD_X_Z=${DTFFT_FORWARD_X_Z}
                                 PRIVATE DTFFT_BACKWARD_X_Z=${DTFFT_BACKWARD_X_Z}
)

if ( DTFFT_ENABLE_PERSISTENT_COMM )
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
    target_compile_definitions(dtfft PRIVATE DTFFT_ENABLE_PERSISTENT_COMM)
  else()
    message(WARNING "MPI implementation do not support collective persistent functions")
  endif()
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
endif()

if ( DTFFT_WITH_CALIPER )
  find_package( caliper REQUIRED )
  message(STATUS "Found caliper: ${caliper_DIR}")
  target_compile_definitions(dtfft PRIVATE DTFFT_WITH_CALIPER)
  target_link_libraries(dtfft PRIVATE caliper)
endif()


add_subdirectory(src)
set_target_properties(dtfft PROPERTIES PUBLIC_HEADER "${DTFFT_HEADERS}")

set(CMAKE_VERBOSE_MAKEFILE OFF)


set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -D__DEBUG -g")
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -std=f2018 -pedantic-errors -fbacktrace -Wall -Werror")
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -Og -g -Waliasing -Wampersand -Wconversion -Wsurprising \
  -Wintrinsics-std -Wtabs -Wintrinsic-shadow -Wline-truncation -Wtarget-lifetime -Winteger-division -Wreal-q-constant -Wunused  \
  -Wrealloc-lhs-all -Wdo-subscript -Wundefined-do-loop -Wextra -ggdb -fopt-info -fopt-info-optall-optimized -fbacktrace -fdump-fortran-optimized\
  -ftree-vectorize -Wimplicit-interface -Wunused-parameter -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -ffpe-summary=none \
  -Warray-bounds -Wimplicit-procedure -Wunderflow -Wuninitialized -fimplicit-none -fdump-core -finit-real=nan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic -Wall -Werror")
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
elseif(CMAKE_Fortran_COMPILER_ID MATCHES intel)
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -warn all -check all -ftrapuv -fpe3 -assume ieee_fpe_flags -debug extended -stand f03 -assume realloc_lhs -traceback -fstack-protector -assume protect_parens -implicitnone")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast -march=native")
endif()


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/dtFFTConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dtFFTConfig.cmake"
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
    FILE dtFFTConfig.cmake)

write_basic_package_version_file(dtFFTConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION} 
    COMPATIBILITY SameMinorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dtFFTConfigVersion.cmake"
    DESTINATION
    "${DTFFT_INSTALL_CMAKEDIR}" )

if(DTFFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
