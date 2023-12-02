cmake_minimum_required(VERSION 3.20)
project(dtFFT VERSION 0.1
              DESCRIPTION "Library to perform FFT on a distibuted memory cluster."
              HOMEPAGE_URL "https://github.com/ShatrovOA/dtFFT"
              LANGUAGES Fortran C CXX)

option(DTFFT_NO_FFTW "Build dtFFT without FFTW support" OFF)
option(DTFFT_ENABLE_MKL "Build dtFFT with MKL DFTI support" OFF)
# option(DTFFT_ENABLE_CUFFT "Build dtFFT with cufft support" OFF)
# option(DTFFT_ENABLE_KFR "Build dtFFT with KFR support" OFF)
option(DTFFT_BUILD_TESTS "Create dtFFT tests" OFF)
option(DTFFT_ENABLE_COVERAGE "Create coverage with gcov utility" OFF)
option(DTFFT_BUILD_SHARED "Build shared library" ON)
option(DTFFT_USE_MPI "Use Fortran `mpi` module instead of `mpi_f08`" OFF)
option(DTFFT_BUILD_C_CXX_API "Create C API" ON)
# option(DTFFT_BUILD_JAVA_API "Create Java API" OFF)
# option(DTFFT_BUILD_PYTHON_API "Create python API" OFF)

include(CheckFortranSourceCompiles)
check_fortran_source_compiles("program test
contains
subroutine test_sub(a)
type(*),  intent(inout) :: a(..)
end subroutine
end program"
HAVE_ASSUMED_RANK_AND_TYPE
SRC_EXT .F90)

if(DTFFT_NO_FFTW)
  add_compile_definitions(NO_FFTW3)
endif()
if(DTFFT_ENABLE_MKL)
  add_compile_definitions(MKL_ENABLED)
endif()

find_package(PkgConfig REQUIRED)
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED)
if(NOT MPI_Fortran_HAVE_F08_MODULE)
  if(NOT MPI_Fortran_HAVE_F90_MODULE)
    message(FATAL_ERROR "Neither `mpi_f08` nor `mpi` modules were found.." )
  endif()
  add_compile_definitions(DTFFT_USE_MPI)
else()
  if(DTFFT_USE_MPI)
    add_compile_definitions(DTFFT_USE_MPI)
  endif()
endif()
include_directories(${MPI_Fortran_MODULE_DIR})
include_directories(${MPI_C_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)


if(DTFFT_BUILD_SHARED)
  add_library(dtfft SHARED "")
else()
  add_library(dtfft STATIC "")
endif()
set(DTFFT_HEADERS "")

add_subdirectory(src)

set(CMAKE_VERBOSE_MAKEFILE OFF)


set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -D__DEBUG -g")
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -m64 -ffree-line-length-none -std=f2018")
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -Og -g -Waliasing -Wampersand -Wconversion -Wsurprising \
  -Wintrinsics-std -Wtabs -Wintrinsic-shadow -Wline-truncation -Wtarget-lifetime -Winteger-division -Wreal-q-constant -Wunused  \
  -Wrealloc-lhs-all -Wdo-subscript -Wundefined-do-loop -Wextra -ggdb -fopt-info -fopt-info-optall-optimized -fbacktrace -fdump-fortran-optimized\
  -ftree-vectorize -Wimplicit-interface -Wunused-parameter -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -ffpe-summary=none \
  -Warray-bounds -Wimplicit-procedure -Wunderflow -Wuninitialized -fimplicit-none -fdump-core -finit-real=nan -g3")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast -march=native")
  if(DTFFT_ENABLE_COVERAGE)
    set(DTFFT_BUILD_TESTS ON)

    add_custom_target(coverage
      COMMAND ${CMAKE_MAKE_PROGRAM} test
      COMMAND gcov ${CMAKE_BINARY_DIR}/CmakeFiles/dtfft.dir/src/*.gcno
      COMMAND mv    ${CMAKE_BINARY_DIR}/*.gcov ${PROJECT_SOURCE_DIR}
    )

    target_link_libraries(dtfft gcov)
    set_target_properties(dtfft
      PROPERTIES
      COMPILE_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
      LINK_FLAGS "-lgcov --coverage"
    )
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES intel)
  set(CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -warn all -check all -ftrapuv -fpe3 -assume ieee_fpe_flags -debug extended")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast -march=native")
endif()


install (TARGETS dtfft
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)


if(DTFFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
