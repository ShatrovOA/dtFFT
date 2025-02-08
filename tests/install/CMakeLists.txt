cmake_minimum_required(VERSION 3.20)
project(dtfftTestUsage LANGUAGES C CXX Fortran)

find_package(dtfft REQUIRED)

if ( DTFFT_WITH_C_CXX_API )
  add_executable(main_c main.c)
  target_link_libraries(main_c dtfft)

  add_executable(main_cxx main.cxx)
  target_link_libraries(main_cxx dtfft)
endif()

add_executable(main_f main.F90)
target_link_libraries(main_f dtfft)
if (DTFFT_WITH_MPI_MODULE)
  target_compile_definitions(main_f PRIVATE -DUSE_MPI)
endif()