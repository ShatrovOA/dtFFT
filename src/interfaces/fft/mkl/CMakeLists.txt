
set(MKL_ARCH intel64)
set(MKL_THREADING sequential)
set(MKL_INTERFACE lp64)
if(DTFFT_BUILD_SHARED)
  set(MKL_LINK dynamic)
else()
  set(MKL_LINK static)
endif()
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

target_compile_options(dtfft PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(dtfft PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(dtfft $<LINK_ONLY:MKL::MKL>)


target_sources(dtfft PRIVATE
  dtfft_executor_mkl_m.F90
  dtfft_interface_mkl_m.F90
  dtfft_interface_mkl_native_m.F90
  dtfft_interface_mkl.c
)