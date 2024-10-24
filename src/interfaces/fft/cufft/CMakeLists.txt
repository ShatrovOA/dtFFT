set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cudalib=cufft")
target_sources(dtfft PRIVATE
  dtfft_executor_cufft_m.F90
  dtfft_interface_cufft_m.F90
  dtfft_interface_cufft_native_m.F90
)