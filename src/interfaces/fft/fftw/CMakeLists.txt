find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)

target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_LIB} ${FFTW_FLOAT_LIB})

target_sources(dtfft PRIVATE
  dtfft_executor_fftw_m.F90
  dtfft_interface_fftw_m.F90
  dtfft_interface_fftw_native_m.F90
)