if ( DTFFT_WITH_OPENMP )
  if ( DTFFT_WITH_FFTW_THREADS )
    find_package(FFTW REQUIRED COMPONENTS FLOAT_THREADS_LIB DOUBLE_THREADS_LIB)
    target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_THREADS_LIB} ${FFTW_FLOAT_THREADS_LIB})
  else()
    find_package(FFTW REQUIRED COMPONENTS FLOAT_OPENMP_LIB DOUBLE_OPENMP_LIB)
    target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_OPENMP_LIB} ${FFTW_FLOAT_OPENMP_LIB})
  endif()
else()
  find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
endif()

target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_LIB} ${FFTW_FLOAT_LIB})

target_sources(dtfft PRIVATE
  dtfft_executor_fftw_m.F90
  dtfft_interface_fftw_m.F90
  dtfft_interface_fftw_native_m.F90
)