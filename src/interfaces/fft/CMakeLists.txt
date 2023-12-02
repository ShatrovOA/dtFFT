if(DTFFT_NO_FFTW)
  target_compile_definitions(dtfft PRIVATE NO_FFTW3)
else()
  add_subdirectory(fftw)
endif()

if(DTFFT_ENABLE_MKL)
  add_subdirectory(mkl)
endif()

# if(DTFFT_ENABLE_KFR)
#   add_subdirectory(kfr)
# endif()