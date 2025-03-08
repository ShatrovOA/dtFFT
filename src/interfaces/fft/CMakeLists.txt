if(DTFFT_WITH_FFTW)
  add_subdirectory(fftw)
endif()

if(DTFFT_WITH_MKL)
  add_subdirectory(mkl)
endif()

if(DTFFT_WITH_CUFFT)
  add_subdirectory(cufft)
endif()

if(DTFFT_WITH_VKFFT)
  add_subdirectory(vkfft)
endif()