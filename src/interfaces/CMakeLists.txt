if(DTFFT_BUILD_C_CXX_API OR DTFFT_BUILD_JAVA_API OR DTFFT_BUILD_PYTHON_API)
  add_subdirectory(api)
endif()

if(NOT DTFFT_TRANSPOSE_ONLY)
  add_subdirectory(fft)
endif()