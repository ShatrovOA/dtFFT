if(DTFFT_BUILD_JAVA_API OR DTFFT_BUILD_PYTHON_API)
  SET(DTFFT_BUILD_C_CXX_API ON)
  if(DTFFT_BUILD_JAVA_API)
    add_subdirectory(java)
  else()
    add_subdirectory(python)
  endif()
endif()

if(DTFFT_BUILD_C_CXX_API)
  target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${PROJECT_SOURCE_DIR}/include>)
  target_sources(dtfft PRIVATE "c/dtfft_api_c.c")
endif()

target_sources(dtfft PRIVATE dtfft_api.F90)