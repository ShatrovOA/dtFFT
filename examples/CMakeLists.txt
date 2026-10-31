file(GLOB SOURCES CONFIGURE_DEPENDS "fortran/*_f.F90")

if(DTFFT_BUILD_C_CXX_API)
  file(GLOB EXAMPLE_C_CXX_SOURCES CONFIGURE_DEPENDS "c/*.c*")
  list(APPEND SOURCES ${EXAMPLE_C_CXX_SOURCES})
endif()

if ( DTFFT_WITH_CUDA AND NOT DTFFT_WITH_MOCK_ENABLED)
  set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}" CACHE STRING "Cuda version detected")
endif()

foreach(example ${SOURCES})
  get_filename_component(example_name ${example} NAME_WLE)
  get_filename_component(extension ${example} EXT)
  dtfft_add_executable(${example_name} ${example})
  target_link_libraries(${example_name} PRIVATE dtfft_example_utils)
  if(extension STREQUAL ".F90")
    set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE Fortran)
  elseif(extension STREQUAL ".c")
    set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE C)
  elseif(extension STREQUAL ".cpp")
    set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  if(DTFFT_WITH_CUDA AND NOT DTFFT_WITH_MOCK_ENABLED)
    target_link_libraries(${example_name} PRIVATE CUDA::cudart)
    if(CMAKE_Fortran_COMPILER_ID MATCHES NVHPC)
      target_compile_options(${example_name}
        PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
      target_link_options(${example_name}
        PRIVATE $<$<LINK_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
    endif()
    if(CMAKE_C_COMPILER_ID MATCHES NVHPC AND CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
      target_compile_options(${example_name}
        PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
      target_link_options(${example_name}
        PRIVATE $<$<LINK_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
    endif()
  endif()

  if(DTFFT_RUNNING_CICD)
    target_compile_definitions(${example_name} PRIVATE DTFFT_RUNNING_CICD)
  endif()
endforeach()