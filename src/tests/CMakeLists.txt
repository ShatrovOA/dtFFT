file(GLOB TEST_SOURCES "*.F90")

set(TEST_SOURCES "test_host_kernels.F90")

# if ( DTFFT_WITH_CUDA )
#   set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}" )
# endif()


foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  get_filename_component(extension ${test} EXT)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} PUBLIC dtfft)
  # target_include_directories(${test_name} PRIVATE ${CMAKE_BINARY_DIR})
  # target_link_libraries(${test_name} PRIVATE m)

  if ( DTFFT_WITH_CUDA )
    target_link_libraries(${test_name} PRIVATE CUDA::cudart)

    if ( CMAKE_Fortran_COMPILER_ID MATCHES NVHPC )
      target_compile_options(${test_name}
        PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
      target_link_options(${test_name}
        PUBLIC $<$<LINK_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
    endif()

    if ( CMAKE_C_COMPILER_ID MATCHES NVHPC AND CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
      if (extension STREQUAL ".c")
        set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE C)
      elseif(extension STREQUAL ".cpp")
        set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
      endif()
      target_compile_options(${test_name}
        PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
      target_link_options(${test_name}
        PUBLIC $<$<LINK_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
    endif()
  endif()

  if ( DTFFT_RUNNING_CICD )
    target_compile_definitions(${test_name} PRIVATE DTFFT_RUNNING_CICD)
  endif()

  add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./${test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()