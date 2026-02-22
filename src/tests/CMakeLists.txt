set(TEST_SOURCES "test_host_kernels.F90")
if ( DTFFT_WITH_CUDA AND NOT SKIP_CUDA_COVERAGE )
  list(APPEND TEST_SOURCES "test_device_kernels.F90")
endif()
if ( DTFFT_WITH_COMPRESSION )
  message("DTFFT_WITH_COMPRESSION = true")
  list(APPEND TEST_SOURCES "test_compression.F90")
endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  get_filename_component(extension ${test} EXT)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} PUBLIC dtfft)

  if ( extension STREQUAL ".F90" )
    set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE Fortran)
  elseif(extension STREQUAL ".c")
    set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE C)
  elseif(extension STREQUAL ".cpp")
    set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./${test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()