set(TEST_SOURCES "test_host_kernels.F90")

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
    COMMAND ./${test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()