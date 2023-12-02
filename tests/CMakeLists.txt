include_directories(${PROJECT_SOURCE_DIR}/src)
file(GLOB TEST_SOURCES "fortran/*_f.F90")


if(DTFFT_BUILD_C_CXX_API)
  file(GLOB TESTS_C_CXX_SOURCES "c/*.c*")
  list(APPEND TEST_SOURCES ${TESTS_C_CXX_SOURCES})
endif()

# if(DTFFT_BUILD_JAVA_API)
#   # add_subdirectory()
# endif()

# if(DTFFT_BUILD_PYTHON_API)
#   add_subdirectory(python)
# endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} dtfft)

  foreach(nproc RANGE 1 8 1)
    add_test( NAME "${test_name}_nproc_${nproc}"
              COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} --oversubscribe ${test_name})
  endforeach()
endforeach()