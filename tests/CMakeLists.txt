include_directories($<$<COMPILE_LANGUAGE:Fortran>:${CMAKE_BINARY_DIR}/mod>)
include_directories(${PROJECT_SOURCE_DIR}/include)
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

if(DTFFT_RUNNING_CICD)
  set(FLAG_OVERSUBSCRIBE "--oversubscribe")
  set(TESTS_MAX_PROC 8)
else()
  set(TESTS_MAX_PROC ${MPIEXEC_MAX_NUMPROCS})
endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} dtfft)

  foreach(nproc RANGE 1 ${TESTS_MAX_PROC} 1)
    add_test(
      NAME "${test_name}_nproc_${nproc}"
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${FLAG_OVERSUBSCRIBE} ./${test_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endforeach()
endforeach()