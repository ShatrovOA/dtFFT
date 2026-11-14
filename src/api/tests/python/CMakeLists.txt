file(GLOB PYTHON_TESTS CONFIGURE_DEPENDS "*.py")

foreach(test ${PYTHON_TESTS})
  get_filename_component(test_name ${test} NAME_WLE)
  foreach(nproc RANGE 1 ${DTFFT_TEST_MAX_PROCS})
    add_test(
      NAME "${test_name}_nproc_${nproc}"
      COMMAND ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        ${nproc}
        ${DTFFT_TEST_MPIEXEC_FLAGS}
        ${Python_EXECUTABLE}
        ${test}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties("${test_name}_nproc_${nproc}" PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}"
    )
  endforeach()
endforeach()
