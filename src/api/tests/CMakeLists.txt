if(NOT DTFFT_BUILD_C_CXX_API)
  return()
endif()

set(TEST_SOURCES "test_plan_save_load_c.c" "test_api_c.c" "test_api_cxx.cpp")

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  # Plan files hold one record per rank, so this has to run at several process counts
  dtfft_add_test(${test_name} ${test}
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}
    MIN_PROCS 1
    NAME_NPROCS
  )
endforeach()

if(DTFFT_BUILD_PYTHON_API)
  add_subdirectory(python)
endif()
