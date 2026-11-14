set(TEST_SOURCES "test_reshape_plans.F90" "test_reshape_handles.F90")

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  dtfft_add_test(${test_name} ${test}
    TEST_UTILS
    WITH_RMA
    LINKER_LANGUAGE Fortran
    MIN_PROCS 1
    NAME_NPROCS
  )
endforeach()