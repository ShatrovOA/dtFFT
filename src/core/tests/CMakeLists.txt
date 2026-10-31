set(TEST_SOURCES
  "test_plan_save_load.F90"
  "test_serializer.F90"
  "test_pencils_f.F90"
  "test_mem_alloc.F90"
)

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  # Saving and loading a plan is inherently collective: the file holds one record per rank
  # and the number of ranks is part of the plan identity. Run across the same range of
  # process counts as the integration tests.
  dtfft_add_test(${test_name} ${test}
    TEST_UTILS
    LINKER_LANGUAGE Fortran
    MIN_PROCS 1
    NAME_NPROCS
  )
endforeach()
