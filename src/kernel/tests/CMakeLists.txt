set(TEST_SOURCES "test_kernel_host.F90")
if ( DTFFT_WITH_CUDA AND NOT SKIP_CUDA_COVERAGE )
  list(APPEND TEST_SOURCES "test_kernel_device.F90")
endif()
if ( DTFFT_WITH_CUDA )
  list(APPEND TEST_SOURCES "test_rtc_block_optimizer.F90")
  list(APPEND TEST_SOURCES "test_rtc_module_cache.F90")
endif()
if ( DTFFT_WITH_COMPRESSION )
  list(APPEND TEST_SOURCES "test_kernel_compressed.F90")
endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  dtfft_add_test(${test_name} ${test}
    TEST_UTILS
    MIN_PROCS 1
    MAX_PROCS 1
  )
endforeach()