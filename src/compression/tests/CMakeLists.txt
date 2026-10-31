set(TEST_SOURCES "")
if ( DTFFT_WITH_COMPRESSION )
    list(APPEND TEST_SOURCES "test_compressors_host.F90")
    if ( DTFFT_WITH_CUDA )
        list(APPEND TEST_SOURCES "test_compressors_device.F90")
    endif()
endif()

foreach(test ${TEST_SOURCES})
    get_filename_component(test_name ${test} NAME_WLE)
    dtfft_add_test(${test_name} ${test}
        TEST_UTILS
        MIN_PROCS 1
        MAX_PROCS 1
        SKIP_RETURN_CODE 77
    )
endforeach()

# if ( DTFFT_WITH_CUSZ )
#   dtfft_add_test(test_compressor_cusz test_compressor_cusz.F90
#     MIN_PROCS 1
#     MAX_PROCS 1
#     SKIP_RETURN_CODE 77
#   )
# endif()

# if ( DTFFT_WITH_CUSZP )
#   dtfft_add_test(test_compressor_cuszp test_compressor_cuszp.F90
#     MIN_PROCS 1
#     MAX_PROCS 1
#     SKIP_RETURN_CODE 77
#   )
# endif()
