target_sources(dtfft PRIVATE
    dtfft_plan.F90
    dtfft_serializer.F90
    dtfft_utils.F90
    dtfft_config.F90
    dtfft_errors.F90
    dtfft_parameters.F90
    dtfft_pencil.F90
)

if ( DTFFT_WITH_CUDA )
    target_sources(dtfft PRIVATE
        interfaces/dtfft_interface_cuda_runtime.F90
        interfaces/dtfft_interface_cuda.F90
    )

    if( USE_NVSHMEM )
        target_sources(dtfft PRIVATE
            interfaces/dtfft_interface_nvshmem.F90
        )
    endif()

    if ( DTFFT_WITH_PROFILER AND NOT DTFFT_WITH_MOCK_ENABLED )
        target_sources(dtfft PRIVATE
            interfaces/dtfft_interface_nvtx.F90
            interfaces/dtfft_interface_nvtx3.c
        )
    endif()

    if ( NOT DTFFT_WITH_MOCK_ENABLED )
        target_sources(dtfft PRIVATE
            dtfft_helpers.c)
    endif()

endif()

if(DTFFT_BUILD_TESTS)
    add_library(test_suite tests/test_suite.F90)
    set_target_properties(test_suite PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dtfft_test_utils_modules"
    )
    target_include_directories(test_suite PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/dtfft_test_utils_modules"
    )
    target_link_libraries(test_suite PRIVATE dtfft)

    add_subdirectory(tests)
endif()