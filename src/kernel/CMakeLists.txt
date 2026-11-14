target_sources(dtfft PRIVATE
    dtfft_abstract_kernel.F90
    dtfft_kernel_host.F90
    impl/dtfft_kernel_host_sm.F90
    impl/dtfft_kernel_host_co_sm.F90
    impl/dtfft_kernel_host_base_sm.F90
    impl/dtfft_kernel_host_block_4_sm.F90
    impl/dtfft_kernel_host_block_8_sm.F90
    impl/dtfft_kernel_host_block_16_sm.F90
    impl/dtfft_kernel_host_block_32_sm.F90
    impl/dtfft_kernel_host_block_64_sm.F90
)

if (DTFFT_WITH_CUDA)
    target_sources(dtfft PRIVATE
        dtfft_kernel_device.F90
        dtfft_rtc_block_optimizer.F90
        dtfft_rtc_module_cache.F90
        dtfft_rtc_module.F90
        interfaces/dtfft_interface_nvrtc.F90
    )
endif()

target_include_directories(dtfft PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(DTFFT_BUILD_TESTS)
    add_subdirectory(tests)
endif()