target_sources(dtfft PRIVATE
  dtfft.F90
  dtfft_abstract_backend.F90
  dtfft_abstract_executor.F90
  dtfft_abstract_kernel.F90
  dtfft_abstract_reshape_handle.F90
  dtfft_backend_mpi.F90
  dtfft_config.F90
  dtfft_errors.F90
  dtfft_kernel_host.F90
  dtfft_pencil.F90
  dtfft_plan.F90
  dtfft_parameters.F90
  dtfft_reshape_handle_datatype.F90
  dtfft_reshape_handle_generic.F90
  dtfft_reshape_plan_base.F90
  dtfft_reshape_plan.F90
  dtfft_transpose_plan.F90
  dtfft_utils.F90
)

if ( DTFFT_WITH_COMPRESSION )
  target_sources(dtfft PRIVATE dtfft_abstract_compressor.F90)
endif()

if ( DTFFT_WITH_ZFP )
  target_sources(dtfft PRIVATE dtfft_compressor_zfp.F90)
endif()

if (DTFFT_WITH_CUDA)
  target_sources(dtfft PRIVATE
    dtfft_kernel_device.F90
    dtfft_nvrtc_block_optimizer.F90
    dtfft_nvrtc_module_cache.F90
    dtfft_nvrtc_module.F90)

  if ( NOT DTFFT_WITH_MOCK_ENABLED )
    target_sources(dtfft PRIVATE
      dtfft_helpers.c)
  endif()

  if (USE_NCCL)
    target_sources(dtfft PRIVATE dtfft_backend_nccl.F90)
  endif()

  if (USE_NVSHMEM)
    target_sources(dtfft PRIVATE dtfft_backend_cufftmp.F90)
  endif()
endif()

add_subdirectory(interfaces)

if(DTFFT_BUILD_TESTS)
  add_subdirectory(tests)
endif()