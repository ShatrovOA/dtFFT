target_sources(dtfft PRIVATE
  dtfft.F90
  dtfft_abstract_executor.F90
  dtfft_abstract_transpose_plan.F90
  dtfft_config.F90
  dtfft_errors.F90
  dtfft_pencil.F90
  dtfft_plan.F90
  dtfft_parameters.F90
  dtfft_transpose_handle_host.F90
  dtfft_transpose_plan_host.F90
  dtfft_utils.F90
)

if (DTFFT_WITH_CUDA)
  target_sources(dtfft PRIVATE
    dtfft_abstract_backend.F90
    dtfft_backend_mpi.F90
    dtfft_helpers.c
    dtfft_nvrtc_block_optimizer.F90
    dtfft_nvrtc_kernel_cache.F90
    dtfft_nvrtc_kernel_generator.F90
    dtfft_nvrtc_kernel.F90
    dtfft_transpose_handle_cuda.F90
    dtfft_transpose_plan_cuda.F90
  )

  if (USE_NCCL)
    target_sources(dtfft PRIVATE
      dtfft_backend_nccl.F90
    )
  endif()

  if (USE_NVSHMEM)
    target_sources(dtfft PRIVATE
      dtfft_backend_cufftmp.F90
    )
  endif()
endif()

add_subdirectory(interfaces)