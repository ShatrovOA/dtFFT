target_sources(dtfft PRIVATE
  dtfft.F90
  dtfft_abstract_executor.F90
  dtfft_abstract_transpose_plan.F90
  dtfft_helpers.c
  dtfft_pencil.F90
  dtfft_plan.F90
  dtfft_parameters.F90
  dtfft_transpose_handle_host.F90
  dtfft_transpose_plan_host.F90
  dtfft_utils.F90
)

if (DTFFT_WITH_CUDA)
  target_sources(dtfft PRIVATE
    dtfft_abstract_gpu_backend.F90
    dtfft_abstract_gpu_backend_selfcopy.F90
    # dtfft_backend_cufftmp.F90
    dtfft_backend_mpi.F90
    dtfft_backend_mpi_a2a.F90
    dtfft_backend_mpi_p2p.F90
    dtfft_backend_mpi_p2p_pipelined.F90
    dtfft_backend_nccl.F90
    dtfft_backend_nccl_pipelined.F90
    dtfft_nvrtc_interfaces.F90
    dtfft_nvrtc_kernel.F90
    dtfft_transpose_handle_cuda.F90
    dtfft_transpose_plan_cuda.F90
  )
endif()

add_subdirectory(interfaces)