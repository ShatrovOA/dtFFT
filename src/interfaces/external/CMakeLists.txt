if ( DTFFT_WITH_CUDA )
  target_sources(dtfft PRIVATE
    dtfft_interface_cuda_runtime.F90
    dtfft_interface_cuda.F90
    dtfft_interface_nvrtc.F90
  )

  if( USE_NVSHMEM )
    target_sources(dtfft PRIVATE
      dtfft_interface_nvshmem.F90
      dtfft_interface_cufft.F90
      # dtfft_interface_nvshmem.cu
    )
  endif()

  if( DTFFT_WITH_CUFFT )
    target_sources(dtfft PRIVATE
      dtfft_interface_cufft.F90
    )
  endif()

  if ( USE_NCCL )
    target_sources(dtfft PRIVATE
      dtfft_interface_nccl.F90
    )
  endif()

  if ( DTFFT_WITH_PROFILER AND NOT DTFFT_WITH_MOCK_ENABLED )
    target_sources(dtfft PRIVATE
      dtfft_interface_nvtx.F90
      dtfft_interface_nvtx3.c
    )
  endif()
endif()

if ( DTFFT_WITH_ZFP )
  target_sources(dtfft PRIVATE dtfft_interface_zfp.F90)
endif()