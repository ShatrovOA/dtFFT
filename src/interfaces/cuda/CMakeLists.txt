target_sources(dtfft PRIVATE
  dtfft_interface_cuda.F90
  dtfft_interface_nvrtc.F90
)

if( USE_NVSHMEM )
  target_sources(dtfft PRIVATE
    dtfft_interface_nvshmem.F90
    dtfft_interface_cufft.F90
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

if ( DTFFT_WITH_PROFILER )
  target_sources(dtfft PRIVATE
    dtfft_interface_nvtx.F90
    dtfft_interface_nvtx3.c
  )
endif()