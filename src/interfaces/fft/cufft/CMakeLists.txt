target_sources(dtfft PRIVATE
  dtfft_executor_cufft_m.F90
)

if ( DTFFT_WITHOUT_NVSHMEM )
  target_link_libraries(dtfft PRIVATE CUDA::cufft)
endif()
