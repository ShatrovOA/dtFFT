target_sources(dtfft PRIVATE
  dtfft_executor_cufft_m.F90
)

# if ( NOT USE_NVSHMEM )
#   target_link_libraries(dtfft PRIVATE CUDA::cufft)
# endif()
