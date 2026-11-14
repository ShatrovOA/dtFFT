target_sources(dtfft PRIVATE
    dtfft_abstract_backend.F90
    dtfft_backend_mpi.F90
)

if (USE_NCCL)
    target_sources(dtfft PRIVATE
        dtfft_backend_nccl.F90
        interfaces/dtfft_interface_nccl.F90
    )
endif()

if (USE_NVSHMEM)
    target_sources(dtfft PRIVATE 
        dtfft_backend_cufftmp.F90)
endif()

if(DTFFT_BUILD_TESTS)
    add_subdirectory(tests)
endif()