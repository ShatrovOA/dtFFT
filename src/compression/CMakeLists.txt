target_sources(dtfft PRIVATE
    dtfft_abstract_compressor.F90
    dtfft_compressor.F90
)

if ( DTFFT_WITH_COMPRESSION )
    target_sources(dtfft PRIVATE
        interfaces/dtfft_compression_utils.cpp
    )
endif()

if ( DTFFT_WITH_ZFP )
    target_sources(dtfft PRIVATE
        dtfft_compressor_zfp.F90
        interfaces/dtfft_interface_zfp.F90
    )
endif()

if ( DTFFT_WITH_SZ3 )
    target_sources(dtfft PRIVATE
        dtfft_compressor_sz3.F90
        interfaces/dtfft_interface_sz3.F90
        interfaces/dtfft_interface_sz3.cpp
    )
endif()

if ( DTFFT_WITH_SZP )
    target_sources(dtfft PRIVATE
        dtfft_compressor_szp.F90
        interfaces/dtfft_interface_szp.F90
        interfaces/dtfft_interface_szp.cpp
    )
endif()

# if ( DTFFT_WITH_CUSZ )
#     target_sources(dtfft PRIVATE
#         dtfft_compressor_cusz.F90
#         impl/dtfft_compressor_cusz_sm.F90
#         interfaces/dtfft_interface_cusz.F90
#         interfaces/dtfft_interface_cusz.cpp
#     )
# endif()

if ( DTFFT_WITH_CUSZP )
    target_sources(dtfft PRIVATE
        dtfft_compressor_cuszp.F90
        interfaces/dtfft_interface_cuszp.F90
        interfaces/dtfft_interface_cuszp.cpp
    )
endif()

if ( DTFFT_WITH_BITCOMP )
    target_sources(dtfft PRIVATE
        dtfft_compressor_bitcomp.F90
        interfaces/dtfft_interface_bitcomp.F90
        interfaces/dtfft_interface_bitcomp.cpp
    )
endif()

if(DTFFT_BUILD_TESTS AND NOT DTFFT_BUILD_PYTHON_API)
    add_subdirectory(tests)
endif()
