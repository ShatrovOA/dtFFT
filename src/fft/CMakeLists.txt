target_sources(dtfft PRIVATE
    dtfft_abstract_executor.F90
    dtfft_executor.F90
)

if(DTFFT_WITH_FFTW)
    target_sources(dtfft PRIVATE
        dtfft_executor_fftw.F90
        interfaces/dtfft_interface_fftw.F90
        interfaces/dtfft_interface_fftw_native.F90
    )

    target_include_directories(dtfft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    if ( DTFFT_WITH_OPENMP )
        include(CheckFFTWFeatures)

        if ( DTFFT_WITH_FFTW_THREADS )
            find_package(FFTW REQUIRED COMPONENTS FLOAT_THREADS_LIB DOUBLE_THREADS_LIB)
            target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_THREADS_LIB} ${FFTW_FLOAT_THREADS_LIB})
            check_fftw_features("${FFTW_INCLUDE_DIRS}" "${FFTW_DOUBLE_THREADS_LIB}")
        else()
            find_package(FFTW REQUIRED COMPONENTS FLOAT_OPENMP_LIB DOUBLE_OPENMP_LIB)
            target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_OPENMP_LIB} ${FFTW_FLOAT_OPENMP_LIB})
            check_fftw_features("${FFTW_INCLUDE_DIRS}" "${FFTW_DOUBLE_OPENMP_LIB}")
        endif()
        if ( FFTW_HAS_PLANNER_NTHREADS )
            target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:FFTW_HAS_PLANNER_NTHREADS>)
        endif()
    else()
        find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
    endif()

    target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
    target_link_libraries(dtfft PRIVATE ${FFTW_DOUBLE_LIB} ${FFTW_FLOAT_LIB})
endif()

if(DTFFT_WITH_CUFFT)
    target_sources(dtfft PRIVATE
        dtfft_executor_cufft.F90
    )

    if ( NOT USE_NVSHMEM )
        target_link_libraries(dtfft PRIVATE CUDA::cufft)
    endif()
endif()

if(DTFFT_WITH_CUFFT OR USE_NVSHMEM)
    target_sources(dtfft PRIVATE
        interfaces/dtfft_interface_cufft.F90
    )
endif()

if(DTFFT_WITH_VKFFT)
    target_sources(dtfft PRIVATE
        dtfft_executor_vkfft.F90
        interfaces/dtfft_interface_vkfft.F90
    )

    if ( NOT VKFFT_DIR )
        message(FATAL_ERROR "VKFFT_DIR is missing. Please pass it to cmake: cmake -DVKFFT_DIR=path/to/location/of/vkFFT.h}")
    endif()

    if ( DTFFT_WITH_CUDA )
        add_library(dtfft_vkfft_cuda MODULE interfaces/dtfft_interface_vkfft_native.c)
        target_include_directories(dtfft_vkfft_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:${VKFFT_DIR}>)
        target_compile_definitions(dtfft_vkfft_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:VKFFT_BACKEND=1>)
        target_link_libraries(dtfft_vkfft_cuda PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
        target_link_libraries(dtfft_vkfft_cuda PRIVATE MPI::MPI_C)

        # add_dependencies(dtfft dtfft_vkfft_cuda)

        set_target_properties(dtfft_vkfft_cuda PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        install(TARGETS dtfft_vkfft_cuda
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()

if(DTFFT_WITH_MKL)
    target_sources(dtfft PRIVATE
        dtfft_executor_mkl.F90
        interfaces/dtfft_interface_mkl.F90
        interfaces/dtfft_interface_mkl_native.F90
        interfaces/dtfft_interface_mkl.c
    )

    set(MKL_ARCH intel64)
    set(MKL_INTERFACE lp64)

    if( DTFFT_WITH_OPENMP )
        set(MKL_THREADING intel_thread)
    else()
        set(MKL_THREADING sequential)
    endif()

    if(DTFFT_BUILD_SHARED)
        set(MKL_LINK dynamic)
    else()
        set(MKL_LINK static)
    endif()
    find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

    target_compile_options(dtfft PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(dtfft PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(dtfft PRIVATE $<LINK_ONLY:MKL::MKL>)
endif()


if(DTFFT_BUILD_TESTS)
    add_subdirectory(tests)
endif()