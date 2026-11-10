target_sources(dtfft PRIVATE
    dtfft.F90
)

add_subdirectory(backend)
add_subdirectory(compression)
add_subdirectory(core)
add_subdirectory(fft)
add_subdirectory(kernel)
add_subdirectory(reshape)

if(DTFFT_BUILD_PYTHON_API OR DTFFT_BUILD_EXAMPLES)
    if(DTFFT_BUILD_SHARED)
        add_library(dtfft_example_utils SHARED)
    else()
        add_library(dtfft_example_utils STATIC)
    endif()
    target_sources(dtfft_example_utils PRIVATE
        "${PROJECT_SOURCE_DIR}/examples/example_utils.F90"
    )
    target_include_directories(dtfft_example_utils PRIVATE
        "${PROJECT_SOURCE_DIR}/examples/include"
        "${PROJECT_SOURCE_DIR}/examples/c"
    )
    set(DTFFT_EXAMPLE_UTILS_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/example_utils_modules")
    set_target_properties(dtfft_example_utils PROPERTIES
        Fortran_MODULE_DIRECTORY "${DTFFT_EXAMPLE_UTILS_MODULE_DIR}"
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(dtfft_example_utils PUBLIC
        "$<BUILD_INTERFACE:${DTFFT_EXAMPLE_UTILS_MODULE_DIR}>"
    )
    target_link_libraries(dtfft_example_utils PRIVATE dtfft)

    if(DTFFT_WITH_CUDA AND NOT DTFFT_WITH_MOCK_ENABLED)
        enable_language(CUDA)
        target_sources(dtfft_example_utils PRIVATE
            "${PROJECT_SOURCE_DIR}/examples/cuda_utils.cu"
        )
        set_target_properties(dtfft_example_utils PROPERTIES CUDA_ARCHITECTURES all)
    endif()

    if(DTFFT_RUNNING_CICD)
        target_compile_definitions(dtfft_example_utils PRIVATE DTFFT_RUNNING_CICD)
    endif()

    if(DTFFT_BUILD_PYTHON_API AND DTFFT_BUILD_SHARED)
        install(TARGETS dtfft_example_utils
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
        if(APPLE)
            set_target_properties(dtfft_example_utils PROPERTIES INSTALL_RPATH "@loader_path/.")
        else()
            set_target_properties(dtfft_example_utils PROPERTIES INSTALL_RPATH "$ORIGIN/.")
        endif()
    endif()
endif()

if(DTFFT_BUILD_C_CXX_API OR DTFFT_BUILD_PYTHON_API)
    add_subdirectory(api)
endif()