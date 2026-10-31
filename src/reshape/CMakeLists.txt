target_sources(dtfft PRIVATE
    dtfft_abstract_reshape_handle.F90
    dtfft_abstract_reshape_plan.F90
    dtfft_reshape_handle_datatype.F90
    dtfft_reshape_handle_generic.F90
    dtfft_reshape_plan.F90
    dtfft_transpose_plan.F90
)

# target_include_directories(dtfft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# add_subdirectory(impl)


if(DTFFT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
