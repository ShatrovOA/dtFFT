get_target_property(LIB_DIR dtfft BINARY_DIR)
set(MODULE_DIR ${LIB_DIR}/mod)

file(GLOB COMMON_SOURCES *.F90)
target_sources(dtfft PRIVATE ${COMMON_SOURCES})

set_target_properties(dtfft PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})

# target_include_directories(dtfft INTERFACE ${MODULE_DIR})
target_link_libraries(dtfft ${MPI_Fortran_LIBRARIES})
target_include_directories(dtfft PRIVATE include)
add_subdirectory(interfaces)