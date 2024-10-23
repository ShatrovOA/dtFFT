file(GLOB COMMON_SOURCES *.F90)
target_sources(dtfft PRIVATE ${COMMON_SOURCES})
target_sources(dtfft PRIVATE "dtfft_helpers.c")
add_subdirectory(interfaces)