pkg_check_modules(FFTW REQUIRED fftw3 fftw3f)
target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(dtfft ${FFTW_LDFLAGS})
message(STATUS "FFTW_FOUND: ${FFTW_FOUND}")
message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}")
message(STATUS "FFTW_CFLAGS: ${FFTW_CFLAGS}")
message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
message(STATUS "FFTW_LIBRARY_DIRS: ${FFTW_LIBRARY_DIRS}")

# file(GLOB FFTW_SOURCES ${FFT_INTERFACES_DIR}/fftw/*.F90)
target_sources(dtfft PRIVATE
dtfft_executor_fftw_m.F90
dtfft_interface_fftw_m.F90
dtfft_interface_fftw_native_m.F90
)