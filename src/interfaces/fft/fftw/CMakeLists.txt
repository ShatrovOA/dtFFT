pkg_check_modules(FFTW REQUIRED fftw3 fftw3f)
target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(dtfft ${FFTW_LDFLAGS})
message(STATUS "FFTW3 includes: ${FFTW_INCLUDE_DIRS}")

# file(GLOB FFTW_SOURCES ${FFT_INTERFACES_DIR}/fftw/*.F90)
target_sources(dtfft PRIVATE
dtfft_executor_fftw_m.F90
dtfft_interface_fftw_m.F90
dtfft_interface_fftw_native_m.F90
)