
set(MKL_INTERFACE lp64)
set(MKL_LINK static)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

target_compile_options(dtfft PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
# target_compile_definitions(dtfft PUBLIC MKL_LP64)

include_directories(dtfft $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(dtfft $<LINK_ONLY:MKL::MKL>)
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input


# include_directories(${MKL_INCLUDE_DIRS})
# link_libraries(${MKL_LDFLAGS})

# message("mkl dirs = $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>")
# message("mkl link = $<LINK_ONLY:MKL::MKL>")
# message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")


# add_compile_definitions(MKL_ENABLED=1)
# file(GLOB MKL_SOURCES ${FFT_INTERFACES_DIR}/mkl/*.F90 ${FFT_INTERFACES_DIR}/mkl/*.c)
# target_sources(dtfft PRIVATE ${MKL_SOURCES})


# pkg_check_modules(FFTW REQUIRED fftw3 fftw3f)
# target_include_directories(dtfft PRIVATE ${FFTW_INCLUDE_DIRS})
# target_link_libraries(dtfft ${FFTW_LDFLAGS})

# file(GLOB FFTW_SOURCES ${FFT_INTERFACES_DIR}/fftw/*.F90)
target_sources(dtfft PRIVATE
dtfft_executor_mkl_m.F90
dtfft_interface_mkl_m.F90
dtfft_interface_mkl.c
)