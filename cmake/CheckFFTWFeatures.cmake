function(check_fftw_features FFTW_INCLUDE_DIRS FFTW_LIBRARIES)
    # Set required includes and libraries for checks
    set(CMAKE_REQUIRED_INCLUDES ${FFTW_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${FFTW_LIBRARIES})

    include(CheckCXXSymbolExists)
    # Check if fftw_planner_nthreads is defined
    check_cxx_symbol_exists(fftw_planner_nthreads "fftw3.h" FFTW_HAS_PLANNER_NTHREADS)

    # Cache results for reuse
    set(FFTW_HAS_PLANNER_NTHREADS ${FFTW_HAS_PLANNER_NTHREADS} CACHE BOOL "NCCL supports fftw_planner_nthreads")
endfunction()