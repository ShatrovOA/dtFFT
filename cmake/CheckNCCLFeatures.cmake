# cmake/CheckNCCLFeatures.cmake
# Module to check NCCL features and version

# Function to check NCCL features
# Arguments:
#   NCCL_INCLUDE_DIRS: Path to NCCL include directory
#   NCCL_LIBRARIES: Path to NCCL libraries
#   CUDA_INCLUDE_DIRS: Path to CUDA Toolkit include directory
#   CUDA_LIBRARY_DIR: Path to CUDA Toolkit library directory
function(check_nccl_features NCCL_INCLUDE_DIRS NCCL_LIBRARIES CUDA_INCLUDE_DIRS CUDA_LIBRARY_DIR)
  # Check if all required arguments are provided
  if (NOT NCCL_INCLUDE_DIRS OR NOT NCCL_LIBRARIES OR NOT CUDA_INCLUDE_DIRS OR NOT CUDA_LIBRARY_DIR)
    message(FATAL_ERROR "check_nccl_features requires NCCL_INCLUDE_DIRS, NCCL_LIBRARIES, CUDA_INCLUDE_DIRS, and CUDA_LIBRARY_DIR")
  endif()

  # Set required includes and libraries for checks
  set(CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${NCCL_LIBRARIES} "${CUDA_LIBRARY_DIR}/libcudart.so")

  include(CheckCXXSymbolExists)
  # Check if NCCL_VERSION_CODE is defined
  check_cxx_symbol_exists(NCCL_VERSION_CODE "nccl.h" NCCL_VERSION_DEFINED)
  # Check if ncclMemAlloc is available
  check_cxx_symbol_exists(ncclMemAlloc "nccl.h" NCCL_HAVE_MEMALLOC)
  # Check if ncclCommRegister is available
  check_cxx_symbol_exists(ncclCommRegister "nccl.h" NCCL_HAVE_COMMREGISTER)

  if (NCCL_VERSION_DEFINED)
    # Generate a temporary C++ file to extract NCCL version
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(WRITE ${file} "
      #include <iostream>
      #include <nccl.h>
      int main() {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;
        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
    ")
    try_run(NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
            RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}"
            LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    if (compile_result AND NCCL_VERSION_MATCHED)
      message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
      # set(NCCL_VERSION ${NCCL_VERSION_FROM_HEADER} CACHE STRING "NCCL version")
    else()
      message(WARNING "Failed to determine NCCL version")
    endif()
  else()
    message(STATUS "NCCL_VERSION_CODE not defined in nccl.h")
  endif()

  # Cache results for reuse
  set(NCCL_VERSION_DEFINED ${NCCL_VERSION_DEFINED} CACHE BOOL "NCCL defines NCCL_VERSION_CODE")
  set(NCCL_HAVE_MEMALLOC ${NCCL_HAVE_MEMALLOC} CACHE BOOL "NCCL supports ncclMemAlloc")
  set(NCCL_HAVE_COMMREGISTER ${NCCL_HAVE_COMMREGISTER} CACHE BOOL "NCCL supports ncclCommRegister")
endfunction()