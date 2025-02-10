# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()
list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR})

if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)  # obtaining NCCL version and some sanity checks
  if ( CUDAToolkit_INCLUDE_DIRS )
  # NCCL Can only be compiled when CUDAToolkit if found and 
  # both CUDAToolkit_INCLUDE_DIRS and CUDAToolkit_LIBRARY_DIR are defined
    set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
    message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")

    set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
    list (APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

    set ( OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    list (APPEND CMAKE_REQUIRED_LIBRARIES ${NCCL_LIBRARIES} ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)

    include(CheckSymbolExists)
    check_symbol_exists(NCCL_VERSION_CODE "nccl.h" NCCL_VERSION_DEFINED)
    check_symbol_exists(ncclMemAlloc "nccl.h" NCCL_HAVE_MEMALLOC)
    check_symbol_exists(ncclCommRegister "nccl.h" NCCL_HAVE_COMMREGISTER)

    if (NCCL_VERSION_DEFINED)
      set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
      file(WRITE ${file} "
        #include <iostream>
        #include <nccl.h>
        int main()
        {
          std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;

          int x;
          ncclGetVersion(&x);
          return x == NCCL_VERSION_CODE;
        }
      ")
      try_run(NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
            RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
            CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}"
            LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
      if( NOT compile_result )
        message(STATUS "Failed to compile ${file}")
      endif()

      if (NOT NCCL_VERSION_MATCHED AND compile_result)
        message(FATAL_ERROR "Found NCCL header version and library version do not match! \
  (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}) Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
      endif()
      if ( compile_result )
        message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
      endif()
    else()
      message(STATUS "Unable to dermine NCCL Version: NCCL_VERSION_CODE is not defined in nccl.h")
    endif ()
    set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})
    set (CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
  else()
    message(STATUS "Unable to determine NCCL Version and capabilities, due to missing CUDAToolkit_INCLUDE_DIRS")
  endif()

  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES NCCL_HAVE_MEMALLOC NCCL_HAVE_COMMREGISTER)
endif()