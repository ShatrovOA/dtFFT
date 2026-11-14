# cmake/FindNCCL.cmake
# Find the NCCL libraries
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

set(NCCL_ROOT "$ENV{NCCL_ROOT}" CACHE PATH "Root directory of NVIDIA NCCL")
set(NCCL_INCLUDE_DIR "$ENV{NCCL_INCLUDE_DIR}" CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR "$ENV{NCCL_LIB_DIR}" CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

if(DEFINED ENV{NCCL_ROOT_DIR} AND NOT "$ENV{NCCL_ROOT_DIR}" STREQUAL "")
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
  if(NOT NCCL_ROOT)
    set(NCCL_ROOT "$ENV{NCCL_ROOT_DIR}")
  endif()
endif()

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR} ${NCCL_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

set(_NCCL_SAVED_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if (USE_STATIC_NCCL)
  message(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  set(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  set(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR} ${NCCL_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}")

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_NCCL_SAVED_LIBRARY_SUFFIXES})
unset(_NCCL_SAVED_LIBRARY_SUFFIXES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if (NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
    )
  endif()
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()