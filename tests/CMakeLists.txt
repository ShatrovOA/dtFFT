file(GLOB TEST_SOURCES "fortran/*_f.F90")

if(DTFFT_BUILD_SHARED)
  add_library(test_utils SHARED fortran/test_utils.F90)
else()
  add_library(test_utils STATIC fortran/test_utils.F90)
endif()
target_link_libraries(test_utils PUBLIC dtfft)


if(DTFFT_BUILD_C_CXX_API)
  file(GLOB TESTS_C_CXX_SOURCES "c/*.c*")
  list(APPEND TEST_SOURCES ${TESTS_C_CXX_SOURCES})
endif()


if(DTFFT_RUNNING_CICD)
  set(TESTS_MAX_PROC 8)

  if ( DTFFT_MPI_VENDOR STREQUAL "openmpi")
    set(FLAG_OVERSUBSCRIBE "--oversubscribe")
  endif()
else()
  set(TESTS_MAX_PROC ${MPIEXEC_MAX_NUMPROCS})
endif()

if ( DTFFT_WITH_CUDA )
  set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  get_filename_component(extension ${test} EXT)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} PRIVATE test_utils)
  target_link_libraries(${test_name} PRIVATE -lm)

  if ( CMAKE_Fortran_COMPILER_ID MATCHES NVHPC )
    target_compile_options(${test_name}
      PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
    target_link_options(${test_name}
      PUBLIC $<$<LINK_LANGUAGE:Fortran>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc -cuda>)
  endif()

  if ( CMAKE_C_COMPILER_ID MATCHES NVHPC AND CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    target_compile_options(${test_name}
      PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
    target_link_options(${test_name}
      PUBLIC $<$<LINK_LANGUAGE:C,CXX>:-gpu=ccall,cuda${CUDA_VERSION} -acc -Minfo=acc>)
  endif()

  foreach(nproc RANGE 1 ${TESTS_MAX_PROC} 1)
    add_test(
      NAME "${test_name}_nproc_${nproc}"
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${FLAG_OVERSUBSCRIBE} ./${test_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endforeach()
endforeach()