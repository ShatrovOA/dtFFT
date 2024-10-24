file(GLOB TEST_SOURCES "fortran/*_f.F90")

if(DTFFT_BUILD_C_CXX_API)
  file(GLOB TESTS_C_CXX_SOURCES "c/*.c*")
  list(APPEND TEST_SOURCES ${TESTS_C_CXX_SOURCES})
endif()

# if(DTFFT_BUILD_JAVA_API)
#   # add_subdirectory()
# endif()

# if(DTFFT_BUILD_PYTHON_API)
#   add_subdirectory(python)
# endif()

if(DTFFT_RUNNING_CICD)
  set(TESTS_MAX_PROC 8)

  if ( DTFFT_MPI_VENDOR STREQUAL "openmpi")
    set(FLAG_OVERSUBSCRIBE "--oversubscribe")
  endif()
else()
  set(TESTS_MAX_PROC ${MPIEXEC_MAX_NUMPROCS})
endif()

foreach(test ${TEST_SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  get_filename_component(extension ${test} EXT)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} PRIVATE dtfft)
  target_link_libraries(${test_name} PRIVATE ${MPI_Fortran_LIBRARIES})
  target_include_directories(${test_name} PRIVATE ${PROJECT_BINARY_DIR})
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${PROJECT_SOURCE_DIR}/src/include>)
  target_include_directories(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${DTFFT_MODULE_DIR}>)
  target_include_directories(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${MPI_Fortran_MODULE_DIR}>)
  target_include_directories(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${MPI_C_INCLUDE_DIRS}>)
  if( extension STREQUAL ".cpp" )
    target_link_libraries(${test_name} PRIVATE ${MPI_CXX_LIBRARIES})
  endif()

  foreach(nproc RANGE 1 ${TESTS_MAX_PROC} 1)
    add_test(
      NAME "${test_name}_nproc_${nproc}"
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${FLAG_OVERSUBSCRIBE} ./${test_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endforeach()
endforeach()