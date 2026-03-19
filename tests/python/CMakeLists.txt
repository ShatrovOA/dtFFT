pybind11_add_module(test_utils_py test_utils_py.cpp)
set_target_properties(test_utils_py PROPERTIES OUTPUT_NAME _dtfft_test_utils)
target_include_directories(test_utils_py PRIVATE
  ${PROJECT_SOURCE_DIR}/tests/c
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}
)
target_link_libraries(test_utils_py PUBLIC test_utils)
install(TARGETS test_utils_py DESTINATION .)
install(TARGETS test_utils DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(APPLE)
  set_target_properties(
    test_utils_py PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
  set_target_properties(
    test_utils PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
else()
  set_target_properties(test_utils_py PROPERTIES INSTALL_RPATH
                                          "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
  set_target_properties(test_utils PROPERTIES INSTALL_RPATH
                                          "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()

file(GLOB SOURCES "*.py")

foreach(test ${SOURCES})
  get_filename_component(test_name ${test} NAME_WLE)
  foreach(nproc RANGE 1 ${TESTS_MAX_PROC} 1)
    add_test(NAME "${test_name}_nproc_${nproc}"
          COMMAND ${MPIEXEC_EXECUTABLE}
          ${MPIEXEC_NUMPROC_FLAG}
          ${nproc}
          ${FLAG_OVERSUBSCRIBE}
          ${Python_EXECUTABLE}
          ${test}
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/python)
    set_tests_properties("${test_name}_nproc_${nproc}" PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}")
  endforeach()
endforeach()