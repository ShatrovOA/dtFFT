# if(DTFFT_BUILD_PYTHON_API)
#   SET(DTFFT_BUILD_C_CXX_API ON)

#   find_package(pybind11 REQUIRED)
#   pybind11_add_module(pydtfft dtfft_api_py.cpp)
#   set_target_properties(pydtfft PROPERTIES OUTPUT_NAME _dtfft)
#   target_include_directories(pydtfft PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR})
#   target_link_libraries(pydtfft PRIVATE dtfft)
#   # install(TARGETS pydtfft
#   #       LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   #       COMPONENT python)

#   # install(TARGETS pydtfft
#   #     LIBRARY DESTINATION dtfft
#   #     ARCHIVE DESTINATION dtfft
#   #     RUNTIME DESTINATION dtfft
#   # )
#   install(TARGETS pydtfft DESTINATION .)
#     if(APPLE)
#     set_target_properties(
#       pydtfft PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
#   else()
#     set_target_properties(pydtfft PROPERTIES INSTALL_RPATH
#                                             "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
#   endif()
# endif()


target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${PROJECT_SOURCE_DIR}/include>)
target_sources(dtfft PRIVATE
  dtfft_api_c.c
  dtfft_api_cxx.cpp
  dtfft_api.F90)