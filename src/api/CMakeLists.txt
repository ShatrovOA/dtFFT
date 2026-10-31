if(DTFFT_BUILD_PYTHON_API)
    set(DTFFT_PYTHON_BUILD_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/dtfft")
    file(MAKE_DIRECTORY "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}")
    file(COPY "${PROJECT_SOURCE_DIR}/src/api/python/"
        DESTINATION "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}"
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi"
    )

    pybind11_add_module(pydtfft dtfft_api_py.cpp)
    set_target_properties(pydtfft PROPERTIES
        OUTPUT_NAME _dtfft
        LIBRARY_OUTPUT_DIRECTORY "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}"
    )
    target_include_directories(pydtfft PRIVATE ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR})
    target_link_libraries(pydtfft PRIVATE dtfft)

    install(TARGETS pydtfft DESTINATION .)
    pybind11_add_module(py_utils dtfft_api_py_utils.cpp)
    set_target_properties(py_utils PROPERTIES
        OUTPUT_NAME _dtfft_example_utils
        LIBRARY_OUTPUT_DIRECTORY "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${DTFFT_PYTHON_BUILD_PACKAGE_DIR}"
    )
    target_include_directories(py_utils PRIVATE
        ${pybind11_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/examples/c
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(py_utils PRIVATE dtfft_example_utils)
    install(TARGETS py_utils DESTINATION .)

    if(APPLE)
        set_target_properties(pydtfft py_utils PROPERTIES
            INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
    else()
        set_target_properties(pydtfft py_utils PROPERTIES
            INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
    endif()
endif()

target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${PROJECT_SOURCE_DIR}/include>)
target_sources(dtfft PRIVATE
    dtfft_api_c.c
    dtfft_api_cxx.cpp
    dtfft_api.F90
)

if(DTFFT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
