target_sources(dtfft PRIVATE
  dtfft_executor_vkfft_m.F90
  dtfft_interface_vkfft_m.F90
)

if ( NOT VKFFT_DIR )
  message(FATAL_ERROR "VKFFT_DIR is missing. Please pass it to cmake: cmake -DVKFFT_DIR=path/to/location/of/vkFFT.h}")
endif()

if ( DTFFT_WITH_CUDA )
  add_library(dtfft_vkfft_cuda MODULE)
  target_sources(dtfft_vkfft_cuda PRIVATE
    dtfft_interface_vkfft_native.c
  )
  target_include_directories(dtfft_vkfft_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:${VKFFT_DIR}>)
  target_compile_definitions(dtfft_vkfft_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:VKFFT_BACKEND=1>)
  target_link_libraries(dtfft_vkfft_cuda PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
  target_link_libraries(dtfft_vkfft_cuda PRIVATE MPI::MPI_C)

  add_dependencies(dtfft dtfft_vkfft_cuda)

  set_target_properties(dtfft_vkfft_cuda PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  install(TARGETS dtfft_vkfft_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()