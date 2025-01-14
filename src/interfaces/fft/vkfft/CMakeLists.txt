target_sources(dtfft PRIVATE
  dtfft_executor_vkfft_m.F90
  dtfft_interface_vkfft_m.F90
  dtfft_interface_vkfft_native.c
)

if ( NOT VKFFT_DIR )
  message(FATAL_ERROR "VKFFT_DIR is missing. Please pass it to cmake: cmake -DVKFFT_DIR=path/to/location/of/vkFFT.h}")
endif()

target_include_directories(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C>:${VKFFT_DIR}>)
target_compile_definitions(dtfft PRIVATE $<$<COMPILE_LANGUAGE:C>:VKFFT_BACKEND=1>)
target_precompile_headers(dtfft PRIVATE ${VKFFT_DIR}/vkFFT.h)
