# Check which cufftMpMakeReshape API is available.
function(check_cufftmp_features CUFFTMP_INCLUDE_DIRS CUDA_INCLUDE_DIRS)
  include(CheckCXXSourceCompiles)

  set(_dtfft_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_INCLUDES ${CUFFTMP_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

  check_cxx_source_compiles("
    #include <cufftMp.h>
    #include <stddef.h>
    #include <type_traits>
    using make_reshape_fn = cufftResult (*)(
      cufftReshapeHandle, size_t, int,
      const long long int *, const long long int *, const long long int *,
      const long long int *, const long long int *, const long long int *,
      void *, cufftMpCommType);
    static_assert(std::is_same<decltype(&cufftMpMakeReshape), make_reshape_fn>::value,
                  \"cuFFTMpMakeReshape signature mismatch\");
    int main() { return 0; }
  " DTFFT_CUFFTMP_API_HAS_RESHAPE_COMM)

  if(DTFFT_CUFFTMP_API_HAS_RESHAPE_COMM)
    message(STATUS "Detected cuFFTMp API: communicator passed to cufftMpMakeReshape")
    set(CUFFTMP_API_HAS_RESHAPE_COMM ON PARENT_SCOPE)
  else()
    check_cxx_source_compiles("
      #include <cufftMp.h>
      #include <stddef.h>
      #include <type_traits>
      using make_reshape_fn = cufftResult (*)(
        cufftReshapeHandle, size_t, int,
        const long long int *, const long long int *, const long long int *,
        const long long int *, const long long int *, const long long int *);
      static_assert(std::is_same<decltype(&cufftMpMakeReshape), make_reshape_fn>::value,
                    \"cuFFTMpMakeReshape signature mismatch\");
      int main() { return 0; }
    " DTFFT_CUFFTMP_API_LEGACY_RESHAPE)
    if(NOT DTFFT_CUFFTMP_API_LEGACY_RESHAPE)
      message(FATAL_ERROR "Unsupported cuFFTMp cufftMpMakeReshape API signature")
    endif()
    message(STATUS "Detected legacy cuFFTMp API: cufftMpAttachReshapeComm")
    set(CUFFTMP_API_HAS_RESHAPE_COMM OFF PARENT_SCOPE)
  endif()

  set(CMAKE_REQUIRED_INCLUDES "${_dtfft_saved_required_includes}")
endfunction()
