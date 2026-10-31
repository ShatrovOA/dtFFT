include(CheckFortranCompilerFlag)

function(dtfft_append_supported_fortran_flags variable)
  set(_dtfft_supported_flags)
  set(_dtfft_unsupported_flags)

  foreach(_dtfft_flag IN LISTS ARGN)
    string(MAKE_C_IDENTIFIER
      "DTFFT_${CMAKE_Fortran_COMPILER_ID}_${_dtfft_flag}"
      _dtfft_check_variable
    )
    check_fortran_compiler_flag("${_dtfft_flag}" "${_dtfft_check_variable}")
    if(${_dtfft_check_variable})
      list(APPEND _dtfft_supported_flags "${_dtfft_flag}")
    else()
      list(APPEND _dtfft_unsupported_flags "${_dtfft_flag}")
    endif()
  endforeach()

  if(_dtfft_unsupported_flags)
    string(JOIN ", " _dtfft_unsupported_flag_string ${_dtfft_unsupported_flags})
    message(STATUS
      "Fortran compiler ${CMAKE_Fortran_COMPILER_ID} does not support optional flags for ${variable}: ${_dtfft_unsupported_flag_string}"
    )
  endif()

  if(_dtfft_supported_flags)
    string(JOIN " " _dtfft_supported_flag_string ${_dtfft_supported_flags})
    set("${variable}" "${${variable}} ${_dtfft_supported_flag_string}" PARENT_SCOPE)
  endif()
endfunction()

set(_dtfft_fortran_flags_module
  "${CMAKE_CURRENT_LIST_DIR}/compiler-flags/${CMAKE_Fortran_COMPILER_ID}.cmake"
)
if(EXISTS "${_dtfft_fortran_flags_module}")
  include("${_dtfft_fortran_flags_module}")
else()
  include("${CMAKE_CURRENT_LIST_DIR}/compiler-flags/Generic.cmake")
endif()

foreach(_lang IN ITEMS C CXX)
  if(CMAKE_${_lang}_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_${_lang}_FLAGS "${CMAKE_${_lang}_FLAGS} -Wpedantic -Wall -Wextra")
  elseif(CMAKE_${_lang}_COMPILER_ID MATCHES "^(Intel|IntelLLVM|NVHPC)$")
    set(CMAKE_${_lang}_FLAGS_RELEASE "${CMAKE_${_lang}_FLAGS_RELEASE} -Ofast")
  endif()
endforeach()
