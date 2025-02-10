!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
module dtfft_interface_mkl_m
!! This module creates C interface with MKL library
use iso_c_binding, only: c_long, c_int, c_ptr, c_f_pointer, c_null_char
implicit none
private

public :: mkl_dfti_create_desc,         &
          mkl_dfti_set_value,           &
          mkl_dfti_commit_desc,         &
          mkl_dfti_execute,             &
          mkl_dfti_free_desc
public :: DftiErrorMessage

  interface
    function DftiErrorMessage_c(error_code)                                   &
      result(message)                                                         &
      bind(C, name="DftiErrorMessage")
    !! Generates an error message.
    import
      integer(c_long),  intent(in), value  :: error_code      !< Completion status of a function.
      type(c_ptr)                          :: message         !< Pointer to message
    end function DftiErrorMessage_c
  end interface

  interface mkl_dfti_set_value
  !! Sets one particular configuration parameter with the specified configuration value.
    function mkl_dfti_set_integer(desc, param, value)                         &
      result(status)                                                          &
      bind(C)
    !! Sets one particular configuration parameter with integer value.
    import
      type(c_ptr),                  value :: desc             !< FFT descriptor.
      integer(c_int),   intent(in), value :: param            !< Configuration parameter.
      integer(c_int),   intent(in), value :: value            !< Configuration value.
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_set_integer

    function mkl_dfti_set_pointer(desc, param, value)                         &
      result(status)                                                          &
      bind(C)
    !! Sets one particular configuration parameter with pointer value.
    import
      type(c_ptr),                  value :: desc             !< FFT descriptor.
      integer(c_int),   intent(in), value :: param            !< Configuration parameter.
      integer(c_long),  intent(in)        :: value(*)         !< Configuration value.
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_set_pointer
  end interface mkl_dfti_set_value

  interface
    function  mkl_dfti_create_desc(precision, domain, dim, length, desc)      &
      result(status)                                                          &
      bind(C)
    !! Allocates the descriptor data structure and initializes it with default configuration values.
    import
      integer(c_int),   intent(in), value :: precision        !< Precision of the transform: DFTI_SINGLE or DFTI_DOUBLE.
      integer(c_int),   intent(in), value :: domain           !< Forward domain of the transform: DFTI_COMPLEX or DFTI_REAL.
      integer(c_long),  intent(in), value :: dim              !< Dimension of the transform.
      integer(c_long),  intent(in)        :: length(*)        !< Length of the transform for a one-dimensional transform. 
                                                              !< Lengths of each dimension for a multi-dimensional transform.
      type(c_ptr)                         :: desc             !< FFT descriptor.
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_create_desc

    function mkl_dfti_commit_desc(desc)                                       &
      result(status)                                                          &
      bind(C)
    !! Performs all initialization for the actual FFT computation.
    import
      type(c_ptr),                  value :: desc             !< FFT descriptor.
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_commit_desc

    function mkl_dfti_execute(desc, in, out, sign)                            &
      result(status)                                                          &
      bind(C)
    !! Computes FFT.
      import
      type(c_ptr),                  value :: desc             !< FFT descriptor.
      type(c_ptr),                  value :: in               !< Data to be transformed 
      type(c_ptr),                  value :: out              !< The transformed data
      integer(c_int),   intent(in), value :: sign             !< Sign of transform
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_execute

    function mkl_dfti_free_desc(desc)                                         &
      result(status)                                                          &
      bind(C)
    !! Frees the memory allocated for a descriptor.
    import
      type(c_ptr),                  value :: desc             !< FFT descriptor.
      integer(c_long)                     :: status           !< Function completion status.
    end function mkl_dfti_free_desc
  endinterface

contains

  function DftiErrorMessage(error_code) result(string)
  !! Generates an error message.
    integer(c_long),    intent(in)  :: error_code       !< Completion status of a function.
    character(len=:),   allocatable :: string           !< Error message
    type(c_ptr)                     :: c_string
    character(len=256), pointer     :: f_string

    c_string = DftiErrorMessage_c(error_code)
    call c_f_pointer(c_string, f_string)
    allocate( string, source=f_string(1:index(f_string, c_null_char) - 1) )
  end function DftiErrorMessage
end module dtfft_interface_mkl_m