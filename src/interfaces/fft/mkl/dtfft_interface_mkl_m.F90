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
!------------------------------------------------------------------------------------------------
!< This module creates interface with MKL library
!------------------------------------------------------------------------------------------------
use iso_c_binding, only: c_long, c_int, c_ptr, c_f_pointer, c_null_char
implicit none
private

public :: mkl_dfti_create_desc,         &
          mkl_dfti_set_value,           &
          mkl_dfti_commit_desc,         &
          mkl_dfti_execute,             &
          mkl_dfti_free_desc, mkl_dfti_execute_forward, mkl_dfti_execute_backward, mkl_dfti_set_pointer

! public :: DftiCreateDescriptor, DftiSetValue, DftiCommitDescriptor, DftiFreeDescriptor
! public :: DftiComputeForward, DftiComputeBackward
public :: DftiErrorMessage

  interface
    ! integer(c_long) function DftiCreateDescriptor(desc, precision, domain, dim, length) bind(C, name="dfti_create_descriptor_highd")
    ! import
    !   type(c_ptr)                       :: desc
    !   integer(c_int), intent(in), value :: precision
    !   integer(c_int), intent(in), value :: domain
    !   integer(c_long),intent(in), value :: dim
    !   integer(c_long),intent(in)        :: length(*)
    ! end function DftiCreateDescriptor

    ! integer(c_long) function DftiSetValue(desc, param, value) bind(C)
    ! import
    !   type(c_ptr),                value :: desc
    !   integer(c_int), intent(in), value :: param
    !   integer(c_int), intent(in), value :: value
    ! end function DftiSetValue

    ! integer(c_long) function DftiCommitDescriptor(desc) bind(C)
    ! import
    !   type(c_ptr),                value :: desc
    ! end function DftiCommitDescriptor

    ! integer(c_long) function DftiFreeDescriptor(desc) bind(C)
    ! import
    !   type(c_ptr)                       :: desc
    ! end function DftiFreeDescriptor

    ! integer(c_long) function DftiComputeForward(desc, in, out) bind(C)
    ! import
    !   type(c_ptr),  value :: desc
    !   type(c_ptr),  value :: in
    !   type(c_ptr),  value :: out
    ! end function DftiComputeForward

    ! integer(c_long) function DftiComputeBackward(desc, in, out) bind(C)
    ! import
    !   type(c_ptr),  value :: desc
    !   type(c_ptr),  value :: in
    !   type(c_ptr),  value :: out
    ! end function DftiComputeBackward

    type(c_ptr) function DftiErrorMessage_c(error_code) bind(C, name="DftiErrorMessage")
    import
      integer(c_long), intent(in),  value  :: error_code
    end function DftiErrorMessage_c
  end interface

  interface
    integer(c_long) function  mkl_dfti_create_desc(precision, domain, dim, length, desc) bind(C)
      import
      integer(c_int), intent(in), value :: precision
      integer(c_int), intent(in), value :: domain
      integer(c_long),intent(in), value :: dim
      integer(c_long),intent(in)        :: length(*)
      type(c_ptr)                       :: desc
    end function mkl_dfti_create_desc

    integer(c_long) function mkl_dfti_set_value(desc, param, value) bind(C)
      import
      type(c_ptr),                value :: desc
      integer(c_int), intent(in), value :: param
      integer(c_int), intent(in), value :: value
    end function mkl_dfti_set_value

    integer(c_long) function mkl_dfti_set_pointer(desc, param, value) bind(C)
      import
      type(c_ptr),                value :: desc
      integer(c_int), intent(in), value :: param
      integer(c_long), intent(in)        :: value(*)
    end function mkl_dfti_set_pointer

    integer(c_long) function mkl_dfti_commit_desc(desc) bind(C)
      import
      type(c_ptr),                value :: desc
    end function mkl_dfti_commit_desc

    integer(c_long) function mkl_dfti_execute(desc, in, out, sign) bind(C)
      import
      type(c_ptr),                value :: desc
      type(c_ptr),                value :: in
      type(c_ptr),                value :: out
      integer(c_int), intent(in), value :: sign
    end function mkl_dfti_execute

    integer(c_long) function mkl_dfti_execute_forward(desc, in, out) bind(C)
      import
      type(c_ptr),                value :: desc
      type(c_ptr),                value :: in
      type(c_ptr),                value :: out
    end function mkl_dfti_execute_forward

    integer(c_long) function mkl_dfti_execute_backward(desc, in, out) bind(C)
      import
      type(c_ptr),                value :: desc
      type(c_ptr),                value :: in
      type(c_ptr),                value :: out
    end function mkl_dfti_execute_backward

    integer(c_long) function mkl_dfti_free_desc(desc) bind(C)
      import
      type(c_ptr),                value :: desc
    end function mkl_dfti_free_desc
  endinterface

contains

  function DftiErrorMessage(error_code) result(string)
    integer(c_long), intent(in)   :: error_code
    character(len=:), allocatable :: string
    type(c_ptr) :: c_string
    character(len=256), pointer :: f_string

    c_string = DftiErrorMessage_c(error_code)
    call c_f_pointer(c_string, f_string)
    allocate( string, source=f_string(1:index(f_string, c_null_char) - 1) )
  end function DftiErrorMessage
end module dtfft_interface_mkl_m