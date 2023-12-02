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
#if defined(_BUILD_DOCS)
#define MKL_ENABLED
#endif
#if defined(MKL_ENABLED)
use dtfft_precisions
use iso_c_binding
implicit none
public

  integer(IP),  bind(C, name="C_DFTI_DOUBLE")                 :: DFTI_DOUBLE
  !! DFTI Double precision
  integer(IP),  bind(C, name="C_DFTI_SINGLE")                 :: DFTI_SINGLE
  !! DFTI Single precision
  integer(IP),  bind(C, name="C_DFTI_NUMBER_OF_TRANSFORMS")   :: DFTI_NUMBER_OF_TRANSFORMS
  !! Number of data sets to be transformed
  integer(IP),  bind(C, name="C_DFTI_PLACEMENT")              :: DFTI_PLACEMENT
  !! Placement of result [DFTI_INPLACE]
  integer(IP),  bind(C, name="C_DFTI_INPUT_DISTANCE")         :: DFTI_INPUT_DISTANCE
  !! Distance between first input elements for multiple transforms
  integer(IP),  bind(C, name="C_DFTI_OUTPUT_DISTANCE")        :: DFTI_OUTPUT_DISTANCE
  !! Distance between first output elements for multiple transforms
  integer(IP),  bind(C, name="C_DFTI_CONJUGATE_EVEN_STORAGE") :: DFTI_CONJUGATE_EVEN_STORAGE
  !! Storage of finite complex-valued sequences in conjugate-even domain [DFTI_COMPLEX_REAL]
  integer(IP),  bind(C, name="C_DFTI_COMPLEX_COMPLEX")        :: DFTI_COMPLEX_COMPLEX
  !! Complex storage type
  integer(IP),  bind(C, name="C_DFTI_COMPLEX")                :: DFTI_COMPLEX
  !! DFTI_FORWARD_DOMAIN is Complex
  integer(IP),  bind(C, name="C_DFTI_REAL")                   :: DFTI_REAL
  !! DFTI_FORWARD_DOMAIN is Real
  integer(IP),  bind(C, name="C_DFTI_INPLACE")                :: DFTI_INPLACE
  !! Result overwrites input
  integer(IP),  bind(C, name="C_DFTI_NOT_INPLACE")            :: DFTI_NOT_INPLACE
  !! Have another place for result

  interface
    subroutine mkl_dfti_create_desc(precision, domain, dim, length, desc) bind(C)
      import
      integer(c_int), intent(in), value :: precision
      integer(c_int), intent(in), value :: domain
      integer(c_int), intent(in), value :: dim
      integer(c_int), intent(in), value :: length
      type(c_ptr)                       :: desc
    end subroutine mkl_dfti_create_desc

    subroutine mkl_dfti_set_value(desc, param, value) bind(C)
      import
      type(c_ptr),                value :: desc
      integer(c_int), intent(in), value :: param
      integer(c_int), intent(in), value :: value
    end subroutine mkl_dfti_set_value

    subroutine mkl_dfti_commit_desc(desc) bind(C)
      import
      type(c_ptr),                value :: desc
    end subroutine mkl_dfti_commit_desc

    subroutine mkl_dfti_execute(desc, in, out, sign) bind(C)
      import
      type(c_ptr),                value :: desc
      type(c_ptr),                value :: in
      type(c_ptr),                value :: out
      integer(c_int), intent(in), value :: sign
    end subroutine mkl_dfti_execute

    subroutine mkl_dfti_free_desc(desc) bind(C)
      import
      type(c_ptr)                       :: desc
    end subroutine mkl_dfti_free_desc
  endinterface
#endif
end module dtfft_interface_mkl_m