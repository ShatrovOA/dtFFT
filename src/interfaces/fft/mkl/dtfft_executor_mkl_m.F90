!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
module dtfft_executor_mkl_m
!------------------------------------------------------------------------------------------------
!< This module describes MKL Wrappers to dtFFT: [[mkl_executor]]
!< https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/fourier-transform-functions/fft-functions.html
!------------------------------------------------------------------------------------------------
#if defined(_BUILD_DOCS)
#define MKL_ENABLED
#endif
#if defined(MKL_ENABLED)
use dtfft_abstract_executor_m
use dtfft_interface_mkl_m
use dtfft_precisions
use dtfft_parameters
use dtfft_info_m
implicit none
private
public :: mkl_executor

  type, extends(abstract_executor)  :: mkl_executor
  private
    integer(IP)                     :: sign   !< Sign of Transform
  contains
  private
    procedure,  public, pass(self)  :: create
    procedure,  public, pass(self)  :: execute
    procedure,  public, pass(self)  :: destroy
    procedure,          pass(self)  :: make_plan
  endtype mkl_executor

  contains

!------------------------------------------------------------------------------------------------
  subroutine make_plan(self, precision, forward_domain, n, how_many, placement, input, output)
!------------------------------------------------------------------------------------------------
!< Creates general MKL plan
!------------------------------------------------------------------------------------------------
    class(mkl_executor),      intent(inout) :: self
    integer(IP),              intent(in)    :: precision          !< MKL Precision
    integer(IP),              intent(in)    :: forward_domain     !< C2C or R2C flag
    integer(IP),              intent(in)    :: n                  !< Size of 1d transform
    integer(IP),              intent(in)    :: how_many           !< Sets DFTI_NUMBER_OF_TRANSFORMS
    integer(IP),              intent(in)    :: placement          !< Sets DFTI_PLACEMENT
    integer(IP),              intent(in)    :: input              !< Sets DFTI_INPUT_DISTANCE
    integer(IP),              intent(in)    :: output             !< Sets DFTI_OUTPUT_DISTANCE
    integer(IP)                             :: mkl_precision      !< MKL Precision value

    if(precision == DTFFT_DOUBLE) then
      mkl_precision = DFTI_DOUBLE
    elseif(precision == DTFFT_SINGLE) then
      mkl_precision = DFTI_SINGLE
    endif

    call mkl_dfti_create_desc(mkl_precision, forward_domain, 1, n, self%plan)
    call mkl_dfti_set_value(self%plan, DFTI_NUMBER_OF_TRANSFORMS, how_many)
    call mkl_dfti_set_value(self%plan, DFTI_PLACEMENT, placement)
    call mkl_dfti_set_value(self%plan, DFTI_INPUT_DISTANCE, input)
    call mkl_dfti_set_value(self%plan, DFTI_OUTPUT_DISTANCE, output)
    call mkl_dfti_set_value(self%plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
    call mkl_dfti_commit_desc(self%plan)
  end subroutine make_plan

  subroutine create(self, fft_type, precision, real_info, complex_info, sign_or_kind)
    class(mkl_executor),      intent(inout) :: self
    integer(IP),              intent(in)    :: fft_type
    integer(IP),              intent(in)    :: precision
    class(info_t), optional,  intent(in)    :: real_info
    class(info_t), optional,  intent(in)    :: complex_info
    integer(IP),   optional,  intent(in)    :: sign_or_kind

    if(.not. present(complex_info)) error stop "dtFFT Internal Error at 'MKL%create': 'complex_info' is missing"
    select case (fft_type)
    case (FFT_C2C)
      if(.not. present(sign_or_kind)) error stop "dtFFT Internal Error at 'MKL%create': 'sign_or_kind' is missing"
      self%sign = sign_or_kind
      call self%make_plan(precision, DFTI_COMPLEX, complex_info%length(1), complex_info%how_many, DFTI_INPLACE, complex_info%length(1), complex_info%length(1))
    case (FFT_R2C)
      if(.not. present(real_info)) error stop "dtFFT Internal Error at 'MKL%create': 'real_info' is missing"
      self%sign = DTFFT_FORWARD
      call self%make_plan(precision, DFTI_REAL, real_info%length(1), real_info%how_many, DFTI_NOT_INPLACE, real_info%length(1), complex_info%length(1))
    case (FFT_C2R)
      if(.not. present(real_info)) error stop "dtFFT Internal Error at 'MKL%create': 'real_info' is missing"
      self%sign = DTFFT_BACKWARD
      call self%make_plan(precision, DFTI_REAL, real_info%length(1), real_info%how_many, DFTI_NOT_INPLACE, complex_info%length(1), real_info%length(1))
    case (FFT_R2R)
      error stop "dtFFT Error: MKL Does not support R2R FFTs"
    endselect
  end subroutine create

  subroutine execute(self, a, b)
    class(mkl_executor),  intent(in)            :: self
    type(*),              intent(inout), target :: a(..)
    type(*),   optional,  intent(inout), target :: b(..)
    type(C_PTR)                                 :: aa, bb

    aa = c_loc(a)
    if(present(b)) then
      bb = c_loc(b)
      call mkl_dfti_execute(self%plan, aa, bb, self%sign)
    else
      call mkl_dfti_execute(self%plan, aa, aa, self%sign)
    endif
  end subroutine execute

  subroutine destroy(self)
    class(mkl_executor),     intent(inout)  :: self

    call mkl_dfti_free_desc(self%plan)
  end subroutine destroy
#endif
end module dtfft_executor_mkl_m