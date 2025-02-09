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
module dtfft_executor_cufft_m
!! This module describes cuFFT Wrappers to dtFFT: ``cufft_executor``
!!
!! https://docs.nvidia.com/cuda/cufft/index.html
use iso_c_binding,                  only: c_ptr, c_int, c_null_ptr, c_loc
use iso_fortran_env,                only: int8, int32
use cudafor,                        only: cudaSuccess
use dtfft_parameters
use dtfft_abstract_executor,        only: abstract_executor, FFT_C2C, FFT_R2C
use dtfft_interface_cufft_m
use dtfft_interface_cufft_native_m
use dtfft_utils,                    only: int_to_str, get_user_stream
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: cufft_executor

  type, extends(abstract_executor) :: cufft_executor
  !! cuFFT FFT Executor
  private
  contains
    procedure :: create_private => create     !< Creates FFT plan via cuFFT Interface
    procedure :: execute_private => execute   !< Executes cuFFT plan
    procedure :: destroy_private => destroy   !< Destroys cuFFT plan
  end type cufft_executor

contains

  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
  !! Creates FFT plan via cuFFT Interface
    class(cufft_executor),            intent(inout) :: self           !< cuFFT FFT Executor
    integer(int8),                    intent(in)    :: fft_rank       !< Rank of fft: 1 or 2
    integer(int8),                    intent(in)    :: fft_type       !< Type of fft: r2r, r2c, c2c
    type(dtfft_precision_t),          intent(in)    :: precision      !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    integer(int32),                   intent(in)    :: idist          !< Distance between the first element of two consecutive signals in a batch of the input data.
    integer(int32),                   intent(in)    :: odist          !< Distance between the first element of two consecutive signals in a batch of the output data.
    integer(int32),                   intent(in)    :: how_many       !< Number of transforms to create
    integer(int32),                   intent(in)    :: fft_sizes(:)   !< Dimensions of transform
    integer(int32),                   intent(in)    :: inembed(:)     !< Storage dimensions of the input data in memory.
    integer(int32),                   intent(in)    :: onembed(:)     !< Storage dimensions of the output data in memory.
    integer(int32),                   intent(inout) :: error_code     !< Error code to be returned to user
    type(dtfft_r2r_kind_t), optional, intent(in)    :: r2r_kinds(:)   !< Kinds of r2r transform
    integer(c_int)                                  :: cufft_type, rnk

    rnk = int(fft_rank, c_int)
    select case (fft_type)
    case (FFT_C2C)
      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_C2C
      else
        cufft_type = CUFFT_Z2Z
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_forward, rnk, fft_sizes, inembed, 1, idist, onembed, 1, odist, cufft_type, how_many) )
      self%plan_backward = self%plan_forward
      self%is_inverse_copied = .true.
    case (FFT_R2C)
      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_R2C
      else
        cufft_type = CUFFT_D2Z
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_forward, rnk, fft_sizes, inembed, 1, idist, onembed, 1, odist, cufft_type, how_many) )

      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_C2R
      else
        cufft_type = CUFFT_Z2D
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_backward, rnk, fft_sizes, onembed, 1, odist, inembed, 1, idist, cufft_type, how_many) )
    case default
      error_code = DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
      if(present(r2r_kinds)) then
      endif
      return
    endselect
    CUFFT_CALL( "cufftSetStream", cufftSetStream(self%plan_forward, get_user_stream()) )
    if ( .not.self%is_inverse_copied ) then
      CUFFT_CALL( "cufftSetStream", cufftSetStream(self%plan_backward, get_user_stream()) )
    endif
  end subroutine create

  subroutine execute(self, a, b, sign)
  !! Executes cuFFT plan
    class(cufft_executor),  intent(in)      :: self           !< cuFFT FFT Executor
    type(c_ptr),            intent(in)      :: a              !< Source pointer
    type(c_ptr),            intent(in)      :: b              !< Target pointer
    integer(int8),          intent(in)      :: sign           !< Sign of transform
    integer(c_int) :: sign_

    sign_ = int(sign, c_int)

    if (self%is_inverse_copied) then
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_forward, a, b, sign_) )
      return
    endif
    if ( sign == FFT_FORWARD ) then
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_forward, a, b, sign_) )
    else
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_backward, a, b, sign_) )
    endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys cuFFT plan
    class(cufft_executor), intent(inout)    :: self           !< cuFFT FFT Executor

    CUFFT_CALL( "cufftDestroy", cufftDestroy(self%plan_forward) )
    if ( .not.self%is_inverse_copied ) then
      CUFFT_CALL( "cufftDestroy", cufftDestroy(self%plan_backward) )
    endif
  end subroutine destroy
end module dtfft_executor_cufft_m