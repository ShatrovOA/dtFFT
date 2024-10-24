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
!------------------------------------------------------------------------------------------------
!< This module describes cuFFT Wrappers to dtFFT: [[cufft_executor]]
!< https://docs.nvidia.com/cuda/cufft/index.html
!------------------------------------------------------------------------------------------------
use iso_fortran_env,                only: error_unit
use cudafor,                        only: c_devptr
use dtfft_parameters,               only: DTFFT_SUCCESS, DTFFT_SINGLE, DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED, DTFFT_FORWARD
use dtfft_precisions,               only: IP
use dtfft_abstract_executor_m,      only: abstract_executor, FFT_C2C, FFT_R2C
use dtfft_interface_cufft_m
use dtfft_interface_cufft_native_m
use dtfft_utils,                    only: CUFFT_SUCCESS, cufftGetErrorString
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: cufft_executor



  type, extends(abstract_executor) :: cufft_executor
  !< cuFFT FFT Executor
  private
    ! integer(IP)                     :: sign                         !< Sign of Transform
  contains
    procedure, pass(self)  :: create_private => create     !< Creates FFT plan via cuFFT Interface
    procedure, pass(self)  :: execute_private => execute   !< Executes cuFFT plan
    procedure, pass(self)  :: destroy_private => destroy   !< Destroys cuFFT plan
  end type cufft_executor

contains

!------------------------------------------------------------------------------------------------
  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
!------------------------------------------------------------------------------------------------
!< Creates FFT plan via cuFFT Interface
!------------------------------------------------------------------------------------------------
    class(cufft_executor),    intent(inout) :: self           !< cuFFT FFT Executor
    integer(IP),              intent(in)    :: fft_rank       !< Rank of fft: 1 or 2
    integer(IP),              intent(in)    :: fft_type       !< Type of fft: r2r, r2c, c2c
    integer(IP),              intent(in)    :: precision      !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    integer(IP),              intent(in)    :: idist          !< Distance between the first element of two consecutive signals in a batch of the input data.
    integer(IP),              intent(in)    :: odist          !< Distance between the first element of two consecutive signals in a batch of the output data.
    integer(IP),              intent(in)    :: how_many       !< Number of transforms to create
    integer(IP),              intent(in)    :: fft_sizes(:)   !< Dimensions of transform
    integer(IP),              intent(in)    :: inembed(:)     !< Storage dimensions of the input data in memory.
    integer(IP),              intent(in)    :: onembed(:)     !< Storage dimensions of the output data in memory.
    integer(IP),              intent(inout) :: error_code     !< Error code to be returned to user
    integer(IP),   optional,  intent(in)    :: r2r_kinds(:)   !< Kinds of r2r transform
    integer(IP)                             :: cufft_type
    integer(IP)                             :: ierr

    select case (fft_type)
    case (FFT_C2C)
      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_C2C
      else
        cufft_type = CUFFT_Z2Z
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_forward, fft_rank, fft_sizes, inembed, 1, idist, onembed, 1, odist, cufft_type, how_many) )
      self%plan_backward = self%plan_forward
      self%is_inverse_copied = .true.
    case (FFT_R2C)
      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_R2C
      else
        cufft_type = CUFFT_D2Z
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_forward, fft_rank, fft_sizes, inembed, 1, idist, onembed, 1, odist, cufft_type, how_many) )

      if ( precision == DTFFT_SINGLE ) then
        cufft_type = CUFFT_C2R
      else
        cufft_type = CUFFT_Z2D
      endif
      CUFFT_CALL( "cufftPlanMany", cufftPlanMany(self%plan_backward, fft_rank, fft_sizes, onembed, 1, odist, inembed, 1, idist, cufft_type, how_many) )
    case default
      error_code = DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
      if(present(r2r_kinds)) then
      endif
      return
    endselect

  end subroutine create

!------------------------------------------------------------------------------------------------
  subroutine execute(self, a, b, sign)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT plan
!------------------------------------------------------------------------------------------------
    class(cufft_executor),  intent(in)      :: self           !< cuFFT FFT Executor
    type(c_devptr),         intent(in)      :: a              !< Source pointer
    type(c_devptr),         intent(in)      :: b              !< Target pointer
    integer(IP),            intent(in)  :: sign
    integer(IP) :: ierr

    if (.not.self%is_inverse_copied) then
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_forward, a, b, sign) )
      return
    endif
    if ( sign == DTFFT_FORWARD ) then
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_forward, a, b, sign) )
    else
      CUFFT_CALL( "cufftXtExec", cufftXtExec(self%plan_backward, a, b, sign) )
    endif
  end subroutine execute

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys cuFFT plan
!------------------------------------------------------------------------------------------------
    class(cufft_executor), intent(inout)    :: self           !< cuFFT FFT Executor
    integer(IP) :: ierr

    CUFFT_CALL( "cufftDestroy", cufftDestroy(self%plan_forward) )
    if ( .not.self%is_inverse_copied ) then
      CUFFT_CALL( "cufftDestroy", cufftDestroy(self%plan_backward) )
    endif
  end subroutine destroy
end module dtfft_executor_cufft_m