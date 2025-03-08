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
#include "dtfft_config.h"
module dtfft_executor_vkfft_m
!! This module describes vkFFT Wrappers to dtFFT: ``vkfft_executor``
!!
!! https://github.com/DTolm/VkFFT/tree/master
use iso_c_binding,                  only: c_ptr, c_int, c_int8_t
use iso_fortran_env,                only: int8, int32
use dtfft_parameters
use dtfft_abstract_executor,        only: abstract_executor, FFT_C2C, FFT_R2C, FFT_R2R
use dtfft_interface_vkfft_m
use dtfft_utils,                    only: get_user_stream
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: vkfft_executor

  type, extends(abstract_executor) :: vkfft_executor
  !! vkFFT FFT Executor
  private
    logical :: is_inverse_required
  contains
    procedure, pass(self)  :: create_private => create     !< Creates FFT plan via vkFFT Interface
    procedure, pass(self)  :: execute_private => execute   !< Executes vkFFT plan
    procedure, pass(self)  :: destroy_private => destroy   !< Destroys vkFFT plan
  end type vkfft_executor

contains

  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
  !! Creates FFT plan via vkFFT Interface
    class(vkfft_executor),            intent(inout) :: self           !< vkFFT FFT Executor
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
    integer(c_int8_t) :: r2c, dct, dst
    integer(c_int)    :: knd, i, dims(2)

    error_code = DTFFT_SUCCESS
    do i = 1, fft_rank
      dims(i) = fft_sizes(fft_rank - i + 1)
    enddo

    r2c = 0
    dct = 0
    dst = 0
    self%is_inverse_required = .false.
    select case ( fft_type )
    case ( FFT_R2C )
      r2c = 1
      self%is_inverse_required = .true.
    case ( FFT_R2R )
      knd = r2r_kinds(1)%val
      do i = 2, fft_rank
        if ( knd /= r2r_kinds(i)%val ) then
          error_code = DTFFT_ERROR_VKFFT_R2R_2D_PLAN
          return
        endif
      enddo
      select case ( knd )
      case ( DTFFT_DCT_1%val )
        dct = 1
      case ( DTFFT_DCT_2%val )
        dct = 2
      case ( DTFFT_DCT_3%val )
        dct = 3
      case ( DTFFT_DCT_4%val )
        dct = 4
      case ( DTFFT_DST_1%val )
        dst = 1
      case ( DTFFT_DST_2%val )
        dst = 2
      case ( DTFFT_DST_3%val )
        dst = 3
      case ( DTFFT_DST_4%val )
        dst = 4
      endselect
    endselect
    call vkfft_create(fft_rank, dims, precision%val, how_many, r2c, int(0, int8), dct, dst, get_user_stream(), self%plan_forward)
    if ( self%is_inverse_required ) then
      call vkfft_create(fft_rank, dims, precision%val, how_many, int(0, int8), r2c, dct, dst, get_user_stream(), self%plan_backward)
    endif
  end subroutine create

  subroutine execute(self, a, b, sign)
  !! Executes vkFFT plan
    class(vkfft_executor),  intent(in)      :: self           !< vkFFT FFT Executor
    type(c_ptr),            intent(in)      :: a              !< Source pointer
    type(c_ptr),            intent(in)      :: b              !< Target pointer
    integer(int8),          intent(in)      :: sign           !< Sign of transform

    if ( self%is_inverse_required .and. sign == FFT_BACKWARD ) then
      call vkfft_execute(self%plan_backward, a, b, sign)
    else
      call vkfft_execute(self%plan_forward, a, b, sign)
    endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys vkFFT plan
    class(vkfft_executor), intent(inout)    :: self           !< vkFFT FFT Executor

    call vkfft_destroy(self%plan_forward)
    if ( self%is_inverse_required ) then
      call vkfft_destroy(self%plan_backward)
    endif
  end subroutine destroy
end module dtfft_executor_vkfft_m