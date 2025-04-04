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
module dtfft_executor_vkfft_m
!! This module describes VkFFT based FFT Executor: [[vkfft_executor]]
!!
!! https://github.com/DTolm/VkFFT/tree/master
use iso_c_binding,                  only: c_ptr, c_int, c_int8_t
use iso_fortran_env,                only: int8, int32, int64
use dtfft_parameters
use dtfft_abstract_executor,        only: abstract_executor, FFT_C2C, FFT_R2C, FFT_R2R
use dtfft_interface_vkfft_m
use dtfft_config,                   only: get_user_stream, get_user_platform
implicit none
private
#include "dtfft_private.h"
public :: vkfft_executor

  type, extends(abstract_executor) :: vkfft_executor
  !! vkFFT FFT Executor
  private
    type(vkfft_wrapper), pointer  :: wrapper => null()
      !! VkFFT Wrapper
    logical                       :: is_inverse_required
      !! Should be create separate inverse FFT Plan or not
  contains
    procedure, pass(self)  :: create_private => create      !! Creates FFT plan via vkFFT Interface
    procedure, pass(self)  :: execute_private => execute    !! Executes vkFFT plan
    procedure, pass(self)  :: destroy_private => destroy    !! Destroys vkFFT plan
    procedure, nopass :: mem_alloc                          !! Dummy method. Raises `error stop`
    procedure, nopass :: mem_free                           !! Dummy method. Raises `error stop`
  end type vkfft_executor

contains

  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
  !! Creates FFT plan via vkFFT Interface
    class(vkfft_executor),            intent(inout) :: self           !! vkFFT FFT Executor
    integer(int8),                    intent(in)    :: fft_rank       !! Rank of fft: 1 or 2
    integer(int8),                    intent(in)    :: fft_type       !! Type of fft: r2r, r2c, c2c
    type(dtfft_precision_t),          intent(in)    :: precision      !! Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    integer(int32),                   intent(in)    :: idist          !! Distance between the first element of two consecutive signals in a batch of the input data.
    integer(int32),                   intent(in)    :: odist          !! Distance between the first element of two consecutive signals in a batch of the output data.
    integer(int32),                   intent(in)    :: how_many       !! Number of transforms to create
    integer(int32),                   intent(in)    :: fft_sizes(:)   !! Dimensions of transform
    integer(int32),                   intent(in)    :: inembed(:)     !! Storage dimensions of the input data in memory.
    integer(int32),                   intent(in)    :: onembed(:)     !! Storage dimensions of the output data in memory.
    integer(int32),                   intent(inout) :: error_code     !! Error code to be returned to user
    type(dtfft_r2r_kind_t), optional, intent(in)    :: r2r_kinds(:)   !! Kinds of r2r transform
    integer(c_int8_t)       :: r2c              !! Is R2C transform required
    integer(c_int8_t)       :: dct              !! Is DCT transform required
    integer(c_int8_t)       :: dst              !! Is DST transform required
    integer(c_int)          :: knd              !! Kind of r2r transform
    integer(c_int)          :: i                !! Loop index
    integer(c_int)          :: dims(2)          !! Dimensions of transform
    integer(c_int)          :: double_precision !! Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    type(dtfft_platform_t)  :: platfrom         !! Platform of the executor

    error_code = DTFFT_SUCCESS
    do i = 1, fft_rank
      dims(i) = fft_sizes(fft_rank - i + 1)
    enddo

    platfrom = get_user_platform()
    CHECK_CALL( load_vkfft(platfrom), error_code )

    if ( platfrom == DTFFT_PLATFORM_CUDA ) then
      self%wrapper => cuda_wrapper
    endif

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
    if ( precision == DTFFT_DOUBLE ) then
      double_precision = 1
    else
      double_precision = 0
    endif
    call self%wrapper%create(fft_rank, dims, double_precision, how_many, r2c, int(0, int8), dct, dst, get_user_stream(), self%plan_forward)
    if ( self%is_inverse_required ) then
      call self%wrapper%create(fft_rank, dims, double_precision, how_many, int(0, int8), r2c, dct, dst, get_user_stream(), self%plan_backward)
    else
      self%plan_backward = self%plan_forward
    endif
  end subroutine create

  subroutine execute(self, a, b, sign)
  !! Executes vkFFT plan
    class(vkfft_executor),  intent(in)      :: self           !! vkFFT FFT Executor
    type(c_ptr),            intent(in)      :: a              !! Source pointer
    type(c_ptr),            intent(in)      :: b              !! Target pointer
    integer(int8),          intent(in)      :: sign           !! Sign of transform

    if ( self%is_inverse_required .and. sign == FFT_BACKWARD ) then
      call self%wrapper%execute(self%plan_backward, a, b, sign)
    else
      call self%wrapper%execute(self%plan_forward, a, b, sign)
    endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys vkFFT plan
    class(vkfft_executor), intent(inout)    :: self           !! vkFFT FFT Executor

    call self%wrapper%destroy(self%plan_forward)
    if ( self%is_inverse_required ) then
      call self%wrapper%destroy(self%plan_backward)
    endif
  end subroutine destroy

  subroutine mem_alloc(alloc_bytes, ptr)
  !! Dummy method. Raises `error stop`
    integer(int64),           intent(in)  :: alloc_bytes  !! Number of bytes to allocate
    type(c_ptr),              intent(out) :: ptr          !! Allocated pointer

    INTERNAL_ERROR("mem_alloc for VkFFT called")
  end subroutine mem_alloc

  subroutine mem_free(ptr)
  !! Dummy method. Raises `error stop`
    type(c_ptr),               intent(in)   :: ptr        !! Pointer to free

    INTERNAL_ERROR("mem_free for VkFFT called")
  end subroutine mem_free
end module dtfft_executor_vkfft_m