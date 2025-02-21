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
module dtfft_abstract_executor
!! This module describes `abstract_executor`: Abstract FFT wrapper class
use iso_c_binding,    only: c_loc, c_ptr, c_int, c_null_ptr, c_associated
use iso_fortran_env,  only: int8, int32, int64
use dtfft_pencil,     only: pencil
use dtfft_parameters
use dtfft_utils
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
implicit none
private
public :: abstract_executor

  integer(int8),  public, parameter :: FFT_C2C = 0
  integer(int8),  public, parameter :: FFT_R2C = 1
  integer(int8),  public, parameter :: FFT_R2R = 2

  integer(int8),  public, parameter :: FFT_1D = 1
  integer(int8),  public, parameter :: FFT_2D = 2

  type, abstract :: abstract_executor
  !< The "most" abstract executor.
  !< All FFT executors are extending this class.
    type(c_ptr)         :: plan_forward
    type(c_ptr)         :: plan_backward
    logical,    private :: is_created = .false.
    logical             :: is_inverse_copied = .false.
  contains
    procedure,  non_overridable,              pass(self), public  :: create               !< Creates FFT plan
    procedure,  non_overridable,              pass(self), public  :: execute              !< Executes plan
    procedure,  non_overridable,              pass(self), public  :: destroy              !< Destroys plan
#ifndef DTFFT_WITH_CUDA
    procedure(mem_alloc_interface), deferred, nopass,     public  :: mem_alloc            !< Allocates aligned memory
    procedure(mem_free_interface),  deferred, nopass,     public  :: mem_free             !< Frees aligned memory
#endif
    procedure(create_interface),    deferred, pass(self)          :: create_private       !< Creates FFT plan
    procedure(execute_interface),   deferred, pass(self)          :: execute_private      !< Executes plan
    procedure(destroy_interface),   deferred, pass(self)          :: destroy_private      !< Destroys plan
  end type abstract_executor

  abstract interface
    subroutine create_interface(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
    !! Creates FFT plan
    import
      class(abstract_executor),         intent(inout) :: self           !< FFT Executor
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
    end subroutine create_interface

    subroutine execute_interface(self, a, b, sign)
    !! Executes plan
    import
      class(abstract_executor), intent(in)  :: self             !< FFT Executor
      type(c_ptr),              intent(in)  :: a                !< Source pointer
      type(c_ptr),              intent(in)  :: b                !< Target pointer
      integer(int8),            intent(in)  :: sign             !< Sign of transform
    end subroutine execute_interface

    subroutine destroy_interface(self)
    !! Destroys plan
    import
      class(abstract_executor), intent(inout) :: self           !< FFT Executor
    end subroutine destroy_interface

#ifndef DTFFT_WITH_CUDA
    subroutine mem_alloc_interface(alloc_bytes, ptr)
    !! Allocates aligned memory
    import
      integer(int64),           intent(in)    :: alloc_bytes
      type(c_ptr),              intent(out)   :: ptr
    end subroutine mem_alloc_interface

    subroutine mem_free_interface(ptr)
    !! Frees aligned memory
    import
     type(c_ptr),               intent(inout) :: ptr
    end subroutine mem_free_interface
#endif
  end interface

contains
  integer(int32) function create(self, fft_rank, fft_type, precision, real_pencil, complex_pencil, r2r_kinds)
  !! Creates FFT plan
    class(abstract_executor),           intent(inout) :: self             !< FFT Executor
    integer(int8),                      intent(in)    :: fft_rank         !< Rank of fft: 1 or 2
    integer(int8),                      intent(in)    :: fft_type         !< Type of fft: r2r, r2c, c2c
    type(dtfft_precision_t),            intent(in)    :: precision        !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    type(pencil),           optional,   intent(in)    :: real_pencil      !< Real data layout
    type(pencil),           optional,   intent(in)    :: complex_pencil   !< Complex data layout
    type(dtfft_r2r_kind_t), optional,   intent(in)    :: r2r_kinds(:)     !< Kinds of r2r transform
    integer(int32),         allocatable   :: fft_sizes(:)     !< Dimensions of transform
    integer(int32),         allocatable   :: inembed(:)       !< 
    integer(int32),         allocatable   :: onembed(:)       !< 
    integer(int32)                        :: idist            !< Distance between the first element of two consecutive signals in a batch of the input data.
    integer(int32)                        :: odist            !< Distance between the first element of two consecutive signals in a batch of the output data.
    integer(int32)                        :: how_many         !< Number of transforms to create

    create = DTFFT_SUCCESS
    if ( self%is_created .and. .not.c_associated(self%plan_forward, c_null_ptr) .and. .not.c_associated(self%plan_backward, c_null_ptr) ) return

    PHASE_BEGIN("Creating FFT", COLOR_FFT)

    self%plan_forward = c_null_ptr
    self%plan_backward = c_null_ptr
    self%is_created = .false.
    self%is_inverse_copied = .false.
    if ( fft_rank /= FFT_1D .and. fft_rank /= FFT_2D ) error stop
    if ( (fft_type == FFT_R2C).and.(.not.present(complex_pencil) .or. .not.present(real_pencil))) error stop
    if ( (fft_type == FFT_R2R).and.(.not.present(real_pencil) .or..not.present(r2r_kinds)) ) error stop

    allocate( fft_sizes(fft_rank), inembed(fft_rank), onembed(fft_rank) )

    how_many = 0
    select case (fft_type)
    case (FFT_C2C)
      select case (fft_rank)
      case (FFT_1D)
        fft_sizes(1) = complex_pencil%counts(1)
      case (FFT_2D)
        fft_sizes(1) = complex_pencil%counts(2)
        fft_sizes(2) = complex_pencil%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      onembed(:) = fft_sizes(:)
      idist = product(fft_sizes)
      odist = idist
      how_many = product(complex_pencil%counts) / idist
    case (FFT_R2C)
      select case ( fft_rank )
      case ( FFT_1D )
        fft_sizes(1) = real_pencil%counts(1)
        onembed(1) = complex_pencil%counts(1)
      case ( FFT_2D )
        fft_sizes(1) = real_pencil%counts(2)
        fft_sizes(2) = real_pencil%counts(1)
        onembed(1) = complex_pencil%counts(2)
        onembed(2) = complex_pencil%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      idist = product(inembed)
      odist = product(onembed)
      how_many = product(real_pencil%counts) / idist
    case (FFT_R2R)
      select case (fft_rank)
      case (FFT_1D)
        fft_sizes(1) = real_pencil%counts(1)
      case (FFT_2D)
        fft_sizes(1) = real_pencil%counts(2)
        fft_sizes(2) = real_pencil%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      onembed(:) = fft_sizes(:)
      idist = product(fft_sizes)
      odist = idist
      how_many = product(real_pencil%counts) / idist
    endselect
    if ( how_many == 0 ) then
      PHASE_END("Creating FFT")
      return
    endif

    call self%create_private(fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, create, r2r_kinds)
    if ( c_associated(self%plan_forward, c_null_ptr) .or. c_associated(self%plan_backward, c_null_ptr) ) error stop "Failed to create FFT Executor"
    if( create == DTFFT_SUCCESS ) self%is_created = .true.
    deallocate( fft_sizes, inembed, onembed )
    PHASE_END("Creating FFT")
  end function create

  subroutine execute(self, a, b, sign)
  !! Executes plan
    class(abstract_executor),     intent(in)    :: self             !< FFT Executor
    type(*),  DEVICE_PTR  target, intent(inout) :: a(..)            !< Source buffer
    type(*),  DEVICE_PTR  target, intent(inout) :: b(..)            !< Target buffer
    integer(int8),                intent(in)    :: sign             !< Sign of transform
    if ( .not.self%is_created ) return
    PHASE_BEGIN("Executing FFT", COLOR_FFT)
    call self%execute_private(c_loc(a), c_loc(b), sign)
    PHASE_END("Executing FFT")
  end subroutine execute

  subroutine destroy(self)
  !! Destroys plan
    class(abstract_executor), intent(inout) :: self             !< FFT Executor
    if ( self%is_created ) call self%destroy_private()
    self%plan_forward = c_null_ptr
    self%plan_backward = c_null_ptr
    self%is_created = .false.
    self%is_inverse_copied = .false.
  end subroutine destroy
end module dtfft_abstract_executor