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
#include "dtfft_cuda.h"
!------------------------------------------------------------------------------------------------
module dtfft_abstract_executor_m
!------------------------------------------------------------------------------------------------
!< This module describes `abstract_executor`: Abstract FFT wrapper class
!------------------------------------------------------------------------------------------------
#ifdef DTFFT_WITH_CUDA
use cudafor,          only: c_devloc, c_devptr
#else
use iso_c_binding,    only: c_loc
#endif
use iso_c_binding,    only: c_ptr, c_int, c_null_ptr, c_associated
use dtfft_info_m,     only: info_t
use dtfft_precisions, only: IP
use dtfft_parameters, only: DTFFT_SUCCESS, DTFFT_FORWARD, DTFFT_BACKWARD
#include "dtfft_profile.h"
implicit none
private
public :: abstract_executor

  integer(IP),  public, parameter :: FFT_C2C = 0
  integer(IP),  public, parameter :: FFT_R2C = 1
  integer(IP),  public, parameter :: FFT_R2R = 2

  integer(IP),  public, parameter :: FFT_1D = 1
  integer(IP),  public, parameter :: FFT_2D = 2

  type, abstract :: abstract_executor
  !< The "most" abstract executor.
  !< All FFT executors are extending this class.
    type(c_ptr)         :: plan_forward
    type(c_ptr)         :: plan_backward
    integer(c_int)      :: fft_type
    logical,    private :: is_created = .false.
    logical             :: is_inverse_copied = .false.
  contains
    procedure,  non_overridable,              pass(self), public    :: create               !< Creates FFT plan
    procedure,  non_overridable,              pass(self), public    :: execute              !< Executes plan
    procedure,  non_overridable,              pass(self), public    :: destroy              !< Destroys plan
    procedure(create_interface),              pass(self), deferred  :: create_private       !< Creates FFT plan
    procedure(execute_interface),             pass(self), deferred  :: execute_private      !< Executes plan
    procedure(destroy_interface),             pass(self), deferred  :: destroy_private      !< Destroys plan
  end type abstract_executor

  abstract interface
!------------------------------------------------------------------------------------------------
    subroutine create_interface(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
!------------------------------------------------------------------------------------------------
!< Creates FFT plan
!------------------------------------------------------------------------------------------------
    import
      class(abstract_executor), intent(inout) :: self           !< FFT Executor
      integer(IP),              intent(in)    :: fft_rank       !< Rank of fft: 1 or 2
      integer(IP),              intent(in)    :: fft_type       !< Type of fft: r2r, r2c, c2r, c2c
      integer(IP),              intent(in)    :: precision      !< Precision of fft: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
      integer(IP),              intent(in)    :: idist          !< Distance between the first element of two consecutive signals in a batch of the input data.
      integer(IP),              intent(in)    :: odist          !< Distance between the first element of two consecutive signals in a batch of the output data.
      integer(IP),              intent(in)    :: how_many       !< Number of transforms to create
      integer(IP),              intent(in)    :: fft_sizes(:)   !< Dimensions of transform
      integer(IP),              intent(in)    :: inembed(:)     !< Storage dimensions of the input data in memory.
      integer(IP),              intent(in)    :: onembed(:)     !< Storage dimensions of the output data in memory.
      integer(IP),              intent(inout) :: error_code     !< Error code to be returned to user
      integer(IP),   optional,  intent(in)    :: r2r_kinds(:)   !< Kinds of r2r transform
    end subroutine create_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_interface(self, a, b, sign)
!------------------------------------------------------------------------------------------------
!< Executes plan
!------------------------------------------------------------------------------------------------
    import
      class(abstract_executor), intent(in)  :: self            !< FFT Executor
      type(C_ADDR),             intent(in)  :: a               !< Source pointer
      type(C_ADDR),             intent(in)  :: b               !< Target pointer
      integer(IP),              intent(in)  :: sign
    end subroutine execute_interface

!------------------------------------------------------------------------------------------------
    subroutine destroy_interface(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan
!------------------------------------------------------------------------------------------------
    import
      class(abstract_executor), intent(inout) :: self           !< FFT Executor
    end subroutine destroy_interface
  end interface

contains
!------------------------------------------------------------------------------------------------
  integer(IP) function create(self, fft_rank, fft_type, precision, real_info, complex_info, r2r_kinds)
!------------------------------------------------------------------------------------------------
!< Creates FFT plan
!------------------------------------------------------------------------------------------------
    class(abstract_executor), intent(inout) :: self             !< FFT Executor
    integer(IP),              intent(in)    :: fft_rank         !< Rank of fft: 1 or 2
    integer(IP),              intent(in)    :: fft_type         !< Type of fft: r2r, r2c, c2c
    integer(IP),              intent(in)    :: precision        !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    ! integer(IP),              intent(in)    :: sign             !< Sign of C2C/R2C transform
    type(info_t),  optional,  intent(in)    :: real_info        !< Real data layout
    type(info_t),  optional,  intent(in)    :: complex_info     !< Complex data layout
    integer(IP),   optional,  intent(in)    :: r2r_kinds(:)     !< Kinds of r2r transform
    integer(IP),   allocatable              :: fft_sizes(:), inembed(:), onembed(:)
    integer(IP) ::  idist, odist, how_many

    create = DTFFT_SUCCESS
    if ( self%is_created .and. .not.c_associated(self%plan_forward, c_null_ptr) .and. .not.c_associated(self%plan_backward, c_null_ptr) ) return

    PHASE_BEGIN("Creating FFT")
    self%plan_forward = c_null_ptr
    self%plan_backward = c_null_ptr
    self%is_created = .false.
    self%is_inverse_copied = .false.
    if ( fft_rank /= FFT_1D .and. fft_rank /= FFT_2D ) error stop
    if ( (fft_type == FFT_R2C).and.(.not.present(complex_info) .or. .not.present(real_info))) error stop
    if ( (fft_type == FFT_R2R).and.(.not.present(real_info) .or..not.present(r2r_kinds)) ) error stop

    allocate( fft_sizes(fft_rank), inembed(fft_rank), onembed(fft_rank) )

    select case (fft_type)
    case (FFT_C2C)
      select case (fft_rank)
      case (FFT_1D)
        fft_sizes(1) = complex_info%counts(1)
      case (FFT_2D)
        fft_sizes(1) = complex_info%counts(2)
        fft_sizes(2) = complex_info%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      onembed(:) = fft_sizes(:)
      idist = product(fft_sizes)
      odist = idist
      how_many = product(complex_info%counts) / idist
    case (FFT_R2C)
      select case ( fft_rank )
      case ( FFT_1D )
        fft_sizes(1) = real_info%counts(1)
        onembed(1) = complex_info%counts(1)
      case ( FFT_2D )
        fft_sizes(1) = real_info%counts(2)
        fft_sizes(2) = real_info%counts(1)
        onembed(1) = complex_info%counts(2)
        onembed(2) = complex_info%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      idist = product(inembed)
      odist = product(onembed)
      how_many = product(real_info%counts) / idist
    case (FFT_R2R)
      select case (fft_rank)
      case (FFT_1D)
        fft_sizes(1) = real_info%counts(1)
      case (FFT_2D)
        fft_sizes(1) = real_info%counts(2)
        fft_sizes(2) = real_info%counts(1)
      endselect
      inembed(:) = fft_sizes(:)
      onembed(:) = fft_sizes(:)
      idist = product(fft_sizes)
      odist = idist
      how_many = product(real_info%counts) / idist
    endselect
    if ( how_many == 0 ) then
      PHASE_END("Creating FFT")
      return
    endif

    self%fft_type = fft_type
    call self%create_private(fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, create, r2r_kinds)
    if ( c_associated(self%plan_forward, c_null_ptr) .or. c_associated(self%plan_backward, c_null_ptr) ) error stop "Failed to create FFT Executor"
    if( create == DTFFT_SUCCESS ) self%is_created = .true.
    deallocate( fft_sizes, inembed, onembed )
    PHASE_END("Creating FFT")
  end function create

!------------------------------------------------------------------------------------------------
  subroutine execute(self, a, b, sign)
!------------------------------------------------------------------------------------------------
!< Executes plan
!------------------------------------------------------------------------------------------------
    class(abstract_executor), intent(in)    :: self             !< FFT Executor
    type(*),      target,     intent(inout)   &
#ifdef DTFFT_WITH_CUDA
      , device                                &
#endif
                                            :: a(..)            !< Source buffer
    type(*),      target,     intent(inout)   &
#ifdef DTFFT_WITH_CUDA
      , device                                &
#endif
                                            :: b(..)            !< Target buffer
    integer(IP),              intent(in)    :: sign
    if ( .not.self%is_created ) return
    PHASE_BEGIN("Executing FFT")
    call self%execute_private(LOC_FUN(a), LOC_FUN(b), sign)
    PHASE_END("Executing FFT")
  end subroutine execute

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan
!------------------------------------------------------------------------------------------------
    class(abstract_executor), intent(inout) :: self             !< FFT Executor
    if ( self%is_created ) call self%destroy_private()
    self%is_created = .false.
    self%is_inverse_copied = .false.
  end subroutine destroy
end module dtfft_abstract_executor_m