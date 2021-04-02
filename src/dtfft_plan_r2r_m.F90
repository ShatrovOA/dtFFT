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
module dtfft_plan_r2r_m
!------------------------------------------------------------------------------------------------
!< This module describes [[dtfft_plan_r2r_2d]] and [[dtfft_plan_r2r_3d]] classes
!------------------------------------------------------------------------------------------------
use dtfft_plan_base_m
use dtfft_abstract_executor_m
use dtfft_executor_fftw_m
use dtfft_precisions
use dtfft_parameters
use iso_fortran_env, only: output_unit
#include "dtfft_mpi.h"
implicit none
private
public :: dtfft_plan_r2r_2d, dtfft_plan_r2r_3d

!------------------------------------------------------------------------------------------------
  type :: r2r_executor
!------------------------------------------------------------------------------------------------
!< Derived datatype to handle external 1d FFT executors
!------------------------------------------------------------------------------------------------
    class(abstract_r2r_executor), allocatable :: r2r        !< Abstract executor
  end type r2r_executor

!------------------------------------------------------------------------------------------------
  type, extends(dtfft_base_plan) :: dtfft_base_plan_r2r
!------------------------------------------------------------------------------------------------
!< Base plan for r2r transform
!------------------------------------------------------------------------------------------------
    type(r2r_executor),   allocatable :: fplans(:)          !< Plans for forward (DTFFT_TRANSPOSED_OUT) transform
    type(r2r_executor),   allocatable :: bplans(:)          !< Plans for backward (DTFFT_TRANSPOSED_IN) transform
  contains
    procedure, pass(self)             :: create_fft_plans   !< Creates 1d plans
    procedure, pass(self)             :: destroy_plan_r2r   !< Destroys 1d plans and releases all internal memory
  end type

!------------------------------------------------------------------------------------------------
  type, extends(dtfft_base_plan_r2r) :: dtfft_plan_r2r_2d
!------------------------------------------------------------------------------------------------
!< Plan for two-dimensional r2r transform
!------------------------------------------------------------------------------------------------
  private
  contains
  private
    procedure, pass(self),     public :: create           => create_r2r_2d        !< Create 2d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_r2r_2d      !< Create 2d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_2d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: execute          => execute_2d           !< Executes 2d plan, double precision
    procedure, pass(self),     public :: execute_f        => execute_f_2d         !< Executes 2d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_2d           !< Destroys 2d plan
  end type dtfft_plan_r2r_2d

!------------------------------------------------------------------------------------------------
  type, extends(dtfft_base_plan_r2r) :: dtfft_plan_r2r_3d
!------------------------------------------------------------------------------------------------
!< Plan for three-dimensional r2r transform
!------------------------------------------------------------------------------------------------
  private
    real(R8P),            allocatable :: work(:)                                  !< Worker buffer used to store Y-pencil, double precision
    real(R4P),            allocatable :: f_work(:)                                !< Worker buffer used to store Y-pencil, single precision
  contains
  private
    procedure, pass(self),     public :: create           => create_r2r_3d        !< Create 3d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_r2r_3d      !< Create 3d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_3d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: get_worker_size  => get_worker_size_3d   !< Returns local sizes, counts and allocation size of optional worker buffer
    procedure, pass(self),     public :: execute          => execute_3d           !< Executes 3d plan, double precision
    procedure, pass(self),     public :: execute_f        => execute_f_3d         !< Executes 3d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_3d           !< Destroys 3d plan
    procedure, pass(self)             :: execute_transposed_out                   !< Executes DTFFT_TRANSPOSED_OUT plan with worker buffer, double precision
    procedure, pass(self)             :: execute_transposed_in                    !< Executes DTFFT_TRANSPOSED_IN plan with worker buffer, double precision
    procedure, pass(self)             :: execute_f_transposed_out                 !< Executes DTFFT_TRANSPOSED_OUT plan with worker buffer, single precision
    procedure, pass(self)             :: execute_f_transposed_in                  !< Executes DTFFT_TRANSPOSED_IN plan with worker buffer, single precision
  end type dtfft_plan_r2r_3d

contains

!------------------------------------------------------------------------------------------------
  subroutine create_fft_plans(self, in_kinds, out_kinds, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates 1d plans with provided executor_type value 
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_r2r), intent(inout) :: self             !< Base R2R Plan
    integer(IP),                intent(in)    :: in_kinds(:)      !< Forward kinds
    integer(IP),                intent(in)    :: out_kinds(:)     !< Backward kinds
    integer(IP),  optional,     intent(in)    :: executor_type    !< Type of External FFT Executor
    integer(IP)                               :: executor         !< External FFT executor
    integer(IP)                               :: d                !< Counter

    allocate(self%fplans(self%dims), self%bplans(self%dims))

    executor = DTFFT_EXECUTOR_FFTW3
    if(present(executor_type)) executor = executor_type
    do d = 1, self%dims
      select case(executor)
      case (DTFFT_EXECUTOR_FFTW3)
#ifndef NO_FFTW3
#ifdef __DEBUG
        if(self%comm_rank == 0 .and. d == 1) write(output_unit, '(a)') "DTFFT is using FFTW3 executor"
#endif
        allocate(fftw_r2r_executor :: self%fplans(d)%r2r)
        allocate(fftw_r2r_executor :: self%bplans(d)%r2r)
#else
        error stop "FFTW3 is disabled in this build"
#endif
      case (DTFFT_EXECUTOR_MKL)
        error stop "DTFFT: MKL currently does not support r2r transforms"
      case (DTFFT_EXECUTOR_CUFFT)
        error stop "DTFFT: CUFFT currently does not support r2r transforms"
      case default
        error stop "DTFFT: Unrecognized executor"
      endselect

      call self%fplans(d)%r2r%create_plan(self%info(d), in_kinds(d), self%precision)
      call self%bplans(d)%r2r%create_plan(self%info(d), out_kinds(d), self%precision)
    enddo
  end subroutine create_fft_plans

!------------------------------------------------------------------------------------------------
  subroutine destroy_plan_r2r(self)
!------------------------------------------------------------------------------------------------
!< Destroys 1d plans and releases all internal memory
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_r2r), intent(inout) :: self           !< Base R2R Plan
    integer(IP)                               :: d              !< Counter

    do d = 1, self%dims
      call self%fplans(d)%r2r%destroy()
      call self%bplans(d)%r2r%destroy()
    enddo
    deallocate(self%fplans)
    deallocate(self%bplans)
    call self%destroy_base_plan()
  end subroutine destroy_plan_r2r

!------------------------------------------------------------------------------------------------
  subroutine create_r2r_2d(self, comm, nx, ny, in_kinds, out_kinds, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan
    TYPE_MPI_COMM,            intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: in_kinds(2)      !< Forward kinds
    integer(IP),              intent(in)    :: out_kinds(2)     !< Backward kinds
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, R8P, [nx, ny], MPI_DOUBLE_PRECISION, DOUBLE_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(in_kinds, out_kinds, executor_type)
    self%is_created = .true.
  end subroutine create_r2r_2d

!------------------------------------------------------------------------------------------------
  subroutine create_f_r2r_2d(self, comm, nx, ny, in_kinds, out_kinds, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional r2r transform, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan
    TYPE_MPI_COMM,            intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: in_kinds(2)      !< Forward kinds
    integer(IP),              intent(in)    :: out_kinds(2)     !< Backward kinds
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, R4P, [nx, ny], MPI_REAL, FLOAT_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(in_kinds, out_kinds, executor_type)
    self%is_created = .true.
  end subroutine create_f_r2r_2d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_2d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation sizes
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan
    integer(IP),  optional,   intent(out)   :: in_starts(:)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: in_counts(:)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: out_starts(:)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: out_counts(:)    !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: alloc_size       !< Maximum number of elements needs to be allocated

    call self%get_local_sizes_internal(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_2d(self, in, out, transpose_type)
!------------------------------------------------------------------------------------------------
!< Executes 2d r2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type   !< Direction of transposition

    call self%check_plan(R8P)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%r2r%execute(in)
      ! Transpose X -> Y
      call self%transpose(self%transpose_out(1), in, out, 'X', 'Y')
      ! 1d FFT Y direction
      call self%fplans(2)%r2r%execute(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Y direction
      call self%bplans(2)%r2r%execute(in)
      ! Transpose Y -> X
      call self%transpose(self%transpose_in(1), in, out, 'Y', 'X')
      ! 1d FFT X direction
      call self%bplans(1)%r2r%execute(out)
    else
      error stop "dtfft_execute_r2r_2d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_2d(self, in, out, transpose_type)
!------------------------------------------------------------------------------------------------
!< Executes 2d r2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type   !< Direction of transposition

    call self%check_plan(R4P)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%r2r%execute_f(in)
      ! Transpose X -> Y
      call self%transpose(self%transpose_out(1), in, out, 'X', 'Y')
      ! 1d FFT Y direction
      call self%fplans(2)%r2r%execute_f(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Y direction
      call self%bplans(2)%r2r%execute_f(in)
      ! Transpose Y -> X
      call self%transpose(self%transpose_in(1), in, out, 'Y', 'X')
      ! 1d FFT X direction
      call self%bplans(1)%r2r%execute_f(out)
    else
      error stop "dtfft_execute_f_r2r_2d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_f_2d

!------------------------------------------------------------------------------------------------
  subroutine destroy_2d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 2d r2r transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_2d), intent(inout) :: self             !< R2R 2D Plan

    call self%destroy_plan_r2r()
  end subroutine destroy_2d

!------------------------------------------------------------------------------------------------
  subroutine create_r2r_3d(self, comm, nx, ny, nz, in_kinds, out_kinds, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    TYPE_MPI_COMM,            intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz               !< Number of points in Z direction
    integer(IP),              intent(in)    :: in_kinds(3)      !< Forward kinds
    integer(IP),              intent(in)    :: out_kinds(3)     !< Backward kinds
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, R8P, [nx, ny, nz], MPI_DOUBLE_PRECISION, DOUBLE_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(in_kinds, out_kinds, executor_type)
    self%is_created = .true.
  end subroutine create_r2r_3d

!------------------------------------------------------------------------------------------------
  subroutine create_f_r2r_3d(self, comm, nx, ny, nz, in_kinds, out_kinds, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    TYPE_MPI_COMM,            intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz               !< Number of points in Z direction
    integer(IP),              intent(in)    :: in_kinds(3)      !< Forward kinds
    integer(IP),              intent(in)    :: out_kinds(3)     !< Backward kinds
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, R4P, [nx, ny, nz], MPI_REAL, FLOAT_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(in_kinds, out_kinds, executor_type)
    self%is_created = .true.
  end subroutine create_f_r2r_3d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_3d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    integer(IP),  optional,   intent(out)   :: in_starts(:)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: in_counts(:)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: out_starts(:)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: out_counts(:)    !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: alloc_size       !< Maximum number of elements needs to be allocated

    call self%get_local_sizes_internal(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_3d

!------------------------------------------------------------------------------------------------
  subroutine get_worker_size_3d(self, starts, counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size of optional worker buffer
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    integer(IP),  optional,   intent(out)   :: starts(:)        !< Starts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: counts(:)        !< Counts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: alloc_size       !< Number of elements to be allocated

    call self%get_worker_size_internal(2, starts, counts, alloc_size)
  end subroutine get_worker_size_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_3d(self, in, out, transpose_type, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type   !< Direction of transposition
    real(R8P),    optional,   intent(inout) :: work(*)          !< Working buffer
    integer(IP)                             :: alloc_size       !< Size of internal work buffer

    call self%check_plan(R8P)
    if(.not. present(work)) then
      if(.not. allocated(self%work)) then
        call self%get_worker_size(alloc_size=alloc_size)
        allocate(self%work(alloc_size))
      endif
    endif
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%r2r%execute(in)
      if(present(work)) then
        call self%execute_transposed_out(in, out, work)
      else
        call self%execute_transposed_out(in, out, self%work)
      endif
      ! 1d FFT Z direction
      call self%fplans(3)%r2r%execute(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Z direction
      call self%bplans(3)%r2r%execute(in)
      if(present(work)) then 
        call self%execute_transposed_in(in, out, work)
      else
        call self%execute_transposed_in(in, out, self%work)
      endif
      ! 1d FFT X direction
      call self%bplans(1)%r2r%execute(out)
    else
      error stop "dtfft_execute_r2r_3d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_3d(self, in, out, transpose_type, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2r plan, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type   !< Direction of transposition
    real(R4P),    optional,   intent(inout) :: work(*)          !< Working buffer
    integer(IP)                             :: alloc_size       !< Size of internal work buffer

    call self%check_plan(R4P)
    if(.not. present(work)) then
      if(.not. allocated(self%f_work)) then
        call self%get_worker_size(alloc_size=alloc_size)
        allocate(self%f_work(alloc_size))
      endif
    endif
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%r2r%execute_f(in)
      if(present(work)) then
        call self%execute_f_transposed_out(in, out, work)
      else
        call self%execute_f_transposed_out(in, out, self%f_work)
      endif
      ! 1d FFT Z direction
      call self%fplans(3)%r2r%execute_f(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Z direction
      call self%bplans(3)%r2r%execute_f(in)
      if(present(work)) then
        call self%execute_f_transposed_in(in, out, work)
      else
        call self%execute_f_transposed_in(in, out, self%f_work)
      endif
      ! 1d FFT X direction
      call self%bplans(1)%r2r%execute_f(out)
    else
      error stop "dtfft_execute_f_r2r: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_f_3d

!------------------------------------------------------------------------------------------------
  subroutine destroy_3d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 3d r2r transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout)  :: self             !< R2R 3D Plan

    if(allocated(self%work))    deallocate(self%work)
    if(allocated(self%f_work))  deallocate(self%f_work)
    call self%destroy_plan_r2r()
  end subroutine destroy_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_transposed_out(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_OUT algorithm, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    real(R8P),                intent(inout) :: work(*)          !< Working buffer

    ! Transpose X -> Y
    call self%transpose(self%transpose_out(1), in, work, 'X', 'Y')
    ! 1d FFT Y direction
    call self%fplans(2)%r2r%execute(work)
    ! Transpose Y -> Z
    call self%transpose(self%transpose_out(2), work, out, 'Y', 'Z')
  end subroutine execute_transposed_out

!------------------------------------------------------------------------------------------------
  subroutine execute_transposed_in(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_IN algorithm, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    real(R8P),                intent(inout) :: work(*)          !< Working buffer

    ! Transpose Z -> Y
    call self%transpose(self%transpose_in(2), in, work, 'Z', 'Y')
    ! 1d FFT Y direction
    call self%bplans(2)%r2r%execute(work)
    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), work, out, 'Y', 'X')
  end subroutine execute_transposed_in

!------------------------------------------------------------------------------------------------
  subroutine execute_f_transposed_out(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_OUT algorithm, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    real(R4P),                intent(inout) :: work(*)          !< Working buffer

    call self%transpose(self%transpose_out(1), in, work, 'X', 'Y')
    ! 1d FFT Y direction
    call self%fplans(2)%r2r%execute_f(work)
    ! Transpose Y -> Z
    call self%transpose(self%transpose_out(2), work, out, 'Y', 'Z')
  end subroutine execute_f_transposed_out

!------------------------------------------------------------------------------------------------
  subroutine execute_f_transposed_in(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_IN algorithm, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r_3d), intent(inout) :: self             !< R2R 3D Plan
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    real(R4P),                intent(inout) :: work(*)          !< Working buffer

    ! Transpose Z -> Y
    call self%transpose(self%transpose_in(2), in, work, 'Z', 'Y')
    ! 1d FFT Y direction
    call self%bplans(2)%r2r%execute_f(work)
    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), work, out, 'Y', 'X')
  end subroutine execute_f_transposed_in
end module dtfft_plan_r2r_m