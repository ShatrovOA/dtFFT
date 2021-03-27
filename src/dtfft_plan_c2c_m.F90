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
module dtfft_plan_c2c_m
!------------------------------------------------------------------------------------------------
!< This module describes [[dtfft_plan_c2c_2d]] and [[dtfft_plan_c2c_3d]] classes
!------------------------------------------------------------------------------------------------
use dtfft_abstract_executor_m
use dtfft_executor_fftw_m
use dtfft_executor_mkl_m
use dtfft_executor_cufft_m
use dtfft_parameters
use dtfft_plan_base_m
use dtfft_precisions
use mpi_f08
use iso_fortran_env, only: output_unit
implicit none
private
public :: dtfft_plan_c2c_2d, dtfft_plan_c2c_3d, dtfft_base_plan_c2c

  type :: c2c_executor
  !< Derived datatype to handle external 1d FFT executors
    class(abstract_c2c_executor), allocatable :: c2c                              !< Abstract executor
  end type c2c_executor

  type, extends(dtfft_base_plan) :: dtfft_base_plan_c2c
  !< Base plan for c2c transform
    integer(IP)                       :: plan_cnt                                 !< Number of FFT plans
    type(c2c_executor),   allocatable :: fplans(:)                                !< Plans for forward (DTFFT_TRANSPOSED_OUT) transform
    type(c2c_executor),   allocatable :: bplans(:)                                !< Plans for backward (DTFFT_TRANSPOSED_IN) transform
    complex(C8P),         allocatable :: work(:)                                  !< Worker buffer used to store intermediate results, double precision
    complex(C4P),         allocatable :: f_work(:)                                !< Worker buffer used to store intermediate results, single precision
  contains  
    procedure, pass(self)             :: create_fft_plans                         !< Creates 1d plans
    procedure, pass(self)             :: destroy_plan_c2c                         !< Destroys 1d plans and releases all internal memory
    procedure, pass(self)             :: execute_transposed_out                   !< Executes DTFFT_TRANSPOSED_OUT plan with worker buffer, double precision
    procedure, pass(self)             :: execute_transposed_in                    !< Executes DTFFT_TRANSPOSED_IN plan with worker buffer, double precision
    procedure, pass(self)             :: execute_f_transposed_out                 !< Executes DTFFT_TRANSPOSED_OUT plan with worker buffer, single precision
    procedure, pass(self)             :: execute_f_transposed_in                  !< Executes DTFFT_TRANSPOSED_IN plan with worker buffer, single precision
    procedure, pass(self)             :: check_and_alloc_work_buffer              !< Checks if user passed optional work buffer and allocates memory if necessary, double precision
    procedure, pass(self)             :: check_and_alloc_work_buffer_f            !< Checks if user passed optional work buffer and allocates memory if necessary, single precision
  end type

  type, extends(dtfft_base_plan_c2c) :: dtfft_plan_c2c_2d
  !< Plan for two-dimensional c2c transform
  private
  contains
  private
    procedure, pass(self),     public :: create           => create_c2c_2d        !< Create 2d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_c2c_2d      !< Create 2d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_2d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: execute          => execute_2d           !< Executes 2d plan, double precision
    procedure, pass(self),     public :: execute_f        => execute_f_2d         !< Executes 2d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_2d           !< Destroys 2d plan
  end type dtfft_plan_c2c_2d

  type, extends(dtfft_base_plan_c2c) :: dtfft_plan_c2c_3d
  !< Plan for three-dimensional c2c transform
  private
  contains
  private
    procedure, pass(self),     public :: create           => create_c2c_3d        !< Create 3d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_c2c_3d      !< Create 3d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_3d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: get_worker_size  => get_worker_size_3d   !< Returns local sizes, counts and allocation size of optional worker buffer
    procedure, pass(self),     public :: execute          => execute_3d           !< Executes 3d plan, double precision
    procedure, pass(self),     public :: execute_f        => execute_f_3d         !< Executes 3d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_3d           !< Destroys 3d plan
  end type dtfft_plan_c2c_3d

contains

!------------------------------------------------------------------------------------------------
  subroutine create_fft_plans(self, number_of_plans, info_inc, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates 1d plans with provided executor_type value 
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    integer(IP),                intent(in)    :: number_of_plans  !< Number of plans to create
    integer(IP),                intent(in)    :: info_inc         !< R2C plan needs to create c2c plan from 2 info
    integer(IP),  optional,     intent(in)    :: executor_type    !< Type of External FFT Executor
    integer(IP)                               :: executor         !< External FFT executor
    integer(IP)                               :: d                !< Counter

    self%plan_cnt = number_of_plans
    allocate(self%fplans(number_of_plans), self%bplans(number_of_plans))

    executor = DTFFT_EXECUTOR_FFTW3
    if(present(executor_type)) executor = executor_type
    do d = 1, number_of_plans
      select case(executor)
      case (DTFFT_EXECUTOR_FFTW3)
#ifndef NO_FFTW3
#ifdef __DEBUG
        if(self%comm_rank == 0 .and. d == 1) write(output_unit, '(a)') "DTFFT is using FFTW3 executor"
#endif
        allocate(fftw_c2c_executor :: self%fplans(d)%c2c)
        allocate(fftw_c2c_executor :: self%bplans(d)%c2c)
#else
        error stop "FFTW3 is disabled in this build"
#endif
      case (DTFFT_EXECUTOR_MKL)
#ifdef MKL_ENABLED
#ifdef __DEBUG
        if(self%comm_rank == 0 .and. d == 1) write(output_unit, '(a)') "DTFFT is using MKL executor"
#endif
        allocate(mkl_c2c_executor :: self%fplans(d)%c2c)
        allocate(mkl_c2c_executor :: self%bplans(d)%c2c)
#else
        error stop "MKL is disabled in this build"
#endif
      case (DTFFT_EXECUTOR_CUFFT)
#ifdef CUFFT_ENABLED
#ifdef __DEBUG
        if(self%comm_rank == 0 .and. d == 1) write(output_unit, '(a)') "DTFFT is using CUFFT executor"
#endif
        allocate(cufft_c2c_executor :: self%fplans(d)%c2c)
        allocate(cufft_c2c_executor :: self%bplans(d)%c2c)
#else
        error stop "CUFFT is disabled in this build"
#endif
      case default
        error stop "DTFFT: Unrecognized executor"
      endselect

      call self%fplans(d)%c2c%create_plan(self%info(d + info_inc), DTFFT_FORWARD, self%precision)
      call self%bplans(d)%c2c%create_plan(self%info(d + info_inc), DTFFT_BACKWARD, self%precision)
    enddo
  end subroutine create_fft_plans

!------------------------------------------------------------------------------------------------
  subroutine destroy_plan_c2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys 1d plans and releases all internal memory
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    integer(IP)                               :: d                !< Counter

    do d = 1, self%plan_cnt
      call self%fplans(d)%c2c%destroy()
      call self%bplans(d)%c2c%destroy()
    enddo
    deallocate(self%fplans)
    deallocate(self%bplans)
    if(allocated(self%work))    deallocate(self%work)
    if(allocated(self%f_work))  deallocate(self%f_work)
    call self%destroy_base_plan()
  end subroutine destroy_plan_c2c

!------------------------------------------------------------------------------------------------
  subroutine check_and_alloc_work_buffer(self, work_id, work)
!------------------------------------------------------------------------------------------------
!< Checks if user passed optional work buffer and allocates memory if necessary, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    integer(IP),                intent(in)    :: work_id          !< id of aligned data
    complex(C8P), optional,     intent(in)    :: work(*)          !< Work buffer
    integer(IP)                               :: alloc_size       !< Number of elements to allocate

    if(.not. present(work)) then
      if(.not. allocated(self%work)) then
        call self%get_worker_size_internal(work_id, alloc_size=alloc_size)
        allocate(self%work(alloc_size))
      endif
    endif
  end subroutine check_and_alloc_work_buffer

!------------------------------------------------------------------------------------------------
  subroutine check_and_alloc_work_buffer_f(self, work_id, work)
!------------------------------------------------------------------------------------------------
!< Checks if user passed optional work buffer and allocates memory if necessary, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    integer(IP),                intent(in)    :: work_id          !< id of aligned data
    complex(C4P), optional,     intent(in)    :: work(*)          !< Work buffer
    integer(IP)                               :: alloc_size       !< Number of elements to allocate

    if(.not. present(work)) then
      if(.not. allocated(self%f_work)) then
        call self%get_worker_size_internal(work_id, alloc_size=alloc_size)
        allocate(self%f_work(alloc_size))
      endif
    endif
  end subroutine check_and_alloc_work_buffer_f

!------------------------------------------------------------------------------------------------
  subroutine execute_transposed_out(self, in, out, work, plan_id)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_OUT algorithm, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    complex(C8P),               intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),               intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P),               intent(inout) :: work(*)          !< Working buffer
    integer(IP),                intent(in)    :: plan_id          !< Id of C2C FFT plan

    ! Transpose X -> Y
    call self%transpose(self%transpose_out(1), in, work, 'X', 'Y')
    ! 1d direct FFT Y direction
    call self%fplans(plan_id)%c2c%execute(work)
    ! Transpose Y -> Z
    call self%transpose(self%transpose_out(2), work, out, 'Y', 'Z')
  end subroutine execute_transposed_out

!------------------------------------------------------------------------------------------------
  subroutine execute_transposed_in(self, in, out, work, plan_id)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_IN algorithm, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    complex(C8P),               intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),               intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P),               intent(inout) :: work(*)          !< Working buffer
    integer(IP),                intent(in)    :: plan_id          !< Id of C2C FFT plan

    ! Transpose Z -> Y
    call self%transpose(self%transpose_in(2), in, work, 'Z', 'Y')
    ! 1d inverse FFT Y direction
    call self%bplans(plan_id)%c2c%execute(work)
    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), work, out, 'Y', 'X')
  end subroutine execute_transposed_in

!------------------------------------------------------------------------------------------------
  subroutine execute_f_transposed_out(self, in, out, work, plan_id)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_OUT algorithm, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    complex(C4P),               intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),               intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P),               intent(inout) :: work(*)          !< Working buffer
    integer(IP),                intent(in)    :: plan_id          !< Id of C2C FFT plan

    ! Transpose X -> Y
    call self%transpose(self%transpose_out(1), in, work, 'X', 'Y')
    ! 1d direct FFT Y direction
    call self%fplans(plan_id)%c2c%execute_f(work)
    ! Transpose Y -> Z
    call self%transpose(self%transpose_out(2), work, out, 'Y', 'Z')
  end subroutine execute_f_transposed_out

!------------------------------------------------------------------------------------------------
  subroutine execute_f_transposed_in(self, in, out, work, plan_id)
!------------------------------------------------------------------------------------------------
!< Executes DTFFT_TRANSPOSED_IN algorithm, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_c2c), intent(inout) :: self             !< Base C2C class
    complex(C4P),               intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),               intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P),               intent(inout) :: work(*)          !< Working buffer
    integer(IP),                intent(in)    :: plan_id          !< Id of C2C FFT plan

    ! Transpose Z -> Y
    call self%transpose(self%transpose_in(2), in, work, 'Z', 'Y')
    ! 1d inverse FFT Y direction
    call self%bplans(plan_id)%c2c%execute_f(work)
    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), work, out, 'Y', 'X')
  end subroutine execute_f_transposed_in

!------------------------------------------------------------------------------------------------
  subroutine create_c2c_2d(self, comm, nx, ny, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class
    type(MPI_Comm),           intent(in)    :: comm               !< Communicator
    integer(IP),              intent(in)    :: nx                 !< Number of points in X direction
    integer(IP),              intent(in)    :: ny                 !< Number of points in Y direction
    integer(IP),  optional,   intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type      !< Type of External FFT Executor

    call self%init_base_plan(comm, C8P, [nx, ny], MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(self%dims, 0, executor_type)
    self%is_created = .true.
  end subroutine create_c2c_2d

!------------------------------------------------------------------------------------------------
  subroutine create_f_c2c_2d(self, comm, nx, ny, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional c2c transform, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class
    type(MPI_Comm),           intent(in)    :: comm               !< Communicator
    integer(IP),              intent(in)    :: nx                 !< Number of points in X direction
    integer(IP),              intent(in)    :: ny                 !< Number of points in Y direction
    integer(IP),  optional,   intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type      !< Type of External FFT Executor

    call self%init_base_plan(comm, C4P, [nx, ny], MPI_COMPLEX, COMPLEX_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(self%dims, 0, executor_type)
    self%is_created = .true.
  end subroutine create_f_c2c_2d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_2d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation sizes
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class
    integer(IP),  optional,   intent(out)   :: in_starts(:)       !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: in_counts(:)       !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: out_starts(:)      !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: out_counts(:)      !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: alloc_size         !< Maximum number of elements needs to be allocated

    call self%get_local_sizes_internal(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_2d(self, in, out, transpose_type)
!------------------------------------------------------------------------------------------------
!< Executes 2d c2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class
    complex(C8P),             intent(inout) :: in(*)              !< Incoming buffer
    complex(C8P),             intent(inout) :: out(*)             !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type     !< Direction of transposition

    call self%check_plan(C8P)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%c2c%execute(in)
      ! Transpose X -> Y
      call self%transpose(self%transpose_out(1), in, out, 'X', 'Y')
      ! 1d FFT Y direction
      call self%fplans(2)%c2c%execute(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Y direction
      call self%bplans(2)%c2c%execute(in)
      ! Transpose Y -> X
      call self%transpose(self%transpose_in(1), in, out, 'Y', 'X')
      ! 1d FFT X direction
      call self%bplans(1)%c2c%execute(out)
    else
      error stop "dtfft_execute_c2c_2d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_2d(self, in, out, transpose_type)
!------------------------------------------------------------------------------------------------
!< Executes 2d c2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class
    complex(C4P),             intent(inout) :: in(*)              !< Incoming buffer
    complex(C4P),             intent(inout) :: out(*)             !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type     !< Direction of transposition

    call self%check_plan(C4P)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%c2c%execute_f(in)
      ! Transpose X -> Y
      call self%transpose(self%transpose_out(1), in, out, 'X', 'Y')
      ! 1d FFT Y direction
      call self%fplans(2)%c2c%execute_f(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Y direction
      call self%bplans(2)%c2c%execute_f(in)
      ! Transpose Y -> X
      call self%transpose(self%transpose_in(1), in, out, 'Y', 'X')
      ! 1d FFT X direction
      call self%bplans(1)%c2c%execute_f(out)
    else
      error stop "dtfft_execute_f_c2c_2d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_f_2d

!------------------------------------------------------------------------------------------------
  subroutine destroy_2d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 2d c2c transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_2d), intent(inout) :: self               !< C2C 2D Class

    call self%destroy_plan_c2c()
  end subroutine destroy_2d

!------------------------------------------------------------------------------------------------
  subroutine create_c2c_3d(self, comm, nx, ny, nz, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    type(MPI_Comm),           intent(in)    :: comm               !< Communicator
    integer(IP),              intent(in)    :: nx                 !< Number of points in X direction
    integer(IP),              intent(in)    :: ny                 !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz                 !< Number of points in Z direction
    integer(IP),  optional,   intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type      !< Type of External FFT Executor

    call self%init_base_plan(comm, C8P, [nx, ny, nz], MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(self%dims, 0, executor_type)
    self%is_created = .true.
  end subroutine create_c2c_3d

!------------------------------------------------------------------------------------------------
  subroutine create_f_c2c_3d(self, comm, nx, ny, nz, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    type(MPI_Comm),           intent(in)    :: comm               !< Communicator
    integer(IP),              intent(in)    :: nx                 !< Number of points in X direction
    integer(IP),              intent(in)    :: ny                 !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz                 !< Number of points in Z direction
    integer(IP),  optional,   intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type      !< Type of External FFT Executor

    call self%init_base_plan(comm, C4P, [nx, ny, nz], MPI_COMPLEX, COMPLEX_STORAGE_SIZE, effort_flag)
    call self%create_fft_plans(self%dims, 0, executor_type)
    self%is_created = .true.
  end subroutine create_f_c2c_3d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_3d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    integer(IP),  optional,   intent(out)   :: in_starts(:)       !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: in_counts(:)       !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: out_starts(:)      !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: out_counts(:)      !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: alloc_size         !< Maximum number of elements needs to be allocated

    call self%get_local_sizes_internal(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_3d

!------------------------------------------------------------------------------------------------
  subroutine get_worker_size_3d(self, starts, counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size of optional worker buffer
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    integer(IP),  optional,   intent(out)   :: starts(:)          !< Starts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: counts(:)          !< Counts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: alloc_size         !< Number of elements to be allocated

    call self%get_worker_size_internal(Y_PENCIL, starts, counts, alloc_size)
  end subroutine get_worker_size_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_3d(self, in, out, transpose_type, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d c2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    complex(C8P),             intent(inout) :: in(*)              !< Incoming buffer
    complex(C8P),             intent(inout) :: out(*)             !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type     !< Direction of transposition
    complex(C8P), optional,   intent(inout) :: work(*)            !< Working buffer

    call self%check_plan(C8P)
    call self%check_and_alloc_work_buffer(Y_PENCIL, work)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%c2c%execute(in)
      if(present(work)) then
        call self%execute_transposed_out(in, out, work, 2)
      else
        call self%execute_transposed_out(in, out, self%work, 2)
      endif
      ! 1d FFT Z direction
      call self%fplans(3)%c2c%execute(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Z direction
      call self%bplans(3)%c2c%execute(in)
      if(present(work)) then 
        call self%execute_transposed_in(in, out, work, 2)
      else
        call self%execute_transposed_in(in, out, self%work, 2)
      endif
      ! 1d FFT X direction
      call self%bplans(1)%c2c%execute(out)
    else
      error stop "dtfft_execute_c2c_3d: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_3d(self, in, out, transpose_type, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d c2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout) :: self               !< C2C 3D Class
    complex(C4P),             intent(inout) :: in(*)              !< Incoming buffer
    complex(C4P),             intent(inout) :: out(*)             !< Outgoing buffer
    integer(IP),              intent(in)    :: transpose_type     !< Direction of transposition
    complex(C4P), optional,   intent(inout) :: work(*)            !< Working buffer

    call self%check_plan(C4P)
    call self%check_and_alloc_work_buffer_f(Y_PENCIL, work)
    if(transpose_type == DTFFT_TRANSPOSE_OUT) then 
      ! 1d FFT X direction
      call self%fplans(1)%c2c%execute_f(in)
      if(present(work)) then
        call self%execute_f_transposed_out(in, out, work, 2)
      else
        call self%execute_f_transposed_out(in, out, self%f_work, 2)
      endif
      ! 1d FFT Z direction
      call self%fplans(3)%c2c%execute_f(out)
    elseif(transpose_type == DTFFT_TRANSPOSE_IN) then 
      ! 1d FFT Z direction
      call self%bplans(3)%c2c%execute_f(in)
      if(present(work)) then
        call self%execute_f_transposed_in(in, out, work, 2)
      else
        call self%execute_f_transposed_in(in, out, self%f_work, 2)
      endif
      ! 1d FFT X direction
      call self%bplans(1)%c2c%execute_f(out)
    else
      error stop "dtfft_execute_f_c2c: Unknown 'transpose_type' parameter"
    endif
  end subroutine execute_f_3d

!------------------------------------------------------------------------------------------------
  subroutine destroy_3d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 3d c2c transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c_3d), intent(inout)  :: self              !< C2C 3D Class

    call self%destroy_plan_c2c()
  end subroutine destroy_3d
end module dtfft_plan_c2c_m