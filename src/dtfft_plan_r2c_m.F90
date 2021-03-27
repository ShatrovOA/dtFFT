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
module dtfft_plan_r2c_m
!------------------------------------------------------------------------------------------------
!< This module describes [[dtfft_plan_r2c_2d]] and [[dtfft_plan_r2c_3d]] classes
!------------------------------------------------------------------------------------------------
use dtfft_abstract_executor_m
use dtfft_executor_cufft_m
use dtfft_executor_fftw_m
use dtfft_executor_mkl_m
use dtfft_info_m
use dtfft_parameters
use dtfft_plan_base_m
use dtfft_plan_c2c_m, only: dtfft_base_plan_c2c
use dtfft_precisions
use mpi_f08
use iso_fortran_env, only: output_unit
implicit none
private
public :: dtfft_plan_r2c_2d,    &
          dtfft_plan_r2c_3d

  type, extends(dtfft_base_plan_c2c)  :: dtfft_base_plan_r2c
  !< Base plan for R2C class
    class(abstract_r2c_executor), allocatable :: r2c          !< R2C FFT
    class(abstract_c2r_executor), allocatable :: c2r          !< C2R FFT
    type(info_t)                              :: real_info    !< Real buffer info
  contains
    procedure, pass(self)             :: r2c_init             !< Creates 'real' info and creates R2C and C2R FFT plans
    procedure, pass(self)             :: r2c_get_local_sizes  !< Returns local sizes, counts and allocation size
    procedure, pass(self)             :: r2c_destroy          !< Destroys plan r2c transform 
  end type dtfft_base_plan_r2c

  type, extends(dtfft_base_plan_r2c)  :: dtfft_plan_r2c_2d
  !< Plan for two-dimensional r2c transform
  private
  contains
  private
    procedure, pass(self),     public :: create           => create_r2c_2d        !< Create 2d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_r2c_2d      !< Create 2d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_2d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: get_worker_size  => get_worker_size_2d   !< Returns local sizes, counts and allocation size of optional worker buffer
    procedure, pass(self),     public :: execute_r2c      => execute_r2c_2d       !< Executes forward 2d plan, double precision
    procedure, pass(self),     public :: execute_c2r      => execute_c2r_2d       !< Executes backward 2d plan, double precision
    procedure, pass(self),     public :: execute_f_r2c    => execute_f_r2c_2d     !< Executes forward 2d plan, single precision
    procedure, pass(self),     public :: execute_f_c2r    => execute_f_c2r_2d     !< Executes backward 2d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_2d           !< Destroys 2d plan
    procedure, pass(self)             :: execute_r2c_transposed                   !< Executes R2C FFT and transposes data to Y pencils, double precision
    procedure, pass(self)             :: execute_f_r2c_transposed                 !< Executes R2C FFT and transposes data to Y pencils, single precision
    procedure, pass(self)             :: execute_c2r_transposed                   !< Transposes data back to X pencils and executes C2R FFT, double precision
    procedure, pass(self)             :: execute_f_c2r_transposed                 !< Transposes data back to X pencils and executes C2R FFT, single precision
  end type dtfft_plan_r2c_2d

  type, extends(dtfft_base_plan_r2c) :: dtfft_plan_r2c_3d
  !< Plan for three-dimensional r2c transform
  private
  contains
  private
    procedure, pass(self),     public :: create           => create_r2c_3d        !< Create 3d plan, double precision
    procedure, pass(self),     public :: create_f         => create_f_r2c_3d      !< Create 3d plan, single precision
    procedure, pass(self),     public :: get_local_sizes  => get_local_sizes_3d   !< Returns local sizes, counts and allocation sizes
    procedure, pass(self),     public :: get_worker_size  => get_worker_size_3d   !< Returns local sizes, counts and allocation size of optional worker buffer
    procedure, pass(self),     public :: execute_r2c      => execute_r2c_3d       !< Executes forward 3d plan, double precision
    procedure, pass(self),     public :: execute_c2r      => execute_c2r_3d       !< Executes backward 3d plan, double precision
    procedure, pass(self),     public :: execute_f_r2c    => execute_f_r2c_3d     !< Executes forward 3d plan, single precision
    procedure, pass(self),     public :: execute_f_c2r    => execute_f_c2r_3d     !< Executes backward 3d plan, single precision
    procedure, pass(self),     public :: destroy          => destroy_3d           !< Destroys 3d plan
  end type dtfft_plan_r2c_3d

contains

!------------------------------------------------------------------------------------------------
  subroutine r2c_init(self, counts, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates 'real' info and creates R2C and C2R FFT plans
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_r2c), intent(inout) :: self           !< Base R2C Class
    integer(IP),                intent(in)    :: counts(:)      !< Global 'real' counts
    integer(IP),  optional,     intent(in)    :: executor_type  !< Type of External FFT Executor
    integer(IP)                               :: executor       !< Type of External FFT Executor

    call self%real_info%init(self%dims, 1, counts, self%comms, self%comm_dims, self%comm_coords)

    executor = DTFFT_EXECUTOR_FFTW3
    if(present(executor_type)) executor = executor_type
    select case(executor)
    case (DTFFT_EXECUTOR_FFTW3)
#ifndef NO_FFTW3
      allocate(fftw_r2c_executor :: self%r2c)
      allocate(fftw_c2r_executor :: self%c2r)
#else
      error stop "FFTW3 is disabled in this build"
#endif
    case (DTFFT_EXECUTOR_MKL)
#ifdef MKL_ENABLED
      allocate(mkl_r2c_executor :: self%r2c)
      allocate(mkl_c2r_executor :: self%c2r)
#else
      error stop "MKL is disabled in this build"
#endif
    case (DTFFT_EXECUTOR_CUFFT)
#ifdef CUFFT_ENABLED
      allocate(cufft_r2c_executor :: self%r2c)
      allocate(cufft_c2r_executor :: self%c2r)
#else
      error stop "CUFFT is disabled in this build"
#endif
    case default
      error stop "DTFFT: Unrecognized executor"
    endselect

    call self%r2c%create_plan(self%real_info, self%info(1), self%precision)
    call self%c2r%create_plan(self%info(1), self%real_info, self%precision)
  end subroutine r2c_init

!------------------------------------------------------------------------------------------------
  subroutine r2c_get_local_sizes(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< This subroutine differs from all other 'get_local_sizes' methods. Returns local sizes, counts and allocation size
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_r2c), intent(in)  :: self             !< Base R2C Class
    integer(IP),  optional,     intent(out) :: in_starts(:)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,     intent(out) :: in_counts(:)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,     intent(out) :: out_starts(:)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,     intent(out) :: out_counts(:)    !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,     intent(out) :: alloc_size       !< Maximum number of elements to be allocated
    integer(IP)                             :: alloc_real       !< Number of elements needed by real buffer

    if(self%is_created) then 
      if(present(in_starts))    in_starts   = self%real_info%starts
      if(present(in_counts))    in_counts   = self%real_info%counts
      alloc_real = product(self%real_info%counts)
      call self%get_local_sizes_internal(out_starts=out_starts, out_counts=out_counts, alloc_size=alloc_size)
      if(present(alloc_size))    alloc_size = max(alloc_real, alloc_size)
    else
      error stop 'DTFFT: error in "get_local_sizes", plan has not been created'
    endif
  end subroutine r2c_get_local_sizes

!------------------------------------------------------------------------------------------------
  subroutine r2c_destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan r2c transform 
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan_r2c), intent(inout) :: self             !< Base R2C Class

    call self%r2c%destroy()
    call self%c2r%destroy()
    call self%real_info%destroy()
  end subroutine r2c_destroy

!------------------------------------------------------------------------------------------------
  subroutine create_r2c_2d(self, comm, nx, ny, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional r2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    type(MPI_Comm),           intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, C8P, [nx / 2 + 1, ny], MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, effort_flag)
    call self%r2c_init([nx, ny], executor_type)
    call self%create_fft_plans(self%dims - 1, 1, executor_type)
    self%is_created = .true.
  end subroutine create_r2c_2d

!------------------------------------------------------------------------------------------------
  subroutine create_f_r2c_2d(self, comm, nx, ny, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for two-dimensional r2c transform, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    type(MPI_Comm),           intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, C4P, [nx / 2 + 1, ny], MPI_COMPLEX, COMPLEX_STORAGE_SIZE, effort_flag)
    call self%r2c_init([nx, ny], executor_type)
    call self%create_fft_plans(self%dims - 1, 1, executor_type)
    self%is_created = .true.
  end subroutine create_f_r2c_2d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_2d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation sizes
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(in)  :: self             !< R2C 2D Class
    integer(IP),  optional,   intent(out) :: in_starts(:)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(:)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(:)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(:)    !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: alloc_size       !< Maximum number of elements to be allocated in case of in-place transform

    call self%r2c_get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_2d

  subroutine get_worker_size_2d(self, starts, counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size of optional worker buffer
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self           !< R2C 2D Class
    integer(IP),  optional,   intent(out)   :: starts(:)      !< Starts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: counts(:)      !< Counts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: alloc_size     !< Number of elements to be allocated

    call self%get_worker_size_internal(X_PENCIL, starts, counts, alloc_size)
  end subroutine get_worker_size_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c_2d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 2d r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),             intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P),  optional,  intent(inout) :: work(*)          !< Work buffer

    call self%check_plan(C8P)
    call self%check_and_alloc_work_buffer(X_PENCIL, work)
    if(present(work)) then
      call self%execute_r2c_transposed(in, work, out)
    else
      call self%execute_r2c_transposed(in, self%work, out)
    endif
    ! 1d c2c FFT
    call self%fplans(1)%c2c%execute(out)
  end subroutine execute_r2c_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r_2d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 2d c2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    complex(C8P),             intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P),  optional,  intent(inout) :: work(*)          !< Work buffer

    call self%check_plan(C8P)
    ! 1d c2c FFT
    call self%bplans(1)%c2c%execute(in)

    call self%check_and_alloc_work_buffer(X_PENCIL, work)
    if(present(work)) then
      call self%execute_c2r_transposed(in, work, out)
    else
      call self%execute_c2r_transposed(in, self%work, out)
    endif
  end subroutine execute_c2r_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c_2d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 2d r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),             intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P),  optional,  intent(inout) :: work(*)          !< Work buffer

    call self%check_plan(C4P)
    call self%check_and_alloc_work_buffer_f(X_PENCIL, work)
    if(present(work)) then
      call self%execute_f_r2c_transposed(in, work, out) 
    else
      call self%execute_f_r2c_transposed(in, self%f_work, out)
    endif
    ! 1d c2c FFT
    call self%fplans(1)%c2c%execute_f(out)
  end subroutine execute_f_r2c_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r_2d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 2d c2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    complex(C4P),             intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P),  optional,  intent(inout) :: work(*)          !< Work buffer

    call self%check_plan(C4P)
    ! 1d c2c FFT
    call self%bplans(1)%c2c%execute_f(in)

    call self%check_and_alloc_work_buffer_f(X_PENCIL, work)
    if(present(work)) then
      call self%execute_f_c2r_transposed(in, work, out) 
    else
      call self%execute_f_c2r_transposed(in, self%f_work, out)
    endif
  end subroutine execute_f_c2r_2d

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c_transposed(self, in, work, out)
!------------------------------------------------------------------------------------------------
!< Executes R2C FFT and transposes data to Y pencils, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),             intent(inout) :: work(*)          !< Work buffer
    complex(C8P),             intent(inout) :: out(*)           !< Outgoing buffer

    ! 1d r2c FFT
    call self%r2c%execute(in, work)
    ! Transpose X -> Y
    call self%transpose(self%transpose_out(1), work, out, 'X', 'Y')
  end subroutine execute_r2c_transposed

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r_transposed(self, in, work, out)
!------------------------------------------------------------------------------------------------
!< Transposes data back to X pencils and executes C2R FFT, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    complex(C8P),             intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),             intent(inout) :: work(*)          !< Work buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer

    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), in, work, 'Y', 'X')
    ! 1d c2r FFT
    call self%c2r%execute(work, out)
  end subroutine execute_c2r_transposed

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c_transposed(self, in, work, out)
!------------------------------------------------------------------------------------------------
!< Executes R2C FFT and transposes data to Y pencils, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),             intent(inout) :: work(*)          !< Work buffer
    complex(C4P),             intent(inout) :: out(*)           !< Outgoing buffer

    ! 1d r2c FFT
    call self%r2c%execute_f(in, work)
    ! Transpose X -> Y
    call self%transpose(self%transpose_out(1), work, out, 'X', 'Y')
  end subroutine execute_f_r2c_transposed

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r_transposed(self, in, work, out)
!------------------------------------------------------------------------------------------------
!< Transposes data back to X pencils and executes C2R FFT, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class
    complex(C4P),             intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),             intent(inout) :: work(*)          !< Work buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer

    ! Transpose Y -> X
    call self%transpose(self%transpose_in(1), in, work, 'Y', 'X')
    ! 1d c2r FFT
    call self%c2r%execute_f(work, out)
  end subroutine execute_f_c2r_transposed

!------------------------------------------------------------------------------------------------
  subroutine destroy_2d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 2d c2c transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_2d), intent(inout) :: self             !< R2C 2D Class

    call self%r2c_destroy()
    call self%destroy_plan_c2c()
  end subroutine destroy_2d

!------------------------------------------------------------------------------------------------
  subroutine create_r2c_3d(self, comm, nx, ny, nz, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    type(MPI_Comm),           intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz               !< Number of points in Z direction
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, C8P, [nx / 2 + 1, ny, nz], MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, effort_flag)
    call self%r2c_init([nx, ny, nz], executor_type)
    call self%create_fft_plans(self%dims - 1, 1, executor_type)
    self%is_created = .true.
  end subroutine create_r2c_3d

!------------------------------------------------------------------------------------------------
  subroutine create_f_r2c_3d(self, comm, nx, ny, nz, effort_flag, executor_type)
!------------------------------------------------------------------------------------------------
!< Creates plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    type(MPI_Comm),           intent(in)    :: comm             !< Communicator
    integer(IP),              intent(in)    :: nx               !< Number of points in X direction
    integer(IP),              intent(in)    :: ny               !< Number of points in Y direction
    integer(IP),              intent(in)    :: nz               !< Number of points in Z direction
    integer(IP),  optional,   intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  optional,   intent(in)    :: executor_type    !< Type of External FFT Executor

    call self%init_base_plan(comm, C4P, [nx / 2 + 1, ny, nz], MPI_COMPLEX, COMPLEX_STORAGE_SIZE, effort_flag)
    call self%r2c_init([nx, ny, nz], executor_type)
    call self%create_fft_plans(self%dims - 1, 1, executor_type)
    self%is_created = .true.
  end subroutine create_f_r2c_3d

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_3d(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    integer(IP),  optional,   intent(out)   :: in_starts(:)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: in_counts(:)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out)   :: out_starts(:)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: out_counts(:)    !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out)   :: alloc_size       !< Maximum number of elements to be allocated in case of in-place transform

    call self%r2c_get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine get_local_sizes_3d

!------------------------------------------------------------------------------------------------
  subroutine get_worker_size_3d(self, starts, counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size of optional worker buffer
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self           !< R2C 3D Class
    integer(IP),  optional,   intent(out)   :: starts(:)      !< Starts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: counts(:)      !< Counts of local portion of Y pencil
    integer(IP),  optional,   intent(out)   :: alloc_size     !< Number of elements to be allocated

    call self%get_worker_size_internal(Y_PENCIL, starts, counts, alloc_size)
  end subroutine get_worker_size_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c_3d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    real(R8P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C8P),             intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P), optional,   intent(inout) :: work(*)          !< Working buffer

    call self%check_plan(C8P)
    ! 1d r2c FFT
    call self%r2c%execute(in, out)

    call self%check_and_alloc_work_buffer(Y_PENCIL, work)
    if(present(work)) then 
      call self%execute_transposed_out(out, out, work, 1)
    else
      call self%execute_transposed_out(out, out, self%work, 1)
    endif
    ! 1d c2c FFT
    call self%fplans(2)%c2c%execute(out)
  end subroutine execute_r2c_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r_3d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    complex(C8P),             intent(inout) :: in(*)            !< Incoming buffer
    real(R8P),                intent(inout) :: out(*)           !< Outgoing buffer
    complex(C8P), optional,   intent(inout) :: work(*)          !< Working buffer

    call self%check_plan(C8P)
    ! 1d c2c FFT
    call self%bplans(2)%c2c%execute(in)
    call self%check_and_alloc_work_buffer(Y_PENCIL, work)
    if(present(work)) then 
      call self%execute_transposed_in(in, in, work, 1)
    else
      call self%execute_transposed_in(in, in, self%work, 1)
    endif
    ! 1d r2c FFT
    call self%c2r%execute(in, out)
  end subroutine execute_c2r_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c_3d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    real(R4P),                intent(inout) :: in(*)            !< Incoming buffer
    complex(C4P),             intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P), optional,   intent(inout) :: work(*)          !< Working buffer

    call self%check_plan(C4P)
    ! 1d r2c FFT
    call self%r2c%execute_f(in, out)
    call self%check_and_alloc_work_buffer_f(Y_PENCIL, work)
    if(present(work)) then 
      call self%execute_f_transposed_out(out, out, work, 1)
    else
      call self%execute_f_transposed_out(out, out, self%f_work, 1)
    endif
    ! 1d c2c FFT
    call self%fplans(2)%c2c%execute_f(out)
  end subroutine execute_f_r2c_3d

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r_3d(self, in, out, work)
!------------------------------------------------------------------------------------------------
!< Executes 3d r2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout) :: self             !< R2C 3D Class
    complex(C4P),             intent(inout) :: in(*)            !< Incoming buffer
    real(R4P),                intent(inout) :: out(*)           !< Outgoing buffer
    complex(C4P), optional,   intent(inout) :: work(*)          !< Working buffer

    call self%check_plan(C4P)
    ! 1d inverse c2c FFT
    call self%bplans(2)%c2c%execute_f(in)
    call self%check_and_alloc_work_buffer_f(Y_PENCIL, work)
    if(present(work)) then 
      call self%execute_f_transposed_in(in, in, work, 1)
    else
      call self%execute_f_transposed_in(in, in, self%f_work, 1)
    endif
    ! 1d r2c FFT
    call self%c2r%execute_f(in, out)
  end subroutine execute_f_c2r_3d

!------------------------------------------------------------------------------------------------
  subroutine destroy_3d(self)
!------------------------------------------------------------------------------------------------
!< Destroys plan for 3d c2c transform
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c_3d), intent(inout)  :: self             !< R2C 3D Class

    call self%r2c_destroy()
    call self%destroy_plan_c2c()
  end subroutine destroy_3d
end module dtfft_plan_r2c_m