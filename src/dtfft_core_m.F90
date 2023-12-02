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
module dtfft_core_m
!------------------------------------------------------------------------------------------------
!< This module describes [[dtfft_core]] class
!------------------------------------------------------------------------------------------------
use dtfft_info_m
use dtfft_parameters
use dtfft_precisions
use dtfft_transpose_m
use dtfft_abstract_executor_m
#include "dtfft.i90"
implicit none
private
public :: dtfft_core, dtfft_plan_c2c, dtfft_plan_r2c, dtfft_plan_r2r, DTFFT_FATAL_ERROR

  type :: fft_executor
    class(abstract_executor), allocatable :: fft
  end type fft_executor

  type, abstract :: dtfft_core
  private
  !< Base plan for all DTFFT plans
    TYPE_MPI_COMM                     :: comm                       !< Grid communicator
    TYPE_MPI_COMM,      allocatable   :: comms(:)                   !< Local 1d communicators
    integer(IP),        allocatable   :: comm_dims(:)               !< Dimensions of grid comm
    integer(IP),        allocatable   :: comm_coords(:)             !< Coordinates of grod comm
    integer(IP)                       :: ndims                      !< Number of global dimensions
    integer(IP)                       :: precision                  !< Precision of transform
    integer(IP)                       :: comm_size                  !< Size of comm
    type(transpose_t),  allocatable   :: transpose_out(:)           !< Classes that perform TRANSPOSED_OUT transposes: XYZ --> YXZ --> ZXY
    type(transpose_t),  allocatable   :: transpose_in(:)            !< Classes that perform TRANSPOSED_IN transposes: ZXY --> YXZ --> XYZ
    type(info_t),       allocatable   :: info(:)                    !< Information about data aligment and datatypes
    logical                           :: is_created = .false.       !< Plan creation flag
    real(R4P),          allocatable   :: aux(:)                     !< Auxilary buffer that can be used in some transforms
    type(fft_executor), allocatable   :: forw_plans(:)              !< Internal fft runner, forward plan
    type(fft_executor), allocatable   :: back_plans(:)              !< Internal fft runner, backward plan
    logical :: is_transpose_plan, is_aux_needed = .false.           !< 
    logical :: is_aux_alloc = .false.
    integer(IP)                       :: effort_flag, storage_size, aux_pencil, executor_type
  contains
  private
    procedure,  pass(self), non_overridable, public :: transpose
    procedure,  pass(self), non_overridable, public :: execute
    procedure,  pass(self),                  public :: destroy
    procedure,  pass(self),                  public :: get_local_sizes
    procedure,  pass(self), non_overridable, public :: get_aux_size
    procedure,  pass(self), non_overridable         :: destroy_internal           !< Destroys core
    procedure,  pass(self), non_overridable         :: get_local_sizes_internal
    procedure,  pass(self), non_overridable         :: check_init_args
    procedure,  pass(self), non_overridable         :: create_transpose_plans
    procedure,  pass(self), non_overridable         :: init => init_core
    procedure,  pass(self), non_overridable         :: transpose_private
    procedure,  pass(self)                          :: execute_2d                 !< Executes 2d plan, single or double
    procedure,  pass(self)                          :: execute_3d                 !< Executes 3d plan, single or double
    procedure,  pass(self)                          :: create_cart_comm           !< Creates cartesian communicator
    procedure,  pass(self)                          :: base_alloc                 !< Alocates nessesary memory
    procedure,  pass(self)                          :: execute_transposed_out     !< 
    procedure,  pass(self)                          :: execute_transposed_in
    procedure,  pass(self)                          :: alloc_fft_plans            !< Allocates `fft_executor` classes
    procedure,  pass(self)                          :: check_aux                  !< 
    procedure,  pass(self)                          :: create_c2c_fft
    ! procedure,  pass(self),   public  :: get_local_sizes_internal   !< Returns local sizes, starts and number of elements to be allocated
    ! procedure,  pass(self),   public  :: get_worker_size_internal   !< Returns local sizes, starts and number of elements to be allocated for the optional worker buffer
    ! procedure,  pass(self),   public  :: check_plan                 !< Checks if plan is created with proper precision
    ! procedure,  pass(self),   public  :: destroy_base_plan          !< Destroys base plan
    ! procedure,  pass(self)            :: create_transpose_plans     !< Creates transposition plans
    ! procedure,  pass(self)            :: create_cart_comm           !< Creates cartesian communicator
    ! procedure,  pass(self)            :: base_alloc                 !< Alocates classes needed by base plan
  end type dtfft_core

  interface
    module subroutine transpose(self, in, out, transpose_type)
    !! Performs single transposition
    !!
    !! Note, that `in` and `out` cannot be the same, otherwise call to MPI will fail
      class(dtfft_core),  intent(inout) :: self           !< Core
      type(*),            intent(in)    :: in(..)         !< Incoming buffer of any rank and kind
      type(*),            intent(inout) :: out(..)        !< Resulting buffer of any rank and kind
      integer(IP),        intent(in)    :: transpose_type !< Type of transposition. One of the:
                                                          !< - DTFFT_TRANSPOSE_X_TO_Y
                                                          !< - DTFFT_TRANSPOSE_Y_TO_X
                                                          !< - DTFFT_TRANSPOSE_Y_TO_Z (only for 3d plan)
                                                          !< - DTFFT_TRANSPOSE_Z_TO_Y (only for 3d plan)
    end subroutine transpose

    module subroutine execute(self, in, out, transpose_type, aux)
    !! Executes plan
    !!
    !! Note that inplace plan execution (`in` == `out`) is available only in these cases:
    !! - R2R 3D
    !! - C2C 3D
    !! - R2C 2D
    !!
    !! In all other cases call to MPI will fail
      class(dtfft_core),  intent(inout) :: self           !< Core
      type(*),            intent(inout) :: in(..)         !< Incoming buffer of any rank and kind
      type(*),            intent(inout) :: out(..)        !< Resulting buffer of any rank and kind
      integer(IP),        intent(in)    :: transpose_type !< Type of transposition. One of the:
                                                          !< - DTFFT_TRANSPOSE_OUT
                                                          !< - DTFFT_TRANSPOSE_IN
      type(*),  optional, intent(inout) :: aux(..)        !< Optional auxiliary buffer. 
                                                          !< Size of buffer must be greater than value returned by `get_aux_size`
    end subroutine execute

    module subroutine destroy(self)
    !! Destroys plan, frees all memory
      class(dtfft_core),  intent(inout) :: self           !< Core
    end subroutine destroy

    module subroutine get_local_sizes(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
    !! Obtain local starts and counts in `real` and `fourier` spaces
      class(dtfft_core),    intent(in)  :: self           !< Core
      integer(IP), optional,intent(out) :: in_starts(:)   !< Start indexes in `real` space (0-based)
      integer(IP), optional,intent(out) :: in_counts(:)   !< Number of elements in `real` space
      integer(IP), optional,intent(out) :: out_starts(:)  !< Start indexes in `fourier` space (0-based)
      integer(IP), optional,intent(out) :: out_counts(:)  !< Number of elements in `fourier` space
      integer(SP), optional,intent(out) :: alloc_size     !< Minimal number of elements required to execute plan
    end subroutine get_local_sizes

    module integer(SP) function get_aux_size(self)
    !! Number of elements required by auxiliary buffer, that can passed to `execute` method
      class(dtfft_core),    intent(in)  :: self           !< Core
    end function get_aux_size

    module subroutine execute_2d(self, in, out, transpose_type, aux)
      class(dtfft_core),  intent(inout) :: self
      type(*),            intent(inout) :: in(..)
      type(*),            intent(inout) :: out(..)
      integer(IP),        intent(in)    :: transpose_type
      type(*),            intent(inout) :: aux(..)
    end subroutine execute_2d

    module subroutine execute_3d(self, in, out, transpose_type, aux)
      class(dtfft_core),  intent(inout) :: self
      type(*),            intent(inout) :: in(..)
      type(*),            intent(inout) :: out(..)
      integer(IP),        intent(in)    :: transpose_type
      type(*),            intent(inout) :: aux(..)
    end subroutine execute_3d

    module subroutine destroy_internal(self)
      class(dtfft_core),  intent(inout) :: self
    end subroutine destroy_internal

    module subroutine get_local_sizes_internal(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
      class(dtfft_core),  intent(in) :: self
      integer(IP),        intent(out), optional :: in_starts(:)
      integer(IP),        intent(out), optional :: in_counts(:)
      integer(IP),        intent(out), optional :: out_starts(:)
      integer(IP),        intent(out), optional :: out_counts(:)
      integer(SP),        intent(out), optional :: alloc_size
    end subroutine get_local_sizes_internal

    module subroutine init_core(self, dims, sngl_type, sngl_storage_size, dbl_type, dbl_storage_size)
      class(dtfft_core),    intent(inout) :: self                 !< Core
      integer(IP),          intent(in)    :: dims(:)              !< Counts of the transform requested
      TYPE_MPI_DATATYPE,    intent(in)    :: sngl_type            !< MPI_Datatype for single precision plan
      integer(IP),          intent(in)    :: sngl_storage_size    !< Number of bytes needed to store single element (single precision)
      TYPE_MPI_DATATYPE,    intent(in)    :: dbl_type             !< MPI_Datatype for double precision plan
      integer(IP),          intent(in)    :: dbl_storage_size     !< Number of bytes needed to store single element (double precision)
    end subroutine init_core

    module subroutine transpose_private(self, in, out, transpose_type)
    !! Performs single transposition
    !!
    !! Note, that `in` and `out` cannot be the same, otherwise call to MPI will fail
      class(dtfft_core),  intent(inout) :: self           !< Core
      type(*),            intent(in)    :: in(..)         !< Incoming buffer of any rank and kind
      type(*),            intent(inout) :: out(..)        !< Resulting buffer of any rank and kind
      integer(IP),        intent(in)    :: transpose_type !< Type of transposition. One of the:
                                                          !< - DTFFT_TRANSPOSE_X_TO_Y
                                                          !< - DTFFT_TRANSPOSE_Y_TO_X
                                                          !< - DTFFT_TRANSPOSE_Y_TO_Z (only for 3d plan)
                                                          !< - DTFFT_TRANSPOSE_Z_TO_Y (only for 3d plan)
    end subroutine transpose_private

    module subroutine check_init_args(self, routine, dims, comm, precision, effort_flag, executor_type)
    !! Check arguments provided by user and sets type values
      class(dtfft_core),        intent(inout) :: self                 !< Core
      character(len=*),         intent(in)    :: routine              !< Called subroutine name
      integer(IP),              intent(in)    :: dims(:)              !< Global dimensions of transform
      TYPE_MPI_COMM,  optional, intent(in)    :: comm                 !< Optional MPI Communicator
      integer(IP),    optional, intent(in)    :: precision            !< Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
      integer(IP),    optional, intent(in)    :: effort_flag          !< DTFFT planner effort flag
      integer(IP),    optional, intent(in)    :: executor_type        !< Type of External FFT Executor
    end subroutine check_init_args

    module subroutine create_transpose_plans(self, tout, tin, comms, base_type, base_storage)
      class(dtfft_core),    intent(inout) :: self                 !< Core
      class(transpose_t),   intent(inout) :: tout(:)              !<
      class(transpose_t),   intent(inout) :: tin(:)               !<
      TYPE_MPI_COMM,        intent(in)    :: comms(:)             !< Array of 1d communicators
      TYPE_MPI_DATATYPE,    intent(in)    :: base_type            !< Base MPI_Datatype
      integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store single element
    end subroutine create_transpose_plans

    module subroutine create_cart_comm(self, comm_dims, comm, comm_coords, local_comms)
      class(dtfft_core),    intent(inout) :: self                 !< Base class
      integer(IP),          intent(in)    :: comm_dims(:)         !< Dims in cartesian communcator
      TYPE_MPI_COMM,        intent(out)   :: comm                 !< Cartesian communcator
      integer(IP),          intent(out)   :: comm_coords(:)       !< Coordinates of current process in cartesian communcator
      TYPE_MPI_COMM,        intent(out)   :: local_comms(:)       !< 1d communicators in cartesian communcator
    end subroutine create_cart_comm

    module subroutine base_alloc(self)
      class(dtfft_core),    intent(inout) :: self                  !< Core
    end subroutine base_alloc

    module subroutine execute_transposed_out(self, in, out, aux)
      class(dtfft_core),  intent(inout) :: self                 !< Core
      type(*),            intent(inout) :: in(..)               !< Incoming buffer
      type(*),            intent(inout) :: out(..)              !< Outgoing buffer
      type(*),            intent(inout) :: aux(..)              !< Working buffer
    end subroutine execute_transposed_out

    module subroutine execute_transposed_in(self, in, out, aux)
      class(dtfft_core),  intent(inout) :: self                 !< Core
      type(*),            intent(inout) :: in(..)               !< Incoming buffer
      type(*),            intent(inout) :: out(..)              !< Outgoing buffer
      type(*),            intent(inout) :: aux(..)              !< Working buffer
    end subroutine execute_transposed_in

    module subroutine alloc_fft_plans(self)
      class(dtfft_core),      intent(inout) :: self             !< Core
    end subroutine alloc_fft_plans

    module subroutine check_aux(self, aux)
      class(dtfft_core),    intent(inout) :: self             !< Core
      type(*),    optional, intent(in)    :: aux(..)
    end subroutine check_aux

    module subroutine create_c2c_fft(self, start, precision)
      class(dtfft_core),    intent(inout) :: self             !< Core
      integer(IP),          intent(in)    :: start            !< Number of plans to create
      integer(IP),          intent(in)    :: precision
    end subroutine create_c2c_fft
  end interface

  type, extends(dtfft_core) :: dtfft_plan_c2c
  private
  contains
  private
    procedure, pass(self),                  public  :: create => create_c2c
    procedure, pass(self), non_overridable          :: create_c2c_internal
  end type dtfft_plan_c2c

  interface
    module subroutine create_c2c(self, dims, comm, precision, effort_flag, executor_type)
    !! C2C Plan Constructor
      class(dtfft_plan_c2c),    intent(inout) :: self               !< C2C Plan
      integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
      TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
      integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
      integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
      integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    end subroutine create_c2c

    module subroutine create_c2c_internal(self, fun, fft_start, dims, comm, precision, effort_flag, executor_type)
      class(dtfft_plan_c2c),    intent(inout) :: self               !< C2C Plan
      character(len=*),         intent(in)    :: fun                !< Called function
      integer(IP),              intent(in)    :: fft_start          !< 1 for c2c, 2 for r2c
      integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
      TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
      integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
      integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
      integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    end subroutine create_c2c_internal
  end interface

  type, extends(dtfft_plan_c2c) :: dtfft_plan_r2c
  private
    type(info_t)  :: real_info
  contains
  private
    procedure,  pass(self), non_overridable, public :: create => create_r2c
    procedure,  pass(self), non_overridable, public :: destroy => destroy_r2c
    procedure,  pass(self), non_overridable         :: execute_3d => execute_3d_r2c
    procedure,  pass(self), non_overridable         :: execute_2d => execute_2d_r2c
    procedure,  pass(self), non_overridable, public :: get_local_sizes => get_local_sizes_r2c
  end type dtfft_plan_r2c

  interface
    module subroutine create_r2c(self, dims, comm, precision, effort_flag, executor_type)
      !! R2C Plan Constructor
      class(dtfft_plan_r2c),    intent(inout) :: self               !< R2C Plan
      integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
      TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
      integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
      integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
      integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    end subroutine create_r2c

    module subroutine destroy_r2c(self)
      class(dtfft_plan_r2c),  intent(inout) :: self
    end subroutine destroy_r2c

    module subroutine execute_3d_r2c(self, in, out, transpose_type, aux)
      class(dtfft_plan_r2c),  intent(inout) :: self                   !< R2C Plan
      type(*),                intent(inout) :: in(..)
      type(*),                intent(inout) :: out(..)
      integer(IP),            intent(in)    :: transpose_type
      type(*),                intent(inout) :: aux(..)
    end subroutine execute_3d_r2c

    module subroutine execute_2d_r2c(self, in, out, transpose_type, aux)
      class(dtfft_plan_r2c),  intent(inout) :: self                   !< R2C Plan
      type(*),                intent(inout) :: in(..)
      type(*),                intent(inout) :: out(..)
      integer(IP),            intent(in)    :: transpose_type
      type(*),                intent(inout) :: aux(..)
    end subroutine execute_2d_r2c

  !------------------------------------------------------------------------------------------------
    module subroutine get_local_sizes_r2c(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
      class(dtfft_plan_r2c),  intent(in)    :: self                   !< R2C Plan
      integer(IP),  optional, intent(out)   :: in_starts(:)           !< Starts of local portion of data in 'real' space
      integer(IP),  optional, intent(out)   :: in_counts(:)           !< Counts of local portion of data in 'real' space
      integer(IP),  optional, intent(out)   :: out_starts(:)          !< Starts of local portion of data in 'fourier' space
      integer(IP),  optional, intent(out)   :: out_counts(:)          !< Counts of local portion of data in 'fourier' space
      integer(SP),  optional, intent(out)   :: alloc_size             !< Maximum number of elements needs to be allocated
    end subroutine get_local_sizes_r2c
  endinterface

  type, extends(dtfft_core) :: dtfft_plan_r2r
  private
  contains
  private
    procedure, pass(self),                  public  :: create => create_r2r
  end type dtfft_plan_r2r

  interface
    module subroutine create_r2r(self, dims, in_kinds, out_kinds, precision, effort_flag, executor_type, comm)
    !! R2R Plan Constructor
      class(dtfft_plan_r2r),    intent(inout) :: self               !< R2R Plan
      integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
      integer(IP),    optional, intent(in)    :: in_kinds(:)        !< Forward kinds
      integer(IP),    optional, intent(in)    :: out_kinds(:)       !< Backward kinds
      integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
      integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
      integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
      TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
    end subroutine create_r2r

    module subroutine DTFFT_FATAL_ERROR(msg, fun)
      character(len=*), intent(in)  :: msg
      character(len=*), intent(in)  :: fun
    end subroutine DTFFT_FATAL_ERROR

    module subroutine DTFFT_DEBUG(msg, fun)
      character(len=*), intent(in)  :: msg
      character(len=*), intent(in)  :: fun
    end subroutine DTFFT_DEBUG
  endinterface
end module dtfft_core_m