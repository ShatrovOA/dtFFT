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
#include "dtfft_config.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
!------------------------------------------------------------------------------------------------
module dtfft_core_m
!------------------------------------------------------------------------------------------------
!< This module describes `dtfft_core`, `dtfft_plan_c2c`, `dtfft_plan_r2c` and `dtfft_plan_r2r` types
!------------------------------------------------------------------------------------------------
use iso_c_binding, only: c_loc
use dtfft_info_m
use dtfft_parameters
use dtfft_precisions
use dtfft_transpose_m
use dtfft_abstract_executor_m
#ifdef DTFFT_WITH_FFTW
use dtfft_executor_fftw_m
#endif
#ifdef DTFFT_WITH_MKL
use dtfft_executor_mkl_m
#endif
#if defined(CUFFT_ENABLED)
use dtfft_executor_cufft_m
#endif
! #ifdef DTFFT_WITH_KFR
! use dtfft_executor_kfr_m
! #endif
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
implicit none
private
public :: dtfft_core, dtfft_plan_c2c, dtfft_plan_r2c, dtfft_plan_r2r

#ifdef __DEBUG
# define DEBUG(msg) call write_debug(msg)
#else
# define DEBUG(msg)
#endif

#define CHECK_INPUT_PARAMETER(param, valid_values, code)          \
  if (.not.any(param == valid_values)) then;                      \
    __FUNC__ = code;                                              \
    return;                                                       \
  endif

#define CHECK_ERROR_AND_RETURN                      \
  if ( ierr /= DTFFT_SUCCESS ) then;                \
    if ( present( error_code ) ) error_code = ierr; \
    return;                                         \
  endif

#define CHECK_OPTIONAL_CALL( func )                 \
  ierr = func;                                      \
  CHECK_ERROR_AND_RETURN

  interface
    function is_same_ptr(ptr1, ptr2) result(bool) bind(C)
#ifdef DTFFT_WITH_CUDA
      use cudafor
#else
      use iso_c_binding, only: c_ptr
#endif
      use iso_c_binding, only: c_bool
      type(C_ADDR), value :: ptr1, ptr2
      logical(c_bool)     :: bool
    end function is_same_ptr
  end interface

  type :: fft_executor
  !< FFT handle
    class(abstract_executor), allocatable :: fft                                  !< Executor
  end type fft_executor

  type, abstract :: dtfft_core
  !< Abstract class for all DTFFT plans
  private
    TYPE_MPI_COMM                     :: comm                                     !< Grid communicator
    TYPE_MPI_COMM,      allocatable   :: comms(:)                                 !< Local 1d communicators
    integer(IP),        allocatable   :: comm_dims(:)                             !< Dimensions of grid comm
    integer(IP),        allocatable   :: comm_coords(:)                           !< Coordinates of grod comm
    integer(IP)                       :: ndims                                    !< Number of global dimensions
    integer(IP)                       :: precision                                !< Precision of transform
    integer(IP)                       :: comm_size                                !< Size of comm
    type(transpose_t),  allocatable   :: transpose_out(:)                         !< Classes that perform TRANSPOSED_OUT transposes: XYZ --> YXZ --> ZXY
    type(transpose_t),  allocatable   :: transpose_in(:)                          !< Classes that perform TRANSPOSED_IN transposes: ZXY --> YXZ --> XYZ
    type(info_t),       allocatable   :: info(:)                                  !< Information about data aligment and datatypes
    real(R4P),          allocatable   :: aux(:)                                   !< Auxiliary buffer
    type(fft_executor), allocatable   :: fft(:)                                   !< Internal fft runners
    integer(IP),        allocatable   :: fft_mapping(:)                           !< Memory and plan creation optimization.
                                                                                  !< In case same FFTs needs to be run in different dimensions
                                                                                  !< only single FFT plan needs to be created
    logical                           :: is_created = .false.                     !< Plan creation flag
    logical                           :: is_transpose_plan = .false.              !< Plan is transpose only
    logical                           :: is_aux_alloc = .false.                   !< Auxiliary buffer is allocated internally
    logical                           :: is_z_slab = .false.                      !< Using Z-slab optimization
                                                                                  !< Only 3D plan.
    integer(IP)                       :: effort_flag                              !< User defined effort flag
    integer(IP)                       :: storage_size                             !< Single element size in bytes
    integer(IP)                       :: executor_type                            !< FFT executor type
  contains
  private
    procedure,  pass(self), non_overridable, public :: transpose                        !< Performs single transposition
    procedure,  pass(self), non_overridable, public :: execute                          !< Executes plan
    procedure,  pass(self), non_overridable, public :: destroy                          !< Destroys plan
    procedure,  pass(self), non_overridable, public :: get_local_sizes                  !< Returns local starts and counts in `real` and `fourier` spaces
    procedure,  pass(self), non_overridable         :: transpose_private                !< Performs single transposition
    procedure,  pass(self), non_overridable         :: execute_private                  !< Executes plan
    procedure,  pass(self), non_overridable         :: get_local_sizes_private          !< Returns local starts and counts in `real` and `fourier` spaces
    procedure,  pass(self), non_overridable         :: check_create_args                !< Check arguments provided to `create` subroutines
    procedure,  pass(self), non_overridable         :: create_transpose_plans           !< Creates all of the transposition types
    procedure,  pass(self), non_overridable         :: test_grid_decomposition          !< Tests performance of transposition plans
    procedure,  pass(self), non_overridable         :: test_mpi_dtypes
    procedure,  pass(self), non_overridable         :: test_forward_n_backward_plans
    procedure,  pass(self), non_overridable         :: measure_transpose_plan
    procedure,  pass(self), non_overridable         :: create_private                   !< Creates core
    procedure,  pass(self), non_overridable         :: create_cart_comm                 !< Creates cartesian communicator
    procedure,  pass(self), non_overridable         :: alloc_fft_plans                  !< Allocates `fft_executor` classes
    procedure,  pass(self), non_overridable         :: check_aux                        !< Checks if aux buffer was passed and if not will allocate one internally
  end type dtfft_core

  type, extends(dtfft_core) :: dtfft_core_c2c
  private
  contains
  private
    procedure, pass(self), non_overridable          :: create_c2c_internal              !< Creates plan for both C2C and R2C
  end type dtfft_core_c2c

  type, extends(dtfft_core_c2c) :: dtfft_plan_c2c
  private
  contains
  private
    procedure, pass(self), non_overridable, public  :: create => create_c2c             !< C2C Plan Constructor
  end type dtfft_plan_c2c

  type, extends(dtfft_core_c2c) :: dtfft_plan_r2c
  private
    type(info_t)  :: real_info
  contains
  private
    procedure,  pass(self), non_overridable, public :: create => create_r2c                   !< R2C Plan Constructor
  end type dtfft_plan_r2c

  type, extends(dtfft_core) :: dtfft_plan_r2r
  private
  contains
  private
    procedure, pass(self),                  public  :: create => create_r2r
  end type dtfft_plan_r2r

contains
!------------------------------------------------------------------------------------------------
  subroutine transpose_private(self, in, out, transpose_type)
!------------------------------------------------------------------------------------------------
!< Performs single transposition
!< 
!< Note, that `in` and `out` cannot be the same, otherwise call to MPI will fail
!------------------------------------------------------------------------------------------------
    class(dtfft_core),      intent(inout) :: self                   !< Abstract plan
    type(*),                intent(in)    &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: in(..)                 !< Incoming buffer of any rank and kind
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: out(..)                !< Resulting buffer of any rank and kind
    integer(IP),            intent(in)    :: transpose_type         !< Type of transposition. One of the:
                                                                    !< - `DTFFT_TRANSPOSE_X_TO_Y`
                                                                    !< - `DTFFT_TRANSPOSE_Y_TO_X`
                                                                    !< - `DTFFT_TRANSPOSE_Y_TO_Z` (only 3D)
                                                                    !< - `DTFFT_TRANSPOSE_Z_TO_Y` (only 3D)
                                                                    !< - `DTFFT_TRANSPOSE_X_TO_Z` (only 3D and slab decomposition in Z direction)
                                                                    !< - `DTFFT_TRANSPOSE_Z_TO_X` (only 3D and slab decomposition in Z direction)
    PHASE_BEGIN('Transpose '//TRANSPOSE_NAMES(transpose_type))
    if ( transpose_type > 0 ) then
      call self%transpose_out(transpose_type)%transpose(in, out)
    else
      call self%transpose_in(abs(transpose_type))%transpose(in, out)
    endif
    PHASE_END('Transpose '//TRANSPOSE_NAMES(transpose_type))
  end subroutine transpose_private

!------------------------------------------------------------------------------------------------
  subroutine transpose(self, in, out, transpose_type, error_code)
!------------------------------------------------------------------------------------------------
!< Performs single transposition
!< 
!< Note, that `in` and `out` cannot be the same, otherwise call to MPI will fail
!------------------------------------------------------------------------------------------------
    class(dtfft_core),      intent(inout) :: self                   !< Abstract plan
    type(*),                intent(in)    &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
      , target                            :: in(..)                 !< Incoming buffer of any rank and kind
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
      , target                            :: out(..)                !< Resulting buffer of any rank and kind
    integer(IP),            intent(in)    :: transpose_type         !< Type of transposition. One of the:
                                                                    !< - `DTFFT_TRANSPOSE_X_TO_Y`
                                                                    !< - `DTFFT_TRANSPOSE_Y_TO_X`
                                                                    !< - `DTFFT_TRANSPOSE_Y_TO_Z` (only for 3d plan)
                                                                    !< - `DTFFT_TRANSPOSE_Z_TO_Y` (only for 3d plan)
                                                                    !< - `DTFFT_TRANSPOSE_X_TO_Z` (only 3D and slab decomposition in Z direction)
                                                                    !< - `DTFFT_TRANSPOSE_Z_TO_X` (only 3D and slab decomposition in Z direction)
                                                                    !<
                                                                    !< [//]: # (ListBreak)
    integer(IP),  optional, intent(out)   :: error_code             !< Optional error code returned to user
    integer(IP) :: ierr

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created )                                  &
      ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( .not.any(transpose_type == VALID_TRANSPOSES)             &
         .or. ( self%ndims == 2 .and. abs(transpose_type) > 1 )   &
         .or. abs(transpose_type) == 3 .and..not.self%is_z_slab)  &
      ierr = DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
    CHECK_ERROR_AND_RETURN
    if ( is_same_ptr(LOC_FUN(in), LOC_FUN(out)) )                 &
      ierr = DTFFT_ERROR_INPLACE_TRANSPOSE
    CHECK_ERROR_AND_RETURN

    REGION_BEGIN("dtfft_transpose")
    call self%transpose_private(in, out, transpose_type)
    REGION_END("dtfft_transpose")
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine transpose

  subroutine execute(self, in, out, transpose_type, aux, error_code)
  !! Executes plan
    class(dtfft_core),      intent(inout) :: self                   !< Abstract plan
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
      , target                            :: in(..)                 !< Incoming buffer of any rank and kind
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
      , target                            :: out(..)                !< Resulting buffer of any rank and kind
    integer(IP),            intent(in)    :: transpose_type         !< Type of transposition. One of the:
                                                                    !< - `DTFFT_TRANSPOSE_OUT`
                                                                    !< - `DTFFT_TRANSPOSE_IN`
                                                                    !<
                                                                    !< [//]: # (ListBreak)
    type(*),      optional, intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
      , target                            :: aux(..)                !< Optional auxiliary buffer.
                                                                    !< Size of buffer must be greater than value 
                                                                    !< returned by `alloc_size` parameter of `get_local_sizes` subroutine
    integer(IP),  optional, intent(out)   :: error_code             !< Optional error code returned to user
    integer(IP) :: ierr
    logical     :: inplace

    inplace = is_same_ptr(LOC_FUN(in), LOC_FUN(out))
    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created )                                                                      &
      ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( .not.any(transpose_type == VALID_FULL_TRANSPOSES) )                                          &
      ierr = DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
    CHECK_ERROR_AND_RETURN
    if ( self%is_transpose_plan .and. self%ndims == 2 .and. inplace )                                 &
      ierr = DTFFT_ERROR_INPLACE_TRANSPOSE
    CHECK_ERROR_AND_RETURN
    if ( present( aux ) ) then
      if ( is_same_ptr(LOC_FUN(in), LOC_FUN(aux)) .or. is_same_ptr(LOC_FUN(out), LOC_FUN(aux)) )      &
        ierr = DTFFT_ERROR_INVALID_AUX
      CHECK_ERROR_AND_RETURN
    endif

    REGION_BEGIN("dtfft_execute")
    call self%check_aux(aux=aux)
    if ( present( aux ) ) then
      call self%execute_private( in, out, transpose_type, aux, inplace )
    else
      call self%execute_private( in, out, transpose_type, self%aux, inplace )
    endif
    REGION_END("dtfft_execute")
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine execute

  subroutine execute_private(self, in, out, transpose_type, aux, inplace)
    class(dtfft_core),      intent(inout) :: self                   !< Abstract plan
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: in(..)                 !< Incoming buffer of any rank and kind
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: out(..)                !< Resulting buffer of any rank and kind
    integer(IP),            intent(in)    :: transpose_type         !< Type of transposition. One of the:
                                                                    !< - `DTFFT_TRANSPOSE_OUT`
                                                                    !< - `DTFFT_TRANSPOSE_IN`
                                                                    !<
                                                                    !< [//]: # (ListBreak)
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: aux(..)                !< Auxiliary buffer.
                                                                    !< Size of buffer must be greater than value
                                                                    !< returned by `alloc_size` parameter of `get_local_sizes` subroutine
    logical,                intent(in)    :: inplace

    if ( self%is_transpose_plan ) then
      select case ( self%ndims )
      case (2)
        select case( transpose_type )
        case ( DTFFT_TRANSPOSE_OUT )
          call self%transpose_private(in, out, DTFFT_TRANSPOSE_X_TO_Y)
        case ( DTFFT_TRANSPOSE_IN )
          call self%transpose_private(in, out, DTFFT_TRANSPOSE_Y_TO_X)
        endselect
      case (3)
        select case( transpose_type )
        case ( DTFFT_TRANSPOSE_OUT )
          if ( inplace .or. .not. self%is_z_slab ) then
            call self%transpose_private(in, aux, DTFFT_TRANSPOSE_X_TO_Y)
            call self%transpose_private(aux, out, DTFFT_TRANSPOSE_Y_TO_Z)
            return
          endif
          call self%transpose_private(in, out, DTFFT_TRANSPOSE_X_TO_Z)
        case ( DTFFT_TRANSPOSE_IN )
          if ( inplace .or. .not. self%is_z_slab ) then
            call self%transpose_private(in, aux, DTFFT_TRANSPOSE_Z_TO_Y)
            call self%transpose_private(aux, out, DTFFT_TRANSPOSE_Y_TO_X)
            return
          endif
          call self%transpose_private(in, out, DTFFT_TRANSPOSE_Z_TO_X)
        endselect
      endselect
      return
    endif ! self%is_transpose_plan

    select case ( transpose_type )
    case ( DTFFT_TRANSPOSE_OUT )
      ! 1d direct FFT X direction || 2d X-Y FFT
      call self%fft(1)%fft%execute(in, aux, DTFFT_FORWARD)
      if ( self%is_z_slab ) then
        ! Transpose X -> Z
        call self%transpose_private(aux, out, DTFFT_TRANSPOSE_X_TO_Z)
        ! 1d direct FFT Z direction
        call self%fft(3)%fft%execute(out, out, DTFFT_FORWARD)
        return
      endif
      ! Transpose X -> Y
      call self%transpose_private(aux, out, DTFFT_TRANSPOSE_X_TO_Y)
      ! 1d FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(out, out, DTFFT_FORWARD)
      if ( self%ndims == 2 ) then
        return
      endif
      ! Transpose Y -> Z
      call self%transpose_private(out, aux, DTFFT_TRANSPOSE_Y_TO_Z)
      ! 1d direct FFT Z direction
      call self%fft(self%fft_mapping(3))%fft%execute(aux, out, DTFFT_FORWARD)
    case ( DTFFT_TRANSPOSE_IN )
      if ( self%is_z_slab ) then
        ! 1d inverse FFT Z direction
        call self%fft(3)%fft%execute(in, in, DTFFT_BACKWARD)
        ! Transpose Z -> X
        call self%transpose_private(in, aux, DTFFT_TRANSPOSE_Z_TO_X)
        ! 2d inverse FFT X-Y direction
        call self%fft(1)%fft%execute(aux, out, DTFFT_BACKWARD)
        return
      endif
      if ( self%ndims == 3 ) then
        ! 1d inverse FFT Z direction
        call self%fft(self%fft_mapping(3))%fft%execute(in, aux, DTFFT_BACKWARD)
        ! Transpose Z -> Y
        call self%transpose_private(aux, in, DTFFT_TRANSPOSE_Z_TO_Y)
      endif
      ! 1d inverse FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(in, in, DTFFT_BACKWARD)
      ! Transpose Y -> X
      call self%transpose_private(in, aux, DTFFT_TRANSPOSE_Y_TO_X)
      ! 1d inverse FFT X direction
      call self%fft(1)%fft%execute(aux, out, DTFFT_BACKWARD)
    endselect
  end subroutine execute_private

  subroutine destroy(self, error_code)
  !! Destroys plan, frees all memory
    class(dtfft_core),      intent(inout) :: self               !< Abstract plan
    integer(IP),  optional, intent(out)   :: error_code         !< Optional Error Code returned to user
    integer(IP)                           :: d                  !< Counter
    integer(IP) :: ierr

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    select type ( self )
    class is ( dtfft_plan_r2c )
      call self%real_info%destroy()
    endselect

    if ( allocated(self%info) ) then
      do d = 1, self%ndims
        call self%info(d)%destroy()
      enddo
      deallocate(self%info)
    endif

    if ( allocated(self%fft) ) then
      do d = 1, self%ndims
        call self%fft(d)%fft%destroy()
        if ( allocated(self%fft(d)%fft) ) deallocate(self%fft(d)%fft)
      enddo
      deallocate(self%fft)
    endif

    if ( allocated(self%aux) )         deallocate(self%aux)
    if ( allocated(self%comm_dims) )   deallocate(self%comm_dims)
    if ( allocated(self%comm_coords) ) deallocate(self%comm_coords)

    self%is_created = .false.
    self%is_aux_alloc = .false.

    block
      logical     :: is_finalized

      call MPI_Finalized(is_finalized, ierr)

      if ( is_finalized ) ierr = DTFFT_ERROR_MPI_FINALIZED
      CHECK_ERROR_AND_RETURN

      if ( allocated(self%transpose_in) .and. allocated(self%transpose_out) ) then
        do d = 1, self%ndims - 1
          call self%transpose_in(d)%destroy()
          call self%transpose_out(d)%destroy()
        enddo
        if ( self%is_z_slab ) then
          call self%transpose_in(self%ndims)%destroy()
          call self%transpose_out(self%ndims)%destroy()
          self%is_z_slab = .false.
        endif
        deallocate(self%transpose_in)
        deallocate(self%transpose_out)
      endif

      if ( allocated(self%comms) ) then
        do d = 1, self%ndims
          call MPI_Comm_free(self%comms(d), ierr)
        enddo
        deallocate(self%comms)
      endif
      call MPI_Comm_free(self%comm, ierr)
    end block
    self%ndims = -1
    if ( present( error_code ) ) error_code = ierr
  end subroutine destroy

  subroutine get_local_sizes(self, in_starts, in_counts, out_starts, out_counts, alloc_size, error_code)
  !! Obtain local starts and counts in `real` and `fourier` spaces
    class(dtfft_core),    intent(in)  :: self                   !< Abstract plan
    integer(IP), optional,intent(out) :: in_starts(:)           !< Start indexes in `real` space (0-based)
    integer(IP), optional,intent(out) :: in_counts(:)           !< Number of elements in `real` space
    integer(IP), optional,intent(out) :: out_starts(:)          !< Start indexes in `fourier` space (0-based)
    integer(IP), optional,intent(out) :: out_counts(:)          !< Number of elements in `fourier` space
    integer(SP), optional,intent(out) :: alloc_size             !< Minimal number of elements required to execute plan
    integer(IP), optional,intent(out) :: error_code             !< Optional error code returned to user
    integer(IP) :: ierr

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    if ( present( error_code ) ) error_code = ierr
    if ( ierr /= DTFFT_SUCCESS ) return

    select type ( self )
    class is (dtfft_plan_r2c)
      if (present( in_starts ) )    in_starts(1:self%ndims)   = self%real_info%starts
      if (present( in_counts ) )    in_counts(1:self%ndims)   = self%real_info%counts
      call self%get_local_sizes_private(self%info, out_starts=out_starts, out_counts=out_counts, alloc_size=alloc_size)
      if ( present( alloc_size ) ) alloc_size = max(int(product(self%real_info%counts), SP), 2 * alloc_size)
    class default
      call self%get_local_sizes_private(self%info, in_starts, in_counts, out_starts, out_counts, alloc_size)
    endselect
  end subroutine get_local_sizes

  subroutine get_local_sizes_private(self, infos, in_starts, in_counts, out_starts, out_counts, alloc_size)
  !! Obtain local starts and counts in `real` and `fourier` spaces
    class(dtfft_core),    intent(in)  :: self                   !< Abstract plan
    type(info_t),         intent(in)  :: infos(:)
    integer(IP), optional,intent(out) :: in_starts(:)           !< Start indexes in `real` space (0-based)
    integer(IP), optional,intent(out) :: in_counts(:)           !< Number of elements in `real` space
    integer(IP), optional,intent(out) :: out_starts(:)          !< Start indexes in `fourier` space (0-based)
    integer(IP), optional,intent(out) :: out_counts(:)          !< Number of elements in `fourier` space
    integer(SP), optional,intent(out) :: alloc_size             !< Minimal number of elements required to execute plan
    integer(IP) :: d

    if ( present(in_starts) )  in_starts(1:self%ndims)   = infos(1)%starts
    if ( present(in_counts) )  in_counts(1:self%ndims)   = infos(1)%counts
    if ( present(out_starts) ) out_starts(1:self%ndims)  = infos(self%ndims)%starts
    if ( present(out_counts) ) out_counts(1:self%ndims)  = infos(self%ndims)%counts
    if ( present(alloc_size) ) alloc_size = maxval([(product(infos(d)%counts), d=1,self%ndims)])
  end subroutine get_local_sizes_private

  integer(IP) function create_private(self, dims, sngl_type, sngl_storage_size, dbl_type, dbl_storage_size, comm, precision, effort_flag, executor_type, kinds)
#define __FUNC__ create_private
!< Creates core
    class(dtfft_core),        intent(inout) :: self                 !< Abstract plan
    integer(IP),              intent(in)    :: dims(:)              !< Counts of the transform requested
    TYPE_MPI_DATATYPE,        intent(in)    :: sngl_type            !< MPI_Datatype for single precision plan
    integer(IP),              intent(in)    :: sngl_storage_size    !< Number of bytes needed to store single element (single precision)
    TYPE_MPI_DATATYPE,        intent(in)    :: dbl_type             !< MPI_Datatype for double precision plan
    integer(IP),              intent(in)    :: dbl_storage_size     !< Number of bytes needed to store single element (double precision)
    TYPE_MPI_COMM,  optional, intent(in)    :: comm                 !< User-defined communicator
    integer(IP),    optional, intent(in)    :: precision            !< Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
    integer(IP),    optional, intent(in)    :: effort_flag          !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type        !< Type of External FFT Executor
    integer(IP),    optional, intent(in)    :: kinds(:)             !< Kinds of R2R transform
    integer(IP)                             :: ierr                 !< Error code
    integer(IP)                             :: d                    !< Counter
    integer(IP)                             :: top_type             !< MPI_Topo_test flag
    integer(IP),            allocatable     :: transposed_dims(:,:) !< Global counts in transposed coordinates
    TYPE_MPI_DATATYPE                       :: base_dtype
    integer(IP)                             :: base_storage
    integer(IP),            allocatable     :: forward_transpose_ids(:)
    integer(IP),            allocatable     :: backward_transpose_ids(:)
    TYPE_MPI_COMM                           :: comm_
    logical                                 :: is_custom_cart_comm
    integer(IP)                             :: n_transpose_plans

    create_private = DTFFT_SUCCESS
    CHECK_INTERNAL_CALL( self%check_create_args(dims, comm, precision, effort_flag, executor_type, kinds) )

    select case ( self%precision )
    case ( DTFFT_SINGLE )
      base_storage = sngl_storage_size
      base_dtype = sngl_type
    case ( DTFFT_DOUBLE )
      base_storage = dbl_storage_size
      base_dtype = dbl_type
    endselect
    self%storage_size = base_storage

    allocate(transposed_dims(self%ndims, self%ndims))
    if ( self%ndims == 2 ) then
      ! Nx x Ny
      transposed_dims(:, 1) = dims(:)
      ! Ny x Nx
      transposed_dims(1, 2) = dims(2)
      transposed_dims(2, 2) = dims(1)
    else
      ! Nx x Ny x Nz
      transposed_dims(:, 1) = dims(:)
      ! Ny x Nx x Nz
      transposed_dims(1, 2) = dims(2)
      transposed_dims(2, 2) = dims(1)
      transposed_dims(3, 2) = dims(3)
      ! Nz x Nx x Ny
      transposed_dims(1, 3) = dims(3)
      transposed_dims(2, 3) = dims(1)
      transposed_dims(3, 3) = dims(2)
    endif

    allocate(self%info(self%ndims))
    allocate(self%comm_dims(self%ndims))
    allocate(self%comms(self%ndims))
    allocate(self%comm_coords(self%ndims))

    comm_ = MPI_COMM_WORLD; if ( present( comm ) ) comm_ = comm
    call MPI_Comm_size(comm_, self%comm_size, ierr)
    call MPI_Topo_test(comm_, top_type, ierr)

    is_custom_cart_comm = .false.
    self%is_z_slab = .false.
    if ( top_type == MPI_CART ) then
      is_custom_cart_comm = .true.
      block
        integer(IP)                 :: ndims                !< Number of dims in user defined cartesian communicator
        integer(IP),  allocatable   :: temp_dims(:)         !< Temporary dims needed by MPI_Cart_get
        integer(IP),  allocatable   :: temp_coords(:)       !< Temporary coordinates needed by MPI_Cart_get
        logical,      allocatable   :: temp_periods(:)      !< Temporary periods needed by MPI_Cart_get

        call MPI_Cartdim_get(comm_, ndims, ierr)
        if ( ndims > self%ndims ) then
          create_private = DTFFT_ERROR_INVALID_COMM_DIMS
          return
        endif
        self%comm_dims(:) = 1
        allocate(temp_dims(ndims), temp_periods(ndims), temp_coords(ndims))
        call MPI_Cart_get(comm_, ndims, temp_dims, temp_periods, temp_coords, ierr)
        if ( ndims == self%ndims ) then
          if ( temp_dims(1) /= 1 ) then
            create_private = DTFFT_ERROR_INVALID_COMM_FAST_DIM
            return
          endif
          self%comm_dims(:) = temp_dims
        elseif ( ndims == self%ndims - 1 ) then
          self%comm_dims(2:) = temp_dims
        elseif ( ndims == self%ndims - 2 ) then
          self%comm_dims(3) = temp_dims(1)
        endif
        deallocate(temp_dims, temp_periods, temp_coords)

        if ( self%ndims == 3 .and. self%comm_dims(2) == 1) then
          self%is_z_slab = .true.
        endif
      endblock
    else
      self%comm_dims(:) = 0
      self%comm_dims(1) = 1
      if ( self%ndims == 3 ) then
        if ( self%comm_size <= dims(3) ) then
          self%comm_dims(2) = 1
          self%comm_dims(3) = self%comm_size
          self%is_z_slab = .true.
        endif
      endif
      call MPI_Dims_create(self%comm_size, self%ndims, self%comm_dims, ierr)

    endif
    if ( self%is_z_slab ) then
      DEBUG("Using Z-slab optimization")
    endif

    n_transpose_plans = self%ndims - 1; if( self%is_z_slab ) n_transpose_plans = n_transpose_plans + 1

    allocate(self%transpose_in(n_transpose_plans))
    allocate(self%transpose_out(n_transpose_plans))

    allocate( forward_transpose_ids( n_transpose_plans ) )
    allocate( backward_transpose_ids( n_transpose_plans ) )

    ! Setting default values
    ! Values are defined during compilation
    forward_transpose_ids(1) = DTFFT_FORWARD_X_Y
    backward_transpose_ids(1) = DTFFT_BACKWARD_X_Y
    if ( self%ndims == 3 ) then
      forward_transpose_ids(2) = DTFFT_FORWARD_Y_Z
      backward_transpose_ids(2) = DTFFT_BACKWARD_Y_Z
      if( self%is_z_slab ) then
        forward_transpose_ids(3) = DTFFT_FORWARD_X_Z
        backward_transpose_ids(3) = DTFFT_BACKWARD_X_Z
      endif
    endif

    ! With custom cart comm we can only search for best Datatypes
    ! only if effort_flag == DTFFT_PATIENT
    if ( (  is_custom_cart_comm                       &
            .or. self%comm_size == 1                  &
            .or. self%ndims == 2                      &
            .or. self%is_z_slab                       &
          ) .and. self%effort_flag == DTFFT_PATIENT ) then
      block
        TYPE_MPI_COMM :: comms(self%ndims), cart_comm
        real(R4P), allocatable :: a(:), b(:)
        integer(IP) :: comm_coords(self%ndims)
        integer(SP) :: alloc_size
        type(info_t) :: infos(self%ndims)
        real(R8P) :: dummy

        call self%create_cart_comm(comm_, self%comm_dims, cart_comm, comm_coords, comms)
        do d = 1, self%ndims
          call infos(d)%init(self%ndims, d, transposed_dims(:,d), comms, self%comm_dims, comm_coords)
        enddo
        call self%get_local_sizes_private(infos, alloc_size=alloc_size)
        alloc_size = alloc_size * base_storage / FLOAT_STORAGE_SIZE

        allocate(a(alloc_size))
        allocate(b(alloc_size))
        ! Populating forward_transpose_ids, backward_transpose_ids with fastest datatype ids
        call self%test_mpi_dtypes(infos, cart_comm, comms, base_dtype, base_storage, a, b, forward_transpose_ids, backward_transpose_ids, dummy)

        call MPI_Comm_free(cart_comm, ierr)
        do d = 1, self%ndims
          call infos(d)%destroy()
          call MPI_Comm_free(comms(d), ierr)
        enddo
        deallocate(a, b)
      endblock
    else if ( self%ndims == 3                           &
              .and. .not.is_custom_cart_comm            &
              .and. .not.self%is_z_slab                 &
              .and. self%effort_flag >= DTFFT_MEASURE   &
              .and. self%comm_size > 1 ) then
      block
        integer(IP) :: square_root, i, current_timer, k
        real(R8P)   :: min_time
        real(R8P),    allocatable :: timers(:)
        integer(IP),  allocatable :: decomps(:,:), forw_ids(:,:), back_ids(:,:)
        real(R8P),    parameter :: MaxR8P  =  huge(1._R8P)

        square_root = int(sqrt(real(self%comm_size, R8P))) + 1
        allocate(timers(2 * square_root))
        allocate(decomps(2, 2 * square_root))
        allocate(forw_ids(2, 2 * square_root))
        allocate(back_ids(2, 2 * square_root))

        current_timer = 1
        do i = 1, square_root - 1
          if ( mod( self%comm_size, i ) /= 0 ) cycle

          call self%test_grid_decomposition(comm_, i, self%comm_size / i, dims, transposed_dims, base_dtype, base_storage, forward_transpose_ids, backward_transpose_ids, current_timer, timers, decomps, forw_ids, back_ids)
          if ( i /= self%comm_size / i) then
            call self%test_grid_decomposition(comm_, self%comm_size / i, i, dims, transposed_dims, base_dtype, base_storage, forward_transpose_ids, backward_transpose_ids, current_timer, timers, decomps, forw_ids, back_ids)
          endif
        enddo

        min_time = MaxR8P
        k = 1
        do i = 1, current_timer - 1
          if ( timers(i) < min_time ) then
            min_time = timers(i)
            k = i
          endif
        enddo

        self%comm_dims(1) = 1
        self%comm_dims(2) = decomps(1, k)
        self%comm_dims(3) = decomps(2, k)
        DEBUG(repeat("*", 50))
        DEBUG("DTFFT_MEASURE: Selected MPI grid 1x"//int_to_str(decomps(1, k))//"x"//int_to_str(decomps(2, k)))
        if ( self%effort_flag == DTFFT_PATIENT ) then
          forward_transpose_ids(:) = forw_ids(:, k)
          backward_transpose_ids(:) = back_ids(:, k)
        else
          DEBUG(repeat("*", 50))
        endif

        deallocate(timers, decomps, forw_ids, back_ids)
      endblock
    endif

    if ( self%effort_flag == DTFFT_PATIENT ) then
      DEBUG(repeat("*", 50))
      DEBUG("DTFFT_PATIENT: Selected transpose ids:")
      DEBUG("    "//TRANSPOSE_NAMES(+1)//": "//int_to_str( forward_transpose_ids(1) ))
      DEBUG("    "//TRANSPOSE_NAMES(-1)//": "//int_to_str( backward_transpose_ids(1) ))
      if ( self%ndims == 3 ) then
        DEBUG("    "//TRANSPOSE_NAMES(+2)//": "//int_to_str( forward_transpose_ids(2) ))
        DEBUG("    "//TRANSPOSE_NAMES(-2)//": "//int_to_str( backward_transpose_ids(2) ))
        if ( self%is_z_slab ) then
          DEBUG("    "//TRANSPOSE_NAMES(+3)//": "//int_to_str( forward_transpose_ids(3) ))
          DEBUG("    "//TRANSPOSE_NAMES(-3)//": "//int_to_str( backward_transpose_ids(3) ))
        endif
      endif
      DEBUG(repeat("*", 50))
    endif

    call self%create_cart_comm(comm_, self%comm_dims, self%comm, self%comm_coords, self%comms)
    do d = 1, self%ndims
      call self%info(d)%init(self%ndims, d, transposed_dims(:,d), self%comms, self%comm_dims, self%comm_coords)
    enddo

    call self%create_transpose_plans(self%transpose_out, self%transpose_in, self%info, self%comms, base_dtype, base_storage, forward_transpose_ids, backward_transpose_ids)
    call self%alloc_fft_plans(kinds)
    deallocate(transposed_dims, forward_transpose_ids, backward_transpose_ids)

    self%is_aux_alloc = .false.
#undef __FUNC__
  end function create_private

  subroutine test_grid_decomposition(self, base_comm, ny, nz, dims, transposed_dims, base_dtype, base_storage, def_forw_ids, def_back_ids, latest_timer_id, timers, decomps, forw_ids, back_ids)
    class(dtfft_core),    intent(in)    :: self                 !< Abstract plan
    TYPE_MPI_COMM,        intent(in)    :: base_comm            !< Base communicator
    integer(IP),          intent(in)    :: ny, nz               !< Number of MPI Processes in Y and Z directions
    integer(IP),          intent(in)    :: dims(:)              !< Global dims
    integer(IP),          intent(in)    :: transposed_dims(:,:) !< Transposed dims
    TYPE_MPI_DATATYPE,    intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(IP),          intent(in)    :: def_forw_ids(:)      !< Default Forward transpose ids
    integer(IP),          intent(in)    :: def_back_ids(:)      !< Default Backward transpose ids
    integer(IP),          intent(inout) :: latest_timer_id      !< Current timer id
    real(R8P),            intent(inout) :: timers(:)            !< Time of current function execution is stored in timers(latest_timer_id)
    integer(IP),          intent(inout) :: decomps(:,:)         !< Current decomposition is stored in decomps(:, latest_timer_id)
    integer(IP),          intent(inout) :: forw_ids(:,:)        !< Best Forward ids are stored in forw_ids(:, latest_timer_id)
    integer(IP),          intent(inout) :: back_ids(:,:)        !< Best Backward ids are stored in back_ids(:, latest_timer_id)
    real(R8P)           :: timer_start, timer_end, global_timer !< Timers
    integer(IP)         :: comm_dims(3), comm_coords(3), d, iter, ierr
    type(transpose_t)   :: transpose_out(2)           !< Classes that perform TRANSPOSED_OUT transposes: XYZ --> YXZ --> ZXY
    type(transpose_t)   :: transpose_in(2)            !< Classes that perform TRANSPOSED_IN transposes: ZXY --> YXZ --> XYZ
    type(info_t)        :: infos(3)
    TYPE_MPI_COMM       :: comm
    TYPE_MPI_COMM       :: comms(3)
    real(R4P), allocatable :: a(:), b(:)
    integer(SP)         :: alloc_size

    if ( ny > dims(2) .or. nz > dims(3) ) return

    PHASE_BEGIN("Testing grid 1x"//int_to_str(ny)//"x"//int_to_str(nz))
    comm_dims(1) = 1
    comm_dims(2) = ny
    comm_dims(3) = nz

    call self%create_cart_comm(base_comm, comm_dims, comm, comm_coords, comms)
    do d = 1, self%ndims
      call infos(d)%init(self%ndims, d, transposed_dims(:,d), comms, comm_dims, comm_coords)
    enddo
    
    call self%get_local_sizes_private(infos, alloc_size=alloc_size)
    alloc_size = alloc_size * base_storage / FLOAT_STORAGE_SIZE

    allocate(a(alloc_size))
    allocate(b(alloc_size))

    if ( self%effort_flag == DTFFT_PATIENT ) then
      call self%test_mpi_dtypes(infos, comm, comms, base_dtype, base_storage, a, b, forw_ids(:, latest_timer_id), back_ids(:, latest_timer_id), timers(latest_timer_id))
    else
      call self%create_transpose_plans(transpose_out, transpose_in, infos, comms, base_dtype, base_storage, def_forw_ids, def_back_ids)

      timer_start = MPI_Wtime()
      do iter = 1, DTFFT_MEASURE_ITERS
        call transpose_out(1)%transpose(a, b)
        call transpose_out(2)%transpose(b, a)

        call transpose_in(2)%transpose(a, b)
        call transpose_in(1)%transpose(b, a)
      enddo
      timer_end = MPI_Wtime()

      call MPI_Allreduce(timer_end - timer_start, global_timer, 1, MPI_REAL8, MPI_SUM, base_comm, ierr)
      timers(latest_timer_id) = global_timer / real(self%comm_size, R8P)

      do d = 1, self%ndims - 1
        call transpose_in(d)%destroy()
        call transpose_out(d)%destroy()
      enddo
    endif
    decomps(1, latest_timer_id) = ny
    decomps(2, latest_timer_id) = nz
    ! DEBUG(repeat("=", 50))
    ! DEBUG("    Average execution time: "//double_to_str(timers(latest_timer_id)))
    latest_timer_id = latest_timer_id + 1

    deallocate(a, b)
    do d = 1, self%ndims
      call infos(d)%destroy()
      call MPI_Comm_free(comms(d), ierr)
    enddo
    call MPI_Comm_free(comm, ierr)
    PHASE_END("Testing grid 1x"//int_to_str(ny)//"x"//int_to_str(nz))
  end subroutine test_grid_decomposition

  subroutine test_mpi_dtypes(self, infos, cart_comm, comms, base_dtype, base_storage, a, b, forward_ids, backward_ids, elapsed_time)
    class(dtfft_core),    intent(in)    :: self             !< Abstract plan
    type(info_t),         intent(in)    :: infos(:)
    TYPE_MPI_COMM,        intent(in)    :: cart_comm
    TYPE_MPI_COMM,        intent(in)    :: comms(:)
    TYPE_MPI_DATATYPE,    intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    real(R4P),            intent(inout) :: a(:)
    real(R4P),            intent(inout) :: b(:)
    integer(IP),          intent(inout) :: forward_ids(:)
    integer(IP),          intent(inout) :: backward_ids(:)
    real(R8P),            intent(out)   :: elapsed_time
    integer(IP) :: dim

    elapsed_time = 0._R8P
    if( self%is_z_slab ) then
      elapsed_time = self%test_forward_n_backward_plans(cart_comm, cart_comm, infos(1), infos(3), base_dtype, base_storage, 3, a, b, forward_ids(3), backward_ids(3))
    else
      do dim = 1, size(infos) - 1
        elapsed_time = elapsed_time + &
          self%test_forward_n_backward_plans(comms(dim + 1), cart_comm, infos(dim), infos(dim + 1), base_dtype, base_storage, dim, a, b, forward_ids(dim), backward_ids(dim))
      enddo
    endif
  end subroutine test_mpi_dtypes

!------------------------------------------------------------------------------------------------
  function test_forward_n_backward_plans(self, comm, cart_comm, from, to, base_dtype, base_storage, transpose_name_id, a, b, forward_id, backward_id) result(elapsed_time)
!------------------------------------------------------------------------------------------------
!< Creates forward and backward transpose plans bases on source and target data distributing,
!< executes them `DTFFT_MEASURE_ITERS` times ( 4 * `DTFFT_MEASURE_ITERS` iterations total )
!< 
!< Returns elapsed time for best plans selected
!------------------------------------------------------------------------------------------------
    class(dtfft_core),    intent(in)    :: self                 !< Abstract plan
    TYPE_MPI_COMM,        intent(in)    :: comm                 !< 1D comm in case of pencils, 3D comm in case of z_slabs
    TYPE_MPI_COMM,        intent(in)    :: cart_comm            !< 3D Cartesian comm
    type(info_t),         intent(in)    :: from                 !< Source meta
    type(info_t),         intent(in)    :: to                   !< Target meta
    TYPE_MPI_DATATYPE,    intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(IP),          intent(in)    :: transpose_name_id    !< ID of transpose name (from -3 to 3, except 0)
    real(R4P),            intent(inout) :: a(:)                 !< Source buffer
    real(R4P),            intent(inout) :: b(:)                 !< Target buffer
    integer(IP),          intent(out)   :: forward_id           !< Best forward plan ID
    integer(IP),          intent(out)   :: backward_id          !< Best backward plan ID
    real(R8P)                           :: elapsed_time         !< Elapsed time for best plans selected
    real(R8P)                           :: forward_time, backward_time, time  !< Timers
    integer(IP)                         :: transpose_id         !< Counter

    forward_time = huge(1._R8P)
    backward_time = huge(1._R8P)

    do transpose_id = 1, 2
      time = self%measure_transpose_plan(comm, cart_comm, from, to, base_dtype, base_storage, transpose_id, transpose_name_id, a, b)
      if ( time < forward_time ) then
        forward_time = time
        forward_id = transpose_id
      endif

      time = self%measure_transpose_plan(comm, cart_comm, to, from, base_dtype, base_storage, transpose_id, -1 * transpose_name_id, a, b)
      if ( time < backward_time ) then
        backward_time = time
        backward_id = transpose_id
      endif
    enddo
    elapsed_time = forward_time + backward_time
  end function test_forward_n_backward_plans

!------------------------------------------------------------------------------------------------
  function measure_transpose_plan(self, comm, cart_comm, from, to, base_dtype, base_storage, transpose_id, transpose_name_id, a, b) result(elapsed_time)
!------------------------------------------------------------------------------------------------
!< Creates transpose plan and executes it `DTFFT_MEASURE_ITERS` times
!< 
!< Returns elapsed time
!------------------------------------------------------------------------------------------------
    class(dtfft_core),    intent(in)    :: self                 !< Abstract plan
    TYPE_MPI_COMM,        intent(in)    :: comm                 !< 1D comm in case of pencils, 3D comm in case of z_slabs
    TYPE_MPI_COMM,        intent(in)    :: cart_comm            !< 3D Cartesian comm
    type(info_t),         intent(in)    :: from                 !< Source meta
    type(info_t),         intent(in)    :: to                   !< Target meta
    TYPE_MPI_DATATYPE,    intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(IP),          intent(in)    :: transpose_id         !< ID of transpose (1 or 2)
    integer(IP),          intent(in)    :: transpose_name_id    !< ID of transpose name (from -3 to 3, except 0)
    real(R4P),            intent(inout) :: a(:)                 !< Source buffer
    real(R4P),            intent(inout) :: b(:)                 !< Target buffer
    real(R8P)                           :: elapsed_time         !< Execution time
    character(len=:),     allocatable   :: phase_name           !< Caliper phase name
    type(transpose_t)                   :: plan                 !< Transpose plan
    real(R8P)                           :: ts, te               !< Timers
    integer(IP)                         :: iter                 !< Counter
    integer(IP)                         :: ierr                 !< Error code

    allocate( phase_name, source="Testing plan "//TRANSPOSE_NAMES(transpose_name_id)//", transpose_id = "//int_to_str(transpose_id) )
    PHASE_BEGIN(phase_name)
    call plan%init(comm, from, to, base_dtype, base_storage, transpose_id)

    ts = MPI_Wtime()
    do iter = 1, DTFFT_MEASURE_ITERS
      call plan%transpose(a, b)
    enddo
    te = MPI_Wtime()
    call MPI_Allreduce(te - ts, elapsed_time, 1, MPI_REAL8, MPI_SUM, cart_comm, ierr)
    elapsed_time = real(elapsed_time, R8P) / real(self%comm_size, R8P)

    call plan%destroy()
    PHASE_END(phase_name)
    deallocate(phase_name)
    ! DEBUG("        Average execution time: "//double_to_str(elapsed_time))
  end function measure_transpose_plan

!------------------------------------------------------------------------------------------------
  integer(IP) function check_create_args(self, dims, comm, precision, effort_flag, executor_type, kinds)
#define __FUNC__ check_create_args
!------------------------------------------------------------------------------------------------
!< Check arguments provided by user and sets private variables
!------------------------------------------------------------------------------------------------
    class(dtfft_core),        intent(inout) :: self             !< Abstract plan
    integer(IP),              intent(in)    :: dims(:)          !< Global dimensions of transform
    TYPE_MPI_COMM,  optional, intent(in)    :: comm             !< Optional MPI Communicator
    integer(IP),    optional, intent(in)    :: precision        !< Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
    integer(IP),    optional, intent(in)    :: effort_flag      !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type    !< Type of External FFT Executor
    integer(IP),    optional, intent(in)    :: kinds(:)         !< Kinds of R2R transform
    integer(IP) :: ierr, top_type, dim

    CHECK_INTERNAL_CALL( dtfft_init() )

    self%ndims = size(dims)
    CHECK_INPUT_PARAMETER(self%ndims, VALID_DIMENSIONS, DTFFT_ERROR_INVALID_N_DIMENSIONS)
    if ( any([(dims(dim) <= 0, dim=1,self%ndims)]) ) then
      check_create_args = DTFFT_ERROR_INVALID_DIMENSION_SIZE
      return
    endif

    if ( present(comm) ) then
      call MPI_Topo_test(comm, top_type, ierr)
      CHECK_INPUT_PARAMETER(top_type, VALID_COMMUNICATORS, DTFFT_ERROR_INVALID_COMM_TYPE)
    endif

    self%precision = DTFFT_DOUBLE
    if ( present(precision) ) then
      CHECK_INPUT_PARAMETER(precision, VALID_PRECISIONS, DTFFT_ERROR_INVALID_PRECISION)
      self%precision = precision
    endif

    self%effort_flag = DTFFT_ESTIMATE
    if ( present(effort_flag) ) then
      CHECK_INPUT_PARAMETER(effort_flag, VALID_EFFORTS, DTFFT_ERROR_INVALID_EFFORT_FLAG)
      self%effort_flag = effort_flag
    endif

    self%is_transpose_plan = .false.
    self%executor_type = DTFFT_EXECUTOR_NONE
    if ( present(executor_type) ) then
      CHECK_INPUT_PARAMETER(executor_type, VALID_EXECUTORS, DTFFT_ERROR_INVALID_EXECUTOR_TYPE)
      self%executor_type = executor_type
    endif
    if ( self%executor_type == DTFFT_EXECUTOR_NONE ) self%is_transpose_plan = .true.

    if ( present(kinds) ) then
      do dim = 1, self%ndims
        if ( .not.any([(kinds(dim) == VALID_R2R_FFTS)]) ) then
          check_create_args = DTFFT_ERROR_INVALID_R2R_KINDS
          return
        endif
      enddo
    endif
#undef __FUNC__
  end function check_create_args

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_plans(self, tout, tin, infos, comms, base_type, base_storage, forward_ids, backward_ids)
!------------------------------------------------------------------------------------------------
!< Creates all of the transposition types
!------------------------------------------------------------------------------------------------
    class(dtfft_core),    intent(in)    :: self                 !< Abstract plan
    type(transpose_t),    intent(inout) :: tout(:)              !< Forward transpose plans
    type(transpose_t),    intent(inout) :: tin(:)               !< Backward transpose plans
    type(info_t),         intent(in)    :: infos(:)             !< Data distributing meta
    TYPE_MPI_COMM,        intent(in)    :: comms(:)             !< Array of 1d communicators
    TYPE_MPI_DATATYPE,    intent(in)    :: base_type            !< Base MPI_Datatype
    integer(IP),          intent(in)    :: base_storage         !< Number of bytes needed to store single element
    integer(IP),          intent(in)    :: forward_ids(:)       !< Types of transpose to perform during forward plan
    integer(IP),          intent(in)    :: backward_ids(:)      !< Types of transpose to perform during backward plan
    integer(IP)                         :: dim                  !< Counter

    do dim = 1, self%ndims - 1
      call tout(dim)%init(comms(dim + 1), infos(dim), infos(dim + 1), base_type, base_storage, forward_ids(dim))
      call tin (dim)%init(comms(dim + 1), infos(dim + 1), infos(dim), base_type, base_storage, backward_ids(dim))
    enddo
    if ( self%is_z_slab ) then
      call tout(3)%init(self%comm, infos(1), infos(3), base_type, base_storage, forward_ids(3))
      call tin (3)%init(self%comm, infos(3), infos(1), base_type, base_storage, backward_ids(3))
    endif
  end subroutine create_transpose_plans

!------------------------------------------------------------------------------------------------
  subroutine create_cart_comm(self, old_comm, comm_dims, comm, comm_coords, local_comms)
!------------------------------------------------------------------------------------------------
!< Creates cartesian communicator
!------------------------------------------------------------------------------------------------
    class(dtfft_core),    intent(in)    :: self                 !< Base class
    TYPE_MPI_COMM,        intent(in)    :: old_comm             !< Communicator to create cartesian from
    integer(IP),          intent(in)    :: comm_dims(:)         !< Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: comm                 !< Cartesian communicator
    integer(IP),          intent(out)   :: comm_coords(:)       !< Coordinates of current process in cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: local_comms(:)       !< 1d communicators in cartesian communicator
    logical,              allocatable   :: periods(:)           !< Grid is not periodic
    logical,              allocatable   :: remain_dims(:)       !< Needed by MPI_Cart_sub
    integer(IP)                         :: dim                  !< Counter
    integer(IP)                         :: comm_rank            !< Rank of current process in cartesian communicator
    integer(IP)                         :: ierr                 !< Error code

    allocate(periods(self%ndims), source = .false.)
    call MPI_Cart_create(old_comm, self%ndims, comm_dims, periods, .true., comm, ierr)
    if ( DTFFT_GET_MPI_VALUE(comm) == DTFFT_GET_MPI_VALUE(MPI_COMM_NULL) ) error stop "comm == MPI_COMM_NULL"
    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Cart_coords(comm, comm_rank, self%ndims, comm_coords, ierr)

    allocate( remain_dims(self%ndims), source = .false. )
    do dim = 1, self%ndims
      remain_dims(dim) = .true.
      call MPI_Cart_sub(comm, remain_dims, local_comms(dim), ierr)
      remain_dims(dim) = .false.
    enddo
    deallocate(remain_dims, periods)
  end subroutine create_cart_comm

!------------------------------------------------------------------------------------------------
  subroutine alloc_fft_plans(self, kinds)
!------------------------------------------------------------------------------------------------
!< Allocates `fft_executor` with required FFT class
!------------------------------------------------------------------------------------------------
    class(dtfft_core),        intent(inout) :: self                 !< Abstract plan
    integer(IP),    optional, intent(in)    :: kinds(:)             !< Kinds of R2R transform
    integer(IP) :: dim, dim2                                        !< Counters

    if ( self%is_transpose_plan ) return

    allocate(self%fft(self%ndims))
    allocate(self%fft_mapping(self%ndims))

    do dim = 1, self%ndims
      self%fft_mapping(dim) = dim

      select case(self%executor_type)
#ifdef DTFFT_WITH_FFTW
      case (DTFFT_EXECUTOR_FFTW3)
        if ( dim == 1 ) then
          DEBUG("Using FFTW3 executor")
        endif
        allocate(fftw_executor :: self%fft(dim)%fft)
#endif
#ifdef DTFFT_WITH_MKL
      case (DTFFT_EXECUTOR_MKL)
        if ( dim == 1 ) then
          DEBUG("Using MKL executor")
        endif
        allocate(mkl_executor :: self%fft(dim)%fft)
#endif
#ifdef DTFFT_WITH_CUFFT
      case (DTFFT_EXECUTOR_CUFFT)
        if ( dim == 1 ) then
          DEBUG("Using CUFFT executor")
        endif
        allocate(cufft_executor :: self%fft(dim)%fft)
#endif
! #ifdef DTFFT_WITH_KFR
!       case (DTFFT_EXECUTOR_KFR)
!         if ( dim == 1 ) then
!            DEBUG("Using KFR executor")
!         endif
!         allocate(kfr_executor :: self%forw_plans(dim)%fft)
!         allocate(kfr_executor :: self%back_plans(dim)%fft)
! #endif
      case default
        error stop "Executor type unrecognized"
      endselect
    enddo
    if( self%is_z_slab ) return

    ! Searching for similar FFT transforms in order to reduce time of plan creation
    ! and reducing memory usage
    ! Most profitable in GPU plan
    do dim = 1, self%ndims
      do dim2 = 1, dim - 1
        if ( dim == dim2 ) cycle
        select type ( self )
        class is ( dtfft_plan_r2c )
          if ( dim == 1 ) cycle
        endselect
        if ( self%info(dim)%counts(1) == self%info(dim2)%counts(1)  &
             .and. product(self%info(dim)%counts) == product(self%info(dim2)%counts) ) then
          if ( present(kinds) ) then
            if ( kinds(dim) == kinds(dim2)) self%fft_mapping(dim) = dim2
          else
            self%fft_mapping(dim) = dim2
          endif
        endif
      enddo
    enddo
  end subroutine alloc_fft_plans

!------------------------------------------------------------------------------------------------
  subroutine check_aux(self, aux)
!------------------------------------------------------------------------------------------------
!< Checks if aux buffer was passed and if not will allocate one internally
!------------------------------------------------------------------------------------------------
    class(dtfft_core),    intent(inout) :: self                 !< Abstract plan
    type(*),    optional, intent(in)    :: aux(..)              !< Auxiliary buffer.
    integer(SP)                         :: alloc_size           !< Number of elements to be allocated
    character(len=100)                  :: debug_msg            !< Logging allocation size

    if ( .not. present(aux) ) then
      if ( .not. self%is_aux_alloc ) then
        call self%get_local_sizes(alloc_size=alloc_size)
        alloc_size = alloc_size * self%storage_size / FLOAT_STORAGE_SIZE
        write(debug_msg, '(a, i0, a)') "Allocating auxiliary buffer of ",alloc_size * FLOAT_STORAGE_SIZE, " bytes"
        DEBUG(debug_msg)
        allocate( self%aux(alloc_size) )
        self%is_aux_alloc = .true.
      endif
    endif
  end subroutine check_aux

!------------------------------------------------------------------------------------------------
  subroutine create_r2r(self, dims, kinds, comm, precision, effort_flag, executor_type, error_code)
!------------------------------------------------------------------------------------------------
!< R2R Plan Constructor
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2r),    intent(inout) :: self               !< R2R Plan
    integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
    integer(IP),    optional, intent(in)    :: kinds(:)           !< Kinds of R2R transform
    TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
    integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
    integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    integer(IP),    optional, intent(out)   :: error_code         !< Optional Error Code returned to user
    integer(IP)                             :: ierr               !< Error code
    integer(IP)                             :: fft_rank           !< Rank of FFT transform
    integer(IP)                             :: dim                !< Counter
    integer(IP)                             :: r2r_kinds(2)       !< Transposed Kinds of R2R transform

    REGION_BEGIN("dtfft_create_r2r")
    CHECK_OPTIONAL_CALL( self%create_private(dims, MPI_REAL, FLOAT_STORAGE_SIZE, MPI_REAL8, DOUBLE_STORAGE_SIZE, comm, precision, effort_flag, executor_type, kinds) )

    if ( .not. self%is_transpose_plan ) then
      if ( .not. present( kinds ) ) ierr = DTFFT_ERROR_MISSING_R2R_KINDS
      CHECK_ERROR_AND_RETURN

      do dim = 1, self%ndims
        r2r_kinds(1) = kinds(dim)
        fft_rank = FFT_1D
        if ( self%is_z_slab .and. dim == 1 ) then
          r2r_kinds(1) = kinds(2)
          r2r_kinds(2) = kinds(1)
          fft_rank = FFT_2D
        endif
        if ( self%is_z_slab .and. dim == 2 ) cycle
        CHECK_OPTIONAL_CALL( self%fft(self%fft_mapping(dim))%fft%create(fft_rank, FFT_R2R, self%precision, real_info=self%info(dim), r2r_kinds=r2r_kinds) )
      enddo
    endif
    if ( present( error_code ) ) error_code = ierr
    self%is_created = .true.
    REGION_END("dtfft_create_r2r")
  end subroutine create_r2r

!------------------------------------------------------------------------------------------------
  subroutine create_c2c(self, dims, comm, precision, effort_flag, executor_type, error_code)
!------------------------------------------------------------------------------------------------
!< C2C Plan Constructor
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_c2c),    intent(inout) :: self               !< C2C Plan
    integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
    TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
    integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
    integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    integer(IP),    optional, intent(out)   :: error_code         !< Optional Error Code returned to user
    integer(IP)                             :: ierr               !< Error code

    REGION_BEGIN("dtfft_create_c2c")
    CHECK_OPTIONAL_CALL( self%create_c2c_internal(1, dims, comm, precision, effort_flag, executor_type) )
    if ( present( error_code ) ) error_code = ierr
    self%is_created = .true.
    REGION_END("dtfft_create_c2c")
  end subroutine create_c2c

!------------------------------------------------------------------------------------------------
  integer(IP) function create_c2c_internal(self, fft_start, dims, comm, precision, effort_flag, executor_type)
#define __FUNC__ create_c2c_internal
!------------------------------------------------------------------------------------------------
!< Creates plan for both C2C and R2C
!------------------------------------------------------------------------------------------------
    class(dtfft_core_c2c),    intent(inout) :: self               !< C2C Plan
    integer(IP),              intent(in)    :: fft_start          !< 1 for c2c, 2 for r2c
    integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
    TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
    integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
    integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    integer(IP)                             :: dim                !< Counter
    integer(IP)                             :: fft_rank           !< Rank of FFT transform

    CHECK_INTERNAL_CALL( self%create_private(dims, MPI_COMPLEX, COMPLEX_STORAGE_SIZE, MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, comm, precision, effort_flag, executor_type) )
    if ( .not. self%is_transpose_plan ) then
      do dim = fft_start, self%ndims
        fft_rank = FFT_1D;  if( self%is_z_slab .and. dim == 1) fft_rank = FFT_2D
        if ( self%is_z_slab .and. dim == 2 ) cycle
        CHECK_INTERNAL_CALL( self%fft(self%fft_mapping(dim))%fft%create(fft_rank, FFT_C2C, self%precision, complex_info=self%info(dim)) )
      enddo
    endif
#undef __FUNC__
  end function create_c2c_internal

  subroutine create_r2c(self, dims, comm, precision, effort_flag, executor_type, error_code)
!------------------------------------------------------------------------------------------------
!< R2C Plan Constructor
!------------------------------------------------------------------------------------------------
    class(dtfft_plan_r2c),    intent(inout) :: self               !< R2C Plan
    integer(IP),              intent(in)    :: dims(:)            !< Global dimensions of transform
    TYPE_MPI_COMM,  optional, intent(in)    :: comm               !< Communicator
    integer(IP),    optional, intent(in)    :: precision          !< Presicion of Transform
    integer(IP),    optional, intent(in)    :: effort_flag        !< DTFFT planner effort flag
    integer(IP),    optional, intent(in)    :: executor_type      !< Type of External FFT Executor
    integer(IP),    optional, intent(out)   :: error_code         !< Optional Error Code returned to user
    integer(IP)                             :: ierr, fft_rank
    integer(IP),    allocatable             :: fixed_dims(:)

    REGION_BEGIN("dtfft_create_r2c")
    allocate(fixed_dims, source=dims)
    fixed_dims(1) = int(dims(1) / 2, IP) + 1
    CHECK_OPTIONAL_CALL( self%create_c2c_internal(2, fixed_dims, comm, precision, effort_flag, executor_type) )
    deallocate(fixed_dims)

    if ( self%is_transpose_plan ) ierr = DTFFT_ERROR_R2C_TRANSPOSE_PLAN
    CHECK_ERROR_AND_RETURN

    call self%real_info%init(self%ndims, 1, dims, self%comms, self%comm_dims, self%comm_coords)
    fft_rank = FFT_1D;  if ( self%is_z_slab ) fft_rank = FFT_2D
    CHECK_OPTIONAL_CALL( self%fft(1)%fft%create(fft_rank, FFT_R2C, self%precision, real_info=self%real_info, complex_info=self%info(1)) )

    if ( present( error_code ) ) error_code = ierr
    self%is_created = .true.
    REGION_END("dtfft_create_r2c")
  end subroutine create_r2c
end module dtfft_core_m