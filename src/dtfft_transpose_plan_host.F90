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
module dtfft_transpose_plan_host
use iso_fortran_env, only: int8, int32, int64, real32, real64, output_unit
use dtfft_abstract_transpose_plan,  only: abstract_transpose_plan, create_cart_comm
use dtfft_pencil,                   only: pencil, get_local_sizes
use dtfft_parameters
use dtfft_transpose_handle_host,    only: transpose_handle_host
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use cudafor
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_profile.h"
#include "dtfft_private.h"
implicit none
private
public :: transpose_plan_host

  integer(int8), save :: FORWARD_PLAN_IDS(3)
  integer(int8), save :: BACKWARD_PLAN_IDS(3)


  type, extends(abstract_transpose_plan) :: transpose_plan_host
  private
#ifdef DTFFT_WITH_CUDA
    type(dtfft_gpu_backend_t)     :: gpu_backend = DTFFT_GPU_BACKEND_MPI_DATATYPE
#endif
    type(transpose_handle_host), allocatable :: in_plans(:)
    type(transpose_handle_host), allocatable :: out_plans(:)
  contains
  ! private
    procedure :: create_private
    procedure :: execute_private
    procedure :: destroy
#ifdef DTFFT_WITH_CUDA
    procedure :: get_gpu_backend
    procedure :: mem_alloc
#endif
    procedure, nopass,      private :: get_plan_execution_time
    procedure, pass(self),  private :: autotune_transpose_id
    procedure, pass(self),  private :: autotune_mpi_datatypes
    procedure, pass(self),  private :: autotune_grid_decomposition
    procedure, pass(self),  private :: autotune_grid
  end type transpose_plan_host

contains
#ifdef DTFFT_WITH_CUDA
  type(dtfft_gpu_backend_t) function get_gpu_backend(self)
    class(transpose_plan_host), intent(in) :: self         !< Transposition class
    get_gpu_backend = self%gpu_backend
  end function get_gpu_backend

  subroutine mem_alloc(self, alloc_bytes, ptr)
  !! Allocates memory via cudaMalloc
    class(transpose_plan_host),     intent(in)  :: self            !< Transposition class
    integer(int64),                 intent(in)  :: alloc_bytes
    type(c_devptr),                 intent(out) :: ptr

    call self%generic_mem_alloc(alloc_bytes, ptr)
  end subroutine mem_alloc
#endif

  function create_private(self, dims, transposed_dims, base_comm, comm_dims, effort, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils) result(error_code)
  !! Creates transposition plans
    class(transpose_plan_host),     intent(inout) :: self                 !< Transposition class
    integer(int32),                 intent(in)    :: dims(:)              !< Global sizes of the transform requested
    integer(int32),                 intent(in)    :: transposed_dims(:,:) !< Transposed sizes of the transform requested
    TYPE_MPI_COMM,                  intent(in)    :: base_comm            !< Base communicator
    integer(int32),                 intent(in)    :: comm_dims(:)         !< Number of MPI Processes in all directions
    type(dtfft_effort_t),           intent(in)    :: effort          !< ``dtFFT`` planner type of effort
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !< Base MPI_Datatype
    integer(int8),                  intent(in)    :: base_storage         !< Number of bytes needed to store single element
    logical,                        intent(in)    :: is_custom_cart_comm  !< Is custom Cartesian communicator provided by user
    TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !< Cartesian communicator
    TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !< Array of 1d communicators
    type(pencil),                   intent(out)   :: pencils(:)           !< Pencils
    integer(int32)                                :: error_code           !< Error code
    integer(int32),                 allocatable   :: best_comm_dims(:)
    integer(int8) :: d, ndims, n_transpose_plans
    integer(int8), allocatable :: best_forward_ids(:), best_backward_ids(:)
    integer(int32) :: comm_size, ierr

    FORWARD_PLAN_IDS(1) = get_datatype_from_env("DTYPE_X_Y"); BACKWARD_PLAN_IDS(1) = get_datatype_from_env("DTYPE_Y_X")
    FORWARD_PLAN_IDS(2) = get_datatype_from_env("DTYPE_Y_Z"); BACKWARD_PLAN_IDS(2) = get_datatype_from_env("DTYPE_Z_Y")
    FORWARD_PLAN_IDS(3) = get_datatype_from_env("DTYPE_X_Z"); BACKWARD_PLAN_IDS(3) = get_datatype_from_env("DTYPE_Z_X")

    ndims = size(dims, kind=int8)

    allocate( best_comm_dims, source=comm_dims )

    n_transpose_plans = ndims - 1_int8; if( self%is_z_slab ) n_transpose_plans = n_transpose_plans + 1_int8
    allocate( best_forward_ids(n_transpose_plans), best_backward_ids(n_transpose_plans) )
    ! Setting default values
    ! Values are defined during compilation
    do d = 1, n_transpose_plans
      best_forward_ids(d) = FORWARD_PLAN_IDS(d)
      best_backward_ids(d) = BACKWARD_PLAN_IDS(d)
    enddo

    call MPI_Comm_size(base_comm, comm_size, ierr)
    ! With custom cart comm we can only search for best Datatypes
    ! only if effort == DTFFT_PATIENT
    if ( (  is_custom_cart_comm                                 &
            .or. comm_size == 1                                 &
            .or. ndims == 2_int8                                &
            .or. self%is_z_slab                                 &
          ) .and. effort == DTFFT_PATIENT ) then
      block
        integer(int32) :: dummy
        real(real64) :: dummy_timer(1)
        integer(int32) :: dummy_decomp(size(comm_dims), 1)
        integer(int8)  :: forw_ids(n_transpose_plans, 1), back_ids(n_transpose_plans, 1)

        dummy = 1
        forw_ids(:, 1) = best_forward_ids(:)
        back_ids(:, 1) = best_backward_ids(:)
        call self%autotune_grid(                                &
          base_comm, comm_dims,                                 &
          dims, transposed_dims,                                &
          effort, base_dtype, base_storage,                &
          dummy, dummy_timer, dummy_decomp, forw_ids, back_ids  &
        )
        best_forward_ids(:) = forw_ids(:, 1)
        best_backward_ids(:) = back_ids(:, 1)
      endblock
    else if ( ndims == 3                                        &
              .and. .not.is_custom_cart_comm                    &
              .and. .not.self%is_z_slab                         &
              .and. effort%val >= DTFFT_MEASURE%val        &
              .and. comm_size > 1 ) then
      call self%autotune_grid_decomposition(                    &
        dims, transposed_dims, base_comm,                       &
        effort, n_transpose_plans,                         &
        base_dtype, base_storage,                               &
        best_comm_dims, best_forward_ids, best_backward_ids     &
      )
    endif

    if ( effort == DTFFT_PATIENT ) then
      WRITE_INFO(repeat("*", 50))
      WRITE_INFO("DTFFT_PATIENT: Selected transpose ids:")
      do d = 1, n_transpose_plans
        WRITE_INFO("    "//TRANSPOSE_NAMES( d)//": "//int_to_str( best_forward_ids(d) ))
        WRITE_INFO("    "//TRANSPOSE_NAMES(-d)//": "//int_to_str( best_backward_ids(d) ))
      enddo
      WRITE_INFO(repeat("*", 50))
    endif

    call create_cart_comm(base_comm, best_comm_dims, cart_comm, comms)
    do d = 1, ndims
      call pencils(d)%create(ndims, d, transposed_dims(:,d), comms)
    enddo

    allocate( self%out_plans(n_transpose_plans), self%in_plans(n_transpose_plans) )

    do d = 1_int8, ndims - 1_int8
      call self%out_plans(d)%create(comms(d + 1), pencils(d), pencils(d + 1), base_dtype, base_storage, best_forward_ids(d))
      call self%in_plans (d)%create(comms(d + 1), pencils(d + 1), pencils(d), base_dtype, base_storage, best_backward_ids(d))
    enddo
    if ( self%is_z_slab ) then
      call self%out_plans(3)%create(cart_comm, pencils(1), pencils(3), base_dtype, base_storage, best_forward_ids(3))
      call self%in_plans (3)%create(cart_comm, pencils(3), pencils(1), base_dtype, base_storage, best_backward_ids(3))
    endif

    deallocate(best_comm_dims)
    deallocate(best_forward_ids, best_backward_ids)
    error_code = DTFFT_SUCCESS
  end function create_private

  subroutine execute_private(self, in, out, transpose_type)
  !! Executes single transposition
    class(transpose_plan_host),     intent(inout) :: self         !< Transposition class
    type(*),  DEVICE_PTR  target,   intent(inout) :: in(..)       !< Incoming buffer of any rank and kind
    type(*),  DEVICE_PTR  target,   intent(inout) :: out(..)      !< Resulting buffer of any rank and kind
    type(dtfft_transpose_type_t),   intent(in)    :: transpose_type !< Type of transpose !< Type of transpose

    if ( transpose_type%val > 0 ) then
      call self%out_plans(transpose_type%val)%transpose(in, out)
    else
      call self%in_plans(abs(transpose_type%val))%transpose(in, out)
    endif
  end subroutine execute_private

  subroutine destroy(self)
  !! Destroys transposition plans
    class(transpose_plan_host),     intent(inout) :: self         !< Transposition class
    integer(int8) :: i

    do i = 1, size(self%in_plans, kind=int8)
      call self%out_plans(i)%destroy()
      call self% in_plans(i)%destroy()
    enddo
    deallocate(self%out_plans)
    deallocate(self% in_plans)
  end subroutine destroy

  subroutine autotune_grid_decomposition(self, dims, transposed_dims, base_comm, effort, n_transpose_plans, base_dtype, base_storage, best_comm_dims, best_forward_ids, best_backward_ids)
    class(transpose_plan_host),   intent(in)    :: self                 !< Abstract plan
    integer(int32),               intent(in)    :: dims(:)              !< Global dims
    integer(int32),               intent(in)    :: transposed_dims(:,:) !< Transposed dims
    TYPE_MPI_COMM,                intent(in)    :: base_comm           !< Base communicator
    type(dtfft_effort_t),         intent(in)    :: effort          !< ``dtFFT`` planner type of effort
    integer(int8),                intent(in)    :: n_transpose_plans
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype            !< Base MPI_Datatype
    integer(int8),                intent(in)    :: base_storage         !< Number of bytes needed to store single element
    integer(int32),               intent(out)   :: best_comm_dims(:)
    integer(int8),                intent(inout) :: best_forward_ids(:), best_backward_ids(:)
    integer(int8)   :: ndims
    integer(int32)  :: comm_size, square_root, i, current_timer, k, ierr
    real(real64)    :: min_time
    real(real64),     allocatable :: timers(:)
    integer(int32),   allocatable :: decomps(:,:)
    integer(int8),    allocatable :: forw_ids(:,:), back_ids(:,:)
    real(real64),    parameter :: MaxR8P  =  huge(1._real64)

    call MPI_Comm_size(base_comm, comm_size, ierr)
    ndims = size(dims, kind=int8)

    square_root = int(sqrt(real(comm_size, real64))) + 1
    allocate(timers(2 * square_root))
    allocate(decomps(2, 2 * square_root))
    allocate(forw_ids(n_transpose_plans, 2 * square_root))
    allocate(back_ids(n_transpose_plans, 2 * square_root))

    do i = 1, 2 * square_root
      do k = 1, n_transpose_plans
        forw_ids(k, i) = FORWARD_PLAN_IDS(k)
        back_ids(k, i) = BACKWARD_PLAN_IDS(k)
      enddo
    enddo

    current_timer = 1
    do i = 1, square_root - 1
      if ( mod( comm_size, i ) /= 0 ) cycle

      call self%autotune_grid(base_comm, [1, i, comm_size / i], dims, transposed_dims, effort, base_dtype, base_storage, current_timer, timers, decomps, forw_ids, back_ids)
      if ( i /= comm_size / i) then
        call self%autotune_grid(base_comm, [1, comm_size / i, i], dims, transposed_dims, effort, base_dtype, base_storage, current_timer, timers, decomps, forw_ids, back_ids)
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

    best_comm_dims(1) = 1
    best_comm_dims(2) = decomps(1, k)
    best_comm_dims(3) = decomps(2, k)
    WRITE_INFO(repeat("*", 50))
    WRITE_INFO("DTFFT_MEASURE: Selected MPI grid 1x"//int_to_str(best_comm_dims(2))//"x"//int_to_str(best_comm_dims(3)))
    if ( effort == DTFFT_PATIENT ) then
      best_forward_ids(:) = forw_ids(:, k)
      best_backward_ids(:) = back_ids(:, k)
    else
      WRITE_INFO(repeat("*", 50))
    endif

    deallocate(timers, decomps, forw_ids, back_ids)
  end subroutine autotune_grid_decomposition

  subroutine autotune_grid(self, base_comm, comm_dims, dims, transposed_dims, effort, base_dtype, base_storage, latest_timer_id, timers, decomps, forw_ids, back_ids)
    class(transpose_plan_host),   intent(in)    :: self                 !< Abstract plan
    TYPE_MPI_COMM,                intent(in)    :: base_comm            !< Base communicator
    integer(int32),               intent(in)    :: comm_dims(:)               !< Number of MPI Processes in Y and Z directions
    integer(int32),               intent(in)    :: dims(:)              !< Global dims
    integer(int32),               intent(in)    :: transposed_dims(:,:) !< Transposed dims
    type(dtfft_effort_t),         intent(in)    :: effort          !< ``dtFFT`` planner type of effort
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(int8),                intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(int32),               intent(inout) :: latest_timer_id      !< Current timer id
    real(real64),                 intent(inout) :: timers(:)            !< Time of current function execution is stored in timers(latest_timer_id)
    integer(int32),               intent(inout) :: decomps(:,:)         !< Current decomposition is stored in decomps(:, latest_timer_id)
    integer(int8),                intent(inout) :: forw_ids(:,:)        !< Best Forward ids are stored in forw_ids(:, latest_timer_id)
    integer(int8),                intent(inout) :: back_ids(:,:)        !< Best Backward ids are stored in back_ids(:, latest_timer_id)
    character(len=:),             allocatable   :: phase_name           !< Caliper phase name
    integer(int32)          :: ierr
    integer(int8)           :: d
    type(pencil), allocatable :: pencils(:)
    TYPE_MPI_COMM           :: comm
    real(real64)            :: tf, tb
    TYPE_MPI_COMM,  allocatable           :: comms(:)
    real(real32), allocatable :: a(:), b(:)
    integer(int64)         :: alloc_size
    integer(int8) :: ndims

    ndims = size(comm_dims, kind=int8)

    if ( ndims == 3 ) then
      if ( comm_dims(2) > dims(2) .or. comm_dims(3) > dims(3) ) return
      allocate( phase_name, source = "Testing grid 1x"//int_to_str(comm_dims(2))//"x"//int_to_str(comm_dims(3)) )
    else
      allocate( phase_name, source = "Testing grid 1x"//int_to_str(comm_dims(2)) )
    endif

    WRITE_INFO(repeat("=", 50))
    WRITE_INFO(phase_name)
    PHASE_BEGIN(phase_name, 0)

    ! comm_dims(1) = 1
    ! comm_dims(2) = ny
    ! comm_dims(3) = nz

    allocate( pencils(ndims) )
    allocate( comms(ndims) )

    call create_cart_comm(base_comm, comm_dims, comm, comms)
    do d = 1, ndims
      call pencils(d)%create(ndims, d, transposed_dims(:,d), comms)
    enddo

    call get_local_sizes(pencils, alloc_size=alloc_size)
    alloc_size = alloc_size * base_storage / FLOAT_STORAGE_SIZE

    allocate(a(alloc_size))
    allocate(b(alloc_size))

    if ( effort == DTFFT_PATIENT ) then
      call self%autotune_mpi_datatypes(pencils, comm, comms, base_dtype, base_storage, a, b, forw_ids(:, latest_timer_id), back_ids(:, latest_timer_id), timers(latest_timer_id))
      WRITE_INFO("Execution time on a grid using fastest transpositions: "//double_to_str(timers(latest_timer_id)))
    else
      timers(latest_timer_id) = 0.0_real64
      do d = 1_int8, ndims - 1_int8
        tf = self%get_plan_execution_time(comms(d + 1), comm, pencils(d), pencils(d + 1), base_dtype, base_storage, FORWARD_PLAN_IDS(d), d, a, b)
        tb = self%get_plan_execution_time(comms(d + 1), comm, pencils(d + 1), pencils(d), base_dtype, base_storage, BACKWARD_PLAN_IDS(d), d, a, b)
        timers(latest_timer_id) = timers(latest_timer_id) + tf + tb
      enddo
      WRITE_INFO("Average execution time on a grid: "//double_to_str(timers(latest_timer_id)))
    endif
    do d = 1_int8, ndims - 1_int8
      decomps(d, latest_timer_id) = comm_dims(d + 1)
      ! decomps(1, latest_timer_id) = comm_dims(2)
      ! decomps(2, latest_timer_id) = comm_dims(3)
    enddo

    WRITE_INFO(repeat("=", 50))
    latest_timer_id = latest_timer_id + 1

    deallocate(a, b)
    do d = 1, ndims
      call pencils(d)%destroy()
      call MPI_Comm_free(comms(d), ierr)
    enddo
    call MPI_Comm_free(comm, ierr)
    deallocate( pencils, comms )
    PHASE_END(phase_name)
  end subroutine autotune_grid

  subroutine autotune_mpi_datatypes(self, pencils, cart_comm, comms, base_dtype, base_storage, a, b, forward_ids, backward_ids, elapsed_time)
    class(transpose_plan_host), intent(in)    :: self               !< Host plan
    type(pencil),               intent(in)    :: pencils(:)         !< Array of pencils
    TYPE_MPI_COMM,              intent(in)    :: cart_comm          !< 3D Cartesian comm
    TYPE_MPI_COMM,              intent(in)    :: comms(:)           !< Array of 1d communicators
    TYPE_MPI_DATATYPE,          intent(in)    :: base_dtype         !< Basic MPI Datatype
    integer(int8),              intent(in)    :: base_storage       !< Number of bytes needed to store Basic MPI Datatype
    real(real32),               intent(inout) :: a(:)               !< Work buffer
    real(real32),               intent(inout) :: b(:)               !< Work buffer
    integer(int8),              intent(inout) :: forward_ids(:)     !< Forward plan IDs
    integer(int8),              intent(inout) :: backward_ids(:)    !< Backward plan IDs
    real(real64),               intent(out)   :: elapsed_time       !< Elapsed time
    integer(int8) :: dim        !< Counter
    integer(int8) :: ndims      !< Number of dimensions

    ndims = size(pencils, kind=int8)
    elapsed_time = 0._real64
    if( self%is_z_slab ) then
      elapsed_time = self%autotune_transpose_id(cart_comm, cart_comm, pencils(1), pencils(3), base_dtype, base_storage, 3_int8, a, b, forward_ids(3), backward_ids(3))
    else
      do dim = 1_int8, size(pencils, kind=int8) - 1_int8
        elapsed_time = elapsed_time + &
          self%autotune_transpose_id(comms(dim + 1), cart_comm, pencils(dim), pencils(dim + 1), base_dtype, base_storage, dim, a, b, forward_ids(dim), backward_ids(dim))
      enddo
    endif
  end subroutine autotune_mpi_datatypes

  function autotune_transpose_id(self, comm, cart_comm, from, to, base_dtype, base_storage, transpose_name_id, a, b, forward_id, backward_id) result(elapsed_time)
  !! Creates forward and backward transpose plans bases on source and target data distributing,
  !! executes them `DTFFT_MEASURE_ITERS` times ( 4 * `DTFFT_MEASURE_ITERS` iterations total )
  !!
  !! Returns elapsed time for best plans selected
    class(transpose_plan_host),   intent(in)    :: self                 !< Abstract plan
    TYPE_MPI_COMM,                intent(in)    :: comm                 !< 1D comm in case of pencils, 3D comm in case of z_slabs
    TYPE_MPI_COMM,                intent(in)    :: cart_comm            !< 3D Cartesian comm
    type(pencil),                 intent(in)    :: from                 !< Source meta
    type(pencil),                 intent(in)    :: to                   !< Target meta
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(int8),                intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(int8),                intent(in)    :: transpose_name_id    !< ID of transpose name (from -3 to 3, except 0)
    real(real32),                 intent(inout) :: a(:)                 !< Source buffer
    real(real32),                 intent(inout) :: b(:)                 !< Target buffer
    integer(int8),                intent(out)   :: forward_id           !< Best forward plan ID
    integer(int8),                intent(out)   :: backward_id          !< Best backward plan ID
    real(real64)                                :: elapsed_time         !< Elapsed time for best plans selected
    real(real64)                                :: forward_time         !< Forward plan execution time
    real(real64)                                :: backward_time        !< Backward plan execution time
    real(real64)                                :: time                 !< Timer
    integer(int8)                               :: datatype_id         !< Counter

    forward_time = huge(1._real64)
    backward_time = huge(1._real64)

    do datatype_id = 1, 2
      time = self%get_plan_execution_time(comm, cart_comm, from, to, base_dtype, base_storage, datatype_id, transpose_name_id, a, b)
      if ( time < forward_time ) then
        forward_time = time
        forward_id = datatype_id
      endif

      time = self%get_plan_execution_time(comm, cart_comm, to, from, base_dtype, base_storage, datatype_id, -1_int8 * transpose_name_id, a, b)
      if ( time < backward_time ) then
        backward_time = time
        backward_id = datatype_id
      endif
    enddo
    elapsed_time = forward_time + backward_time
  end function autotune_transpose_id

  function get_plan_execution_time(comm, cart_comm, from, to, base_dtype, base_storage, datatype_id, transpose_name_id, a, b) result(elapsed_time)
  !! Creates transpose plan and executes it `DTFFT_MEASURE_WARMUP_ITERS` + `DTFFT_MEASURE_ITERS` times
  !!
  !! Returns elapsed time
    TYPE_MPI_COMM,                intent(in)    :: comm                 !< 1D comm in case of pencils, 3D comm in case of z_slabs
    TYPE_MPI_COMM,                intent(in)    :: cart_comm            !< 3D Cartesian comm
    type(pencil),                 intent(in)    :: from                 !< Source meta
    type(pencil),                 intent(in)    :: to                   !< Target meta
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype           !< Basic MPI Datatype
    integer(int8),                intent(in)    :: base_storage         !< Number of bytes needed to store Basic MPI Datatype
    integer(int8),                intent(in)    :: datatype_id          !< ID of transpose (1 or 2)
    integer(int8),                intent(in)    :: transpose_name_id    !< ID of transpose name (from -3 to 3, except 0)
    real(real32),                 intent(inout) :: a(:)                 !< Source buffer
    real(real32),                 intent(inout) :: b(:)                 !< Target buffer
    real(real64)                                :: elapsed_time         !< Execution time
    character(len=:),             allocatable   :: phase_name           !< Caliper phase name
    type(transpose_handle_host)                 :: plan                 !< Transpose plan
    real(real64)                                :: ts, te               !< Timers
    integer(int32)                              :: iter                 !< Counter
    integer(int32)                              :: ierr                 !< Error code
    integer(int32)                              :: comm_size            !< Size of ``cart_comm``
    integer(int32)                              :: n_warmup_iters
    integer(int32)                              :: n_iters

    allocate( phase_name, source="    Testing plan "//TRANSPOSE_NAMES(transpose_name_id)//", datatype_id = "//int_to_str(datatype_id) )
    PHASE_BEGIN(phase_name, 0)
    WRITE_INFO(phase_name)
    call plan%create(comm, from, to, base_dtype, base_storage, datatype_id)

    n_warmup_iters = get_iters_from_env(.true.)
    n_iters = get_iters_from_env(.false.)

    do iter = 1, n_warmup_iters
      call plan%transpose(a, b)
    enddo

    ts = MPI_Wtime()
    do iter = 1, n_iters
      call plan%transpose(a, b)
    enddo
    te = MPI_Wtime()
    call MPI_Allreduce(te - ts, elapsed_time, 1, MPI_REAL8, MPI_SUM, cart_comm, ierr)
    call MPI_Comm_size(cart_comm, comm_size, ierr)
    elapsed_time = real(elapsed_time, real64) / real(comm_size, real64)

    call plan%destroy()
    PHASE_END(phase_name)
    deallocate(phase_name)
    WRITE_INFO("        Average execution time: "//double_to_str(elapsed_time))
  end function get_plan_execution_time
end module dtfft_transpose_plan_host