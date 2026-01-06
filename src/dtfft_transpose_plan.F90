!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
module dtfft_transpose_plan
!! This module describes [[transpose_plan]] class
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend,               only: backend_helper
use dtfft_abstract_reshape_handle,        only: reshape_container, create_args
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_pencil,                         only: pencil, pencil_init, get_local_sizes
use dtfft_reshape_plan_base,              only: allocate_plans, reshape_plan_base, execute_autotune, destroy_plans
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_profile.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: transpose_plan

  integer(int8), save :: FORWARD_PLAN_IDS(3)
    !! Default data types for forward transpositions
  integer(int8), save :: BACKWARD_PLAN_IDS(3)
    !! Default data types for backward transpositions
  logical,       save :: ARE_DATATYPES_SET = .false.
    !! Are default data types set

  type, extends(reshape_plan_base) :: transpose_plan
  !! Transpose Plan class
  !! This class is a container for transposition plans
  private
    logical                   :: is_z_slab        !! Is Z-slab optimization enabled
    logical                   :: is_y_slab        !! Is Y-slab optimization enabled
  contains
    procedure,  non_overridable,  pass(self), public :: create            !! Creates transpose plan
    procedure,  non_overridable,  pass(self), public :: get_z_slab        !! Returns .true. if Z-slab optimization is enabled
    procedure,  non_overridable,  pass(self), public :: get_y_slab        !! Returns .true. if Y-slab optimization is enabled
  end type transpose_plan

contains

  function create(self, platform, dims, base_comm, effort, base_dtype, base_storage, cart_comm, comms, pencils, ipencil) result(error_code)
  !! Creates transposition plan
    class(transpose_plan),          intent(inout) :: self           !! Transposition class
    type(dtfft_platform_t),         intent(in)    :: platform       !! Platform to create plan for
    integer(int32),                 intent(in)    :: dims(:)        !! Global sizes of the transform requested
    TYPE_MPI_COMM,                  intent(in)    :: base_comm      !! Base communicator
    type(dtfft_effort_t),           intent(in)    :: effort         !! ``dtFFT`` planner type of effort
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype     !! Base MPI_Datatype
    integer(int64),                 intent(in)    :: base_storage   !! Number of bytes needed to store single element
    TYPE_MPI_COMM,                  intent(out)   :: cart_comm      !! Cartesian communicator
    TYPE_MPI_COMM,                  intent(out)   :: comms(:)       !! Array of 1d communicators
    type(pencil),                   intent(out)   :: pencils(:)     !! Data distributing meta
    type(pencil_init),    optional, intent(in)    :: ipencil        !! Pencil passed by user
    integer(int32)                                :: error_code     !! Error code
    ! integer(int32),   allocatable   :: transposed_dims(:,:) !! Global counts in transposed coordinates
    logical           :: cond1    !! First condition for Z-slab optimization
    logical           :: cond2    !! Second condition for Z-slab optimization
    integer(int32),   allocatable   :: comm_dims(:)   !! Dims in cartesian communicator
    integer(int8)     :: ndims      !! Number of dimensions
    integer(int32)    :: comm_size  !! Number of MPI processes
    integer(int32)    :: top_type   !! Topology type
    integer(int32)    :: ierr       !! Error code
    logical           :: is_custom_cart_comm  !! Custom cartesian communicator provided by user
    integer(int8)     :: d
    TYPE_MPI_COMM     :: base_comm_
    integer(int8) :: n_transpose_plans
    integer(int32), allocatable :: best_decomposition(:)
    logical :: pencils_created
    real(real64) :: ts, te
    integer(int8) :: best_forward_ids(3), best_backward_ids(3)
    logical :: invalid_grid_selected
    logical :: z_slab_from_conf, y_slab_from_conf
    logical :: exec_autotune

    error_code = DTFFT_SUCCESS
    CHECK_CALL( self%init(platform, effort), error_code )

    call MPI_Comm_size(base_comm, comm_size, ierr)
    call MPI_Topo_test(base_comm, top_type, ierr)
    base_comm_ = base_comm

    ndims = size(dims, kind=int8)
    allocate( comm_dims(ndims) )
    comm_dims(:) = 0

    is_custom_cart_comm = .false.
    self%is_z_slab = .false.; z_slab_from_conf = get_conf_z_slab_enabled()
    self%is_y_slab = .false.; y_slab_from_conf = get_conf_y_slab_enabled()

    invalid_grid_selected = .false.
    if ( present(ipencil) ) then
      is_custom_cart_comm = .true.
      do d = 1, ndims
        call MPI_Comm_size(ipencil%comms(d), comm_dims(d), ierr)
      enddo
      if ( comm_dims(1) /= 1 ) then
        error_code = DTFFT_ERROR_INVALID_COMM_FAST_DIM
        return
      endif
      if ( ndims == 3 ) then
        if (comm_dims(2) == 1 .and. z_slab_from_conf ) then
          self%is_z_slab = .true.
          base_comm_ = ipencil%comms(3)
        else if ( comm_dims(3) == 1 .and. y_slab_from_conf ) then
          self%is_y_slab = .true.
          base_comm_ = ipencil%comms(2)
        endif
      endif
    else ! ipencil not present
      if ( top_type == MPI_CART ) then
        is_custom_cart_comm = .true.
        block
          integer(int32)                 :: grid_ndims           ! Number of dims in user defined cartesian communicator
          integer(int32),  allocatable   :: temp_dims(:)         ! Temporary dims needed by MPI_Cart_get
          integer(int32),  allocatable   :: temp_coords(:)       ! Temporary coordinates needed by MPI_Cart_get
          logical,         allocatable   :: temp_periods(:)      ! Temporary periods needed by MPI_Cart_get

          call MPI_Cartdim_get(base_comm_, grid_ndims, ierr)
          if ( grid_ndims > ndims ) then
            error_code = DTFFT_ERROR_INVALID_COMM_DIMS
            return
          endif
          comm_dims(:) = 1
          allocate(temp_dims(grid_ndims), temp_periods(grid_ndims), temp_coords(grid_ndims))
          call MPI_Cart_get(base_comm_, grid_ndims, temp_dims, temp_periods, temp_coords, ierr)
          if ( grid_ndims == ndims ) then
            if ( temp_dims(1) /= 1 ) then
              error_code = DTFFT_ERROR_INVALID_COMM_FAST_DIM
              return
            endif
            comm_dims(:) = temp_dims
          elseif ( grid_ndims == ndims - 1 ) then
            comm_dims(2:) = temp_dims
          elseif ( grid_ndims == ndims - 2 ) then
            comm_dims(3) = temp_dims(1)
          endif
          deallocate(temp_dims, temp_periods, temp_coords)

          do d = 2, ndims
            if ( comm_dims(d) > dims(d) ) then
              WRITE_WARN("Number of MPI processes in direction "//to_str(d)//" greater then number of physical points: "//to_str(comm_dims(d))//" > "//to_str(dims(d)))
            endif
          enddo
          if ( ndims == 3 ) then
            if (comm_dims(2) == 1 .and. z_slab_from_conf ) then
              self%is_z_slab = .true.
            else if ( comm_dims(3) == 1 .and. y_slab_from_conf ) then
              self%is_y_slab = .true.
            endif
          endif
        endblock
      else !  top_type /= MPI_CART
        comm_dims(:) = 0
        comm_dims(1) = 1
        cond1 = comm_size <= dims(ndims)
        cond2 = comm_size <= dims(1) .and. comm_size <= dims(2)
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
          cond1 = DEF_TILE_SIZE <= dims(ndims) / comm_size
          cond2 = DEF_TILE_SIZE <= dims(1) / comm_size .and. DEF_TILE_SIZE <= dims(2) / comm_size
        endif
#endif
        if ( ndims == 3 ) then
          if ( cond1 .and. z_slab_from_conf ) then
            comm_dims(2) = 1
            comm_dims(3) = comm_size
            self%is_z_slab = .true.
          else if ( cond2 .and. y_slab_from_conf ) then
            comm_dims(2) = comm_size
            comm_dims(3) = 1
            self%is_y_slab = .true.
          else if ( cond1 ) then
            comm_dims(2) = 1
            comm_dims(3) = comm_size
          else if ( cond2 ) then
            comm_dims(2) = comm_size
            comm_dims(3) = 1
          endif
        endif
        call MPI_Dims_create(comm_size, int(ndims, int32), comm_dims, ierr)
        if(dims(ndims - 1) < comm_dims(ndims - 1) .or. dims(ndims) < comm_dims(ndims) ) then
          WRITE_WARN("Unable to create correct grid decomposition.")
          invalid_grid_selected = .true.
          ! WRITE_WARN("Fallback to Z slab is used")
          ! comm_dims(ndims - 1) = 1
          ! comm_dims(ndims) = comm_size
        endif
      endif
    endif

    error_code = DTFFT_SUCCESS
    if ( .not. ARE_DATATYPES_SET) then
      FORWARD_PLAN_IDS(1) = get_datatype_from_env("DTYPE_X_Y"); BACKWARD_PLAN_IDS(1) = get_datatype_from_env("DTYPE_Y_X")
      FORWARD_PLAN_IDS(2) = get_datatype_from_env("DTYPE_Y_Z"); BACKWARD_PLAN_IDS(2) = get_datatype_from_env("DTYPE_Z_Y")
      FORWARD_PLAN_IDS(3) = get_datatype_from_env("DTYPE_X_Z"); BACKWARD_PLAN_IDS(3) = get_datatype_from_env("DTYPE_Z_X")
      ARE_DATATYPES_SET = .true.
    endif

    best_forward_ids(:) = FORWARD_PLAN_IDS(:)
    best_backward_ids(:) = BACKWARD_PLAN_IDS(:)

    self%platform = platform
    self%backend = get_conf_backend()

    allocate( best_decomposition(ndims) )
    best_decomposition(:) = comm_dims(:)
    call MPI_Comm_size(base_comm_, comm_size, ierr)
    if ( comm_size == 1 .and. self%backend /= DTFFT_BACKEND_MPI_DATATYPE ) self%backend = BACKEND_NOT_SET

    pencils_created = .false.
    if ( ndims == 2 .or. is_custom_cart_comm .or. self%is_z_slab .or. self%is_y_slab ) then
      pencils_created = .true.
      call create_pencils_and_comm(dims, base_comm_, comm_dims, cart_comm, comms, pencils, ipencil=ipencil)
    endif

    ts = MPI_Wtime()

    exec_autotune = (effort%val >= DTFFT_MEASURE%val .and. .not.pencils_created ) .or. (effort%val >= DTFFT_PATIENT%val) .and. comm_size > 1
    if ( exec_autotune ) then
      PHASE_BEGIN("Autotune transpose plan", COLOR_AUTOTUNE)
      WRITE_INFO("Starting autotune of transpose plans...")
    endif

    if ( effort%val >= DTFFT_PATIENT%val .and. comm_size > 1 .and. .not.invalid_grid_selected) then
      if ( pencils_created ) then
        call run_autotune_backend(                                                                              &
          platform, comms, cart_comm, effort, base_dtype, pencils, base_storage, self%stream, self%is_z_slab,   &
          best_forward_ids, best_backward_ids, best_backend=self%backend)
      else
        call autotune_grid_decomposition(                                                                       &
          platform, dims, base_comm_, effort, base_dtype, base_storage,                        &
          self%stream, best_forward_ids, best_backward_ids, best_decomposition, best_backend=self%backend)
      endif
    else if ( ndims == 3                                &
      .and. .not.is_custom_cart_comm                    &
      .and. .not.self%is_z_slab                         &
      .and. .not.self%is_y_slab                         &
      .and. effort == DTFFT_MEASURE                     &
      .and. comm_size > 1 ) then

      call autotune_grid_decomposition(                                                                         &
        platform, dims, base_comm_, effort, base_dtype, base_storage,                          &
        self%stream, best_forward_ids, best_backward_ids, best_decomposition, backend=self%backend)
    endif

    if ( exec_autotune ) then
      PHASE_END("Autotune transpose plan")
    endif
    te = MPI_Wtime()

    if ( effort%val >= DTFFT_MEASURE%val .and. ndims > 2 .and. comm_size > 1 ) then
      if ( self%is_z_slab ) then
        WRITE_INFO("Skipped search of MPI processor grid due to Z-slab optimization enabled")
      else if ( self%is_y_slab ) then
        WRITE_INFO("Skipped search of MPI processor grid due to Y-slab optimization enabled")
      else if ( is_custom_cart_comm ) then
        WRITE_INFO("Skipped search of MPI processor grid due to custom grid provided")
      else if ( invalid_grid_selected ) then
        WRITE_INFO("Skipped search of MPI processor grid due to lack of work per process")
      else
        WRITE_INFO("DTFFT_MEASURE: Selected MPI processor grid 1x"//to_str(best_decomposition(2))//"x"//to_str(best_decomposition(3)))
      endif
    endif
    if ( effort%val >= DTFFT_PATIENT%val .and. comm_size > 1 ) then
      WRITE_INFO("DTFFT_PATIENT: Selected backend is "//dtfft_get_backend_string(self%backend))
    endif

    n_transpose_plans = ndims - 1_int8; if( self%is_z_slab ) n_transpose_plans = n_transpose_plans + 1_int8

    if ( effort%val >= DTFFT_PATIENT%val .and. self%backend == DTFFT_BACKEND_MPI_DATATYPE .and. comm_size > 1 ) then
      WRITE_INFO("DTFFT_PATIENT: Selected transpose ids:")
      do d = 1, n_transpose_plans
        WRITE_INFO("    "//TRANSPOSE_NAMES( d)//": "//to_str( best_forward_ids(d) ))
        WRITE_INFO("    "//TRANSPOSE_NAMES(-d)//": "//to_str( best_backward_ids(d) ))
      enddo
    endif
    if ( exec_autotune ) then
      WRITE_INFO("Time spent on autotune: "//to_str(te - ts)//" [s]")
    endif

    if ( .not.pencils_created ) then
      call create_pencils_and_comm(dims, base_comm_, best_decomposition, cart_comm, comms, pencils)
    endif
    ts = MPI_Wtime()
    allocate( self%plans(-1 * n_transpose_plans:n_transpose_plans) )
    call allocate_plans(self%plans, self%backend)
    call self%helper%create(platform, cart_comm, comms, is_backend_nccl(self%backend), pencils)

    block
      type(create_args) :: args

      args%platform = platform
      args%helper = self%helper
      args%effort = effort
      args%backend = self%backend
      args%force_effort = .false.
      args%base_type = base_dtype
      args%base_storage = base_storage

      do d = 1_int8, ndims - 1_int8
        args%datatype_id = best_forward_ids(d)
        call self%plans(d)%p%create(pencils(d), pencils(d + 1), args)
        args%datatype_id = best_backward_ids(d)
        call self%plans(-d)%p%create(pencils(d + 1), pencils(d), args)
      enddo
      if ( self%is_z_slab ) then
        args%datatype_id = best_forward_ids(3)
        call self%plans(3)%p%create(pencils(1), pencils(3), args)
        args%datatype_id = best_backward_ids(3)
        call self%plans(-3)%p%create(pencils(3), pencils(1), args)
      endif
    endblock
    te = MPI_Wtime()
    WRITE_INFO("Time spent creating final transpose plans: "//to_str(te - ts)//" [s]")

    call get_local_sizes(pencils, alloc_size=self%min_buffer_size)
    self%min_buffer_size = self%min_buffer_size * (base_storage / FLOAT_STORAGE_SIZE)
    allocate( self%names(CONF_DTFFT_TRANSPOSE_Z_TO_X:CONF_DTFFT_TRANSPOSE_X_TO_Z) )
    do d = CONF_DTFFT_TRANSPOSE_Z_TO_X, CONF_DTFFT_TRANSPOSE_X_TO_Z
      self%names(d) = string(TRANSPOSE_NAMES(d))
    end do
    deallocate( best_decomposition, comm_dims )
    error_code = DTFFT_SUCCESS
  end function create

  pure logical function get_z_slab(self)
  !! Returns .true. if Z-slab optimization is enabled
    class(transpose_plan),   intent(in)    :: self      !! Transposition class
    get_z_slab = self%is_z_slab
  end function get_z_slab

  pure logical function get_y_slab(self)
  !! Returns .true. if Y-slab optimization is enabled
    class(transpose_plan),   intent(in)    :: self      !! Transposition class
    get_y_slab = self%is_y_slab
  end function get_y_slab

  subroutine autotune_grid_decomposition(                                                             &
    platform, dims, base_comm, effort, base_dtype, base_storage, stream,             &
    best_forward_ids, best_backward_ids, best_decomposition, backend, min_execution_time, best_backend)
  !! Runs through all possible grid decompositions and selects the best one based on the lowest average execution time
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to use
    integer(int32),                   intent(in)    :: dims(:)
      !! Global sizes of the transform requested
    TYPE_MPI_COMM,                    intent(in)    :: base_comm
      !! 3D comm
    type(dtfft_effort_t),             intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
      !! Base MPI_Datatype
    integer(int64),                   intent(in)    :: base_storage
      !! Number of bytes needed to store single element
    type(dtfft_stream_t),             intent(in)    :: stream
      !! Stream to use
    integer(int8),                    intent(inout) :: best_forward_ids(:)
      !! Best Datatype ids for forward plan
    integer(int8),                    intent(inout) :: best_backward_ids(:)
      !! Best Datatype ids for backward plan
    integer(int32),                   intent(out)   :: best_decomposition(:)
      !! Best decomposition found
    type(dtfft_backend_t),  optional, intent(in)    :: backend
      !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
    real(real32),           optional, intent(out)   :: min_execution_time
      !! Elapsed time for best plan selected
    type(dtfft_backend_t),  optional, intent(out)   :: best_backend
      !! Best backend selected
    integer(int8)   :: ndims
    type(dtfft_backend_t) :: best_backend_
    integer(int32)  :: comm_size, square_root, i, current_timer, k, ierr
    real(real32) :: current_time, elapsed_time
    real(real32), allocatable :: timers(:)
    integer(int32),   allocatable :: decomps(:,:)
    type(dtfft_backend_t), allocatable :: backends(:)
    integer(int8),  allocatable :: forw_ids(:,:)        !! Best Forward ids are stored in forw_ids(:, latest_timer_id)
    integer(int8),  allocatable :: back_ids(:,:)        !! Best Backward ids are stored in back_ids(:, latest_timer_id)
    integer(int8) :: forw(3), back(3)

    call MPI_Comm_size(base_comm, comm_size, ierr)
    ndims = size(dims, kind=int8)

    square_root = int(sqrt(real(comm_size, real64))) + 1
    allocate(timers(2 * square_root))
    allocate(decomps(2, 2 * square_root))
    allocate(backends(2 * square_root))
    allocate(forw_ids(3, 2 * square_root))
    allocate(back_ids(3, 2 * square_root))

    current_timer = 0
    do i = 1, square_root - 1
      if ( mod( comm_size, i ) /= 0 ) cycle

      forw(:) = best_forward_ids(:)
      back(:) = best_backward_ids(:)

      call autotune_grid(                                                 &
        platform, dims, base_comm, effort, base_dtype,   &
        [1, i, comm_size / i], base_storage, stream, .false., forw, back, &
        backend=backend, best_time=current_time,best_backend=best_backend_)
      if ( current_time > 0.0 ) then
        current_timer = current_timer + 1
        timers(current_timer) = current_time
        decomps(1, current_timer) = i
        decomps(2, current_timer) = comm_size / i
        forw_ids(:, current_timer) = forw(:)
        back_ids(:, current_timer) = back(:)
        backends(current_timer) = best_backend_
      endif
      if ( i /= comm_size / i) then
        forw(:) = best_forward_ids(:)
        back(:) = best_backward_ids(:)

        call autotune_grid(                                                 &
          platform, dims, base_comm, effort, base_dtype,   &
          [1, comm_size / i, i], base_storage, stream, .false., forw, back, &
          backend=backend, best_time=current_time,best_backend=best_backend_)
        if ( current_time > 0.0 ) then
          current_timer = current_timer + 1
          timers(current_timer) = current_time
          decomps(1, current_timer) = comm_size / i
          decomps(2, current_timer) = i
          forw_ids(:, current_timer) = forw(:)
          back_ids(:, current_timer) = back(:)
          backends(current_timer) = best_backend_
        endif
      endif
    enddo

    elapsed_time = MAX_REAL32
    k = 1
    do i = 1, current_timer
      if ( timers(i) < elapsed_time ) then
        elapsed_time = timers(i)
        k = i
      endif
    enddo

    best_decomposition(1) = 1
    best_decomposition(2) = decomps(1, k)
    best_decomposition(3) = decomps(2, k)

    best_forward_ids(:) = forw_ids(:, k)
    best_backward_ids(:) = back_ids(:, k)
    if ( present(best_backend) ) best_backend = backends(k)
    if ( present(min_execution_time) ) min_execution_time = elapsed_time
    deallocate( timers, decomps, backends, forw_ids, back_ids )
  end subroutine autotune_grid_decomposition

  subroutine autotune_grid(                                                                                       &
    platform, dims, base_comm, effort, base_dtype, comm_dims, base_storage, stream, is_z_slab,   &
    best_forward_ids, best_backward_ids, backend, best_time, best_backend)
  !! Creates cartesian grid and runs various backends on it. Returns best backend and execution time
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to create plan for
    integer(int32),                   intent(in)    :: dims(:)
      !! Global sizes of the transform requested
    TYPE_MPI_COMM,                    intent(in)    :: base_comm
      !! Basic communicator to create 3d grid from
    type(dtfft_effort_t),             intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
      !! Base MPI_Datatype
    integer(int32),                   intent(in)    :: comm_dims(:)
      !! Number of processors in each dimension
    integer(int64),                   intent(in)    :: base_storage
      !! Number of bytes needed to store single element
    type(dtfft_stream_t),             intent(in)    :: stream
      !! Stream to use
    logical,                          intent(in)    :: is_z_slab
      !! Is Z-slab optimization enabled
    integer(int8),                    intent(inout) :: best_forward_ids(:)
      !! Best Datatype ids for forward plan
    integer(int8),                    intent(inout) :: best_backward_ids(:)
      !! Best Datatype ids for backward plan
    type(dtfft_backend_t),  optional, intent(in)    :: backend
      !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
    type(dtfft_backend_t),  optional, intent(out)   :: best_backend
      !! Best backend selected for the grid
    real(real32),           optional, intent(out)   :: best_time
      !! Elapsed time for best plan selected
    type(pencil), allocatable :: pencils(:)
    character(len=:),             allocatable   :: phase_name           !! Caliper phase name
    integer(int8) :: d, ndims
    TYPE_MPI_COMM, allocatable :: comms(:)
    TYPE_MPI_COMM :: cart_comm
    integer(int32) :: mpi_ierr

    best_time = -1.0
    ndims = size(dims, kind=int8)

    if ( ndims == 3 ) then
      if ( comm_dims(2) > dims(2) .or. comm_dims(3) > dims(3) ) return
      allocate( phase_name, source = "Testing grid 1x"//to_str(comm_dims(2))//"x"//to_str(comm_dims(3)) )
    else
      allocate( phase_name, source = "Testing grid 1x"//to_str(comm_dims(2)) )
    endif

    WRITE_INFO("")
    WRITE_INFO(phase_name)
    REGION_BEGIN(phase_name, COLOR_AUTOTUNE)

    allocate( comms(ndims), pencils(ndims) )
    call create_pencils_and_comm(dims, base_comm, comm_dims, cart_comm, comms, pencils)

    call run_autotune_backend(                                                                            &
      platform, comms, cart_comm, effort, base_dtype, pencils, base_storage, stream, is_z_slab,           &
      best_forward_ids, best_backward_ids, backend=backend, best_time=best_time, best_backend=best_backend)

    do d = 1, ndims
      call pencils(d)%destroy()
      call MPI_Comm_free(comms(d), mpi_ierr)
    enddo
    call MPI_Comm_free(cart_comm, mpi_ierr)
    deallocate( comms, pencils )
    REGION_END(phase_name)
  end subroutine autotune_grid

  subroutine run_autotune_backend(                                                                &
    platform, comms, cart_comm, effort, base_dtype, pencils, base_storage, stream, is_z_slab,     &
    best_forward_ids, best_backward_ids, backend, best_time, best_backend)
  !! Runs autotune for all backends
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to create plan for
    TYPE_MPI_COMM,                    intent(in)    :: comms(:)
      !! 1D comms
    TYPE_MPI_COMM,                    intent(in)    :: cart_comm
      !! 3D Cartesian comm
    type(dtfft_effort_t),             intent(in)    :: effort
      !!
    TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
      !! Base MPI_Datatype
    type(pencil),                     intent(in)    :: pencils(:)
      !! Source meta
    integer(int64),                   intent(in)    :: base_storage
      !! Number of bytes needed to store single element
    type(dtfft_stream_t),             intent(in)    :: stream
      !! Stream to use
    logical,                          intent(in)    :: is_z_slab
      !! Is Z-slab optimization enabled
    integer(int8),                    intent(inout) :: best_forward_ids(:)
      !! Best Datatype ids for forward plan
    integer(int8),                    intent(inout) :: best_backward_ids(:)
      !! Best Datatype ids for backward plan
    type(dtfft_backend_t),  optional, intent(in)    :: backend
      !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
    real(real32),           optional, intent(out)   :: best_time
      !! Elapsed time for best backend
    type(dtfft_backend_t),  optional, intent(out)   :: best_backend
      !! Best backend selected
    type(dtfft_backend_t),  allocatable :: backends_to_run(:)
    type(dtfft_backend_t) :: current_backend_id, best_backend_
    logical :: is_udb !! Used defined backend
    real(real32) :: execution_time, best_time_
    integer(int32) :: b
    type(reshape_container), allocatable :: plans(:)
    integer(int8) :: i, n_transpose_plans
    logical :: is_aux_alloc
    integer(int64)         :: alloc_size
    logical :: nccl_enabled
#ifdef DTFFT_WITH_CUDA
    logical :: nvshmem_enabled
#endif
    character(len=:), allocatable :: testing_phase
    type(backend_helper)                      :: helper
    integer(int64) :: min_buffer_size
    type(create_args) :: create_kwargs
    logical :: pipe_enabled, mpi_enabled, dtype_enabled

    if ( present(backend) ) then
      allocate( backends_to_run(1) )
      backends_to_run(1) = backend
      is_udb = .true.
    else
      allocate( backends_to_run, source=VALID_BACKENDS )
      is_udb = .false.
    endif
    best_backend_ = backends_to_run(1)

    if ( is_z_slab ) then
      n_transpose_plans = 1
    else
      n_transpose_plans = size(pencils, kind=int8) - 1_int8
    endif

    allocate( plans(2 * n_transpose_plans) )

    call get_local_sizes(pencils, alloc_size=alloc_size)
    alloc_size = alloc_size * base_storage
    min_buffer_size = alloc_size / FLOAT_STORAGE_SIZE

    pipe_enabled = get_conf_pipelined_enabled()
    dtype_enabled = get_conf_datatype_enabled()
    mpi_enabled = get_conf_mpi_enabled()
    nccl_enabled = .false.
#ifdef DTFFT_WITH_CUDA
    nccl_enabled = ( platform == DTFFT_PLATFORM_CUDA .and. get_conf_nccl_enabled() ) .or. ( is_udb .and. is_backend_nccl( backends_to_run(1) ) )
    nvshmem_enabled = get_conf_nvshmem_enabled()
#endif

    call helper%create(platform, cart_comm, comms, nccl_enabled, pencils)

    create_kwargs%effort = DTFFT_ESTIMATE
    create_kwargs%force_effort = .true.
    create_kwargs%platform = platform
    create_kwargs%helper = helper
    create_kwargs%base_type = base_dtype
    create_kwargs%base_storage = base_storage

    best_time_ = MAX_REAL32

    do b = 1, size(backends_to_run)
      current_backend_id = backends_to_run(b)
      if ( .not.is_udb) then
        if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. platform == DTFFT_PLATFORM_CUDA ) cycle
        if ( is_backend_pipelined(current_backend_id) .and. .not.pipe_enabled ) cycle
        if ( is_backend_mpi(current_backend_id) .and. .not.mpi_enabled .and. .not.current_backend_id == DTFFT_BACKEND_MPI_DATATYPE ) cycle
        if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. .not.dtype_enabled ) cycle
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
          if ( is_backend_nvshmem(current_backend_id) .and. .not.nvshmem_enabled ) cycle
          if ( is_backend_nccl(current_backend_id) .and. .not.nccl_enabled) cycle
        else
          if ( is_backend_nccl(current_backend_id) .or. is_backend_nvshmem(current_backend_id) ) cycle
        endif
#endif
      endif

      call allocate_plans(plans, current_backend_id)

      testing_phase = "Testing backend "//dtfft_get_backend_string(current_backend_id)
      REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
      WRITE_INFO(testing_phase)

      is_aux_alloc = .false.
      if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. effort%val >= DTFFT_PATIENT%val) then
        call run_autotune_datatypes(create_kwargs, alloc_size, helper, pencils, is_z_slab, best_forward_ids, best_backward_ids, execution_time)
      else
        create_kwargs%backend = current_backend_id

        if ( is_z_slab ) then
          create_kwargs%datatype_id = best_forward_ids(3)
          call plans(1)%p%create(pencils(1), pencils(3), create_kwargs)
          create_kwargs%datatype_id = best_backward_ids(3)
          call plans(2)%p%create(pencils(3), pencils(1), create_kwargs)
        else
          do i = 1, n_transpose_plans
            create_kwargs%datatype_id = best_forward_ids(i)
            call plans(i)%p%create(pencils(i), pencils(i + 1), create_kwargs)
            create_kwargs%datatype_id = best_backward_ids(i)
            call plans(i + n_transpose_plans)%p%create(pencils(i + 1), pencils(i), create_kwargs)
          enddo
        endif
        execution_time = execute_autotune(plans, cart_comm, current_backend_id, platform, helper, stream, alloc_size)
      endif

      if ( execution_time < best_time_ ) then
        best_time_ = execution_time
        best_backend_ = current_backend_id
      endif

      call destroy_plans(plans)
      REGION_END("Testing backend "//dtfft_get_backend_string(current_backend_id))
    enddo

    deallocate( plans )
    call helper%destroy()

    if ( present(best_time)) best_time = best_time_
    if ( present(best_backend) ) best_backend = best_backend_
  end subroutine run_autotune_backend

  subroutine run_autotune_datatypes(                                                              &
    create_kwargs, buffer_size, helper, pencils,                                                    &
    is_z_slab, best_forward_ids, best_backward_ids, elapsed_time)
    type(create_args),                intent(inout) :: create_kwargs
    integer(int64),                   intent(in)    :: buffer_size
    type(backend_helper),             intent(inout) :: helper
      !! Base MPI_Datatype
    type(pencil),                     intent(in)    :: pencils(:)
      !! Source meta
    logical,                          intent(in)    :: is_z_slab
      !! Is Z-slab optimization enabled
    integer(int8),                    intent(out)   :: best_forward_ids(:)
      !! Best Datatype ids for forward plan
    integer(int8),                    intent(out)   :: best_backward_ids(:)
      !! Best Datatype ids for backward plan
    real(real32),                     intent(out)   :: elapsed_time
      !! Elapsed time for best plans selected in [ms]
    integer(int8) :: dim        !! Counter
    integer(int8) :: ndims      !! Number of dimensions

    ndims = size(pencils, kind=int8)
    elapsed_time = 0._real32
    if( is_z_slab ) then
      elapsed_time = autotune_transpose_id(create_kwargs, buffer_size, helper, pencils(1), pencils(3),                        &
        3_int8, best_forward_ids(3), best_backward_ids(3))
    else
      do dim = 1_int8, ndims - 1_int8
        elapsed_time = elapsed_time                                                               &
        + autotune_transpose_id(create_kwargs, buffer_size, helper, pencils(dim), pencils(dim + 1),                           &
            dim, best_forward_ids(dim), best_backward_ids(dim))
      enddo
    endif
    WRITE_INFO("  Execution time on a grid using fastest transpositions: "//to_str(real(elapsed_time, real64))//" [ms]")
  end subroutine run_autotune_datatypes

  function autotune_transpose_id(create_kwargs, buffer_size, helper, from, to, transpose_name_id, forward_id, backward_id) result(elapsed_time)
  !! Creates forward and backward transpose plans for backend `DTFFT_BACKEND_MPI_DATATYPE` based on source and target data distributions and,
  !! executes them `DTFFT_MEASURE_ITERS` times ( 4 * `DTFFT_MEASURE_ITERS` iterations total ) + 4 * `DTFFT_MEASURE_WARMUP_ITERS` warmup iterations
  !!
  !! Returns elapsed time for best plans selected
    type(create_args),            intent(inout) :: create_kwargs
    integer(int64),               intent(in)    :: buffer_size
    type(backend_helper),         intent(inout) :: helper               !! Backend helper
    type(pencil),                 intent(in)    :: from                 !! Source meta
    type(pencil),                 intent(in)    :: to                   !! Target meta
    integer(int8),                intent(in)    :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
    integer(int8),                intent(out)   :: forward_id           !! Best forward plan ID
    integer(int8),                intent(out)   :: backward_id          !! Best backward plan ID
    real(real32)                                :: elapsed_time         !! Elapsed time for best plans selected
    real(real32)                                :: forward_time         !! Forward plan execution time
    real(real32)                                :: backward_time        !! Backward plan execution time
    real(real32)                                :: time                 !! Timer
    integer(int8)                               :: datatype_id          !! Counter

    forward_time = huge(1._real32)
    backward_time = huge(1._real32)

    do datatype_id = 1, 2
      time = get_plan_execution_time(create_kwargs, buffer_size, helper, from, to, datatype_id, transpose_name_id)
      if ( time < forward_time ) then
        forward_time = time
        forward_id = datatype_id
      endif

      time = get_plan_execution_time(create_kwargs, buffer_size, helper, to, from, datatype_id, -1_int8 * transpose_name_id)
      if ( time < backward_time ) then
        backward_time = time
        backward_id = datatype_id
      endif
    enddo
    elapsed_time = forward_time + backward_time
  end function autotune_transpose_id

  function get_plan_execution_time(create_kwargs, buffer_size, helper, from, to,  datatype_id, transpose_name_id) result(elapsed_time)
  !! Creates transpose plan for backend `DTFFT_BACKEND_MPI_DATATYPE` and executes it `DTFFT_MEASURE_WARMUP_ITERS` + `DTFFT_MEASURE_ITERS` times
  !!
  !! Returns elapsed time
    type(create_args),            intent(inout) :: create_kwargs
    integer(int64),               intent(in)    :: buffer_size
    type(backend_helper),         intent(inout) :: helper               !! Backend helper
    type(pencil),                 intent(in)    :: from                 !! Source meta
    type(pencil),                 intent(in)    :: to                   !! Target meta
    integer(int8),                intent(in)    :: datatype_id          !! ID of transpose (1 or 2)
    integer(int8),                intent(in)    :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
    real(real32)                                :: elapsed_time         !! Execution time [ms]
    character(len=:),             allocatable   :: phase_name           !! Caliper phase name
    type(reshape_container) :: plan(1)

    allocate( phase_name, source="  Testing plan "//TRANSPOSE_NAMES(transpose_name_id)//", datatype_id = "//to_str(datatype_id) )
    REGION_BEGIN(phase_name, 0)
    WRITE_INFO(phase_name)

    create_kwargs%datatype_id = datatype_id

    call allocate_plans(plan, DTFFT_BACKEND_MPI_DATATYPE)

    call plan(1)%p%create(from, to, create_kwargs)
    elapsed_time = execute_autotune(plan, helper%comms(1), DTFFT_BACKEND_MPI_DATATYPE, DTFFT_PLATFORM_HOST, helper, NULL_STREAM, buffer_size, 4)

    call destroy_plans(plan)
    REGION_END(phase_name)
    deallocate(phase_name)
  end function get_plan_execution_time

  subroutine get_permutations(ndims, dperm, cperm)
    integer(int8),  intent(in)  :: ndims
    integer(int8), allocatable :: dperm(:,:)
    integer(int8), allocatable :: cperm(:,:)

    allocate(dperm(ndims, ndims), cperm(ndims, ndims))
    if ( ndims == 2_int8 ) then
      dperm(1, 1) = 1
      dperm(2, 1) = 2
      cperm(:, 1) = dperm(:, 1)

      dperm(1, 2) = 2
      dperm(2, 2) = 1

      cperm(:, 2) = cperm(:, 1)
    else
      dperm(1, 1) = 1
      dperm(2, 1) = 2
      dperm(3, 1) = 3
      cperm(:, 1) = dperm(:, 1)

      dperm(1, 2) = 2
      dperm(2, 2) = 3
      dperm(3, 2) = 1

      cperm(1, 2) = 1
      cperm(2, 2) = 3
      cperm(3, 2) = 2

      dperm(1, 3) = 3
      dperm(2, 3) = 1
      dperm(3, 3) = 2

      cperm(:, 3) = cperm(:, 1)
    endif
  end subroutine get_permutations

  subroutine create_pencils_and_comm(dims, old_comm, comm_dims, comm, local_comms, pencils, ipencil)
  !! Creates cartesian communicator
    integer(int32),       intent(in)            :: dims(:)              !! Global dimensions
    TYPE_MPI_COMM,        intent(in)            :: old_comm             !! Communicator to create cartesian from
    integer(int32),       intent(in)            :: comm_dims(:)         !! Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: comm                 !! Cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: local_comms(:)       !! 1d communicators in cartesian communicator
    type(pencil),         intent(out)           :: pencils(:)           !! Data distributing meta
    type(pencil_init),    intent(in), optional  :: ipencil              !! Pencil passed by user
    integer(int8)         :: ndims              !! Number of dimensions
    integer(int8)         :: d                  !! Counter
    integer(int8) :: i, j
    integer(int8),  allocatable :: dperm(:,:), cperm(:, :)
    integer(int32), allocatable :: transposed_dims(:,:)
    TYPE_MPI_COMM,  allocatable :: transposed_comms(:,:)
    integer(int32), allocatable :: lstarts(:), lcounts(:)

    ndims = size(comm_dims, kind=int8)
    call create_cart_comm(old_comm, comm_dims, comm, local_comms, ipencil=ipencil)
    call get_permutations(ndims, dperm, cperm)

    allocate( transposed_dims(ndims, ndims), transposed_comms(ndims, ndims) )
    do i = 1, ndims
      do j = 1, ndims
        transposed_dims(j, i) = dims(dperm(j, i))
        transposed_comms(j, i) = local_comms(cperm(j, i))
      enddo
    enddo
    deallocate( dperm, cperm )

    if ( present(ipencil) ) then
      allocate(lstarts, source=ipencil%starts)
      allocate(lcounts, source=ipencil%counts)
      do d = 1, ndims
        call pencils(d)%create(ndims, d, transposed_dims(:, d), transposed_comms(:, d), lstarts=lstarts, lcounts=lcounts)
        lcounts(:) = pencils(d)%counts(:)
        lstarts(:) = pencils(d)%starts(:)
      enddo

      deallocate(lstarts, lcounts)
    else
      do d = 1, ndims
        call pencils(d)%create(ndims, d, transposed_dims(:,d), transposed_comms(:, d))
      enddo
    endif

    deallocate(transposed_dims, transposed_comms)
  end subroutine create_pencils_and_comm

  subroutine create_cart_comm(old_comm, comm_dims, comm, local_comms, ipencil)
  !! Creates cartesian communicator
    TYPE_MPI_COMM,        intent(in)            :: old_comm             !! Communicator to create cartesian from
    integer(int32),       intent(in)            :: comm_dims(:)         !! Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: comm                 !! Cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: local_comms(:)       !! 1d communicators in cartesian communicator
    type(pencil_init),    intent(in), optional  :: ipencil              !! Pencil passed by user
    logical,              allocatable   :: periods(:)           !! Grid is not periodic
    logical,              allocatable   :: remain_dims(:)       !! Needed by MPI_Cart_sub
    integer(int8)                       :: dim                  !! Counter
    integer(int32)                      :: ierr                 !! Error code
    integer(int8)                       :: ndims
    TYPE_MPI_COMM              :: temp_cart_comm
    TYPE_MPI_COMM, allocatable :: temp_comms(:)

    ndims = size(comm_dims, kind=int8)

    if ( present( ipencil ) ) then
      call MPI_Comm_dup(old_comm, comm, ierr)
      do dim = 1, ndims
        call MPI_Comm_dup(ipencil%comms(dim), local_comms(dim), ierr)
      enddo
      return
    endif
    allocate(periods(ndims), source = .false.)

    call MPI_Cart_create(old_comm, int(ndims, int32), comm_dims, periods, .true., temp_cart_comm, ierr)
    call create_subcomm_include_all(temp_cart_comm, comm)
    if ( GET_MPI_VALUE(comm) == GET_MPI_VALUE(MPI_COMM_NULL) ) INTERNAL_ERROR("comm == MPI_COMM_NULL")

    allocate(temp_comms(ndims))

    allocate( remain_dims(ndims), source = .false. )
    do dim = 1, ndims
      remain_dims(dim) = .true.
      call MPI_Cart_sub(temp_cart_comm, remain_dims, temp_comms(dim), ierr)
      call create_subcomm_include_all(temp_comms(dim), local_comms(dim))
      if ( GET_MPI_VALUE(local_comms(dim)) == GET_MPI_VALUE(MPI_COMM_NULL) ) INTERNAL_ERROR("local_comms(dim) == MPI_COMM_NULL: dim = "//to_str(dim))
      remain_dims(dim) = .false.
    enddo
    call MPI_Comm_free(temp_cart_comm, ierr)
    do dim = 1, ndims
      call MPI_Comm_free(temp_comms(dim), ierr)
    enddo
    deallocate(temp_comms)
    deallocate(remain_dims, periods)
  end subroutine create_cart_comm
end module dtfft_transpose_plan
