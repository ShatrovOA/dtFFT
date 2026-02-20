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
#ifdef DTFFT_WITH_COMPRESSION
use dtfft_abstract_compressor
#endif
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
        ! integer(int8) :: best_forward_ids(3), best_backward_ids(3)
        logical :: invalid_grid_selected
        logical :: z_slab_from_conf, y_slab_from_conf
        logical :: exec_autotune
        type(dtfft_transpose_mode_t) :: transpose_mode, best_transpose_mode
        type(dtfft_transpose_mode_t) :: fmodes(3), bmodes(3)
        type(dtfft_backend_t)        :: fbacks(3), bbacks(3)

        error_code = DTFFT_SUCCESS
#ifdef DTFFT_WITH_COMPRESSION
        CHECK_CALL( self%init(platform, effort, get_conf_backend(), DTFFT_PATIENT, get_conf_transpose(), base_dtype, base_storage), error_code )
#else
        CHECK_CALL( self%init(platform, effort, get_conf_backend(), DTFFT_PATIENT), error_code )
#endif

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
                endif
            endif
        endif

        transpose_mode = get_conf_transpose_mode()
        best_transpose_mode = transpose_mode
        fmodes(:) = transpose_mode
        bmodes(:) = transpose_mode

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
            if ( self%backend == DTFFT_BACKEND_ADAPTIVE ) then
                if ( pencils_created ) then
                    call run_autotune_backend(                                                                                                                  &
                        platform, comms, cart_comm, effort, transpose_mode, base_dtype, pencils, base_storage, self%stream, self%is_z_slab, self%is_y_slab,     &
                        fmodes, bmodes, backend=self%backend, fbacks=fbacks, bbacks=bbacks)
                else
                    call autotune_grid_decomposition(                                                                                                           &
                        platform, dims, base_comm_, effort, transpose_mode, base_dtype, base_storage,                                                           &
                        self%stream, fmodes, bmodes, best_decomposition, backend=self%backend, fbacks=fbacks, bbacks=bbacks)
                endif
            else
                if ( pencils_created ) then
                    call run_autotune_backend(                                                                                                                  &
                        platform, comms, cart_comm, effort, transpose_mode, base_dtype, pencils, base_storage, self%stream, self%is_z_slab, self%is_y_slab,     &
                        fmodes, bmodes, best_backend=self%backend, best_transpose_mode=best_transpose_mode)
                else
                    call autotune_grid_decomposition(                                                                                                           &
                        platform, dims, base_comm_, effort, transpose_mode, base_dtype, base_storage,                                                           &
                        self%stream, fmodes, bmodes, best_decomposition, best_backend=self%backend, best_transpose_mode=best_transpose_mode)
                endif
            endif
        else if ( ndims == 3                                                                                                                                    &
                .and. .not.is_custom_cart_comm                                                                                                                  &
                .and. .not.self%is_z_slab                                                                                                                       &
                .and. .not.self%is_y_slab                                                                                                                       &
                .and. effort == DTFFT_MEASURE                                                                                                                   &
                .and. comm_size > 1 ) then

            call autotune_grid_decomposition(                                                                                                                   &
                platform, dims, base_comm_, effort, transpose_mode, base_dtype, base_storage,                                                                   &
                self%stream, fmodes, bmodes, best_decomposition, backend=self%backend)
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
            WRITE_INFO("DTFFT_PATIENT: Selected transpose modes:")
            do d = 1, n_transpose_plans
                WRITE_INFO("    "//TRANSPOSE_NAMES( d)//": "//trim(TRANSPOSE_MODE_NAMES(fmodes(d)%val)))
                WRITE_INFO("    "//TRANSPOSE_NAMES(-d)//": "//trim(TRANSPOSE_MODE_NAMES(bmodes(d)%val)))
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
        if ( self%backend == DTFFT_BACKEND_ADAPTIVE ) then
            block
                type(dtfft_backend_t), allocatable :: backs(:)

                allocate( backs(-1 * n_transpose_plans:n_transpose_plans) )
                do d = 1, n_transpose_plans
                    backs(-1 * n_transpose_plans + d - 1) = bbacks(n_transpose_plans - d + 1)
                    backs(d) = fbacks(d)
                enddo
                backs(0) = BACKEND_NOT_SET

                call allocate_plans(self%plans, backends=backs)

                deallocate(backs)
            endblock
        else
            call allocate_plans(self%plans, self%backend)
        endif
        call self%helper%create(platform, cart_comm, comms, is_backend_nccl(self%backend), pencils)

        block
            type(create_args) :: args
            logical :: is_diff_transpose_mode, is_adaptive

            args%platform = platform
            args%helper = self%helper
            args%effort = effort
            args%backend = self%backend
            args%force_effort = .false.
            args%base_type = base_dtype
            args%base_storage = base_storage
            args%transpose_mode = best_transpose_mode
#ifdef DTFFT_WITH_COMPRESSION
            args%compression_config = get_conf_transpose()
#endif
            is_adaptive = self%backend == DTFFT_BACKEND_ADAPTIVE
            is_diff_transpose_mode = is_adaptive .or. self%backend == DTFFT_BACKEND_MPI_DATATYPE

            do d = 1_int8, ndims - 1_int8
                if ( is_diff_transpose_mode ) args%transpose_mode = fmodes(d)
                if ( is_adaptive ) args%backend = fbacks(d)
                call self%plans(d)%p%create(pencils(d), pencils(d + 1), args)
                if ( is_diff_transpose_mode ) args%transpose_mode = bmodes(d)
                if ( is_adaptive ) args%backend = bbacks(d)
                call self%plans(-d)%p%create(pencils(d + 1), pencils(d), args)
            enddo
            if ( self%is_z_slab ) then
                if ( is_diff_transpose_mode ) args%transpose_mode = fmodes(3)
                if ( is_adaptive ) args%backend = fbacks(3)
                call self%plans(3)%p%create(pencils(1), pencils(3), args)
                if ( is_diff_transpose_mode ) args%transpose_mode = bmodes(3)
                if ( is_adaptive ) args%backend = bbacks(3)
                call self%plans(-3)%p%create(pencils(3), pencils(1), args)
            endif
        endblock
        te = MPI_Wtime()
        WRITE_INFO("Time spent on creating final transpose plans: "//to_str(te - ts)//" [s]")

        call get_local_sizes(pencils, alloc_size=self%min_buffer_size)
        self%min_buffer_size = self%min_buffer_size * (base_storage / FLOAT_STORAGE_SIZE)
        allocate( self%names(CONF_DTFFT_TRANSPOSE_Z_TO_X:CONF_DTFFT_TRANSPOSE_X_TO_Z), self%colors(CONF_DTFFT_TRANSPOSE_Z_TO_X:CONF_DTFFT_TRANSPOSE_X_TO_Z) )
        do d = CONF_DTFFT_TRANSPOSE_Z_TO_X, CONF_DTFFT_TRANSPOSE_X_TO_Z
            self%names(d) = string("Transpose "//TRANSPOSE_NAMES(d))
            self%colors(d) = COLOR_TRANSPOSE_PALLETTE(d)
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

    subroutine autotune_grid_decomposition(                                                                 &
        platform, dims, base_comm, effort, transpose_mode, base_dtype, base_storage, stream,                &
        fmodes, bmodes, best_decomposition, backend, min_execution_time, best_backend, best_transpose_mode, &
        fbacks, bbacks)
    !! Runs through all possible grid decompositions and selects the best one based on the lowest average execution time
        type(dtfft_platform_t),           intent(in)    :: platform
        !! Platform to use
        integer(int32),                   intent(in)    :: dims(:)
        !! Global sizes of the transform requested
        TYPE_MPI_COMM,                    intent(in)    :: base_comm
        !! 3D comm
        type(dtfft_effort_t),             intent(in)    :: effort
        !! Effort level for the plan creation
        type(dtfft_transpose_mode_t),     intent(in)    :: transpose_mode
        !! Transpose mode to use
        TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
        !! Base MPI_Datatype
        integer(int64),                   intent(in)    :: base_storage
        !! Number of bytes needed to store single element
        type(dtfft_stream_t),             intent(in)    :: stream
        !! Stream to use
        type(dtfft_transpose_mode_t),     intent(inout) :: fmodes(:)
            !! Best transpose modes for forward plan
        type(dtfft_transpose_mode_t),     intent(inout) :: bmodes(:)
            !! Best transpose modes for backward plan
        integer(int32),                   intent(out)   :: best_decomposition(:)
        !! Best decomposition found
        type(dtfft_backend_t),  optional, intent(in)    :: backend
        !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
        real(real32),           optional, intent(out)   :: min_execution_time
        !! Elapsed time for best plan selected
        type(dtfft_backend_t),  optional, intent(out)   :: best_backend
        !! Best backend selected
        type(dtfft_transpose_mode_t), optional, intent(out) :: best_transpose_mode
        !! Best transpose mode selected
        type(dtfft_backend_t),  optional, intent(out)   :: fbacks(:)
        !! Best backends for forward plans
        type(dtfft_backend_t),  optional, intent(out)   :: bbacks(:)
        !! Best backends for backward plans
        integer(int8)   :: ndims
        type(dtfft_backend_t) :: best_backend_
        type(dtfft_transpose_mode_t) :: best_transpose_mode_
        integer(int32)  :: comm_size, square_root, i, current_timer, k, ierr
        real(real32) :: current_time, elapsed_time
        real(real32), allocatable :: timers(:)
        integer(int32),   allocatable :: decomps(:,:)
        type(dtfft_backend_t), allocatable :: backends(:)
        type(dtfft_transpose_mode_t), allocatable :: transpose_modes(:)
        type(dtfft_transpose_mode_t),  allocatable :: forw_modes(:,:)        !! Best Forward ids are stored in forw_ids(:, latest_timer_id)
        type(dtfft_transpose_mode_t),  allocatable :: back_modes(:,:)        !! Best Backward ids are stored in back_ids(:, latest_timer_id)
        type(dtfft_transpose_mode_t) :: forw(3), back(3)
        type(dtfft_backend_t), allocatable :: fbacks_(:,:)
        type(dtfft_backend_t), allocatable :: bbacks_(:,:)
        type(dtfft_backend_t) :: fbacks_local(3), bbacks_local(3)

        call MPI_Comm_size(base_comm, comm_size, ierr)
        ndims = size(dims, kind=int8)

        square_root = int(sqrt(real(comm_size, real64))) + 1
        allocate(timers(2 * square_root))
        allocate(decomps(2, 2 * square_root))
        allocate(backends(2 * square_root))
        allocate(transpose_modes(2 * square_root))
        allocate(forw_modes(3, 2 * square_root))
        allocate(back_modes(3, 2 * square_root))
        allocate(fbacks_(3, 2 * square_root))
        allocate(bbacks_(3, 2 * square_root))

        current_timer = 0
        do i = 1, square_root - 1
            if ( mod( comm_size, i ) /= 0 ) cycle

            forw(:) = fmodes(:)
            back(:) = bmodes(:)

            call autotune_grid(                                                                                                                         &
                platform, dims, base_comm, effort, transpose_mode, base_dtype, [1, i, comm_size / i], base_storage, stream, forw, back,        &
                backend=backend, best_time=current_time, best_backend=best_backend_, best_transpose_mode=best_transpose_mode_, fbacks=fbacks_local, bbacks=bbacks_local)
            if ( current_time > 0.0 ) then
                current_timer = current_timer + 1
                timers(current_timer) = current_time
                decomps(1, current_timer) = i
                decomps(2, current_timer) = comm_size / i
                forw_modes(:, current_timer) = forw(:)
                back_modes(:, current_timer) = back(:)
                backends(current_timer) = best_backend_
                transpose_modes(current_timer) = best_transpose_mode_
                fbacks_(:, current_timer) = fbacks_local(:)
                bbacks_(:, current_timer) = bbacks_local(:)
            endif

            if ( i /= comm_size / i) then
                forw(:) = fmodes(:)
                back(:) = bmodes(:)

                call autotune_grid(                                                                                                                     &
                    platform, dims, base_comm, effort, transpose_mode, base_dtype, [1, comm_size / i, i], base_storage, stream, forw, back,    &
                    backend=backend, best_time=current_time, best_backend=best_backend_, best_transpose_mode=best_transpose_mode_, fbacks=fbacks_local, bbacks=bbacks_local)
                if ( current_time > 0.0 ) then
                    current_timer = current_timer + 1
                    timers(current_timer) = current_time
                    decomps(1, current_timer) = comm_size / i
                    decomps(2, current_timer) = i
                    forw_modes(:, current_timer) = forw(:)
                    back_modes(:, current_timer) = back(:)
                    backends(current_timer) = best_backend_
                    transpose_modes(current_timer) = best_transpose_mode_
                    fbacks_(:, current_timer) = fbacks_local(:)
                    bbacks_(:, current_timer) = bbacks_local(:)
                endif
            endif
        enddo

        ! This may happen if all `autotune_grid` return without executing due to grid constrains
        if ( current_timer == 0 ) then
            WRITE_WARN("Unable to execute grid autotuning. Setting default values")
            current_timer = 1
            elapsed_time = 0.0_real32

            call MPI_Dims_create(comm_size, 2, decomps(:, 1), ierr)
            if ( present( backend ) ) then
                if ( backend /= DTFFT_BACKEND_ADAPTIVE ) then
                    backends(1) = backend
                else
                    backends(1) = get_correct_backend(BACKEND_NOT_SET)
                endif
            else
                backends(1) = get_correct_backend(BACKEND_NOT_SET)
            endif
            if ( present(fbacks) .and. present(bbacks) ) then
                fbacks_(:, 1) = BACKEND_NOT_SET;    fbacks_(:, 1) = get_correct_backend(fbacks_(:, 1))
                bbacks_(:, 1) = BACKEND_NOT_SET;    bbacks_(:, 1) = get_correct_backend(bbacks_(:, 1))
            endif
            forw_modes(:, 1) = transpose_mode
            back_modes(:, 1) = transpose_mode
            transpose_modes(1) = transpose_mode
        endif

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

        fmodes(:) = forw_modes(:, k)
        bmodes(:) = back_modes(:, k)
        if ( present(best_backend) ) best_backend = backends(k)
        if ( present(min_execution_time) ) min_execution_time = elapsed_time
        if ( present(best_transpose_mode) ) best_transpose_mode = transpose_modes(k)
        if ( present(fbacks) ) fbacks(:) = fbacks_(:, k)
        if ( present(bbacks) ) bbacks(:) = bbacks_(:, k)
        deallocate( timers, decomps, backends, transpose_modes, forw_modes, back_modes, fbacks_, bbacks_ )
    end subroutine autotune_grid_decomposition

    subroutine autotune_grid(                                                                                       &
        platform, dims, base_comm, effort, transpose_mode, base_dtype, comm_dims, base_storage, stream,             &
        fmodes, bmodes, backend, best_time, best_backend, best_transpose_mode, fbacks, bbacks)
    !! Creates cartesian grid and runs various backends on it. Returns best backend and execution time
        type(dtfft_platform_t),           intent(in)    :: platform
        !! Platform to create plan for
        integer(int32),                   intent(in)    :: dims(:)
        !! Global sizes of the transform requested
        TYPE_MPI_COMM,                    intent(in)    :: base_comm
        !! Basic communicator to create 3d grid from
        type(dtfft_effort_t),             intent(in)    :: effort
        !! Effort level for the plan creation
        type(dtfft_transpose_mode_t),     intent(in)    :: transpose_mode
        !! Transpose mode to use
        TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
        !! Base MPI_Datatype
        integer(int32),                   intent(in)    :: comm_dims(:)
        !! Number of processors in each dimension
        integer(int64),                   intent(in)    :: base_storage
        !! Number of bytes needed to store single element
        type(dtfft_stream_t),             intent(in)    :: stream
        !! Stream to use
        type(dtfft_transpose_mode_t),     intent(inout) :: fmodes(:)
            !! Best transpose modes for forward plan
        type(dtfft_transpose_mode_t),     intent(inout) :: bmodes(:)
            !! Best transpose modes for backward plan
        type(dtfft_backend_t),  optional, intent(in)    :: backend
        !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
        type(dtfft_backend_t),  optional, intent(out)   :: best_backend
        !! Best backend selected for the grid
        real(real32),           optional, intent(out)   :: best_time
        !! Elapsed time for best plan selected
        type(dtfft_transpose_mode_t), optional, intent(out) :: best_transpose_mode
        !! Best transpose mode selected
        type(dtfft_backend_t),  optional,       intent(out) :: fbacks(:)
        !! Best backends for forward plans
        type(dtfft_backend_t),  optional,       intent(out) :: bbacks(:)
        !! Best backends for backward plans
        type(pencil), allocatable :: pencils(:)
        character(len=:),             allocatable   :: phase_name           !! Caliper phase name
        integer(int8) :: d, ndims
        TYPE_MPI_COMM, allocatable :: comms(:)
        TYPE_MPI_COMM :: cart_comm
        integer(int32) :: mpi_ierr
        integer(int8), allocatable :: cperm(:,:), dperm(:,:)

        best_time = -1.0
        ndims = size(dims, kind=int8)

        if ( ndims == 3 ) then
            call get_permutations(ndims, dperm, cperm)
            do d = 1, ndims
                if ( dims(dperm(2, d)) < comm_dims(cperm(2, d)) .or. dims(dperm(3, d)) < comm_dims(cperm(3, d)) ) then
                    deallocate(dperm, cperm)
                    return
                endif
            enddo
            deallocate(dperm, cperm)
            allocate( phase_name, source = "Testing grid 1x"//to_str(comm_dims(2))//"x"//to_str(comm_dims(3)) )
        else
            allocate( phase_name, source = "Testing grid 1x"//to_str(comm_dims(2)) )
        endif

        WRITE_INFO("")
        WRITE_INFO(phase_name)
        REGION_BEGIN(phase_name, COLOR_AUTOTUNE)

        allocate( comms(ndims), pencils(ndims) )
        call create_pencils_and_comm(dims, base_comm, comm_dims, cart_comm, comms, pencils)

        call run_autotune_backend(platform, comms, cart_comm, effort, transpose_mode, base_dtype, pencils, base_storage, stream, .false., .false.,   &
            fmodes, bmodes, backend, best_time, best_backend, best_transpose_mode, fbacks, bbacks)

        do d = 1, ndims
            call pencils(d)%destroy()
            call MPI_Comm_free(comms(d), mpi_ierr)
        enddo
        call MPI_Comm_free(cart_comm, mpi_ierr)
        deallocate( comms, pencils )
        REGION_END(phase_name)
    end subroutine autotune_grid

    subroutine run_autotune_backend(                                                                                            &
        platform, comms, cart_comm, effort, transpose_mode, base_dtype, pencils, base_storage, stream, is_z_slab, is_y_slab,    &
        fmodes, bmodes, backend, best_time, best_backend, best_transpose_mode, fbacks, bbacks)
    !! Runs autotune for all backends
        type(dtfft_platform_t),           intent(in)    :: platform         !! Platform to create plan for
        TYPE_MPI_COMM,                    intent(in)    :: comms(:)         !! 1D comms
        TYPE_MPI_COMM,                    intent(in)    :: cart_comm        !! 3D Cartesian comm
        type(dtfft_effort_t),             intent(in)    :: effort           !! Effort level for the plan creation
        type(dtfft_transpose_mode_t),     intent(in)    :: transpose_mode   !! Transpose mode to use
        TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype       !! Base MPI_Datatype
        type(pencil),                     intent(in)    :: pencils(:)       !! Layouts
        integer(int64),                   intent(in)    :: base_storage     !! Number of bytes needed to store single element
        type(dtfft_stream_t),             intent(in)    :: stream           !! Stream to use
        logical,                          intent(in)    :: is_z_slab        !! Is Z-slab optimization enabled
        logical,                          intent(in)    :: is_y_slab        !! Is Y-slab optimization enabled
        type(dtfft_transpose_mode_t),     intent(inout) :: fmodes(:)
            !! Best transpose modes for forward plan
        type(dtfft_transpose_mode_t),     intent(inout) :: bmodes(:)
            !! Best transpose modes for backward plan
        type(dtfft_backend_t),  optional, intent(in)    :: backend
        !! GPU Backend to test. Should be passed only when effort is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
        real(real32),           optional, intent(out)   :: best_time
        !! Elapsed time for best backend
        type(dtfft_backend_t),  optional, intent(out)   :: best_backend
        !! Best backend selected
        type(dtfft_transpose_mode_t), optional, intent(out) :: best_transpose_mode
        !! Best transpose mode selected
        type(dtfft_backend_t),  optional,       intent(out) :: fbacks(:)
        type(dtfft_backend_t),  optional,       intent(out) :: bbacks(:)
        type(dtfft_backend_t),  allocatable :: backends_to_run(:), fbacks_(:), bbacks_(:)
        type(dtfft_backend_t) :: current_backend_id, best_backend_
        type(dtfft_transpose_mode_t) :: best_transpose_mode_
        logical :: is_udb !! Used defined backend
        real(real32) :: execution_time, best_time_
        integer(int32) :: b
        ! type(reshape_container), allocatable :: plans(:)
        integer(int8) :: j !i, j, n_transpose_plans
        ! logical :: is_aux_alloc
        ! integer(int64)         :: alloc_size
        logical :: nccl_enabled
#ifdef DTFFT_WITH_CUDA
        logical :: nvshmem_enabled
#endif
        type(dtfft_backend_t) :: backend_
        type(backend_helper)                      :: helper
        type(create_args) :: create_kwargs
        logical :: pipe_enabled, mpi_enabled, dtype_enabled, rma_enabled, fused_enabled
        logical :: compressed_enabled
#ifdef DTFFT_WITH_COMPRESSION
        type(dtfft_compression_config_t) :: conf
#endif
        integer(int32) :: n_transpose_plans
        real(real32), allocatable :: ftimes(:), btimes(:)

        n_transpose_plans = size(pencils, kind=int8) - 1_int8
        if ( is_z_slab ) n_transpose_plans = n_transpose_plans + 1

        allocate( fbacks_(n_transpose_plans), source=BACKEND_NOT_SET )
        allocate( bbacks_(n_transpose_plans), source=BACKEND_NOT_SET )
        allocate( ftimes(n_transpose_plans), source=MAX_REAL32 )
        allocate( btimes(n_transpose_plans), source=MAX_REAL32 )

        fbacks_(:) = BACKEND_NOT_SET;   fbacks_(:) = get_correct_backend(fbacks_)
        bbacks_(:) = BACKEND_NOT_SET;   bbacks_(:) = get_correct_backend(bbacks_)
        ftimes(:) = MAX_REAL32
        btimes(:) = MAX_REAL32
        if ( is_z_slab ) then
            ftimes(1:2) = 0.0_real32
            btimes(1:2) = 0.0_real32
        else if ( is_y_slab ) then
            ftimes(2) = 0.0_real32
            btimes(2) = 0.0_real32
        endif

        backend_ = BACKEND_NOT_SET
        is_udb = .false.
        if ( present(backend) ) then
            backend_ = backend
            if ( backend == DTFFT_BACKEND_ADAPTIVE ) then
                allocate( backends_to_run, source=VALID_BACKENDS(1:size(VALID_BACKENDS) - 1) )
            else
                allocate( backends_to_run(1) )
                backends_to_run(1) = backend
                is_udb = .true.
            endif
        else
            allocate( backends_to_run, source=VALID_BACKENDS(1:size(VALID_BACKENDS) - 1) )
        endif
        best_backend_ = backends_to_run(1)

        pipe_enabled = get_conf_pipelined_enabled()
        dtype_enabled = get_conf_datatype_enabled()
        mpi_enabled = get_conf_mpi_enabled()
        rma_enabled = get_conf_rma_enabled()
        fused_enabled = get_conf_fused_enabled()
        nccl_enabled = .false.
#ifdef DTFFT_WITH_CUDA
        nccl_enabled = ( platform == DTFFT_PLATFORM_CUDA .and. get_conf_nccl_enabled() ) .or. ( is_udb .and. is_backend_nccl( backends_to_run(1) ) )
        nvshmem_enabled = get_conf_nvshmem_enabled()
#endif

        compressed_enabled = .false.
#ifdef DTFFT_WITH_COMPRESSION
        conf = get_conf_transpose()
        compressed_enabled = get_conf_compression_enabled() .and. conf%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE
#endif

        call helper%create(platform, cart_comm, comms, nccl_enabled, pencils)

        create_kwargs%effort = DTFFT_ESTIMATE
        create_kwargs%force_effort = .true.
        create_kwargs%platform = platform
        create_kwargs%helper = helper
        create_kwargs%base_type = base_dtype
        create_kwargs%base_storage = base_storage
        create_kwargs%transpose_mode = transpose_mode
#ifdef DTFFT_WITH_COMPRESSION
        create_kwargs%compression_config = conf
#endif

        best_transpose_mode_ = transpose_mode
        fmodes(:) = transpose_mode
        bmodes(:) = transpose_mode

        best_time_ = MAX_REAL32

        do b = 1, size(backends_to_run)
            current_backend_id = backends_to_run(b)
            if ( .not.is_udb) then
                if ( is_backend_pipelined(current_backend_id) .and. .not.pipe_enabled ) cycle
                if ( is_backend_mpi(current_backend_id) .and. .not.mpi_enabled .and. .not.current_backend_id == DTFFT_BACKEND_MPI_DATATYPE ) cycle
                if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. .not.dtype_enabled ) cycle
                if ( is_backend_rma(current_backend_id) .and. .not.rma_enabled ) cycle
                if ( is_backend_fused(current_backend_id) .and. .not. fused_enabled) cycle
                if ( is_backend_compressed(current_backend_id) .and. .not. compressed_enabled ) cycle
                if ( current_backend_id == DTFFT_BACKEND_ADAPTIVE ) cycle
#ifdef DTFFT_WITH_CUDA
                if ( platform == DTFFT_PLATFORM_CUDA ) then
                    if ( is_backend_nvshmem(current_backend_id) .and. .not.nvshmem_enabled ) cycle
                    if ( is_backend_nccl(current_backend_id) .and. .not.nccl_enabled) cycle
                    if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE ) cycle
                else
                    if ( is_backend_nccl(current_backend_id) .or. is_backend_nvshmem(current_backend_id) ) cycle
                endif
#endif
            endif

            create_kwargs%backend = current_backend_id

            if ( backend_ == DTFFT_BACKEND_ADAPTIVE ) then
                block
                    type(dtfft_transpose_mode_t), allocatable :: lfmodes(:), lbmodes(:)
                    real(real32), allocatable :: lftimes(:), lbtimes(:)
                    integer(int32) :: i

                    allocate( lftimes(n_transpose_plans), lbtimes(n_transpose_plans) )
                    allocate( lfmodes(n_transpose_plans), lbmodes(n_transpose_plans) )

                    call execute_single(backend_, effort, create_kwargs, helper, pencils, is_z_slab, is_y_slab, lfmodes, lbmodes, lftimes, lbtimes)
                    if ( is_z_slab ) then
                        if ( lftimes(3) < ftimes(3) ) then
                            fmodes(3) = lfmodes(3)
                            fbacks_(3) = current_backend_id
                            ftimes(3) = lftimes(3)
                        endif
                        if ( lbtimes(3) < btimes(3) ) then
                            bmodes(3) = lbmodes(3)
                            bbacks_(3) = current_backend_id
                            btimes(3) = lbtimes(3)
                        endif
                    else
                        do i = 1, n_transpose_plans
                            if ( lftimes(i) < ftimes(i) ) then
                                fmodes(i) = lfmodes(i)
                                fbacks_(i) = current_backend_id
                                ftimes(i) = lftimes(i)
                            endif
                            if ( lbtimes(i) < btimes(i) ) then
                                bmodes(i) = lbmodes(i)
                                bbacks_(i) = current_backend_id
                                btimes(i) = lbtimes(i)
                            endif
                        enddo
                    endif

                    deallocate(lftimes, lbtimes, lfmodes, lbmodes)
                endblock
            else

                if ( .not.(current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .or. platform == DTFFT_PLATFORM_CUDA) .and. effort == DTFFT_EXHAUSTIVE ) then
                    do j = CONF_DTFFT_TRANSPOSE_MODE_PACK, CONF_DTFFT_TRANSPOSE_MODE_UNPACK
                        create_kwargs%transpose_mode = dtfft_transpose_mode_t(j)
                        execution_time = execute_many(cart_comm, create_kwargs, pencils, helper, is_z_slab, is_y_slab, effort, stream, fmodes, bmodes)
                        if ( execution_time < best_time_ ) then
                            best_transpose_mode_ = dtfft_transpose_mode_t(j)
                        endif
                    enddo
                else
                    execution_time = execute_many(cart_comm, create_kwargs, pencils, helper, is_z_slab, is_y_slab, effort, stream, fmodes, bmodes)
                endif

                if ( execution_time < best_time_ ) then
                    best_time_ = execution_time
                    best_backend_ = current_backend_id
                endif
            endif
        enddo

        call helper%destroy()

        if ( present(best_time)) then
            if ( backend_ == DTFFT_BACKEND_ADAPTIVE ) then
                best_time = sum(ftimes) + sum(btimes)
            else
                best_time = best_time_
            endif
        endif
        if ( present(best_backend) ) best_backend = best_backend_
        if ( present(best_transpose_mode) ) best_transpose_mode = best_transpose_mode_
        if ( present(fbacks) ) fbacks(:size(fbacks_)) = fbacks_(:)
        if ( present(bbacks) ) bbacks(:size(bbacks_)) = bbacks_(:)
    end subroutine run_autotune_backend

    real(real32) function execute_many(cart_comm, create_kwargs, pencils, helper, is_z_slab, is_y_slab, effort, stream, fmodes, bmodes)
    !! 
        TYPE_MPI_COMM,                  intent(in)    :: cart_comm      !! 3D Cartesian comm
        type(create_args),              intent(inout) :: create_kwargs  !! Create arguments
        type(pencil),                   intent(in)    :: pencils(:)     !! Layouts
        type(backend_helper),           intent(inout) :: helper
        logical,                        intent(in)    :: is_z_slab      !! Is Z-slab optimization enabled
        logical,                        intent(in)    :: is_y_slab      !! Is Y-slab optimization enabled
        type(dtfft_effort_t),           intent(in)    :: effort         !! Effort level for the plan creation
        type(dtfft_stream_t),           intent(in)    :: stream         !! Stream to use
        type(dtfft_transpose_mode_t),   intent(inout) :: fmodes(:)      !! Best transpose modes for forward plan
        type(dtfft_transpose_mode_t),   intent(inout) :: bmodes(:)      !! Best transpose modes for backward plan
        type(reshape_container), allocatable :: plans(:)
        integer(int64)         :: alloc_size
        integer(int32) :: i, n_transpose_plans
        character(len=:), allocatable :: testing_phase

        if ( is_z_slab ) then
            n_transpose_plans = 1
        else
            n_transpose_plans = size(pencils, kind=int8) - 1_int8
        endif

        if ( is_y_slab ) then
            n_transpose_plans = n_transpose_plans - 1_int8
        endif

        allocate( plans(2 * n_transpose_plans) )

        call allocate_plans(plans, create_kwargs%backend)
        if ( create_kwargs%backend == DTFFT_BACKEND_MPI_DATATYPE .or. create_kwargs%platform == DTFFT_PLATFORM_CUDA ) then
            allocate( testing_phase, source="Testing backend "//dtfft_get_backend_string(create_kwargs%backend) )
        else
            allocate( testing_phase, source="Testing backend "//dtfft_get_backend_string(create_kwargs%backend)//", transpose mode = "//trim(TRANSPOSE_MODE_NAMES(create_kwargs%transpose_mode%val)) )
        endif
        REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
        WRITE_INFO(testing_phase)

        call get_local_sizes(pencils, alloc_size=alloc_size)
        alloc_size = alloc_size * create_kwargs%base_storage

        if ( create_kwargs%backend == DTFFT_BACKEND_MPI_DATATYPE .and. effort%val >= DTFFT_PATIENT%val) then
            block
                real(real32) :: ftimes(3), btimes(3)
                call execute_single(DTFFT_BACKEND_MPI_DATATYPE, effort, create_kwargs, helper, pencils, is_z_slab, is_y_slab, fmodes, bmodes, ftimes, btimes, .true.)
                execute_many = sum(ftimes) + sum(btimes)
                WRITE_INFO("  Execution time on a grid using fastest transpositions: "//to_str(real(execute_many, real64))//" [ms]")
            endblock
        else
            if ( is_z_slab ) then
                call plans(1)%p%create(pencils(1), pencils(3), create_kwargs)
                call plans(2)%p%create(pencils(3), pencils(1), create_kwargs)
            else
                do i = 1, n_transpose_plans
                    call plans(i)%p%create(pencils(i), pencils(i + 1), create_kwargs)
                    call plans(i + n_transpose_plans)%p%create(pencils(i + 1), pencils(i), create_kwargs)
                enddo
            endif
            execute_many = execute_autotune(plans, cart_comm, create_kwargs%backend, create_kwargs%platform, helper, stream, alloc_size)
        endif

        call destroy_plans(plans)
        deallocate(plans)
        REGION_END(testing_phase)
        deallocate(testing_phase)
    end function execute_many

    subroutine execute_single(                                                                                      &
        base_backend, effort, create_kwargs, helper, pencils,                                                       &
        is_z_slab, is_y_slab, fmodes, bmodes, ftimes, btimes, phase_created)
    !! Executes autotuning over all single transpose plans and returns best transpose modes and their execution times
        type(dtfft_backend_t),          intent(in)    :: base_backend   !! Backend to test
        type(dtfft_effort_t),           intent(in)    :: effort         !! Effort level for the plan creation
        type(create_args),              intent(inout) :: create_kwargs  !! Create arguments
        type(backend_helper),           intent(inout) :: helper         !! Base MPI_Datatype
        type(pencil),                   intent(in)    :: pencils(:)     !! Pencils metadata
        logical,                        intent(in)    :: is_z_slab      !! Is Z-slab optimization enabled
        logical,                        intent(in)    :: is_y_slab      !! Is Y-slab optimization enabled
        type(dtfft_transpose_mode_t),   intent(inout) :: fmodes(:)      !! Best transpose modes for forward plan
        type(dtfft_transpose_mode_t),   intent(inout) :: bmodes(:)      !! Best transpose modes for backward plan
        real(real32),                   intent(out)   :: ftimes(:)      !! Elapsed time for best forward plans selected in [ms]
        real(real32),                   intent(out)   :: btimes(:)      !! Elapsed time for best backward plans selected in [ms]
        logical,         optional,      intent(in)    :: phase_created  !! Whether caliper phase was created
        integer(int8)                   :: dim              !! Counter
        integer(int8)                   :: ndims            !! Number of dimensions
        character(len=:),   allocatable :: testing_phase    !! Caliper phase name
        logical :: phase_created_

        ndims = size(pencils, kind=int8)
        if ( is_y_slab ) ndims = ndims - 1_int8

        ftimes(:) = 0._real32
        btimes(:) = 0._real32

        phase_created_ = .false.; if ( present(phase_created) ) phase_created_ = phase_created
        if ( .not.phase_created_ ) then
            allocate( testing_phase, source="Testing backend "//dtfft_get_backend_string(create_kwargs%backend) )
            REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
            WRITE_INFO(testing_phase)
        endif

        if( is_z_slab ) then
            call execute_single_transpose_modes(base_backend, effort, create_kwargs, helper, pencils(1), pencils(3), 3_int8, fmodes(3), bmodes(3), ftimes(3), btimes(3))
        else
            do dim = 1_int8, ndims - 1_int8
                call execute_single_transpose_modes(base_backend, effort, create_kwargs, helper, pencils(dim), pencils(dim + 1), dim, fmodes(dim), bmodes(dim), ftimes(dim), btimes(dim))
            enddo
        endif
        if ( .not.phase_created_ ) then
            REGION_END(testing_phase)
            deallocate( testing_phase )
        endif
    end subroutine execute_single

    subroutine execute_single_transpose_modes(base_backend, effort, create_kwargs, helper, from, to, transpose_name_id, fmode, bmode, forward_time, backward_time)
    !! Creates forward and backward transpose plans for backend, specified by `create_kwargs`, based on source and target data distributions and,
    !! executes them `DTFFT_MEASURE_ITERS` times ( 4 * `DTFFT_MEASURE_ITERS` iterations total ) + 4 * `DTFFT_MEASURE_WARMUP_ITERS` warmup iterations
    !!
    !! Returns elapsed time for best plans selected
        type(dtfft_backend_t),          intent(in)      :: base_backend         !! 
        type(dtfft_effort_t),           intent(in)      :: effort
        type(create_args),              intent(inout)   :: create_kwargs
        type(backend_helper),           intent(inout)   :: helper               !! Backend helper
        type(pencil),                   intent(in)      :: from                 !! Source meta
        type(pencil),                   intent(in)      :: to                   !! Target meta
        integer(int8),                  intent(in)      :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
        type(dtfft_transpose_mode_t),   intent(out)     :: fmode           !! Best forward plan ID
        type(dtfft_transpose_mode_t),   intent(out)     :: bmode          !! Best backward plan ID
        real(real32),                   intent(out)     :: forward_time         !! Forward plan execution time
        real(real32),                   intent(out)     :: backward_time        !! Backward plan execution time
        real(real32)                    :: time                 !! Timer
        integer(int32)                  :: transpose_mode          !! Counter
        type(dtfft_transpose_mode_t)    :: tmode

        forward_time = huge(1._real32)
        backward_time = huge(1._real32)

        if ( base_backend == DTFFT_BACKEND_MPI_DATATYPE .or. effort == DTFFT_EXHAUSTIVE ) then
            do transpose_mode = CONF_DTFFT_TRANSPOSE_MODE_PACK, CONF_DTFFT_TRANSPOSE_MODE_UNPACK
                tmode = dtfft_transpose_mode_t(transpose_mode)
                create_kwargs%transpose_mode = tmode

                time = run_execute_single(create_kwargs, helper, from, to, transpose_name_id)
                if ( time < forward_time ) then
                    forward_time = time
                    fmode = tmode
                endif

                time = run_execute_single(create_kwargs, helper, to, from, -1_int8 * transpose_name_id)
                if ( time < backward_time ) then
                    backward_time = time
                    bmode = tmode
                endif
            enddo
        else
            forward_time = run_execute_single(create_kwargs, helper, from, to, transpose_name_id); fmode = create_kwargs%transpose_mode
            backward_time = run_execute_single(create_kwargs, helper, to, from, -1_int8 * transpose_name_id); bmode = create_kwargs%transpose_mode
        endif
    end subroutine execute_single_transpose_modes

    function run_execute_single(create_kwargs, helper, from, to, transpose_name_id) result(elapsed_time)
    !! Creates transpose plan for backend, specified by `create_kwargs` and executes it `DTFFT_MEASURE_WARMUP_ITERS` + `DTFFT_MEASURE_ITERS` times
    !!
    !! Returns elapsed time
        type(create_args),            intent(inout) :: create_kwargs
        type(backend_helper),         intent(inout) :: helper               !! Backend helper
        type(pencil),                 intent(in)    :: from                 !! Source meta
        type(pencil),                 intent(in)    :: to                   !! Target meta
        integer(int8),                intent(in)    :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
        real(real32)                                :: elapsed_time         !! Execution time [ms]
        character(len=:),             allocatable   :: phase_name           !! Caliper phase name
        type(reshape_container) :: plan(1)
        integer(int64)  :: buffer_size

        allocate( phase_name, source="  Testing plan "//TRANSPOSE_NAMES(transpose_name_id)//", transpose mode = "//trim(TRANSPOSE_MODE_NAMES(create_kwargs%transpose_mode%val)) )
        REGION_BEGIN(phase_name, 0)
        WRITE_INFO(phase_name)

        call allocate_plans(plan, create_kwargs%backend)

        call plan(1)%p%create(from, to, create_kwargs)
        buffer_size = create_kwargs%base_storage * max( product(from%counts), product(to%counts) )
        elapsed_time = execute_autotune(plan, helper%comms(1), create_kwargs%backend, DTFFT_PLATFORM_HOST, helper, NULL_STREAM, buffer_size, 4)

        call destroy_plans(plan)
        REGION_END(phase_name)
        deallocate(phase_name)
    end function run_execute_single

    subroutine get_permutations(ndims, dperm, cperm)
    !! Returns data and communicator permutations for given number of dimensions
        integer(int8),              intent(in)  :: ndims        !! Number of dimensions
        integer(int8), allocatable, intent(out) :: dperm(:,:)   !! Data permutations
        integer(int8), allocatable, intent(out) :: cperm(:,:)   !! Communicator permutations

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
        if ( GET_MPI_VALUE(comm) == GET_MPI_VALUE(MPI_COMM_NULL) ) then
            INTERNAL_ERROR("comm == MPI_COMM_NULL")
        endif

        allocate(temp_comms(ndims))

        allocate( remain_dims(ndims), source = .false. )
        do dim = 1, ndims
            remain_dims(dim) = .true.
            call MPI_Cart_sub(temp_cart_comm, remain_dims, temp_comms(dim), ierr)
            call create_subcomm_include_all(temp_comms(dim), local_comms(dim))
            if ( GET_MPI_VALUE(local_comms(dim)) == GET_MPI_VALUE(MPI_COMM_NULL) ) then
                INTERNAL_ERROR("local_comms(dim) == MPI_COMM_NULL: dim = "//to_str(dim))
            endif
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
