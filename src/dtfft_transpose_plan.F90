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
#include "dtfft.f03"
module dtfft_transpose_plan
!! This module describes [[transpose_plan]] class
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend,               only: backend_helper
use dtfft_abstract_transpose_handle,      only: abstract_transpose_handle, create_args, execute_args
use dtfft_config
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
use dtfft_kernel_device,                  only: DEF_TILE_SIZE
use dtfft_interface_cuda,                 only: load_cuda
use dtfft_interface_nvrtc,                only: load_nvrtc
# ifdef NCCL_HAVE_COMMREGISTER
use dtfft_abstract_backend,               only: NCCL_REGISTER_PREALLOC_SIZE
# endif
# ifdef DTFFT_WITH_NVSHMEM
use dtfft_interface_nvshmem
# endif
# ifdef DTFFT_WITH_NCCL
use dtfft_interface_nccl
# endif
#endif
use dtfft_parameters
use dtfft_pencil,                         only: pencil, pencil_init, get_local_sizes
use dtfft_transpose_handle_generic,       only: transpose_handle_generic
use dtfft_transpose_handle_datatype,      only: transpose_handle_datatype
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_profile.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: transpose_plan

  type :: plan_t
  !! This type is a container for allocatable transpose handles
    class(abstract_transpose_handle), allocatable :: p  !! Transpose handle
  end type plan_t

  integer(int8), save :: FORWARD_PLAN_IDS(3)
    !! Default data types for forward transpositions
  integer(int8), save :: BACKWARD_PLAN_IDS(3)
    !! Default data types for backward transpositions
  logical,       save :: ARE_DATATYPES_SET = .false.
    !! Are default data types set

  type :: transpose_plan
  !! Transpose Plan class
  !! This class is a container for transposition plans
  private
    type(dtfft_backend_t)     :: backend
      !! Backend
    type(backend_helper)      :: helper
      !! Backend helper
    logical                   :: is_z_slab
      !! Z-slab optimization flag (for 3D transforms)
    integer(int64)            :: min_buffer_size
      !! Minimal buffer size for transposition
    type(dtfft_platform_t)    :: platform
      !! Platform used for transposition
    type(dtfft_stream_t)      :: stream
      !! CUDA stream
    type(c_ptr)               :: aux
      !! Auxiliary memory
    real(real32), pointer     :: paux(:)
      !! Pointer to auxiliary memory
    logical                   :: is_aux_alloc = .false.
      !! Is auxiliary memory allocated
    type(plan_t), allocatable :: plans(:)
    !! Plans for each transposition
  contains
  private
    procedure,  non_overridable,  pass(self), public :: create            !! Creates transpose plan
    procedure,  non_overridable,  pass(self), public :: execute           !! Executes transposition
    procedure,  non_overridable,  pass(self), public :: execute_end       !! Finishes asynchronous transposition
    procedure,  non_overridable,  pass(self), public :: get_async_active  !! Returns .true. if any of the plans is running asynchronously
    procedure,  non_overridable,  pass(self), public :: destroy           !! Destroys transpose plan
    procedure,  non_overridable,  pass(self), public :: get_aux_size      !! Returns auxiliary buffer size
    procedure,  non_overridable,  pass(self), public :: get_backend       !! Returns backend id
    procedure,  non_overridable,  pass(self), public :: get_z_slab        !! Returns .true. if Z-slab optimization is enabled
    procedure,  non_overridable,  pass(self), public :: mem_alloc         !! Allocates memory based on selected backend
    procedure,  non_overridable,  pass(self), public :: mem_free          !! Frees memory allocated with mem_alloc
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
    integer(int32),   allocatable   :: transposed_dims(:,:) !! Global counts in transposed coordinates
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


    call MPI_Comm_size(base_comm, comm_size, ierr)
    call MPI_Topo_test(base_comm, top_type, ierr)
    base_comm_ = base_comm

    ndims = size(dims, kind=int8)
    allocate( comm_dims(ndims) )
    comm_dims(:) = 0

    is_custom_cart_comm = .false.
    self%is_z_slab = .false.

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
      if ( ndims == 3 .and. comm_dims(2) == 1 .and. get_conf_z_slab_enabled() ) then
        self%is_z_slab = .true.
        base_comm_ = ipencil%comms(3)
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
          if ( ndims == 3 .and. comm_dims(2) == 1 .and. get_conf_z_slab_enabled() ) then
            self%is_z_slab = .true.
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
        if ( ndims == 3 .and. cond1 ) then
          comm_dims(2) = 1
          comm_dims(3) = comm_size
          self%is_z_slab = get_conf_z_slab_enabled()
        else if (ndims == 3 .and. cond2 ) then
          comm_dims(2) = comm_size
          comm_dims(3) = 1
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

    allocate(transposed_dims(ndims, ndims))
    if ( ndims == 2 ) then
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

    if ( platform == DTFFT_PLATFORM_HOST ) then
      if ( .not. get_conf_datatype_enabled() .and. .not. get_conf_mpi_enabled() .and. effort == DTFFT_PATIENT) then
        error_code = DTFFT_ERROR_BACKENDS_DISABLED
        return
      endif
#ifdef DTFFT_WITH_CUDA
    else
      if ( .not.get_conf_mpi_enabled() .and. .not.get_conf_nccl_enabled() .and. .not.get_conf_nvshmem_enabled() .and. effort == DTFFT_PATIENT) then
        error_code = DTFFT_ERROR_BACKENDS_DISABLED
        return
      endif

      CHECK_CALL( load_cuda(), error_code )
      CHECK_CALL( load_nvrtc(), error_code )
      self%stream = get_conf_stream()

! # ifdef DTFFT_WITH_NVSHMEM
!       if ( is_backend_nvshmem(self%backend) .or. (get_conf_nvshmem_enabled() .and. effort == DTFFT_PATIENT) ) then
!         call nvshmem_init(base_comm_)
!       endif
! # endif
#endif
    endif



    ! if ( platform == DTFFT_PLATFORM_HOST .and. (is_backend_nccl(self%backend) .or. is_backend_nvshmem(self%backend)) ) then
    ! ! Do not raise error here, just fallback to MPI_P2P
    !   self%backend = DTFFT_BACKEND_MPI_DATATYPE
    ! endif

    allocate( best_decomposition(ndims) )
    best_decomposition(:) = comm_dims(:)
    call MPI_Comm_size(base_comm_, comm_size, ierr)
    if ( comm_size == 1 .and. self%backend /= DTFFT_BACKEND_MPI_DATATYPE ) self%backend = BACKEND_NOT_SET

    pencils_created = .false.
    if ( ndims == 2 .or. is_custom_cart_comm .or. self%is_z_slab ) then
      pencils_created = .true.
      call create_pencils_and_comm(transposed_dims, base_comm_, comm_dims, cart_comm, comms, pencils, ipencil=ipencil)
    endif

    ts = MPI_Wtime()

    if ( effort == DTFFT_PATIENT .and. comm_size > 1 .and. .not.invalid_grid_selected) then
      if ( pencils_created ) then
        call run_autotune_backend(                                                                              &
          platform, comms, cart_comm, effort, base_dtype, pencils, base_storage, self%stream, self%is_z_slab,   &
          best_forward_ids, best_backward_ids, best_backend=self%backend)
      else
        call autotune_grid_decomposition(                                                                       &
          platform, dims, transposed_dims, base_comm_, effort, base_dtype, base_storage,                        &
          self%stream, best_forward_ids, best_backward_ids, best_decomposition, best_backend=self%backend)
      endif
    else if ( ndims == 3                                &
      .and. .not.is_custom_cart_comm                    &
      .and. .not.self%is_z_slab                         &
      .and. effort == DTFFT_MEASURE                     &
      .and. comm_size > 1 ) then

      call autotune_grid_decomposition(                                                                         &
        platform, dims, transposed_dims, base_comm_, effort, base_dtype, base_storage,                          &
        self%stream, best_forward_ids, best_backward_ids, best_decomposition, backend=self%backend)
    endif
    te = MPI_Wtime()

    if ( effort%val >= DTFFT_MEASURE%val .and. ndims > 2 .and. comm_size > 1 ) then
      WRITE_INFO(repeat("*", 50))
      if ( self%is_z_slab ) then
        WRITE_INFO("Skipped search of MPI processor grid due to Z-slab optimization enabled")
      else if ( is_custom_cart_comm ) then
        WRITE_INFO("Skipped search of MPI processor grid due to custom grid provided")
      else if ( invalid_grid_selected ) then
        WRITE_INFO("Skipped search of MPI processor grid due to lack of work per process")
      else
        WRITE_INFO("DTFFT_MEASURE: Selected MPI processor grid 1x"//to_str(best_decomposition(2))//"x"//to_str(best_decomposition(3)))
      endif
    endif
    if ( effort == DTFFT_PATIENT .and. comm_size > 1 ) then
      WRITE_INFO("DTFFT_PATIENT: Selected backend is "//dtfft_get_backend_string(self%backend))
    endif

    n_transpose_plans = ndims - 1_int8; if( self%is_z_slab ) n_transpose_plans = n_transpose_plans + 1_int8

    if ( effort == DTFFT_PATIENT .and. self%backend == DTFFT_BACKEND_MPI_DATATYPE ) then
      WRITE_INFO("DTFFT_PATIENT: Selected transpose ids:")
      do d = 1, n_transpose_plans
        WRITE_INFO("    "//TRANSPOSE_NAMES( d)//": "//to_str( best_forward_ids(d) ))
        WRITE_INFO("    "//TRANSPOSE_NAMES(-d)//": "//to_str( best_backward_ids(d) ))
      enddo
      WRITE_INFO(repeat("*", 50))
    endif
    if ( effort%val >= DTFFT_MEASURE%val .and. comm_size > 1 ) then
      WRITE_INFO("Time spent on autotune: "//to_str(te - ts)//" [s]")
    endif

    if ( .not.pencils_created ) then
      call create_pencils_and_comm(transposed_dims, base_comm_, best_decomposition, cart_comm, comms, pencils)
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

      do d = 1_int8, ndims - 1_int8
        args%datatype_id = best_forward_ids(d)
        call self%plans(d)%p%create(pencils(d), pencils(d + 1), base_storage, args)
        args%datatype_id = best_backward_ids(d)
        call self%plans(-d)%p%create(pencils(d + 1), pencils(d), base_storage, args)
      enddo
      if ( self%is_z_slab ) then
        args%datatype_id = best_forward_ids(3)
        call self%plans(3)%p%create(pencils(1), pencils(3), base_storage, args)
        args%datatype_id = best_backward_ids(3)
        call self%plans(-3)%p%create(pencils(3), pencils(1), base_storage, args)
      endif
    endblock
    te = MPI_Wtime()
    WRITE_INFO("Time spent creating final plans: "//to_str(te - ts)//" [s]")

    call get_local_sizes(pencils, alloc_size=self%min_buffer_size)
    self%min_buffer_size = self%min_buffer_size * (base_storage / FLOAT_STORAGE_SIZE)
    call alloc_and_set_aux(platform, self%helper, self%backend, cart_comm, self%aux, self%paux, self%plans, self%is_aux_alloc)

    deallocate( best_decomposition, comm_dims, transposed_dims )
    error_code = DTFFT_SUCCESS
  end function create

  logical function get_async_active(self)
  !! Returns .true. if any of the plans is running asynchronously
    class(transpose_plan),   intent(in)    :: self           !! Transposition class
    integer(int32)  :: i

    get_async_active = .false.
    do i = lbound(self%plans, dim=1), ubound(self%plans, dim=1)
      if ( allocated( self%plans(i)%p ) ) then
        get_async_active = get_async_active .or. self%plans(i)%p%get_async_active()
      endif
    enddo
  end function get_async_active

  subroutine execute(self, in, out, transpose_type, exec_type, error_code)
  !! Executes transposition
    class(transpose_plan),        intent(inout) :: self           !! Transposition class
    type(c_ptr),                  intent(in)    :: in             !! Incoming buffer
    type(c_ptr),                  intent(in)    :: out            !! Resulting buffer
    type(dtfft_transpose_t),      intent(in)    :: transpose_type !! Type of transpose to execute
    type(async_exec_t),           intent(in)    :: exec_type      !! Type of execution (sync/async)
    integer(int32),    optional,  intent(out)   :: error_code     !! Error code
    real(real32), pointer :: pin(:)   !! Source buffer
    real(real32), pointer :: pout(:)  !! Destination buffer
    type(execute_args)    :: kwargs   !! Additional arguments for execution
    integer(int32)        :: ierr     !! Error code

    REGION_BEGIN('Transpose '//TRANSPOSE_NAMES(transpose_type%val), COLOR_TRANSPOSE_PALLETTE(transpose_type%val))
    call c_f_pointer(in, pin, [self%min_buffer_size])
    call c_f_pointer(out, pout, [self%min_buffer_size])
    kwargs%exec_type = exec_type
    kwargs%stream = self%stream
    if ( self%is_aux_alloc ) then
      kwargs%p1 => self%paux
    else
      ! Pointer is unused. Pointing to something in order to avoid runtime null pointer errors
      kwargs%p1 => pin
    endif
    call self%plans(transpose_type%val)%p%execute(pin, pout, kwargs, ierr)
    if( present( error_code ) ) error_code = ierr
    REGION_END('Transpose '//TRANSPOSE_NAMES(transpose_type%val))
  end subroutine execute

  subroutine execute_end(self, in, out, transpose_type, error_code)
  !! Finishes asynchronous transposition
    class(transpose_plan),        intent(inout) :: self           !! Transposition class
    type(c_ptr),                  intent(in)    :: in             !! Incoming buffer
    type(c_ptr),                  intent(in)    :: out            !! Resulting buffer
    type(dtfft_transpose_t),      intent(in)    :: transpose_type !! Type of transpose
    integer(int32),               intent(out)   :: error_code     !! Error code
    real(real32),   pointer :: pin(:)   !! Source buffer
    real(real32),   pointer :: pout(:)  !! Destination buffer
    type(execute_args)      :: kwargs   !! Additional arguments for execution

    call c_f_pointer(in, pin, [self%min_buffer_size])
    call c_f_pointer(out, pout, [self%min_buffer_size])

    kwargs%p1 => pin
    kwargs%p2 => pout
    kwargs%stream = self%stream
    call self%plans(transpose_type%val)%p%execute_end(kwargs, error_code)
  end subroutine execute_end

  subroutine destroy(self)
  !! Destroys transposition plans
    class(transpose_plan),    intent(inout) :: self         !! Transposition class
    integer(int32) :: ierr

    if ( self%is_aux_alloc ) then
      call self%mem_free(self%aux, ierr)
      self%paux => null()
      self%is_aux_alloc = .false.
    endif

    if ( allocated( self%plans ) ) then
      call destroy_plans(self%plans)
      deallocate( self%plans )
    endif

    call self%helper%destroy()
! # ifdef DTFFT_WITH_NVSHMEM
!       if ( is_backend_nvshmem(self%backend) ) then
!         call nvshmem_finalize()
!       endif
! # endif
! #ifdef DTFFT_WITH_CUDA
!     if ( self%platform == DTFFT_PLATFORM_CUDA  ) then
!       call cache%cleanup()
!     endif
! #endif
  end subroutine destroy

  logical function get_z_slab(self)
  !! Returns .true. if Z-slab optimization is enabled
    class(transpose_plan),   intent(in)    :: self      !! Transposition class
    get_z_slab = self%is_z_slab
  end function get_z_slab

  subroutine allocate_plans(plans, backend)
  !! Allocates array of plans
    type(plan_t),           intent(inout) :: plans(:)   !! Plans to allocate
    type(dtfft_backend_t),  intent(in)    :: backend    !! Backend to use
    integer(int32) :: i

    do i = 1, size(plans)
      if ( backend == DTFFT_BACKEND_MPI_DATATYPE ) then
        allocate( transpose_handle_datatype :: plans(i)%p )
      else
        allocate( transpose_handle_generic :: plans(i)%p )
      endif
    enddo
  end subroutine allocate_plans

  subroutine destroy_plans(plans)
  !! Destroys array of plans
    type(plan_t),           intent(inout) :: plans(:) !! Plans to destroy
    integer(int32) :: i

    do i = 1, size(plans)
      if( allocated(plans(i)%p) ) then
        call plans(i)%p%destroy()
        deallocate(plans(i)%p)
      endif
    enddo
  end subroutine destroy_plans

  subroutine autotune_grid_decomposition(                                                             &
    platform, dims, transposed_dims, base_comm, effort, base_dtype, base_storage, stream,             &
    best_forward_ids, best_backward_ids, best_decomposition, backend, min_execution_time, best_backend)
  !! Runs through all possible grid decompositions and selects the best one based on the lowest average execution time
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to use
    integer(int32),                   intent(in)    :: dims(:)
      !! Global sizes of the transform requested
    integer(int32),                   intent(in)    :: transposed_dims(:,:)
      !! Transposed dimensions
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
        platform, dims, transposed_dims, base_comm, effort, base_dtype,   &
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
          platform, dims, transposed_dims, base_comm, effort, base_dtype,   &
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
    platform, dims, transposed_dims, base_comm, effort, base_dtype, comm_dims, base_storage, stream, is_z_slab,   &
    best_forward_ids, best_backward_ids, backend, best_time, best_backend)
  !! Creates cartesian grid and runs various backends on it. Returns best backend and execution time
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to create plan for
    integer(int32),                   intent(in)    :: dims(:)
      !! Global sizes of the transform requested
    integer(int32),                   intent(in)    :: transposed_dims(:,:)
      !! Transposed dimensions
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
    call create_pencils_and_comm(transposed_dims, base_comm, comm_dims, cart_comm, comms, pencils)

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
    real(real32) :: execution_time, avg_execution_time, best_time_
    integer(int32) :: iter, comm_size, mpi_ierr, b, ierr
    ! type(transpose_handle_generic),  allocatable   :: plans(:)
    type(plan_t), allocatable :: plans(:)
    integer(int8) :: i, n_transpose_plans
    type(c_ptr) :: in, out, aux
    real(real32), pointer :: pin(:), pout(:), paux(:)
    logical :: is_aux_alloc
    real(real64) :: ts, te
    ! , need_aux
    integer(int64)         :: alloc_size
#ifdef DTFFT_WITH_CUDA
    type(cudaEvent) :: timer_start, timer_stop
#endif
    character(len=:), allocatable :: testing_phase
    type(backend_helper)                      :: helper
    integer(int32) :: n_warmup_iters, n_iters
    integer(int64) :: min_buffer_size
    type(create_args) :: create_kwargs
    type(execute_args) :: execute_kwargs
    integer(int32) :: first
    logical :: pipe_enabled, mpi_enabled, dtype_enabled

    if ( present(backend) ) then
      allocate( backends_to_run(1) )
      backends_to_run(1) = backend
      is_udb = .true.
    else
      if ( platform == DTFFT_PLATFORM_HOST ) then
        first = 1
      else
        first = 2
      endif
      allocate(backends_to_run(size(VALID_BACKENDS(first:))))
      do b = 1, size(backends_to_run)
        backends_to_run(b) = VALID_BACKENDS(first - 1 + b)
      enddo
      is_udb = .false.
    endif
    best_backend_ = backends_to_run(1)

    if ( is_z_slab ) then
      n_transpose_plans = 1
    else
      n_transpose_plans = size(pencils, kind=int8) - 1_int8
    endif

    allocate( plans(2 * n_transpose_plans) )

    call MPI_Comm_size(cart_comm, comm_size, mpi_ierr)

    call helper%create(platform, cart_comm, comms, any(is_backend_nccl(backends_to_run)), pencils)

    call get_local_sizes(pencils, alloc_size=alloc_size)
    alloc_size = alloc_size * base_storage
    min_buffer_size = alloc_size / FLOAT_STORAGE_SIZE

    create_kwargs%effort = DTFFT_ESTIMATE
    create_kwargs%force_effort = .true.
    create_kwargs%platform = platform
    create_kwargs%helper = helper
    create_kwargs%base_type = base_dtype

    execute_kwargs%exec_type = EXEC_BLOCKING
    execute_kwargs%stream = stream

#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaEventCreate(timer_start) )
      CUDA_CALL( cudaEventCreate(timer_stop) )
    endif
#endif

    n_warmup_iters = get_conf_measure_warmup_iters()
    n_iters = get_conf_measure_iters()
    pipe_enabled = get_conf_pipelined_enabled()
    dtype_enabled = get_conf_datatype_enabled()
    mpi_enabled = get_conf_mpi_enabled()

    best_time_ = MAX_REAL32

    do b = 1, size(backends_to_run)
      current_backend_id = backends_to_run(b)

      if ( ((is_backend_pipelined(current_backend_id) .and. .not.pipe_enabled)                           &
            .or.(is_backend_mpi(current_backend_id) .and. .not.mpi_enabled)                             &
            .or.(current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. .not.dtype_enabled))             &
            .and. .not.is_udb) cycle
#ifdef DTFFT_WITH_CUDA
      if ( platform == DTFFT_PLATFORM_CUDA ) then
        if ( ( is_backend_nvshmem(current_backend_id) .and. .not.get_conf_nvshmem_enabled())            &
              .and. .not.is_udb) cycle
      else
        if ( is_backend_nccl(current_backend_id) .or. is_backend_nvshmem(current_backend_id) ) cycle
      endif
#endif
      call allocate_plans(plans, current_backend_id)

      if ( .not. is_backend_nvshmem(current_backend_id) ) then
        call alloc_mem(platform, helper, current_backend_id, cart_comm, alloc_size, in, ierr); DTFFT_CHECK(ierr)
        call alloc_mem(platform, helper, current_backend_id, cart_comm, alloc_size, out, ierr); DTFFT_CHECK(ierr)

        call c_f_pointer(in, pin, [min_buffer_size])
        call c_f_pointer(out, pout, [min_buffer_size])
      endif

      testing_phase = "Testing backend "//dtfft_get_backend_string(current_backend_id)
      REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
      WRITE_INFO(testing_phase)

      is_aux_alloc = .false.
      if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. effort == DTFFT_PATIENT) then
        call run_autotune_datatypes(helper, base_dtype, pencils, base_storage, is_z_slab, best_forward_ids, best_backward_ids, pin, pout, avg_execution_time)
      else
        create_kwargs%backend = current_backend_id

        if ( is_z_slab ) then
          create_kwargs%datatype_id = best_forward_ids(3)
          call plans(1)%p%create(pencils(1), pencils(3), base_storage, create_kwargs)
          create_kwargs%datatype_id = best_backward_ids(3)
          call plans(2)%p%create(pencils(3), pencils(1), base_storage, create_kwargs)
        else
          do i = 1, n_transpose_plans
            create_kwargs%datatype_id = best_forward_ids(i)
            call plans(i)%p%create(pencils(i), pencils(i + 1), base_storage, create_kwargs)
            create_kwargs%datatype_id = best_backward_ids(i)
            call plans(i + n_transpose_plans)%p%create(pencils(i + 1), pencils(i), base_storage, create_kwargs)
          enddo
        endif

        if ( is_backend_nvshmem(current_backend_id) ) then
          !! Symmetric heap can be allocated after nvshmem_init, which is done during plan creation
          call alloc_mem(platform, helper, current_backend_id, cart_comm, alloc_size, in, ierr); DTFFT_CHECK(ierr)
          call alloc_mem(platform, helper, current_backend_id, cart_comm, alloc_size, out, ierr); DTFFT_CHECK(ierr)

          call c_f_pointer(in, pin, [min_buffer_size])
          call c_f_pointer(out, pout, [min_buffer_size])
        endif

        call alloc_and_set_aux(platform, helper, current_backend_id, cart_comm, aux, paux, plans, is_aux_alloc)
        if( is_aux_alloc ) then
          execute_kwargs%p1 => paux
        else
          execute_kwargs%p1 => pout
        endif

        REGION_BEGIN("Warmup", COLOR_TRANSPOSE)
        do iter = 1, n_warmup_iters
          do i = 1, 2_int8 * n_transpose_plans
            call plans(i)%p%execute(pin, pout, execute_kwargs, ierr)
          enddo
        enddo
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
          CUDA_CALL( cudaStreamSynchronize(stream) )
        endif
#endif
        REGION_END("Warmup")

        call MPI_Barrier(cart_comm, mpi_ierr)

        REGION_BEGIN("Measure", COLOR_EXECUTE)
        if ( platform == DTFFT_PLATFORM_HOST ) then
          ts = MPI_Wtime()
#ifdef DTFFT_WITH_CUDA
        else
          CUDA_CALL( cudaEventRecord(timer_start, stream) )
#endif
        endif
        do iter = 1, n_iters
          do i = 1, 2_int8 * n_transpose_plans
            call plans(i)%p%execute(pin, pout, execute_kwargs, ierr)
          enddo
        enddo
        if ( platform == DTFFT_PLATFORM_HOST ) then
          te = MPI_Wtime()
          execution_time = real(te - ts, real32) * 1000._real32
#ifdef DTFFT_WITH_CUDA
        else
          CUDA_CALL( cudaEventRecord(timer_stop, stream) )
          CUDA_CALL( cudaEventSynchronize(timer_stop) )
          CUDA_CALL( cudaEventElapsedTime(execution_time, timer_start, timer_stop) )
#endif
        endif
        REGION_END("Measure")
        avg_execution_time = report_timings(cart_comm, execution_time, n_iters)
      endif
      ! execution_time = execution_time / real(n_iters, real32)


      ! call MPI_Allreduce(execution_time, min_execution_time, 1, MPI_REAL4, MPI_MIN, cart_comm, mpi_ierr)
      ! call MPI_Allreduce(execution_time, max_execution_time, 1, MPI_REAL4, MPI_MAX, cart_comm, mpi_ierr)
      ! call MPI_Allreduce(execution_time, avg_execution_time, 1, MPI_REAL4, MPI_SUM, cart_comm, mpi_ierr)

      ! avg_execution_time = avg_execution_time / real(comm_size, real32)

      ! WRITE_INFO("  max: "//to_str(real(max_execution_time, real64))//" [ms]")
      ! WRITE_INFO("  min: "//to_str(real(min_execution_time, real64))//" [ms]")
      ! WRITE_INFO("  avg: "//to_str(real(avg_execution_time, real64))//" [ms]")

      if ( avg_execution_time < best_time_ ) then
        best_time_ = avg_execution_time
        best_backend_ = current_backend_id
      endif

      call free_mem(platform, helper, current_backend_id, in, ierr)
      call free_mem(platform, helper, current_backend_id, out, ierr)
      if ( is_aux_alloc ) then
        call free_mem(platform, helper, current_backend_id, aux, ierr)
        is_aux_alloc = .false.
      endif

      call destroy_plans(plans)
      REGION_END("Testing backend "//dtfft_get_backend_string(current_backend_id))
    enddo

    deallocate( plans )
#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaEventDestroy(timer_start) )
      CUDA_CALL( cudaEventDestroy(timer_stop) )
    endif
#endif
    call helper%destroy()

    if ( present(best_time)) best_time = best_time_
    if ( present(best_backend) ) best_backend = best_backend_
  end subroutine run_autotune_backend

  subroutine run_autotune_datatypes(                                                              &
    helper, base_dtype, pencils, base_storage,                                                    &
    is_z_slab, best_forward_ids, best_backward_ids, a, b, elapsed_time)
    type(backend_helper),             intent(inout) :: helper
    TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
      !! Base MPI_Datatype
    type(pencil),                     intent(in)    :: pencils(:)
      !! Source meta
    integer(int64),                   intent(in)    :: base_storage
      !! Number of bytes needed to store single element
    logical,                          intent(in)    :: is_z_slab
      !! Is Z-slab optimization enabled
    integer(int8),                    intent(out)   :: best_forward_ids(:)
      !! Best Datatype ids for forward plan
    integer(int8),                    intent(out)   :: best_backward_ids(:)
      !! Best Datatype ids for backward plan
    real(real32),                     intent(inout) :: a(:)
      !! Source buffer
    real(real32),                     intent(inout) :: b(:)
      !! Target buffer
    real(real32),                     intent(out)   :: elapsed_time
      !! Elapsed time for best plans selected in [ms]
    integer(int8) :: dim        !! Counter
    integer(int8) :: ndims      !! Number of dimensions

    ndims = size(pencils, kind=int8)
    elapsed_time = 0._real32
    if( is_z_slab ) then
      elapsed_time = autotune_transpose_id(helper, pencils(1), pencils(3),                        &
        base_dtype, base_storage, 3_int8, a, b, best_forward_ids(3), best_backward_ids(3))
    else
      do dim = 1_int8, ndims - 1_int8
        elapsed_time = elapsed_time                                                               &
        + autotune_transpose_id(helper, pencils(dim), pencils(dim + 1),                           &
            base_dtype, base_storage, dim, a, b, best_forward_ids(dim), best_backward_ids(dim))
      enddo
    endif
    WRITE_INFO("  Execution time on a grid using fastest transpositions: "//to_str(real(elapsed_time, real64))//" [ms]")
  end subroutine run_autotune_datatypes

  function autotune_transpose_id(helper, from, to, base_dtype, base_storage, transpose_name_id, a, b, forward_id, backward_id) result(elapsed_time)
  !! Creates forward and backward transpose plans for backend `DTFFT_BACKEND_MPI_DATATYPE` based on source and target data distributions and,
  !! executes them `DTFFT_MEASURE_ITERS` times ( 4 * `DTFFT_MEASURE_ITERS` iterations total ) + 4 * `DTFFT_MEASURE_WARMUP_ITERS` warmup iterations
  !!
  !! Returns elapsed time for best plans selected
    type(backend_helper),         intent(inout) :: helper               !! Backend helper
    type(pencil),                 intent(in)    :: from                 !! Source meta
    type(pencil),                 intent(in)    :: to                   !! Target meta
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype           !! Basic MPI Datatype
    integer(int64),               intent(in)    :: base_storage         !! Number of bytes needed to store Basic MPI Datatype
    integer(int8),                intent(in)    :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
    real(real32),                 intent(inout) :: a(:)                 !! Source buffer
    real(real32),                 intent(inout) :: b(:)                 !! Target buffer
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
      time = get_plan_execution_time(helper, from, to, base_dtype, base_storage, datatype_id, transpose_name_id, a, b)
      if ( time < forward_time ) then
        forward_time = time
        forward_id = datatype_id
      endif

      time = get_plan_execution_time(helper, to, from, base_dtype, base_storage, datatype_id, -1_int8 * transpose_name_id, a, b)
      if ( time < backward_time ) then
        backward_time = time
        backward_id = datatype_id
      endif
    enddo
    elapsed_time = forward_time + backward_time
  end function autotune_transpose_id

  function get_plan_execution_time(helper, from, to, base_dtype, base_storage, datatype_id, transpose_name_id, a, b) result(elapsed_time)
  !! Creates transpose plan for backend `DTFFT_BACKEND_MPI_DATATYPE` and executes it `DTFFT_MEASURE_WARMUP_ITERS` + `DTFFT_MEASURE_ITERS` times
  !!
  !! Returns elapsed time
    type(backend_helper),         intent(inout) :: helper               !! Backend helper
    type(pencil),                 intent(in)    :: from                 !! Source meta
    type(pencil),                 intent(in)    :: to                   !! Target meta
    TYPE_MPI_DATATYPE,            intent(in)    :: base_dtype           !! Basic MPI Datatype
    integer(int64),               intent(in)    :: base_storage         !! Number of bytes needed to store Basic MPI Datatype
    integer(int8),                intent(in)    :: datatype_id          !! ID of transpose (1 or 2)
    integer(int8),                intent(in)    :: transpose_name_id    !! ID of transpose name (from -3 to 3, except 0)
    real(real32),                 intent(inout) :: a(:)                 !! Source buffer
    real(real32),                 intent(inout) :: b(:)                 !! Target buffer
    real(real32)                                :: elapsed_time         !! Execution time [ms]
    real(real32)                                :: time                 !! Timer
    character(len=:),             allocatable   :: phase_name           !! Caliper phase name
    type(transpose_handle_datatype)             :: plan                 !! Transpose plan
    real(real64)                                :: ts, te               !! Timers
    integer(int32)                              :: iter                 !! Counter
    integer(int32)                              :: ierr                 !! Error code
    type(create_args)   :: create_kwargs
    type(execute_args) :: exec_kwargs
    integer(int32) :: n_iters

    allocate( phase_name, source="  Testing plan "//TRANSPOSE_NAMES(transpose_name_id)//", datatype_id = "//to_str(datatype_id) )
    REGION_BEGIN(phase_name, 0)
    WRITE_INFO(phase_name)

    create_kwargs%base_type = base_dtype
    create_kwargs%datatype_id = datatype_id
    create_kwargs%helper = helper

    call plan%create(from, to, base_storage, create_kwargs)

    exec_kwargs%exec_type = EXEC_BLOCKING

    do iter = 1, get_conf_measure_warmup_iters()
      call plan%execute(a, b, exec_kwargs, ierr)
    enddo
    call MPI_Barrier(helper%comms(1), ierr)
    n_iters = get_conf_measure_iters()
    ts = MPI_Wtime()
    do iter = 1, n_iters
      call plan%execute(a, b, exec_kwargs, ierr)
    enddo
    te = MPI_Wtime()

    time = real(te - ts, real32) * 1000._real32
    elapsed_time = report_timings(helper%comms(1), time, n_iters, 4)

    call plan%destroy()
    REGION_END(phase_name)
    deallocate(phase_name)
  end function get_plan_execution_time

  function report_timings(comm, elapsed_time, n_iters, space_count) result(max_time)
    TYPE_MPI_COMM,            intent(in)  :: comm
    real(real32),             intent(in)  :: elapsed_time
    integer(int32),           intent(in)  :: n_iters
    integer(int32), optional, intent(in)  :: space_count
    real(real32)                          :: max_time
    real(real32) :: execution_time, min_time, avg_time
    integer(int32) :: ierr, comm_size, space_count_

    execution_time = elapsed_time / real(n_iters, real32)
    space_count_ = 2; if ( present(space_count) ) space_count_ = space_count

    call MPI_Allreduce(execution_time, min_time, 1, MPI_REAL4, MPI_MIN, comm, ierr)
    call MPI_Allreduce(execution_time, max_time, 1, MPI_REAL4, MPI_MAX, comm, ierr)
    call MPI_Allreduce(execution_time, avg_time, 1, MPI_REAL4, MPI_SUM, comm, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)

    avg_time = avg_time / real(comm_size, real32)

    WRITE_INFO(repeat(" ", space_count_)//"max: "//to_str(real(max_time, real64))//" [ms]")
    WRITE_INFO(repeat(" ", space_count_)//"min: "//to_str(real(min_time, real64))//" [ms]")
    WRITE_INFO(repeat(" ", space_count_)//"avg: "//to_str(real(avg_time, real64))//" [ms]")
  end function report_timings

  function get_aux_size(self) result(aux_size)
  !! Returns maximum auxiliary memory size needed by transpose plan
    class(transpose_plan), intent(in)    :: self  !! Transposition class
    integer(int64) :: aux_size

    aux_size = get_aux_size_generic(self%plans)
  end function get_aux_size

  function get_aux_size_generic(plans) result(aux_size)
  !! Returns maximum auxiliary memory size needed by plans
    type(plan_t),   intent(in)  :: plans(:)       !! Transpose plans
    integer(int64)              :: aux_size       !! Maximum auxiliary memory size needed
    integer(int64), allocatable :: worksizes(:)   !! Auxiliary memory sizes needed by each plan
    integer(int32)  :: n_transpose_plans, i

    n_transpose_plans = size(plans)
    allocate( worksizes( n_transpose_plans ), source=0_int64 )

    do i = 1, n_transpose_plans
      if ( allocated( plans(i)%p ) ) then
        worksizes(i) = plans(i)%p%get_aux_size()
      endif
    enddo
    aux_size = maxval(worksizes)
    deallocate(worksizes)
  end function get_aux_size_generic

  subroutine alloc_and_set_aux(platform, helper, backend, cart_comm, aux, paux, plans, is_aux_alloc)
  !! Allocates auxiliary memory according to the backend and sets it to the plans
    type(dtfft_platform_t),       intent(in)                :: platform
    type(backend_helper),         intent(inout)             :: helper       !! Backend helper
    type(dtfft_backend_t),        intent(in)                :: backend      !! GPU backend
    TYPE_MPI_COMM,                intent(in)                :: cart_comm    !! Cartesian communicator
    type(c_ptr),                  intent(inout)             :: aux          !! Allocatable auxiliary memory
    real(real32),     pointer,    intent(inout)             :: paux(:)      !! Pointer to auxiliary memory
    type(plan_t),                 intent(in)                :: plans(:)
    logical                                                 :: is_aux_alloc !! Is auxiliary memory allocated
    integer(int64) :: max_work_size_local, max_work_size_global
    integer(int32)  :: mpi_ierr
    integer(int32) :: alloc_ierr

    max_work_size_local = get_aux_size_generic(plans)
    call MPI_Allreduce(max_work_size_local, max_work_size_global, 1, MPI_INTEGER8, MPI_MAX, cart_comm, mpi_ierr)

    is_aux_alloc = .false.
    paux => null()
    if ( max_work_size_global > 0 ) then
      call alloc_mem(platform, helper, backend, cart_comm, max_work_size_global, aux, alloc_ierr)
      DTFFT_CHECK(alloc_ierr)
      call c_f_pointer(aux, paux, [max_work_size_global / 4_int64])
      is_aux_alloc = .true.
    endif
  end subroutine alloc_and_set_aux

  subroutine create_pencils_and_comm(transposed_dims, old_comm, comm_dims, comm, local_comms, pencils, ipencil)
  !! Creates cartesian communicator
    integer(int32),       intent(in)            :: transposed_dims(:,:) !! Global counts in transposed coordinates
    TYPE_MPI_COMM,        intent(in)            :: old_comm             !! Communicator to create cartesian from
    integer(int32),       intent(in)            :: comm_dims(:)         !! Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: comm                 !! Cartesian communicator
    TYPE_MPI_COMM,        intent(out)           :: local_comms(:)       !! 1d communicators in cartesian communicator
    type(pencil),         intent(out)           :: pencils(:)           !! Data distributing meta
    type(pencil_init),    intent(in), optional  :: ipencil              !! Pencil passed by user
    integer(int8)         :: ndims              !! Number of dimensions
    integer(int8)         :: d                  !! Counter
    integer(int8)         :: order(3)

    ndims = size(comm_dims, kind=int8)
    order = [1_int8, 3_int8, 2_int8]

    call create_cart_comm(old_comm, comm_dims, comm, local_comms, ipencil=ipencil)
    if ( present(ipencil) ) then
      block
        integer(int32), allocatable :: lstarts(:), lcounts(:)

        allocate(lstarts, source=ipencil%starts)
        allocate(lcounts, source=ipencil%counts)
        do d = 1, ndims
          if( ndims == 3 .and. d == 2 ) then
            call pencils(d)%create(ndims, d, transposed_dims(:,d), local_comms, lstarts=lstarts, lcounts=lcounts, order=order)
          else
            call pencils(d)%create(ndims, d, transposed_dims(:,d), local_comms, lstarts=lstarts, lcounts=lcounts)
          endif
          lcounts(:) = pencils(d)%counts(:)
          lstarts(:) = pencils(d)%starts(:)
        enddo

        deallocate(lstarts, lcounts)
      endblock
    else
      do d = 1, ndims
        if( ndims == 3 .and. d == 2 ) then
          call pencils(d)%create(ndims, d, transposed_dims(:,d), local_comms, order=order)
        else
          call pencils(d)%create(ndims, d, transposed_dims(:,d), local_comms)
        endif
      enddo
    endif
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

  type(dtfft_backend_t) function get_backend(self)
  !! Returns plan GPU backend
    class(transpose_plan), intent(in)    :: self           !! Transposition class
    get_backend = self%backend
  end function get_backend

  subroutine mem_alloc(self, comm, alloc_bytes, ptr, error_code)
  !! Allocates memory based on selected backend
    class(transpose_plan),  intent(inout) :: self           !! Transposition class
    TYPE_MPI_COMM,          intent(in)    :: comm           !! MPI communicator
    integer(int64),         intent(in)    :: alloc_bytes    !! Number of bytes to allocate
    type(c_ptr),            intent(out)   :: ptr            !! Pointer to the allocated memory
    integer(int32),         intent(out)   :: error_code     !! Error code

    call alloc_mem(self%platform, self%helper, self%backend, comm, alloc_bytes, ptr, error_code)
  end subroutine mem_alloc

  subroutine mem_free(self, ptr, error_code)
  !! Frees memory allocated with mem_alloc
    class(transpose_plan),  intent(inout) :: self           !! Transposition class
    type(c_ptr),            intent(in)    :: ptr            !! Pointer to the memory to free
    integer(int32),         intent(out)   :: error_code     !! Error code

    call free_mem(self%platform, self%helper, self%backend, ptr, error_code)
  end subroutine mem_free

  subroutine alloc_mem(platform, helper, backend, comm, alloc_bytes, ptr, error_code)
  !! Allocates memory based on ``backend``
    type(dtfft_platform_t), intent(in)    :: platform
    type(backend_helper),   intent(inout) :: helper         !! Backend helper
    type(dtfft_backend_t),  intent(in)    :: backend        !! GPU backend
    TYPE_MPI_COMM,          intent(in)    :: comm           !! MPI communicator
    integer(int64),         intent(in)    :: alloc_bytes    !! Number of bytes to allocate
    type(c_ptr),            intent(out)   :: ptr            !! Pointer to the allocated memory
    integer(int32),         intent(out)   :: error_code     !! Error code
#ifdef DTFFT_WITH_CUDA
    integer(int64)  :: free_mem_avail, total_mem_avail
#endif

    error_code = DTFFT_SUCCESS
    if ( platform == DTFFT_PLATFORM_HOST ) then
      ptr = mem_alloc_host(alloc_bytes)
#ifdef DTFFT_WITH_CUDA
    else
      CUDA_CALL( cudaMemGetInfo(free_mem_avail, total_mem_avail) )
# ifdef DTFFT_DEBUG
      block
        integer(int64) :: min_mem, max_mem, min_free_mem, max_free_mem
        integer(int32) :: mpi_err

        call MPI_Allreduce(alloc_bytes, max_mem, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
        call MPI_Allreduce(alloc_bytes, min_mem, 1, MPI_INTEGER8, MPI_MIN, comm, mpi_err)
        call MPI_Allreduce(free_mem_avail, max_free_mem, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
        call MPI_Allreduce(free_mem_avail, min_free_mem, 1, MPI_INTEGER8, MPI_MIN, comm, mpi_err)
        WRITE_DEBUG("Trying to allocate "//to_str(min_mem)//"/"//to_str(max_mem)//" (min/max) bytes for backend: '"//dtfft_get_backend_string(backend)//"'")
        WRITE_DEBUG("Free memory available: "//to_str(min_free_mem)//"/"//to_str(max_free_mem)//" (min/max) bytes")
      endblock
# endif
      if ( alloc_bytes > free_mem_avail ) then
        error_code = DTFFT_ERROR_ALLOC_FAILED
        return
      endif
      if ( is_backend_nccl(backend) ) then
# ifdef DTFFT_WITH_NCCL
#   ifdef NCCL_HAVE_MEMALLOC
        error_code = ncclMemAlloc(ptr, alloc_bytes)
#   else
        error_code = cudaMalloc(ptr, alloc_bytes)
#   endif
#   ifdef NCCL_HAVE_COMMREGISTER
        if ( error_code == cudaSuccess .and. helper%should_register ) then
          block
            type(c_ptr), allocatable :: temp(:,:)
            type(c_ptr) :: handle

            if ( size(helper%nccl_register, dim=2) == helper%nccl_register_size ) then
              allocate( temp(2, helper%nccl_register_size + NCCL_REGISTER_PREALLOC_SIZE) )
              temp(2, 1:helper%nccl_register_size) = helper%nccl_register(2, 1:helper%nccl_register_size)
              deallocate( helper%nccl_register )
              call move_alloc(temp, helper%nccl_register)
            endif
            helper%nccl_register_size = helper%nccl_register_size + 1

            NCCL_CALL( ncclCommRegister(helper%nccl_comm, ptr, alloc_bytes, handle) )
            helper%nccl_register(1, helper%nccl_register_size) = ptr
            helper%nccl_register(2, helper%nccl_register_size) = handle
            WRITE_DEBUG("Registered pointer "//to_str(transfer(ptr, int64)))
          endblock
        endif
#   endif
# else
        INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
    else if ( is_backend_nvshmem(backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
      block
        integer(int64)  :: max_alloc_bytes
        integer(int32)  :: mpi_err
        call MPI_Allreduce(alloc_bytes, max_alloc_bytes, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
        ptr = nvshmem_malloc(max_alloc_bytes)
        if ( is_null_ptr(ptr) ) error_code = DTFFT_ERROR_ALLOC_FAILED
      endblock
# else
      INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
    else
      error_code = cudaMalloc(ptr, alloc_bytes)
    endif
#endif
    endif
    if ( error_code /= DTFFT_SUCCESS ) error_code = DTFFT_ERROR_ALLOC_FAILED
  end subroutine alloc_mem

  subroutine free_mem(platform, helper, backend, ptr, error_code)
  !! Frees memory based on ``backend``
    type(dtfft_platform_t),         intent(in)    :: platform
    type(backend_helper),           intent(inout) :: helper         !! Backend helper
    type(dtfft_backend_t),          intent(in)    :: backend        !! GPU backend
    type(c_ptr),                    intent(in)    :: ptr            !! Pointer to the memory to free
    integer(int32),                 intent(out)   :: error_code     !! Error code

    error_code = DTFFT_SUCCESS
    if ( platform == DTFFT_PLATFORM_HOST ) then
      call mem_free_host(ptr)
#ifdef DTFFT_WITH_CUDA
    else
      if ( is_backend_nccl(backend) ) then
# ifdef NCCL_HAVE_COMMREGISTER
      if ( helper%should_register ) then
        block
          integer(int32) :: i

          do i = 1, size(helper%nccl_register, dim=2)
            if ( .not. is_same_ptr(ptr, helper%nccl_register(1, i)) ) cycle
            NCCL_CALL( ncclCommDeregister(helper%nccl_comm, helper%nccl_register(2, i)) )
            helper%nccl_register(1, i) = c_null_ptr
            helper%nccl_register(2, i) = c_null_ptr
            helper%nccl_register_size = helper%nccl_register_size - 1
            WRITE_DEBUG("Pointer "//to_str(transfer(ptr, int64))//" has been removed from registry")
          enddo
        endblock
      endif
# endif
# ifdef DTFFT_WITH_NCCL
#   ifdef NCCL_HAVE_MEMALLOC
      error_code = ncclMemFree(ptr)
#   else
      error_code = cudaFree(ptr)
#   endif
# else
      INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
    else if ( is_backend_nvshmem(backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
      call nvshmem_free(ptr)
# else
      INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
    else
      error_code = cudaFree(ptr)
    endif
#endif
  endif
    if ( error_code /= DTFFT_SUCCESS ) error_code = DTFFT_ERROR_FREE_FAILED
  end subroutine free_mem
end module dtfft_transpose_plan
