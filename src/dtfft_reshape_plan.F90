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
module dtfft_reshape_plan
!! This module describes [[reshape_plan]] class
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend,               only: backend_helper
use dtfft_abstract_reshape_handle,        only: abstract_reshape_handle, reshape_container, create_args
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_pencil,                         only: dtfft_pencil_t, pencil, pencil_init, get_local_sizes
use dtfft_reshape_plan_base,              only: reshape_plan_base, allocate_plans, destroy_plans, execute_autotune
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_profile.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: reshape_plan


  integer(int32), parameter :: NEIGHBOR_GROUP = 1
  integer(int32), parameter :: STRIDED_GROUP = 2


  type, extends(reshape_plan_base) :: reshape_plan
  !! Reshape Plan class
  !! This class is a container for transposition plans
  private
    integer(int32), allocatable :: init_grid(:)
    integer(int32), allocatable :: final_grid(:)
    TYPE_MPI_COMM,  allocatable :: comms(:)
  contains
  private
    procedure,  non_overridable,  pass(self), public :: create            !! Creates transpose plan
    procedure,  non_overridable,  pass(self), public :: get_grid          !! Returns grid for specified reshape
    procedure,  non_overridable,  pass(self), public :: destroy
  end type reshape_plan

contains

  subroutine destroy(self)
    class(reshape_plan),            intent(inout) :: self
    integer(int32) :: d, ierr

    if ( allocated(self%init_grid) ) deallocate(self%init_grid)
    if ( allocated(self%final_grid) ) deallocate(self%final_grid)
    call self%reshape_plan_base%destroy()
    if ( allocated(self%comms) ) then
      do d = 2, 3
        call MPI_Comm_free(self%comms(d), ierr)
      enddo
      deallocate(self%comms)
    endif
  end subroutine destroy

  subroutine get_grid(self, dim, grid)
  !! Returns grid decomposition for specified dimension
    class(reshape_plan),  target, intent(in)  :: self
      !! Reshape plan
    integer(int8),                intent(in)  :: dim
      !! Dimension: 1 for initial grid, other values for final grid
    integer(int32),     pointer,  intent(out) :: grid(:)
      !! Pointer to grid array

    if ( dim == 1 ) then
      grid => self%init_grid
    else
      grid => self%final_grid
    endif
  end subroutine get_grid

  function create(self, platform, ipencil, pencils, comm, local_comms, base_dtype, base_storage, effort, backend, base_init_dtype, base_init_storage, bricks, is_final_enabled) result(error_code)
  !! Creates reshape plan
  !!
  !! This function initializes the reshape plan by creating communicators for brick layouts,
  !! setting up grid decompositions, and allocating reshape operation plans.
    class(reshape_plan),            intent(inout) :: self
      !! Reshape plan to be initialized
    type(dtfft_platform_t),         intent(in)    :: platform
      !! Platform to create plan for (HOST or CUDA)
    type(pencil_init),              intent(in)    :: ipencil
      !! Pencil decomposition passed by user
    type(pencil),                   intent(in)    :: pencils(:)
      !! Array of pencil decompositions for different layouts
    TYPE_MPI_COMM,                  intent(in)    :: comm
      !! Global MPI communicator
    TYPE_MPI_COMM,                  intent(in)    :: local_comms(:)
      !! Local MPI communicators for each dimension
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype
      !! Base MPI datatype for complex data
    integer(int64),                 intent(in)    :: base_storage
      !! Number of bytes needed to store single complex element
    type(dtfft_effort_t),           intent(in)    :: effort
      !! dtFFT planner effort level (ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE)
    type(dtfft_backend_t),          intent(in)    :: backend
      !! Communication backend to use
    TYPE_MPI_DATATYPE,              intent(in)    :: base_init_dtype
      !! Base MPI datatype for real data
    integer(int64),                 intent(in)    :: base_init_storage
      !! Number of bytes needed to store single real element
    type(pencil),                   intent(out)   :: bricks(:)
      !! Pencils describing brick data distribution
    logical,                        intent(out)   :: is_final_enabled
      !! Flag indicating if final reshape in Fourier space is enabled
    integer(int32)                                :: error_code
      !! Error code: DTFFT_SUCCESS on success
    integer(int8)     :: ndims      !! Number of dimensions
    integer(int8)     :: d
    integer(int32)    :: c_size, ierr
    integer(int32)    :: bsize
    TYPE_MPI_COMM, allocatable     :: temp_comms(:)
    integer(int64) :: min_buffer_size_real, min_buffer_size_complex
    real(real64) :: ts, te

    error_code = DTFFT_SUCCESS
    CHECK_CALL( self%init(platform, effort), error_code )

    ndims = size(ipencil%dims, kind=int8)
    call bricks(1)%create(ndims, 1_int8, ipencil%dims, ipencil%comms, lstarts=ipencil%starts, lcounts=ipencil%counts)

    call MPI_Comm_size(ipencil%comms(ndims), c_size, ierr)

    allocate( self%init_grid(ndims), self%final_grid(ndims) )
    do d = 1, ndims
      call MPI_Comm_size(ipencil%comms(d), self%init_grid(d), ierr)
    enddo

    allocate( temp_comms(ndims), self%comms(3) )

    call MPI_Comm_dup(ipencil%comms(1), self%comms(2), ierr)
    call create_custom_comm(local_comms(ndims), c_size, NEIGHBOR_GROUP, self%comms(3))
    call create_custom_comm(local_comms(ndims), c_size, STRIDED_GROUP, temp_comms(ndims))
    if ( ndims == 3 ) then
      temp_comms(2) = local_comms(2)
    endif

    call MPI_Comm_rank(local_comms(ndims), c_size, ierr)
    call MPI_Comm_rank(self%comms(3), bsize, ierr)

    block
      integer(int32) :: slow_count, min_start
      TYPE_MPI_COMM :: final_comms(ndims)
      integer(int32) :: final_sizes(ndims)

      call MPI_Allreduce(pencils(ndims)%counts(ndims), slow_count, 1, MPI_INTEGER, MPI_SUM, self%comms(3), ierr)

      final_comms(1) = self%comms(3)
      final_sizes(1) = ipencil%dims(ndims)
      if ( ndims == 2 ) then
        final_comms(2) = MPI_COMM_SELF
        final_sizes(2) = slow_count
      else  
        final_comms(2) = MPI_COMM_SELF
        final_comms(3) = MPI_COMM_SELF

        final_sizes(2) = pencils(ndims)%counts(2)
        final_sizes(3) = slow_count
      endif

      call bricks(2)%create(ndims, ndims, final_sizes, final_comms, pencils(ndims)%starts, pencils(ndims)%counts, .true.)

      call MPI_Allreduce(pencils(ndims)%starts(ndims), min_start, 1, MPI_INTEGER, MPI_MIN, self%comms(3), ierr)
      bricks(2)%starts(ndims) = bricks(2)%starts(ndims) + min_start

      if ( ndims == 3 ) then
        bricks(2)%starts(2) = pencils(ndims)%starts(2)
      endif
    end block

    temp_comms(1) = self%comms(3)
    call MPI_Comm_size(temp_comms(1), self%final_grid(ndims), ierr)
    call MPI_Comm_size(temp_comms(2), self%final_grid(1), ierr)

    if ( ndims == 3 ) then
      call MPI_Comm_size(temp_comms(3), self%final_grid(2), ierr)
    endif
    is_final_enabled = .false.
    if ( self%final_grid(ndims) > 1 .and. get_conf_fourier_reshape_enabled() ) is_final_enabled = .true.

    call MPI_Comm_free(temp_comms(ndims), ierr)
    deallocate( temp_comms )

    self%backend = get_conf_reshape_backend()
    if ( .not. is_backend_compatible(backend, self%backend) .and. effort%val < DTFFT_PATIENT%val ) then
      WRITE_WARN("Incompatible reshape backend detected, it has been ignored")
      self%backend = get_compatible(backend, platform)
    endif

    if ( effort%val < DTFFT_EXHAUSTIVE%val .and. platform == DTFFT_PLATFORM_HOST ) then
      if ( is_backend_nccl(self%backend) .or. is_backend_nvshmem(self%backend) ) then
        error_code = DTFFT_ERROR_INVALID_PLATFORM_BACKEND
        return
      endif
    endif

    call get_local_sizes([bricks(1), pencils(1)], alloc_size=min_buffer_size_real)
    call get_local_sizes([bricks(2), pencils(ndims)], alloc_size=min_buffer_size_complex)

    min_buffer_size_real = min_buffer_size_real * (base_init_storage / FLOAT_STORAGE_SIZE)
    min_buffer_size_complex = min_buffer_size_complex * (base_storage / FLOAT_STORAGE_SIZE)
    self%min_buffer_size = max(min_buffer_size_real, min_buffer_size_complex)

    if ( effort == DTFFT_EXHAUSTIVE ) then
      ts = MPI_Wtime()
  
      PHASE_BEGIN("Autotune reshape plan", COLOR_AUTOTUNE)
      WRITE_INFO("Starting autotune of reshape plans...")
      call autotune_reshape_plan(platform, comm, self%comms, base_init_dtype, base_init_storage, base_dtype, base_init_storage, &
        bricks, pencils, self%stream, self%min_buffer_size * FLOAT_STORAGE_SIZE, backend, self%backend)
      PHASE_END("Autotune reshape plan")

      te = MPI_Wtime()
      WRITE_INFO("Time spent on autotune: "//to_str(te - ts)//" [s]")
    endif

    ts = MPI_Wtime()
    call self%helper%create(platform, comm, self%comms, is_backend_nccl(self%backend), [bricks(1), bricks(2), pencils(1), pencils(ndims)])
    call create_reshape_plans(self%plans, self%backend, platform, self%helper, effort, .false., &
         base_init_dtype, base_init_storage, base_dtype, base_storage, bricks, pencils)

    te = MPI_Wtime()
    WRITE_INFO("Time spent on creating final reshape plans: "//to_str(te - ts)//" [s]")


    allocate( self%names(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS:CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS) )
    do d = CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS
      self%names(d) = string(RESHAPE_NAMES(d))
    end do
  end function create

  subroutine create_custom_comm(old_comm, new_size, group_type, new_comm)
  !! Creates custom MPI communicator by splitting processes into groups
  !!
  !! This subroutine divides processes from an existing communicator into groups
  !! based on the specified group type: neighbor groups or strided groups.
    TYPE_MPI_COMM,  intent(in)  :: old_comm
      !! Original MPI communicator to split
    integer(int32), intent(in)  :: new_size
      !! Size of each group in the new communicator
    integer(int32), intent(in)  :: group_type
      !! Type of grouping: NEIGHBOR_GROUP (consecutive ranks) or STRIDED_GROUP (interleaved ranks)
    TYPE_MPI_COMM,  intent(out) :: new_comm
      !! Newly created MPI communicator
    integer(int32)  :: ierror
    integer(int32) :: old_rank, old_size, color

    call MPI_Comm_rank(old_comm, old_rank, ierror)
    call MPI_Comm_size(old_comm, old_size, ierror)

    select case(group_type)
    case(NEIGHBOR_GROUP)
      color = old_rank / new_size
    case(STRIDED_GROUP)
      color = mod(old_rank, new_size)
    end select
    call MPI_Comm_split(old_comm, color, old_rank, new_comm, ierror)
  end subroutine create_custom_comm

  pure logical function is_backend_compatible(backend1, backend2)
  !! Checks if two communication backends are compatible
  !!
  !! Backends are compatible if they belong to the same family:
  !! MPI backends are compatible with each other, NCCL with NCCL, NVSHMEM with NVSHMEM.
  !!
  !! @return .true. if backends are compatible, .false. otherwise
    type(dtfft_backend_t), intent(in) :: backend1
      !! First backend to compare
    type(dtfft_backend_t), intent(in) :: backend2
      !! Second backend to compare

    is_backend_compatible = .true.
    if ( is_backend_mpi(backend1) .and. is_backend_mpi(backend2) ) return
    if ( is_backend_nccl(backend1) .and. is_backend_nccl(backend2) ) return
    if ( is_backend_nvshmem(backend1) .and. is_backend_nvshmem(backend2) ) return
    is_backend_compatible = .false.
  end function is_backend_compatible

  pure type(dtfft_backend_t) function get_compatible(backend, platform)
  !! Returns a compatible reshape backend for the given main backend and platform
  !!
  !! Selects an appropriate reshape backend based on the main backend family:
  !! - For MPI backends: returns MPI_P2P (GPU) or MPI_DATATYPE (CPU)
  !! - For NCCL backends: returns NCCL
  !! - For NVSHMEM backends: returns CUFFTMP
  !!
  !! @return Compatible backend for reshape operations
    type(dtfft_backend_t),  intent(in) :: backend
      !! Input backend from main FFT plan
    type(dtfft_platform_t), intent(in) :: platform
      !! Execution platform (HOST or CUDA)

    if ( is_backend_mpi(backend) ) then
      if ( platform == DTFFT_PLATFORM_CUDA ) then
        get_compatible = DTFFT_BACKEND_MPI_P2P
      else
        get_compatible = DTFFT_BACKEND_MPI_DATATYPE
      endif
    else if ( is_backend_nccl(backend) ) then
      get_compatible = DTFFT_BACKEND_NCCL
    else if ( is_backend_nvshmem(backend) ) then
      get_compatible = DTFFT_BACKEND_CUFFTMP
    endif
  end function get_compatible

  subroutine create_reshape_plans(plans, backend, platform, helper, effort, force_effort, base_init_dtype, base_init_storage, base_dtype, base_storage, bricks, pencils)
  !! Creates and allocates all reshape operation plans
  !!
  !! This subroutine allocates and initializes plans for all four reshape operations:
  !! - X_BRICKS_TO_PENCILS: brick to pencil in X dimension (real space)
  !! - X_PENCILS_TO_BRICKS: pencil to brick in X dimension (real space)
  !! - Z_PENCILS_TO_BRICKS: pencil to brick in Z dimension (Fourier space)
  !! - Z_BRICKS_TO_PENCILS: brick to pencil in Z dimension (Fourier space)
    type(reshape_container),  allocatable,  intent(inout) :: plans(:)
      !! Array of reshape plan containers to be allocated and initialized
    type(dtfft_backend_t),                  intent(in)    :: backend
      !! Communication backend to use for reshape operations
    type(dtfft_platform_t),                 intent(in)    :: platform
      !! Execution platform (HOST or CUDA)
    type(backend_helper),                   intent(inout) :: helper
      !! Backend helper for communication setup
    type(dtfft_effort_t),                   intent(in)    :: effort
      !! dtFFT planner effort level
    logical,                                intent(in)    :: force_effort
      !! 
    TYPE_MPI_DATATYPE,                      intent(in)    :: base_init_dtype
      !! Base MPI datatype for real space data
    integer(int64),                         intent(in)    :: base_init_storage
      !! Number of bytes needed to store single real element
    TYPE_MPI_DATATYPE,                      intent(in)    :: base_dtype
      !! Base MPI datatype for Fourier space data
    integer(int64),                         intent(in)    :: base_storage
      !! Number of bytes needed to store single complex element
    type(pencil),                           intent(in)    :: bricks(:)
      !! Pencils describing brick data distribution
    type(pencil),                           intent(in)    :: pencils(:)
      !! Array of pencil decompositions
     type(create_args) :: args

    allocate( plans(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS:CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS) )

    call allocate_plans(plans, backend)

    args%platform = platform
    args%helper = helper
    args%effort = effort
    args%backend = backend
    args%force_effort = force_effort

    args%base_type = base_init_dtype
    args%base_storage = base_init_storage
    call plans(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS)%p%create(bricks(1), pencils(1), args)
    call plans(CONF_DTFFT_RESHAPE_X_PENCILS_TO_BRICKS)%p%create(pencils(1), bricks(1), args)

    args%base_type = base_dtype
    args%base_storage = base_storage
    call plans(CONF_DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS)%p%create(pencils(size(pencils)), bricks(2), args)
    call plans(CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS)%p%create(bricks(2), pencils(size(pencils)), args)
  end subroutine create_reshape_plans

  subroutine autotune_reshape_plan(                                                                &
    platform, base_comm, comms, base_init_dtype, base_init_storage, base_dtype, base_storage, bricks, pencils, stream,     &
    buffer_size, transpose_backend, best_backend)
  !! Runs autotune for all backends
    type(dtfft_platform_t),           intent(in)    :: platform
      !! Platform to create plan for
    TYPE_MPI_COMM,                    intent(in)    :: base_comm
      !! 3D Cartesian comm
    TYPE_MPI_COMM,                    intent(in)    :: comms(:)
      !! 1D comms
    TYPE_MPI_DATATYPE,                intent(in)    :: base_init_dtype
      !! Base MPI datatype for real space data
    integer(int64),                   intent(in)    :: base_init_storage
      !! Number of bytes needed to store single real element
    TYPE_MPI_DATATYPE,                intent(in)    :: base_dtype
      !! Base MPI datatype for Fourier space data
    integer(int64),                   intent(in)    :: base_storage
      !! Number of bytes needed to store single complex element
    type(pencil),                     intent(in)    :: bricks(:)
      !! Pencils describing brick data distribution
    type(pencil),                     intent(in)    :: pencils(:)
      !! Array of pencil decompositions
    type(dtfft_stream_t),             intent(in)    :: stream
      !! Stream to use
    integer(int64),                   intent(in)    :: buffer_size
      !! Size of the buffer to use during autotune (in bytes)
    type(dtfft_backend_t),            intent(in)    :: transpose_backend
      !! Backend used in transpose plans
    type(dtfft_backend_t),            intent(out)   :: best_backend
      !! Best backend selected
    type(dtfft_backend_t),  allocatable :: backends_to_run(:)
    type(dtfft_backend_t) :: current_backend_id
    real(real32) :: execution_time, best_time_
    integer(int32) :: b
    type(reshape_container), allocatable :: plans(:)
    logical :: nccl_enabled
#ifdef DTFFT_WITH_CUDA
    logical :: nvshmem_enabled
#endif
    character(len=:), allocatable :: testing_phase
    type(backend_helper)                      :: helper
    logical :: pipe_enabled, mpi_enabled, dtype_enabled

    allocate( backends_to_run, source=VALID_BACKENDS )

    nccl_enabled = .false.
#ifdef DTFFT_WITH_CUDA
    nccl_enabled = platform == DTFFT_PLATFORM_CUDA .and. get_conf_nccl_enabled()
    nvshmem_enabled = get_conf_nvshmem_enabled()
#endif

    call helper%create(platform, base_comm, comms, nccl_enabled, [bricks(1), bricks(2), pencils(1), pencils(size(pencils))])

    pipe_enabled = get_conf_pipelined_enabled()
    dtype_enabled = get_conf_datatype_enabled()
    mpi_enabled = get_conf_mpi_enabled()

    best_time_ = MAX_REAL32

    do b = 1, size(backends_to_run)
      current_backend_id = backends_to_run(b)
      if ( .not. is_backend_compatible(transpose_backend, current_backend_id) ) cycle
      if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. platform == DTFFT_PLATFORM_CUDA ) cycle
      if ( is_backend_pipelined(current_backend_id) .and. .not.pipe_enabled ) cycle
      if ( is_backend_mpi(current_backend_id) .and. .not.mpi_enabled .and. .not.current_backend_id == DTFFT_BACKEND_MPI_DATATYPE ) cycle
      if ( current_backend_id == DTFFT_BACKEND_MPI_DATATYPE .and. .not.dtype_enabled ) cycle
#ifdef DTFFT_WITH_CUDA
      if ( platform == DTFFT_PLATFORM_CUDA ) then
        if ( is_backend_nvshmem(current_backend_id) .and. .not.nvshmem_enabled ) cycle
        if ( is_backend_nccl(current_backend_id) .and. .not.nccl_enabled) cycle
        ! DTFFT_BACKEND_CUFFTMP == DTFFT_BACKEND_CUFFTMP_PIPELINED for this plan
        if ( current_backend_id == DTFFT_BACKEND_CUFFTMP_PIPELINED ) cycle
      else
        if ( is_backend_nccl(current_backend_id) .or. is_backend_nvshmem(current_backend_id) ) cycle
      endif
#endif

      testing_phase = "Testing backend "//dtfft_get_backend_string(current_backend_id)
      REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
      WRITE_INFO(testing_phase)

      call create_reshape_plans(plans, current_backend_id, platform, helper, DTFFT_ESTIMATE, .true., base_init_dtype, base_init_storage, base_dtype, base_storage, bricks, pencils)

      execution_time = execute_autotune(plans, base_comm, current_backend_id, platform, helper, stream, buffer_size)
      if ( execution_time < best_time_ ) then
        best_time_ = execution_time
        best_backend = current_backend_id
      endif
      call destroy_plans(plans)
      deallocate( plans )
      REGION_END("Testing backend "//dtfft_get_backend_string(current_backend_id))
    enddo

    call helper%destroy()

  end subroutine autotune_reshape_plan
end module dtfft_reshape_plan
