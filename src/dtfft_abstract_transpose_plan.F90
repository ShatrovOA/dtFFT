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
module dtfft_abstract_transpose_plan
!! This module describes Abstraction for all Tranpose plans: [[abstract_transpose_plan]]
use iso_c_binding,      only: c_ptr, c_null_ptr
use iso_fortran_env,    only: int8, int32, int64, error_unit, output_unit
use dtfft_config
use dtfft_pencil,       only: pencil
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_abstract_backend, only: backend_helper
#ifdef NCCL_HAVE_COMM_REGISTER
use dtfft_abstract_backend, only: NCCL_REGISTER_PREALLOC_SIZE
#endif
use dtfft_nvrtc_kernel,     only: DEF_TILE_SIZE
use dtfft_interface_cuda_runtime
# ifdef DTFFT_WITH_NVSHMEM
use dtfft_interface_nvshmem
# endif
# ifdef DTFFT_WITH_NCCL
use dtfft_interface_nccl
# endif
#endif
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: abstract_transpose_plan
public :: create_cart_comm
#ifdef DTFFT_WITH_CUDA
public :: alloc_mem, free_mem
#endif

  type, abstract :: abstract_transpose_plan
  !! The most Abstract Transpose Plan
#ifdef DTFFT_WITH_CUDA
    type(dtfft_gpu_backend_t)     :: gpu_backend = DTFFT_GPU_BACKEND_MPI_DATATYPE
    type(backend_helper)          :: helper
#endif
    logical :: is_z_slab  !! Z-slab optimization flag (for 3D transforms)
  contains
    procedure,                            pass(self),           public  :: create           !! Create transposition plan
    procedure,                            pass(self),           public  :: execute          !! Executes transposition
    procedure(create_interface),          pass(self), deferred          :: create_private   !! Creates overriding class
    procedure(execute_interface),         pass(self), deferred          :: execute_private  !! Executes overriding class
    procedure(destroy_interface),         pass(self), deferred, public  :: destroy          !! Destroys overriding class
#ifdef DTFFT_WITH_CUDA
    procedure,   non_overridable,         pass(self),           public  :: get_gpu_backend  !! Returns backend id
    procedure,   non_overridable,         pass(self),           public  :: mem_alloc        !! Allocates memory based on selected backend
    procedure,   non_overridable,         pass(self),           public  :: mem_free         !! Frees memory allocated with mem_alloc
#endif
  end type abstract_transpose_plan


  interface
    function create_interface(self, dims, transposed_dims, base_comm, comm_dims, effort, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils) result(error_code)
    !! Creates transposition plans
    import
      class(abstract_transpose_plan), intent(inout) :: self                 !! Transposition class
      integer(int32),                 intent(in)    :: dims(:)              !! Global sizes of the transform requested
      integer(int32),                 intent(in)    :: transposed_dims(:,:) !! Transposed sizes of the transform requested
      TYPE_MPI_COMM,                  intent(in)    :: base_comm            !! Base MPI communicator
      integer(int32),                 intent(in)    :: comm_dims(:)         !! Dims in cartesian communicator
      type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner type of effort
      TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !! Base MPI_Datatype
      integer(int8),                  intent(in)    :: base_storage         !! Number of bytes needed to store single element
      logical,                        intent(in)    :: is_custom_cart_comm  !! Custom cartesian communicator provided by user
      TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !! Cartesian communicator
      TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !! Array of 1d communicators
      type(pencil),                   intent(out)   :: pencils(:)           !! Data distributing meta
      integer(int32)                                :: error_code           !! Error code
    end function create_interface

    subroutine execute_interface(self, in, out, transpose_type)
    !! Executes single transposition
    import
      class(abstract_transpose_plan), intent(inout) :: self           !! Transposition class
      type(*),              target,   intent(inout) :: in(..)         !! Incoming buffer of any rank and kind
      type(*),              target,   intent(inout) :: out(..)        !! Resulting buffer of any rank and kind
      type(dtfft_transpose_type_t),   intent(in)    :: transpose_type !! Type of transpose
    end subroutine execute_interface

    subroutine destroy_interface(self)
    !! Destroys transposition plans
    import
      class(abstract_transpose_plan), intent(inout) :: self         !! Transposition class
    end subroutine destroy_interface
  endinterface

contains

  function create(self, dims, base_comm_, effort, base_dtype, base_storage, cart_comm, comms, pencils) result(error_code)
  !! Creates transposition plans
    class(abstract_transpose_plan), intent(inout) :: self                 !! Transposition class
    integer(int32),                 intent(in)    :: dims(:)              !! Global sizes of the transform requested
    TYPE_MPI_COMM,                  intent(in)    :: base_comm_           !! Base communicator
    type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner type of effort
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !! Base MPI_Datatype
    integer(int8),                  intent(in)    :: base_storage         !! Number of bytes needed to store single element
    TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !! Cartesian communicator
    TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !! Array of 1d communicators
    type(pencil),                   intent(out)   :: pencils(:)           !! Data distributing meta
    integer(int32)                                :: error_code
    integer(int32),               allocatable     :: transposed_dims(:,:) !! Global counts in transposed coordinates
    logical :: cond1, cond2

    integer(int32),  allocatable :: comm_dims(:)
    integer(int8) :: ndims
    integer(int32) :: comm_size, top_type, ierr
    logical :: is_custom_cart_comm

    call MPI_Comm_size(base_comm_, comm_size, ierr)
    call MPI_Topo_test(base_comm_, top_type, ierr)

    ndims = size(dims, kind=int8)
    allocate( comm_dims(ndims) )

    is_custom_cart_comm = .false.
    self%is_z_slab = .false.
    if ( top_type == MPI_CART ) then
      is_custom_cart_comm = .true.
      block
        integer(int32)                 :: grid_ndims           !! Number of dims in user defined cartesian communicator
        integer(int32),  allocatable   :: temp_dims(:)         !! Temporary dims needed by MPI_Cart_get
        integer(int32),  allocatable   :: temp_coords(:)       !! Temporary coordinates needed by MPI_Cart_get
        logical,         allocatable   :: temp_periods(:)      !! Temporary periods needed by MPI_Cart_get
        integer(int8) :: d

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
            WRITE_WARN("Number of MPI processes in direction "//int_to_str(d)//" greater then number of physical points: "//int_to_str(comm_dims(d))//" > "//int_to_str(dims(d)))
          endif
        enddo
        if ( ndims == 3 .and. comm_dims(2) == 1 .and. get_z_slab_flag() ) then
          self%is_z_slab = .true.
        endif
      endblock
    else
      comm_dims(:) = 0
      comm_dims(1) = 1
#ifdef DTFFT_WITH_CUDA
      if ( get_user_platform() == DTFFT_PLATFORM_HOST ) then
        cond1 = comm_size <= dims(ndims)
        cond2 = comm_size <= dims(1) .and. comm_size <= dims(2)
      else
        cond1 = DEF_TILE_SIZE <= dims(ndims) / comm_size
        cond2 = DEF_TILE_SIZE <= dims(1) / comm_size .and. DEF_TILE_SIZE <= dims(2) / comm_size
      endif
#else
        cond1 = comm_size <= dims(ndims)
        cond2 = comm_size <= dims(1) .and. comm_size <= dims(2)
#endif

      if ( ndims == 3 .and. cond1 ) then
        comm_dims(2) = 1
        comm_dims(3) = comm_size
        self%is_z_slab = get_z_slab_flag()
      else if (ndims == 3 .and. cond2 ) then
          comm_dims(2) = comm_size
          comm_dims(3) = 1
        endif
      call MPI_Dims_create(comm_size, int(ndims, int32), comm_dims, ierr)
      if(dims(ndims - 1) < comm_dims(ndims - 1) .or. dims(ndims) < comm_dims(ndims) ) then
        WRITE_WARN("Unable to create correct grid decomposition.")
        ! WRITE_WARN("Fallback to Z slab is used")
        ! comm_dims(ndims - 1) = 1
        ! comm_dims(ndims) = comm_size
      endif
    endif
    if ( self%is_z_slab ) then
      WRITE_INFO("Using Z-slab optimization")
    endif

    ndims = size(dims, kind=int8)

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

    error_code = self%create_private(dims, transposed_dims, base_comm_, comm_dims, effort, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils)
    if ( error_code /= DTFFT_SUCCESS ) return

    deallocate( transposed_dims )
    deallocate( comm_dims )
  end function create

  subroutine execute(self, in, out, transpose_type)
  !! Executes single transposition
    class(abstract_transpose_plan), intent(inout) :: self         !! Transposition class
    type(*),                        intent(inout) :: in(..)       !! Incoming buffer of any rank and kind
    type(*),                        intent(inout) :: out(..)      !! Resulting buffer of any rank and kind
    type(dtfft_transpose_type_t),   intent(in)    :: transpose_type !! Type of transpose

    PHASE_BEGIN('Transpose '//TRANSPOSE_NAMES(transpose_type%val), COLOR_TRANSPOSE_PALLETTE(transpose_type%val))
    call self%execute_private(in, out, transpose_type)
    PHASE_END('Transpose '//TRANSPOSE_NAMES(transpose_type%val))
  end subroutine execute

#ifdef DTFFT_WITH_CUDA
  type(dtfft_gpu_backend_t) function get_gpu_backend(self)
    class(abstract_transpose_plan), intent(in)    :: self         !! Transposition class
    get_gpu_backend = self%gpu_backend
  end function get_gpu_backend

  subroutine mem_alloc(self, comm, alloc_bytes, ptr, error_code)
    !! Allocates memory based on selected backend
    class(abstract_transpose_plan), intent(inout) :: self            !! Transposition class
    TYPE_MPI_COMM,                  intent(in)    :: comm
    integer(int64),                 intent(in)    :: alloc_bytes
    type(c_ptr),                    intent(out)   :: ptr
    integer(int32),                 intent(out)   :: error_code

    call alloc_mem(self%helper, self%gpu_backend, comm, alloc_bytes, ptr, error_code)
  end subroutine mem_alloc

  subroutine mem_free(self, ptr, error_code)
    class(abstract_transpose_plan), intent(inout) :: self            !! Transposition class
    type(c_ptr),                    intent(in)    :: ptr
    integer(int32),                 intent(out)   :: error_code

    call free_mem(self%helper, self%gpu_backend, ptr, error_code)
  end subroutine mem_free

  subroutine alloc_mem(helper, gpu_backend, comm, alloc_bytes, ptr, error_code)
  !! Allocates memory based on ``gpu_backend``
    type(backend_helper),           intent(inout) :: helper
    type(dtfft_gpu_backend_t),      intent(in)    :: gpu_backend
    TYPE_MPI_COMM,                  intent(in)    :: comm
    integer(int64),                 intent(in)    :: alloc_bytes
    type(c_ptr),                    intent(out)   :: ptr
    integer(int32),                 intent(out)   :: error_code
    integer(int32)  :: ierr

    error_code = DTFFT_SUCCESS
    ierr = cudaSuccess

    if ( is_backend_nccl(gpu_backend) ) then
#ifdef DTFFT_WITH_NCCL
# ifdef NCCL_HAVE_MEMALLOC
      ierr = ncclMemAlloc(ptr, alloc_bytes)
# else
      ierr = cudaMalloc(ptr, alloc_bytes)
# endif
# ifdef NCCL_HAVE_COMM_REGISTER
      if ( ierr == cudaSuccess .and. helper%should_register) then
        block
          type(c_ptr), allocatable :: temp(:,:)
          type(c_ptr) :: handle

          if ( size(helper%nccl_register, dim=1) == helper%nccl_register_size ) then
            allocate( temp(helper%nccl_register_size, 2) )
            temp(:,:) = helper%nccl_register(:,:)
            deallocate( helper%nccl_register )
            allocate( helper%nccl_register(helper%nccl_register_size + NCCL_REGISTER_PREALLOC_SIZE, 2) )
            helper%nccl_register(:helper%nccl_register_size,:) = temp(:,:)
            deallocate( temp )
          endif
          helper%nccl_register_size = helper%nccl_register_size + 1

          NCCL_CALL( "ncclCommRegister", ncclCommRegister(helper%nccl_comm, ptr, alloc_bytes, handle) )
          helper%nccl_register(helper%nccl_register_size, 1) = ptr
          helper%nccl_register(helper%nccl_register_size, 2) = handle
        endblock
      endif
# endif
#else
      error stop "not DTFFT_WITH_NCCL"
#endif
    else if ( is_backend_nvshmem(gpu_backend) ) then
#ifdef DTFFT_WITH_NVSHMEM
      block
        integer(int64)  :: max_alloc_bytes
        call MPI_Allreduce(alloc_bytes, max_alloc_bytes, 1, MPI_INTEGER8, MPI_MAX, comm, ierr)
        ptr = nvshmem_malloc(max_alloc_bytes)
        if ( is_null_ptr(ptr) ) error_code = DTFFT_ERROR_ALLOC_FAILED
      endblock
#else
      error stop "not DTFFT_WITH_NVSHMEM"
#endif
    else
      ierr = cudaMalloc(ptr, alloc_bytes)
    endif
    if ( ierr /= cudaSuccess ) error_code = DTFFT_ERROR_ALLOC_FAILED
  end subroutine alloc_mem

  subroutine free_mem(helper, gpu_backend, ptr, error_code)
  !! Frees memory based on ``gpu_backend``
    type(backend_helper),           intent(inout) :: helper
    type(dtfft_gpu_backend_t),      intent(in)    :: gpu_backend
    type(c_ptr),                    intent(in)    :: ptr
    integer(int32),                 intent(out)   :: error_code
    integer(int32)  :: ierr

    error_code = DTFFT_SUCCESS
    ierr = cudaSuccess
    if ( is_backend_nccl(gpu_backend) ) then
#ifdef NCCL_HAVE_COMM_REGISTER
      if ( helper%should_register ) then
        block
          integer(int32) :: i

          do i = 1, size(helper%nccl_register, dim=1)
            if ( .not. is_same_ptr(ptr, helper%nccl_register(i, 1)) ) cycle
            NCCL_CALL( "ncclCommDeregister", ncclCommDeregister(helper%nccl_comm, helper%nccl_register(i, 2)) )
            helper%nccl_register(i, 1) = c_null_ptr
            helper%nccl_register(i, 2) = c_null_ptr
            helper%nccl_register_size = helper%nccl_register_size - 1
          enddo
        endblock
      endif
#endif
#ifdef DTFFT_WITH_NCCL
# ifdef NCCL_HAVE_MEMALLOC
      ierr = ncclMemFree(ptr)
# else
      ierr = cudaFree(ptr)
# endif
#else
  error stop "not DTFFT_WITH_NCCL"
#endif
    else if ( is_backend_nvshmem(gpu_backend) ) then
#ifdef DTFFT_WITH_NVSHMEM
      call nvshmem_free(ptr)
#else
      error stop "not DTFFT_WITH_NVSHMEM"
#endif
    else
      ierr = cudaFree(ptr)
    endif
    if ( ierr /= cudaSuccess ) error_code = DTFFT_ERROR_FREE_FAILED
  end subroutine free_mem
#endif

  subroutine create_cart_comm(old_comm, comm_dims, comm, local_comms)
  !! Creates cartesian communicator
    TYPE_MPI_COMM,        intent(in)    :: old_comm             !! Communicator to create cartesian from
    integer(int32),       intent(in)    :: comm_dims(:)         !! Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: comm                 !! Cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: local_comms(:)       !! 1d communicators in cartesian communicator
    logical,              allocatable   :: periods(:)           !! Grid is not periodic
    logical,              allocatable   :: remain_dims(:)       !! Needed by MPI_Cart_sub
    integer(int8)                       :: dim                  !! Counter
    integer(int32)                      :: ierr                 !! Error code
    integer(int8)                       :: ndims

    ndims = size(comm_dims, kind=int8)

    allocate(periods(ndims), source = .false.)
    call MPI_Cart_create(old_comm, int(ndims, int32), comm_dims, periods, .false., comm, ierr)
    if ( DTFFT_GET_MPI_VALUE(comm) == DTFFT_GET_MPI_VALUE(MPI_COMM_NULL) ) error stop "comm == MPI_COMM_NULL"

    allocate( remain_dims(ndims), source = .false. )
    do dim = 1, ndims
      remain_dims(dim) = .true.
      call MPI_Cart_sub(comm, remain_dims, local_comms(dim), ierr)
      remain_dims(dim) = .false.
    enddo
    deallocate(remain_dims, periods)

! #ifdef DTFFT_WITH_CUDA
!   block
!     integer(int32) :: comm_rank, comm_size, host_size, host_rank, proc_name_size, n_ranks_processed, n_names_processed, processing_id, n_total_ranks_processed
!     integer(int32) :: min_val, max_val, i, j, k, min_dim, max_dim
!     TYPE_MPI_COMM  :: host_comm
!     integer(int32) :: top_type
!     character(len=MPI_MAX_PROCESSOR_NAME) :: proc_name, processing_name
!     character(len=MPI_MAX_PROCESSOR_NAME), allocatable :: all_names(:), processed_names(:)
!     integer(int32), allocatable :: all_sizes(:), processed_ranks(:), groups(:,:)
!     TYPE_MPI_GROUP :: base_group, temp_group

!     call MPI_Comm_rank(comm, comm_rank, ierr)
!     call MPI_Comm_size(comm, comm_size, ierr)
!     call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, host_comm, ierr)
!     call MPI_Comm_rank(host_comm, host_rank, ierr)
!     call MPI_Comm_size(host_comm, host_size, ierr)
!     call MPI_Comm_free(host_comm, ierr)
!     call MPI_Topo_test(old_comm, top_type, ierr)
!     call MPI_Allreduce(MPI_IN_PLACE, host_size, 1, MPI_INTEGER4, MPI_MAX, comm, ierr)

!     if ( ndims == 2 .or. host_size == 1 .or. any(comm_dims(2:) == 1) .or. top_type == MPI_CART) then
!       return
!     endif

!     do dim = 2, ndims
!       call MPI_Comm_free(local_comms(dim), ierr)
!     enddo

!     call MPI_Comm_group(comm, base_group, ierr)
!     call MPI_Group_rank(base_group, comm_rank, ierr)

!     allocate( all_names(comm_size), processed_names(comm_size), all_sizes(comm_size), processed_ranks(comm_size) )

!     call MPI_Get_processor_name(proc_name, proc_name_size, ierr)
!     ! Obtaining mapping of which process sits on which node
!     call MPI_Allgather(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, comm, ierr)
!     call MPI_Allgather(host_size, 1, MPI_INTEGER4, all_sizes, 1, MPI_INTEGER4, comm, ierr)

!     if ( comm_dims(2) >= comm_dims(3) ) then
!       min_val = comm_dims(3)
!       max_val = comm_dims(2)
!       min_dim = 3
!       max_dim = 2
!     else
!       min_val = comm_dims(2)
!       max_val = comm_dims(3)
!       min_dim = 2
!       max_dim = 3
!     endif

!     allocate( groups(min_val, max_val) )

!     processed_ranks(:) = -1

!     processing_id = 1
!     processing_name = all_names(processing_id)
!     n_ranks_processed = 0
!     n_names_processed = 0
!     n_total_ranks_processed = 0
!     do j = 0, max_val - 1
!       do i = 0, min_val - 1
!         if ( n_ranks_processed == all_sizes(processing_id) ) then
!           n_names_processed = n_names_processed + 1
!           processed_names(n_names_processed) = processing_name
!           processing_id = 0
!           n_ranks_processed = 0
!           do while(.true.)
!             processing_id = processing_id + 1
!             if ( processing_id > comm_size ) exit
!             processing_name = all_names(processing_id)
!             if ( .not. any(processing_name == processed_names(:n_names_processed)) ) exit
!           enddo
!         endif
!         do k = 1, comm_size
!           if ( processing_name == all_names(k) .and. .not.any(k - 1 == processed_ranks)) exit
!         enddo
!         n_ranks_processed = n_ranks_processed + 1
!         groups(i + 1, j + 1) = k - 1
!         n_total_ranks_processed = n_total_ranks_processed + 1
!         processed_ranks(n_total_ranks_processed) = k - 1
!       enddo
!     enddo

!     do j = 0, max_val - 1
!       do i = 0, min_val - 1
!         if ( any(comm_rank == groups(:, j + 1)) ) then
!           call MPI_Group_incl(base_group, min_val, groups(:, j + 1), temp_group, ierr)
!           call MPI_Comm_create(comm, temp_group, local_comms(min_dim), ierr)
!           call MPI_Group_free(temp_group, ierr)
!         endif
!       enddo
!     enddo

!     do i = 0, min_val - 1
!       do j = 0, max_val - 1
!         if ( any(comm_rank == groups(i + 1, :)) ) then
!           call MPI_Group_incl(base_group, max_val, groups(i + 1, :), temp_group, ierr)
!           call MPI_Comm_create(comm, temp_group, local_comms(max_dim), ierr)
!           call MPI_Group_free(temp_group, ierr)
!         endif
!       enddo
!     enddo

!     deallocate(all_names, processed_names, all_sizes, processed_ranks, groups)

!   endblock
! #endif
  end subroutine create_cart_comm
end module dtfft_abstract_transpose_plan