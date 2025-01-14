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
!! This module defines most Abstract Transpose Plan: `abstract_transpose_plan`
use iso_fortran_env,    only: int8, int32, error_unit, output_unit
use dtfft_pencil,       only: pencil
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_nvrtc_kernel, only: DEF_TILE_SIZE
#endif
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: abstract_transpose_plan
public :: create_cart_comm

  type, abstract :: abstract_transpose_plan
  !! The most Abstract Transpose Plan
    logical :: is_z_slab  !< Z-slab optimization flag (for 3D transforms)
  contains
    procedure,                    pass(self),           public          :: create           !< Create transposition plan
    procedure,                    pass(self),           public          :: execute          !< Executes transposition
    procedure(create_interface),  pass(self), deferred                  :: create_private   !< Creates overriding class
    procedure(execute_interface), pass(self), deferred                  :: execute_private  !< Executes overriding class
    procedure(destroy_interface), pass(self), deferred, public          :: destroy          !< Destroys overriding class
#ifdef DTFFT_WITH_CUDA
    procedure(get_backend_id_interface),  pass(self), deferred, public  :: get_backend_id   !< Returns backend id
#endif
  end type abstract_transpose_plan


  interface
    function create_interface(self, dims, transposed_dims, base_comm, comm_dims, effort_flag, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils) result(error_code)
    !! Creates transposition plans
    import
      class(abstract_transpose_plan), intent(inout) :: self                 !< Transposition class
      integer(int32),                 intent(in)    :: dims(:)              !< Global sizes of the transform requested
      integer(int32),                 intent(in)    :: transposed_dims(:,:) !< Transposed sizes of the transform requested
      TYPE_MPI_COMM,                  intent(in)    :: base_comm            !< Base MPI communicator
      integer(int32),                 intent(in)    :: comm_dims(:)         !< Dims in cartesian communicator
      integer(int8),                  intent(in)    :: effort_flag          !< DTFFT planner effort flag
      TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !< Base MPI_Datatype
      integer(int8),                  intent(in)    :: base_storage         !< Number of bytes needed to store single element
      logical,                        intent(in)    :: is_custom_cart_comm  !< Custom cartesian communicator provided by user
      TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !< Cartesian communicator
      TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !< Array of 1d communicators
      type(pencil),                   intent(out)   :: pencils(:)           !< Data distributing meta
      integer(int32)                                :: error_code           !< Error code
    end function create_interface

    subroutine execute_interface(self, in, out, transpose_id)
    !! Executes single transposition
    import
      class(abstract_transpose_plan), intent(inout) :: self         !< Transposition class
      type(*),    DEVICE_PTR          intent(in)    :: in(..)       !< Incoming buffer of any rank and kind
      type(*),    DEVICE_PTR          intent(inout) :: out(..)      !< Resulting buffer of any rank and kind
      integer(int8),                  intent(in)    :: transpose_id !< Type of transpose
    end subroutine execute_interface

    subroutine destroy_interface(self)
    !! Destroys transposition plans
    import
      class(abstract_transpose_plan), intent(inout) :: self         !< Transposition class
    end subroutine destroy_interface

#ifdef DTFFT_WITH_CUDA
    integer(int8) function get_backend_id_interface(self)
    import
      class(abstract_transpose_plan), intent(in) :: self            !< Transposition class
    end function get_backend_id_interface
#endif
  endinterface

contains

  function create(self, dims, base_comm_, effort_flag, base_dtype, base_storage, cart_comm, comms, pencils) result(error_code)
  !! Creates transposition plans
    class(abstract_transpose_plan), intent(inout) :: self                 !< Transposition class
    integer(int32),                 intent(in)    :: dims(:)              !< Global sizes of the transform requested
    TYPE_MPI_COMM,                  intent(in)    :: base_comm_           !< Base communicator
    integer(int8),                  intent(in)    :: effort_flag          !< DTFFT planner effort flag
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !< Base MPI_Datatype
    integer(int8),                  intent(in)    :: base_storage         !< Number of bytes needed to store single element
    TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !< Cartesian communicator
    TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !< Array of 1d communicators
    type(pencil),                   intent(out)   :: pencils(:)             !< Data distributing meta
    integer(int32)                                :: error_code
    integer(int32),               allocatable     :: transposed_dims(:,:) !< Global counts in transposed coordinates

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
        integer(int32)                 :: grid_ndims           !< Number of dims in user defined cartesian communicator
        integer(int32),  allocatable   :: temp_dims(:)         !< Temporary dims needed by MPI_Cart_get
        integer(int32),  allocatable   :: temp_coords(:)       !< Temporary coordinates needed by MPI_Cart_get
        logical,         allocatable   :: temp_periods(:)      !< Temporary periods needed by MPI_Cart_get
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
          if ( comm_dims(d) > dims(d) ) WRITE_WARN("Number of MPI processes in direction "//int_to_str(d)//" greater then number of physical points: "//int_to_str(comm_dims(d))//" > "//int_to_str(dims(d)))
        enddo
        if ( ndims == 3 .and. comm_dims(2) == 1 .and. get_z_slab_enabled() ) then
          self%is_z_slab = .true.
        endif
      endblock
    else
      comm_dims(:) = 0
      comm_dims(1) = 1
      if ( ndims == 3                                   &
#ifdef DTFFT_WITH_CUDA
        .and. DEF_TILE_SIZE <= dims(ndims) / comm_size  &
#else
        .and. comm_size <= dims(ndims)                  &
#endif
      ) then
        comm_dims(2) = 1
        comm_dims(3) = comm_size

        self%is_z_slab = get_z_slab_enabled()
      else if (ndims == 3 .and.                                                         &
#ifdef DTFFT_WITH_CUDA
        DEF_TILE_SIZE <= dims(1) / comm_size .and. DEF_TILE_SIZE <= dims(2) / comm_size &
#else
        comm_size <= dims(1) .and. comm_size <= dims(2)                                 &
#endif
        ) then
          comm_dims(2) = comm_size
          comm_dims(3) = 1
        endif
      call MPI_Dims_create(comm_size, int(ndims, int32), comm_dims, ierr)
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

    error_code = self%create_private(dims, transposed_dims, base_comm_, comm_dims, effort_flag, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils)
    if ( error_code /= DTFFT_SUCCESS ) return

    deallocate( transposed_dims )
    deallocate( comm_dims )
  end function create

  subroutine execute(self, in, out, transpose_id)
  !! Executes single transposition
    class(abstract_transpose_plan), intent(inout) :: self         !< Transposition class
    type(*),    DEVICE_PTR          intent(in)    :: in(..)       !< Incoming buffer of any rank and kind
    type(*),    DEVICE_PTR          intent(inout) :: out(..)      !< Resulting buffer of any rank and kind
    integer(int8),                  intent(in)    :: transpose_id !< Type of transpose

    PHASE_BEGIN('Transpose '//TRANSPOSE_NAMES(transpose_id), COLOR_TRANSPOSE_PALLETTE(transpose_id))
    call self%execute_private(in, out, transpose_id)
    PHASE_END('Transpose '//TRANSPOSE_NAMES(transpose_id))
  end subroutine execute

  subroutine create_cart_comm(old_comm, comm_dims, comm, local_comms)
  !! Creates cartesian communicator
    TYPE_MPI_COMM,        intent(in)    :: old_comm             !< Communicator to create cartesian from
    integer(int32),       intent(in)    :: comm_dims(:)         !< Dims in cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: comm                 !< Cartesian communicator
    TYPE_MPI_COMM,        intent(out)   :: local_comms(:)       !< 1d communicators in cartesian communicator
    logical,              allocatable   :: periods(:)           !< Grid is not periodic
    logical,              allocatable   :: remain_dims(:)       !< Needed by MPI_Cart_sub
    integer(int8)                       :: dim                  !< Counter
    integer(int32)                      :: ierr                 !< Error code
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

#ifdef DTFFT_WITH_CUDA
  block
    integer(int32) :: comm_rank, comm_size, host_size, host_rank, proc_name_size, n_ranks_processed, n_names_processed, processing_id, n_total_ranks_processed
    integer(int32) :: min_val, max_val, i, j, k, min_dim, max_dim
    TYPE_MPI_COMM  :: host_comm
    ! TYPE_MPI_COMM  :: comms(3)
    integer(int32) :: top_type
    character(len=MPI_MAX_PROCESSOR_NAME) :: proc_name, processing_name
    character(len=MPI_MAX_PROCESSOR_NAME), allocatable :: all_names(:), processed_names(:)
    integer(int32), allocatable :: all_sizes(:), processed_ranks(:), groups(:,:)
    TYPE_MPI_GROUP :: base_group, temp_group

    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)
    call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, host_comm, ierr)
    call MPI_Comm_rank(host_comm, host_rank, ierr)
    call MPI_Comm_size(host_comm, host_size, ierr)
    call MPI_Comm_free(host_comm, ierr)
    call MPI_Topo_test(old_comm, top_type, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, host_size, 1, MPI_INTEGER4, MPI_MAX, comm, ierr)

    if ( ndims == 2 .or. host_size == 1 .or. any(comm_dims(2:) == 1) .or. top_type == MPI_CART) then
      return
    endif

    do dim = 2, ndims
      ! call MPI_Comm_dup(local_comms(dim), comms(dim), ierr)
      call MPI_Comm_free(local_comms(dim), ierr)
    enddo
    if ( comm_rank == 0 ) then
      print*,'MPI_Comm_free local_comms'
    endif

    call MPI_Comm_group(comm, base_group, ierr)
    call MPI_Group_rank(base_group, comm_rank, ierr)

    allocate( all_names(comm_size), processed_names(comm_size), all_sizes(comm_size), processed_ranks(comm_size) )
    
    call MPI_Get_processor_name(proc_name, proc_name_size, ierr)
    ! Obtaining mapping of which process sits on which node
    call MPI_Allgather(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, comm, ierr)
    call MPI_Allgather(host_size, 1, MPI_INTEGER4, all_sizes, 1, MPI_INTEGER4, comm, ierr)

    ! if ( comm_rank == 0 ) then
    !   print*,'MPI_Allgather proc_name: ',all_names
    !   print*,'MPI_Allgather host_size: ',all_sizes
    ! endif
    if ( comm_dims(2) >= comm_dims(3) ) then
      min_val = comm_dims(3)
      max_val = comm_dims(2)
      min_dim = 3
      max_dim = 2
    else
      min_val = comm_dims(2)
      max_val = comm_dims(3)
      min_dim = 2
      max_dim = 3
    endif
    ! min_val = min(comm_dims(2), comm_dims(3))
    ! max_val = max(comm_dims(2), comm_dims(3))

    allocate( groups(min_val, max_val) )

    ! if ( comm_rank == 0 ) then
    !   print*,'min_val: ',min_val, 'min_dim = ',min_dim
    !   print*,'max_val: ',max_val, 'max_dim = ',max_dim
    ! endif

    processed_ranks(:) = -1

    processing_id = 1
    processing_name = all_names(processing_id)
    n_ranks_processed = 0
    n_names_processed = 0
    n_total_ranks_processed = 0
    do j = 0, max_val - 1
      do i = 0, min_val - 1
        ! if ( comm_rank == 0 ) then
        !   print*,'i = ',i,'processing_id = ',processing_id,'n_ranks_processed = ',n_ranks_processed
        ! endif
        if ( n_ranks_processed == all_sizes(processing_id) ) then
          n_names_processed = n_names_processed + 1
          processed_names(n_names_processed) = processing_name
          processing_id = 0
          n_ranks_processed = 0
          do while(.true.)
            processing_id = processing_id + 1
            if ( processing_id > comm_size ) exit
            processing_name = all_names(processing_id)
            if ( .not. any(processing_name == processed_names(:n_names_processed)) ) exit
          enddo
        endif
        do k = 1, comm_size
          ! if ( comm_rank == 0 ) then
          !   print*,'k = ', k, 'processing_name = ',trim(processing_name), '  all_names(k) = ',trim(all_names(k))
          ! endif
          if ( processing_name == all_names(k) .and. .not.any(k - 1 == processed_ranks)) exit
        enddo
        ! if ( comm_rank == 0 ) then
        !   print*,'exited'
        ! endif
        ! 
        ! print*,'k = ',k
        ! processing_id = k + 1
        n_ranks_processed = n_ranks_processed + 1
        groups(i + 1, j + 1) = k - 1
        n_total_ranks_processed = n_total_ranks_processed + 1
        processed_ranks(n_total_ranks_processed) = k - 1
      enddo
      ! if ( comm_rank == 0 ) then
      !   'group'
      ! endif

      ! print*,'group = ',group
    enddo
    
    ! if ( comm_rank == 0 ) then
    !   print*,'groups = ',groups
    ! endif
    do j = 0, max_val - 1
      do i = 0, min_val - 1
        if ( any(comm_rank == groups(:, j + 1)) ) then
          call MPI_Group_incl(base_group, min_val, groups(:, j + 1), temp_group, ierr)
          call MPI_Comm_create(comm, temp_group, local_comms(min_dim), ierr)
          call MPI_Group_free(temp_group, ierr)
        endif
      enddo
    enddo

    do i = 0, min_val - 1
      do j = 0, max_val - 1
        if ( any(comm_rank == groups(i + 1, :)) ) then
          call MPI_Group_incl(base_group, max_val, groups(i + 1, :), temp_group, ierr)
          call MPI_Comm_create(comm, temp_group, local_comms(max_dim), ierr)
          call MPI_Group_free(temp_group, ierr)
        endif
      enddo
    enddo

    deallocate(all_names, processed_names, all_sizes, processed_ranks, groups)
    ! call MPI_Cart_coords(comm, comm_rank, 3, coords, ierr)
    

  endblock
#endif
  end subroutine create_cart_comm
end module dtfft_abstract_transpose_plan