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
module dtfft_pencil
!! This module describes private [[pencil]] and public [[dtfft_pencil_t]] classes
use iso_fortran_env
use iso_c_binding
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#include "_dtfft_cuda.h"
#endif
implicit none
private
public :: pencil, pencil_init
public :: dtfft_pencil_t, dtfft_pencil_c
public :: get_local_sizes
public :: get_transpose_type, get_reshape_type
public :: pencil_c2f, pencil_f2c
public :: from_bricks

  type :: dtfft_pencil_t
  !! Structure to hold pencil decomposition info
    integer(int8)               :: dim
      !! Aligned dimension id
    integer(int8)               :: ndims = 0
      !! Number of dimensions
    integer(int32), allocatable :: starts(:)
      !! Local starts, starting from 0 for both C and Fortran
    integer(int32), allocatable :: counts(:)
      !! Local counts of data, in elements
    integer(int64)              :: size
      !! Total number of elements in a pencil
    logical,        private     :: is_created = .false.
      !! Is pencil created
  contains
    ! final :: destroy_pencil_t !! Finalizer
    procedure, pass(self) :: destroy => destroy_pencil_t
      !! Destroys pencil
  end type dtfft_pencil_t

  interface dtfft_pencil_t
  !! Type bound constuctor for dtfft_pencil_t
    module procedure create_pencil_t    !! Creates pencil object, that can be used to create dtFFT plans
  end interface dtfft_pencil_t

  type, bind(C) :: dtfft_pencil_c
  !! Structure to hold pencil decomposition info
    integer(c_int8_t)   :: dim        !! Aligned dimension id
    integer(c_int8_t)   :: ndims      !! Number of dimensions
    integer(c_int32_t)  :: starts(3)  !! Local starts, starting from 0 for both C and Fortran
    integer(c_int32_t)  :: counts(3)  !! Local counts of data, in elements
    integer(c_size_t)   :: size       !! Total number of elements in a pencil
  end type dtfft_pencil_c

  type :: pencil
  !! Class that describes information about data layout
    integer(int8)                :: aligned_dim       !! Position of aligned dimension. For example: X pencil aligned_dim = 1, Z pencil aligned_dim = 3
    integer(int8)                :: rank              !! Rank of buffer: 2 or 3
    integer(int32), allocatable  :: starts(:)         !! Local starts, starting from 0 for both C and Fortran
    integer(int32), allocatable  :: counts(:)         !! Local counts of data, in elements
    logical                      :: is_even           !! Is data evenly distributed across processes
    logical                      :: is_distributed    !! Is aligned data distributed across multiple processes
  contains
  private
    procedure, pass(self),  public  :: create         !! Creates pencil
    procedure, pass(self),  public  :: destroy        !! Destroys pencil
    ! procedure, pass(self),  public  :: output         !! Writes pencil data to stdout
    !                                                   !! Used only for debugging purposes
    procedure, pass(self),  public  :: make_public    !! Creates public object that users can use to create own FFT backends
  end type pencil

  type :: pencil_init
  !! Class that describes information about data layout
  !!
  !! It is an extension of dtfft_pencil_t with additional fields
    TYPE_MPI_COMM,  allocatable :: comms(:)     !! 1D communicators for each dimension
    integer(int32), allocatable :: starts(:)    !! Local starts
    integer(int32), allocatable :: counts(:)    !! Local counts
    integer(int32), allocatable :: dims(:)      !! Global dimensions of entire region
  contains
  private
    procedure, pass(self),  public  :: create => create_pencil_init   !! Creates and validates pencil passed by user to plan constructors
    procedure, pass(self),  public  :: destroy => destroy_pencil_init !! Destroys pencil_init
    procedure, pass(self),  public  :: from_bricks                    !! Creates pencil_init from brick decomposition
  end type pencil_init

contains
  subroutine create(self, rank, aligned_dim, counts, comms, lstarts, lcounts, is_distributed)
  !! Creates pencil
    class(pencil),                  intent(inout) :: self             !! Pencil
    integer(int8),                  intent(in)    :: rank             !! Rank of buffer
    integer(int8),                  intent(in)    :: aligned_dim      !! Position of aligned dimension
    integer(int32),                 intent(in)    :: counts(:)        !! Global counts
    TYPE_MPI_COMM,                  intent(in)    :: comms(:)         !! Grid communicators
    integer(int32),       optional, intent(in)    :: lstarts(:)       !! Local starts
    integer(int32),       optional, intent(in)    :: lcounts(:)       !! Local counts
    logical,              optional, intent(in)    :: is_distributed    !! Is data distributed
    ! integer(int8),        optional, intent(in)    :: dim_to_keep
    integer(int8)                     :: d                !! Counter
    logical, allocatable              :: is_even(:)       !! Even distribution flag
    integer(int8)               :: comm_id
    integer(int32), allocatable :: starts(:), sizes(:)
    integer(int32) :: fast_comm_size, ierr
    logical :: is_distributed_
    integer(int32) :: start, count
    logical :: one, two, three

    call self%destroy()
    allocate(sizes(rank))
    allocate(starts(rank))
    allocate(is_even(rank))
    self%aligned_dim = aligned_dim
    self%rank = rank
    is_distributed_ = .false.; if ( present(is_distributed) ) is_distributed_ = is_distributed
    if ( present(lstarts) .and. present(lcounts) ) then
      if ( aligned_dim == 1 ) then
        if ( is_distributed_ ) then
          do d = 1, rank
            call get_local_size(counts(d), comms(d), starts(d), sizes(d))
          enddo
        else
          starts(:) = lstarts(:)
          sizes(:) = lcounts(:)
        endif
      else
        do d = 1, rank
          if ( rank == 3 ) then
            one = aligned_dim == 2 .and. d == 2
            two = aligned_dim == 3 .and. d == 2
            three = aligned_dim == 3 .and. d == 2 .and. is_distributed_
            if ( one .or. two .or. three) then
              if ( one .or. two ) then
                start = lstarts(3)
                count = lcounts(3)
              else
                start = lstarts(2)
                count = lcounts(2)
              endif
              call get_local_size(counts(d), comms(d), starts(d), sizes(d), start, count)
            else
              call get_local_size(counts(d), comms(d), starts(d), sizes(d))
            endif


            ! if ( aligned_dim == 2 .and. d == 2 ) then
            !   call get_local_size(counts(d), comms(3), starts(d), sizes(d), lstarts(3), lcounts(3))
            ! else if ( aligned_dim == 2 .and. d == 3 ) then
            !   call get_local_size(counts(d), comms(2), starts(d), sizes(d))
            ! else if ( aligned_dim == 3 .and. d == 2 .and. is_distributed_) then
            !   call get_local_size(counts(d), comms(d), starts(d), sizes(d), lstarts(2), lcounts(2))
            ! else if ( aligned_dim == 3 .and. d == 2 ) then
            !   call get_local_size(counts(d), comms(2), starts(d), sizes(d), lstarts(3), lcounts(3))

            ! else
            !   call get_local_size(counts(d), comms(d), starts(d), sizes(d))
            ! endif
          else
            call get_local_size(counts(d), comms(d), starts(d), sizes(d))
          endif
        enddo
      endif
    else
      do d = 1, rank
        ! comm_id = d
        ! if ( .not. is_distributed_ ) then
        !   if ( rank == 3 ) then
        !     if ( aligned_dim == 2 .and. d == 2 ) then
        !       comm_id = 3
        !     else if ( aligned_dim == 2 .and. d == 3 ) then
        !       comm_id = 2
        !     endif
        !   endif
        ! endif
        call get_local_size(counts(d), comms(d), starts(d), sizes(d))
      enddo
    endif
    do d = 1, rank
      ! comm_id = d
      ! if ( .not. is_distributed_ ) then
      !   if ( rank == 3 ) then
      !     if ( aligned_dim == 2 .and. d == 2 ) then
      !       comm_id = 3
      !     else if ( aligned_dim == 2 .and. d == 3 ) then
      !       comm_id = 2
      !     endif
      !   endif
      ! endif
      is_even(d) = check_if_even(sizes(d), comms(d))
    enddo
    allocate(self%counts(rank))
    allocate(self%starts(rank))
    do d = 1, rank
      ! order_ = d
      ! if ( present(order) ) order_ = order(d)
      self%counts(d) = sizes(d)
      self%starts(d) = starts(d)
    enddo
    self%is_even = all(is_even)
    deallocate(is_even, starts, sizes)

    call MPI_Comm_size(comms(1), fast_comm_size, ierr)
    allocate( starts(fast_comm_size) )
    call MPI_Allgather(self%starts(1), 1, MPI_INTEGER, starts, 1, MPI_INTEGER, comms(1), ierr)
    self%is_distributed = any( starts /= starts(1) )
    deallocate( starts )
  end subroutine create

  subroutine destroy(self)
  !! Destroys pencil
    class(pencil),      intent(inout) :: self             !! Pencil

    if ( allocated(self%counts) ) deallocate(self%counts)
    if ( allocated(self%starts) ) deallocate(self%starts)
  end subroutine destroy

  subroutine get_local_size(n_global, comm, start, count, start_to_keep, size_to_keep)
  !! Computes local portions of data based on global count and position inside grid communicator
    integer(int32),           intent(in)  :: n_global             !! Global number of points
    TYPE_MPI_COMM,            intent(in)  :: comm                 !! Grid communicator
    integer(int32),           intent(out) :: start                !! Local start
    integer(int32),           intent(out) :: count                !! Local count
    integer(int32), optional, intent(in)  :: start_to_keep  !! Start to keep in case of user defined decomposition
    integer(int32), optional, intent(in)  :: size_to_keep   !! Size to keep in case of user defined decomposition
    integer(int32), allocatable   :: shift(:)             !! Work buffer
    integer(int32)                :: comm_dim             !! Number of MPI processes along n_global
    integer(int32)                :: comm_rank            !! Rank of current MPI process
    integer(int32)                :: res                  !! Residual from n_global / comm_dim
    integer(int32)                :: i                    !! Counter
    integer(int32)                :: ierr                 !! Error code

    call MPI_Comm_size(comm, comm_dim, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)

    res = mod(n_global, comm_dim)
    start = 0

    if ( comm_dim == 1 ) then
      count = n_global
      return
    elseif( present(size_to_keep) ) then
      count = size_to_keep
    elseif ( comm_rank >= comm_dim - res ) then
      count = int(n_global / comm_dim, int32) + 1
    else
      count = int(n_global / comm_dim, int32)
    endif
    if ( present(start_to_keep) ) then
      start = start_to_keep
    else
      allocate( shift(comm_dim) )
      call MPI_Allgather(count, 1, MPI_INTEGER, shift, 1, MPI_INTEGER, comm, ierr)

      do i = 0, comm_rank - 1
        start = start + shift(i + 1)
      end do
      deallocate(shift)
    endif
  end subroutine get_local_size

  logical function check_if_even(count, comm)
  !! Checks if data is evenly distributed across processes
    integer(int32), intent(in)    :: count                !! Local count
    TYPE_MPI_COMM,  intent(in)    :: comm                 !! Grid communicator
    integer(int32)                :: comm_size            !! Number of MPI processes
    integer(int32), allocatable   :: shift(:)             !! Work buffer
    integer(int32)                :: ierr                 !! Error code

    call MPI_Comm_size(comm, comm_size, ierr)

    if ( comm_size == 1 ) then
      check_if_even = .true.
      return
    end if
    allocate( shift(comm_size) )
    call MPI_Allgather(count, 1, MPI_INTEGER, shift, 1, MPI_INTEGER, comm, ierr)
    check_if_even = all(shift == shift(1))
    deallocate(shift)
  end function check_if_even

!   subroutine output(self, name, vec)
!   !! Writes pencil data to stdout
!     class(pencil),                intent(in)  :: self                 !! Pencil
!     character(len=*),             intent(in)  :: name                 !! Name of pencil
!     real(real32),    target,      intent(in)  :: vec(:)               !! Device pointer to data
!     integer(int32)                            :: iter                 !! Iteration counter
!     integer(int32)                            :: i,j,k,ijk            !! Counters
!     integer(int32)                            :: comm_size            !! Number of MPI processes
!     integer(int32)                            :: comm_rank            !! Rank of current MPI process
!     integer(int32)                            :: ierr                 !! Error code
! #ifdef DTFFT_WITH_CUDA
!     real(real32),    target,      allocatable :: buf(:)               !! Host buffer
! #endif

!     call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
!     call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

! #ifdef DTFFT_WITH_CUDA
!     allocate( buf( product(self%counts) ) )
!     if ( is_device_ptr(c_loc(vec)) ) then
!       CUDA_CALL( cudaDeviceSynchronize())
!       CUDA_CALL( cudaMemcpy(c_loc(buf), c_loc(vec), int(real32, int64) * product(self%counts), cudaMemcpyDeviceToHost) )
!     endif
! #endif

!     do iter = 0, comm_size - 1
!       call MPI_Barrier(MPI_COMM_WORLD, ierr)
!       if ( iter == comm_rank ) then
!         write(output_unit,'(a)') name
!         do k = 0, self%counts(3) - 1
!           do j = 0, self%counts(2) - 1
!             ijk = k * self%counts(2) * self%counts(1) + j * self%counts(1)
! #ifdef DTFFT_WITH_CUDA
!             write(output_unit,'(2i5, *(f9.2))') j, k, (buf(ijk + i + 1), i=0,self%counts(1) - 1)
! #else
!             write(output_unit,'(2i5, *(f9.2))') j, k, (vec(ijk + i + 1), i=0,self%counts(1) - 1)
! #endif
!           enddo
!           write(output_unit, '(a)') ' '
!           flush(output_unit)
!         enddo
!         write(output_unit, '(a)') ' '
!         write(output_unit, '(a)') ' '
!         flush(output_unit)
!       endif
!       call MPI_Barrier(MPI_COMM_WORLD, ierr)
!     enddo

! #ifdef DTFFT_WITH_CUDA
!     deallocate(buf)
! #endif
!   end subroutine output

  type(dtfft_pencil_t) function make_public(self)
  !! Creates public object that users can use to create own FFT backends
    class(pencil),  intent(in)  :: self                 !! Pencil
    integer(int8) :: i  !! Counter

    make_public%dim = self%aligned_dim
    make_public%ndims = self%rank
    if ( allocated(make_public%counts) ) deallocate( make_public%counts )
    if ( allocated(make_public%starts) ) deallocate( make_public%starts )
    allocate(make_public%counts(1:self%rank), source=self%counts)
    allocate(make_public%starts(1:self%rank), source=self%starts)

    make_public%size = 1_int64
    do i = 1, make_public%ndims
      make_public%size = make_public%size * int(make_public%counts(i), int64)
    enddo
  end function make_public

  type(dtfft_pencil_t) function create_pencil_t(starts, counts)
  !! Creates pencil object, that can be used to create dtFFT plans
    integer(int32), intent(in)    :: starts(:)         !! Local starts, starting from 0 for both C and Fortran
    integer(int32), intent(in)    :: counts(:)         !! Local counts of data, in elements

    call create_pencil_t%destroy()
    create_pencil_t%ndims = size(starts, kind=int8)
    allocate(create_pencil_t%starts, source=starts)
    allocate(create_pencil_t%counts, source=counts)
    create_pencil_t%is_created = .true.
  end function  create_pencil_t

  subroutine destroy_pencil_t(self)
  !! Destroys pencil
    class(dtfft_pencil_t), intent(inout) :: self !! Public pencil

    if ( allocated(self%counts) ) deallocate( self%counts )
    if ( allocated(self%starts) ) deallocate( self%starts )
    self%dim = -1
    self%ndims = -1
    self%size = -1
    self%is_created = .false.
  end subroutine destroy_pencil_t

  ! subroutine destroy_pencil_t(self)
  ! !! Destroys pencil
  !   type(dtfft_pencil_t), intent(inout) :: self !! Public pencil

  !   call self%destroy()
  ! end subroutine destroy_pencil_t

  subroutine pencil_f2c(pencil, c_pencil)
  !! Converts Fortran pencil to C pencil
    type(dtfft_pencil_t), intent(in)  :: pencil     !! Fortran pencil
    type(dtfft_pencil_c), intent(out) :: c_pencil   !! C pencil

    c_pencil%dim = pencil%dim
    c_pencil%ndims = pencil%ndims
    c_pencil%size = pencil%size
    c_pencil%starts(1:pencil%ndims) = pencil%starts(:)
    c_pencil%counts(1:pencil%ndims) = pencil%counts(:)
  end subroutine pencil_f2c

  subroutine pencil_c2f(c_pencil, pencil, error_code)
  !! Converts C pencil to Fortran pencil
    type(dtfft_pencil_c), intent(in)  :: c_pencil   !! C pencil
    type(dtfft_pencil_t), intent(out) :: pencil     !! Fortran pencil
    integer(int32),       intent(out) :: error_code !! Error code

    error_code = DTFFT_SUCCESS
    if ( .not. any(c_pencil%ndims == [2, 3]) ) then
      error_code = DTFFT_ERROR_PENCIL_NOT_INITIALIZED
      return
    endif

    pencil = dtfft_pencil_t(c_pencil%starts(1:c_pencil%ndims), c_pencil%counts(1:c_pencil%ndims))
  end subroutine pencil_c2f

  subroutine get_local_sizes(pencils, in_starts, in_counts, out_starts, out_counts, alloc_size, is_y_slab)
  !! Obtain local starts and counts in `real` and `fourier` spaces
    type(pencil),             intent(in)  :: pencils(:)             !! Array of pencils
    integer(int32), optional, intent(out) :: in_starts(:)           !! Start indexes in `real` space (0-based)
    integer(int32), optional, intent(out) :: in_counts(:)           !! Number of elements in `real` space
    integer(int32), optional, intent(out) :: out_starts(:)          !! Start indexes in `fourier` space (0-based)
    integer(int32), optional, intent(out) :: out_counts(:)          !! Number of elements in `fourier` space
    integer(int64), optional, intent(out) :: alloc_size             !! Minimal number of elements required to execute plan
    logical,        optional, intent(in)  :: is_y_slab              !! Is Y-slab optimization used
    integer(int8)                         :: d                      !! Counter
    integer(int8)                         :: ndims                  !! Number of dimensions
    logical                               :: is_y_slab_             !! Is Y-slab optimization used
    integer(int8)                         :: out_dim                !! Aligned dimension of output pencil
    integer(int8)                         :: n_pencils

    ndims = size(pencils(1)%counts, kind=int8)
    n_pencils = size(pencils, kind=int8)
    is_y_slab_ = .false.; if ( present(is_y_slab) ) is_y_slab_ = is_y_slab
    out_dim = min(ndims, n_pencils)
    if ( is_y_slab_ .and. ndims == 3 ) out_dim = 2
    if ( present(in_starts) )  in_starts(1:ndims)   = pencils(1)%starts(1:ndims)
    if ( present(in_counts) )  in_counts(1:ndims)   = pencils(1)%counts(1:ndims)
    if ( present(out_starts) ) out_starts(1:ndims)  = pencils(out_dim)%starts(1:ndims)
    if ( present(out_counts) ) out_counts(1:ndims)  = pencils(out_dim)%counts(1:ndims)
    if ( present(alloc_size) ) alloc_size = maxval([(product(pencils(d)%counts), d=1,n_pencils)])
  end subroutine get_local_sizes

  pure function get_transpose_type(send, recv) result(transpose_type)
  !! Determines transpose ID based on pencils
    type(pencil),     intent(in)  :: send           !! Send pencil
    type(pencil),     intent(in)  :: recv           !! Receive pencil
    type(dtfft_transpose_t)       :: transpose_type !! Transpose ID

    transpose_type = dtfft_transpose_t(0)
    if (send%aligned_dim == 1 .and. recv%aligned_dim == 2) then
      transpose_type = DTFFT_TRANSPOSE_X_TO_Y
    else if (recv%aligned_dim == 1 .and. send%aligned_dim == 2) then
      transpose_type = DTFFT_TRANSPOSE_Y_TO_X
    else if (send%aligned_dim == 1 .and. recv%aligned_dim == 3) then
      transpose_type = DTFFT_TRANSPOSE_X_TO_Z
    else if (recv%aligned_dim == 1 .and. send%aligned_dim == 3) then
      transpose_type = DTFFT_TRANSPOSE_Z_TO_X
    else if (send%aligned_dim == 2 .and. recv%aligned_dim == 3) then
      transpose_type = DTFFT_TRANSPOSE_Y_TO_Z
    else if (recv%aligned_dim == 2 .and. send%aligned_dim == 3) then
      transpose_type = DTFFT_TRANSPOSE_Z_TO_Y
    endif
  end function get_transpose_type

  pure function get_reshape_type(send, recv) result(reshape_type)
  !! Determines reshape ID based on pencils
    type(pencil),     intent(in)  :: send           !! Send pencil
    type(pencil),     intent(in)  :: recv           !! Receive pencil
    type(dtfft_reshape_t)         :: reshape_type   !! Reshape ID

    reshape_type = dtfft_reshape_t(0)
    if (send%aligned_dim == 1 .and. recv%aligned_dim == 1) then
      if ( send%is_distributed ) then
        reshape_type = DTFFT_RESHAPE_X_BRICKS_TO_PENCILS
      else 
        reshape_type = DTFFT_RESHAPE_X_PENCILS_TO_BRICKS
      end if
    else
      if ( recv%is_distributed ) then
        reshape_type = DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS
      else
        reshape_type = DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS
      endif
    endif
  end function get_reshape_type

  function from_bricks(self, platform, bricks, comm) result(error_code)
    class(pencil_init),     intent(inout) :: self
    type(dtfft_platform_t), intent(in)  :: platform       !! Platform to create plan for
    type(pencil_init),      intent(in)  :: bricks
    TYPE_MPI_COMM,          intent(in)  :: comm         !! MPI Communicator passed to plan constructors
    integer(int32)                        :: error_code   !! Error code
    integer(int32) :: top_type, ierr, comm_rank
    type(dtfft_pencil_t) :: p
    logical :: is_nice_grid_found
    integer(int8)     :: ndims      !! Number of dimensions
    integer(int32),   allocatable   :: comm_dims(:)   !! Dims in cartesian communicator
    integer(int32),   allocatable   :: pencil_dims(:), pencil_starts(:)
    TYPE_MPI_COMM,    allocatable   :: pencil_comms(:)
    integer(int8)     :: d
    integer(int32) :: i, y_size, z_size, tile_size, fast_dim_size
    type(pencil) :: temp_pencil

    error_code = DTFFT_SUCCESS

    call MPI_Topo_test(comm, top_type, ierr)

    ndims = size(bricks%dims, kind=int8)
    allocate( comm_dims(ndims) )
    do d = 1, ndims
      call MPI_Comm_size(bricks%comms(d), comm_dims(d), ierr)
    enddo
    fast_dim_size = comm_dims(1)
    call MPI_Comm_rank(comm, comm_rank, ierr)

    tile_size = 4 ! for no reason at all
#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      tile_size = DEF_TILE_SIZE
    endif
#endif

    if ( top_type == MPI_CART .and. ndims == 3 ) then
      block
        integer(int32)                 :: grid_ndims           ! Number of dims in user defined cartesian communicator
        integer(int32),  allocatable   :: temp_dims(:)         ! Temporary dims needed by MPI_Cart_get
        integer(int32),  allocatable   :: temp_coords(:)       ! Temporary coordinates needed by MPI_Cart_get
        logical,         allocatable   :: temp_periods(:)      ! Temporary periods needed by MPI_Cart_get
        integer(int32)                 :: dims3d(3)

        call MPI_Cartdim_get(comm, grid_ndims, ierr)
        if ( grid_ndims > ndims ) then
          error_code = DTFFT_ERROR_INVALID_COMM_DIMS
          return
        endif
        allocate(temp_dims(grid_ndims), temp_periods(grid_ndims), temp_coords(grid_ndims))
        call MPI_Cart_get(comm, grid_ndims, temp_dims, temp_periods, temp_coords, ierr)
        if ( product(temp_dims) /= product(comm_dims) ) then
          error_code = DTFFT_ERROR_INVALID_CART_COMM
          return
        endif
        dims3d(:) = 1
        if ( grid_ndims == ndims ) then
          if ( temp_dims(1) /= 1 ) then
            error_code = DTFFT_ERROR_INVALID_COMM_FAST_DIM
            return
          endif
          dims3d(:) = temp_dims
        elseif ( grid_ndims == ndims - 1 ) then
          dims3d(2:) = temp_dims(:)
        elseif ( grid_ndims == ndims - 2 ) then
          dims3d(3) = temp_dims(1)
        endif
        deallocate(temp_dims, temp_periods, temp_coords)


        if ( dims3d(2) < comm_dims(2) .or. dims3d(3) < comm_dims(3) ) then
          error_code = DTFFT_ERROR_INVALID_CART_COMM
          return
        endif

        y_size = dims3d(2) / comm_dims(2)
        z_size = dims3d(3) / comm_dims(3)

        if ( y_size * z_size /= fast_dim_size ) then
          error_code = DTFFT_ERROR_INVALID_CART_COMM
          return
        endif

        if ( y_size * comm_dims(2) /= dims3d(2) .or. z_size * comm_dims(3) /= dims3d(3) ) then
          error_code = DTFFT_ERROR_INVALID_CART_COMM
          return
        endif
      endblock
    else
      ! Nx / P0  Ny / P1  Nz / P2
      ! -> reshape
      ! Nx  Ny / (P1 * Q1)  Nz / (P2 * Q2)
      ! -> transpose
      ! Ny  Nz / (P2 * Q2)  Nx / (P1 * Q1)
      ! -> transpose
      ! Nz  Nx / (P1 * Q1)  Ny / (P2 * Q2)
      ! -> reshape
      ! Nz / P2  Nx / (P1 * Q1)  Ny / Q2

      is_nice_grid_found = .false.
      if ( ndims == 3 .and. bricks%counts(ndims) > tile_size * fast_dim_size ) then
      !  .and. mod(bricks%counts(ndims), fast_dim_size) == 0 ) then
        y_size = 1
        z_size = fast_dim_size
        is_nice_grid_found = .true.
      else if ( (bricks%counts(2) > tile_size * fast_dim_size ) .or. ndims == 2 ) then
        ! .and. mod(bricks%counts(2), fast_dim_size) == 0
        y_size = fast_dim_size
        z_size = 1
        is_nice_grid_found = .true.
      else
        do i = 2, fast_dim_size
          if ( mod(fast_dim_size, i) /= 0 ) cycle
          if ( bricks%counts(ndims) < tile_size * i .or. mod(bricks%counts(ndims), i) /= 0 ) cycle
          if ( bricks%counts(2    ) < tile_size * i .or. mod(bricks%counts(2    ), i) /= 0 ) cycle

          is_nice_grid_found = .true.
          y_size = min(i, fast_dim_size / i)
          z_size = max(i, fast_dim_size / i)
          exit
        enddo
      endif

      ! In order to all processes have same decomposition
      call MPI_Bcast(is_nice_grid_found, 1, MPI_LOGICAL, 0, comm, ierr)

      if ( is_nice_grid_found ) then
        call MPI_Bcast(y_size, 1, MPI_INTEGER4, 0, comm, ierr)
        call MPI_Bcast(z_size, 1, MPI_INTEGER4, 0, comm, ierr)
      else
        block
          integer(int32) :: temp_dims(2)

          temp_dims(:) = 0
          call MPI_Dims_create(fast_dim_size, int(ndims - 1, int32), temp_dims, ierr)
          WRITE_WARN("Unable to find good grid decomposition, using MPI_Dims_create")
          y_size = temp_dims(1)
          z_size = temp_dims(2)
        endblock
      endif
    endif

    allocate( pencil_dims(ndims), pencil_starts(ndims) )
    pencil_dims(1) = bricks%dims(1)
    pencil_dims(2:) = bricks%counts(2:)

    pencil_starts(1) = 0

    allocate( pencil_comms(ndims) )

    pencil_comms(1) = MPI_COMM_SELF
    if (y_size == 1) then
      pencil_comms(2) = MPI_COMM_SELF
      pencil_comms(3) = bricks%comms(1)

      pencil_starts(2) = bricks%starts(2)
      call MPI_Allreduce(bricks%starts(3), pencil_starts(3), 1, MPI_INTEGER, MPI_MIN, pencil_comms(3), ierr)
    else if ( z_size == 1 ) then
      pencil_comms(2) = bricks%comms(1)

      call MPI_Allreduce(bricks%starts(2), pencil_starts(2), 1, MPI_INTEGER, MPI_MIN, pencil_comms(2), ierr)

      if ( ndims == 3 ) then
        pencil_comms(3) = MPI_COMM_SELF
        pencil_starts(3) = bricks%starts(3)
      endif
    else
      block
        integer(int32) :: dims(2)
        logical        :: remain_dims(2), periods(2), reorder
        TYPE_MPI_COMM :: cart_comm

        dims(1) = y_size
        dims(2) = z_size
        periods(1) = .false.
        periods(2) = .false.
        reorder = .false.
        call MPI_Cart_create(bricks%comms(1), 2, dims, periods, reorder, cart_comm, ierr)

        remain_dims(1) = .true.
        remain_dims(2) = .false.
        call MPI_Cart_sub(cart_comm, remain_dims, pencil_comms(2), ierr)

        remain_dims(1) = .false.
        remain_dims(2) = .true.
        call MPI_Cart_sub(cart_comm, remain_dims, pencil_comms(3), ierr)

        call MPI_Comm_free(cart_comm, ierr)
        ! pencil_dims(2:) = bricks%dims(2:)

        ! call MPI_Allreduce(bricks%starts(2), pencil_starts(2), 1, MPI_INTEGER, MPI_MIN, bricks%comms(2), ierr)
        ! call MPI_Allreduce(bricks%starts(3), pencil_starts(3), 1, MPI_INTEGER, MPI_MIN, bricks%comms(3), ierr)

        call MPI_Allreduce(bricks%starts(2), pencil_starts(2), 1, MPI_INTEGER, MPI_MIN, pencil_comms(2), ierr)
        call MPI_Allreduce(bricks%starts(3), pencil_starts(3), 1, MPI_INTEGER, MPI_MIN, pencil_comms(3), ierr)
      endblock
    endif

    call temp_pencil%create(ndims, 1_int8, pencil_dims, pencil_comms)
    temp_pencil%starts(:) = temp_pencil%starts(:) + pencil_starts(:)

    do d = 2, ndims
      if ( pencil_comms(d) /= bricks%comms(1) .and. pencil_comms(d) /= MPI_COMM_SELF ) then
        call MPI_Comm_free(pencil_comms(d), ierr)
      endif
    enddo

    p = dtfft_pencil_t(temp_pencil%starts, temp_pencil%counts)

    ierr = self%create(p, comm)

    if ( ierr /= DTFFT_SUCCESS ) INTERNAL_ERROR("from_bricks: unable to create pencil")
    call temp_pencil%destroy()
    call p%destroy()
    deallocate( pencil_dims, pencil_comms, comm_dims )
  end function from_bricks

  function create_pencil_init(self, pencil, comm) result(error_code)
  !! Creates and validates pencil passed by user to plan constructors
    class(pencil_init),     intent(inout) :: self         !! Internal pencil representation based on dtfft_pencil_t
    type(dtfft_pencil_t),   intent(in)    :: pencil       !! Pencil passed by user to plan constructors
    TYPE_MPI_COMM,          intent(in)    :: comm         !! MPI Communicator passed to plan constructors
    integer(int32)                        :: error_code   !! Error code
    integer(int32)              :: comm_rank          !! Rank of current MPI process
    integer(int32)              :: comm_size, csizes(3), psize   !! Size of communicator
    integer(int32)              :: ndims              !! Number of dimensions
    integer(int32)              :: ierr               !! Error code from MPI calls
    integer(int32), allocatable :: all_starts(:,:)    !! All starts gathered from all processes
    integer(int32), allocatable :: all_counts(:,:)    !! All counts gathered from all processes
    integer(int32), allocatable :: fixed_dims(:)      !! Fixed dimensions for 1D communicators
    integer(int32)              :: i, j, d            !! Counters
    integer(int32)              :: p1, p2, d1, d2     !! Counters

    error_code = DTFFT_SUCCESS
    if (.not. pencil%is_created) then
      error_code = DTFFT_ERROR_PENCIL_NOT_INITIALIZED
    end if
    CHECK_ERROR_AND_RETURN_AGG(comm)

    ndims = size(pencil%starts)
    if (ndims /= size(pencil%counts)) then
      error_code = DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH
    end if
    CHECK_ERROR_AND_RETURN_AGG(comm)

    if (ndims < 2 .or. ndims > 3) then
      error_code = DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES
    end if
    CHECK_ERROR_AND_RETURN_AGG(comm)

    if (any(pencil%starts < 0)) then
      error_code = DTFFT_ERROR_PENCIL_INVALID_STARTS
    end if
    CHECK_ERROR_AND_RETURN_AGG(comm)

    if (any(pencil%counts < 0)) then
      error_code = DTFFT_ERROR_PENCIL_INVALID_COUNTS
    end if
    CHECK_ERROR_AND_RETURN_AGG(comm)

    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)

    allocate(self%dims(ndims))

    allocate(all_starts(ndims, comm_size))
    allocate(all_counts(ndims, comm_size))

    call MPI_Allgather(pencil%starts, ndims, MPI_INTEGER, all_starts, ndims, MPI_INTEGER, comm, ierr)
    call MPI_Allgather(pencil%counts, ndims, MPI_INTEGER, all_counts, ndims, MPI_INTEGER, comm, ierr)

    ! Computing global dimensions
    do d = 1, ndims
      self%dims(d) = maxval(all_starts(d,:) + all_counts(d,:))
    enddo

    do p1 = 1, comm_size
      ! Skip processes with zero volume
      if (any(all_counts(:, p1) == 0)) cycle

      do p2 = p1 + 1, comm_size
        ! Skip processes with zero volume
        if (any(all_counts(:, p2) == 0)) cycle

        do d1 = 1, ndims
          do d2 = d1 + 1, ndims
            ! If two processes have the same start on two different axes,
            ! then their blocks must have the same size on these axes.
            if (all_starts(d1, p1) == all_starts(d1, p2) .and. &
                all_starts(d2, p1) == all_starts(d2, p2)) then

              if ((all_counts(d1, p1) /= all_counts(d1, p2) .or. &
                  all_counts(d2, p1) /= all_counts(d2, p2))) then
                    error_code = DTFFT_ERROR_PENCIL_SHAPE_MISMATCH
              endif
            endif
          enddo
        enddo
      enddo
    enddo
    CHECK_ERROR_AND_RETURN_AGG(comm)

    ! Check intersection of pencils (overlap check already handles zero-volume pencils)
    do i = 1, comm_size
      do j = i + 1, comm_size
        if (check_overlap(all_starts(:, i), all_counts(:, i), all_starts(:, j), all_counts(:, j), ndims)) then
          error_code = DTFFT_ERROR_PENCIL_OVERLAP
        endif
      enddo
    enddo
    CHECK_ERROR_AND_RETURN_AGG(comm)
    ! Check continuity of pencils
    if (.not. check_continuity(all_starts, all_counts, self%dims, comm_size)) then
      error_code = DTFFT_ERROR_PENCIL_NOT_CONTINUOUS
    endif
    CHECK_ERROR_AND_RETURN_AGG(comm)

    allocate(self%starts, source=pencil%starts)
    allocate(self%counts, source=pencil%counts)
    allocate(self%comms(ndims))
    allocate(fixed_dims(ndims - 1))

    ! Create 1D communicators for each dimension
    do d = 1, ndims
      j = 1
      do i = 1, ndims
        if (i /= d) then
          fixed_dims(j) = i
          j = j + 1
        endif
      enddo
      call create_1d_comm(self%starts, all_starts, fixed_dims, comm, self%comms(d), all_counts)
    enddo

    psize = 1
    do d = 1, ndims
      call MPI_Comm_size(self%comms(d), csizes(d), ierr)
      psize = psize * csizes(d)
    enddo
    if ( psize /= comm_size ) then
      do d = 1, ndims
        WRITE_ERROR("d = "//to_str(d)//": comm_size = "//to_str(csizes(d)))
      enddo
      INTERNAL_ERROR("create_pencil_init: psize /= comm_size")
    endif
    deallocate(fixed_dims)
    deallocate(all_starts)
    deallocate(all_counts)
  end function create_pencil_init

  subroutine destroy_pencil_init(self)
  !! Destroys pencil_init
    class(pencil_init), intent(inout) :: self   !!  Internal pencil representation based on dtfft_pencil_t

    if (allocated(self%starts)) deallocate(self%starts)
    if (allocated(self%counts)) deallocate(self%counts)
    if (allocated(self%comms)) then
      block
        integer(int32) :: ierr, i
        do i = 1, size(self%comms)
          call MPI_Comm_free(self%comms(i), ierr)
        enddo
      end block
      deallocate(self%comms)
    end if
    if (allocated(self%dims)) deallocate(self%dims)
  end subroutine destroy_pencil_init


  pure logical function check_overlap(lbounds1, sizes1, lbounds2, sizes2, ndims)
  !! Check if two pencols overlap in ndims-dimensional space
    integer(int32), intent(in) :: lbounds1(:)   !! Lower bounds of first pencil
    integer(int32), intent(in) :: sizes1(:)     !! Sizes of first pencil
    integer(int32), intent(in) :: lbounds2(:)   !! Lower bounds of second pencil
    integer(int32), intent(in) :: sizes2(:)     !! Sizes of second pencil
    integer(int32), intent(in) :: ndims         !! Number of dimensions
    integer :: d

    ! If either pencil has zero size in any dimension, there is no overlap
    if (any(sizes1(1:ndims) == 0) .or. any(sizes2(1:ndims) == 0)) then
      check_overlap = .false.
      return
    endif

    check_overlap = .true.

    do d = 1, ndims
      ! If there is no intersection in one of the dimensions, then there is no intersection in the whole space
      if (lbounds1(d) + sizes1(d) <= lbounds2(d) .or. lbounds2(d) + sizes2(d) <= lbounds1(d)) then
        check_overlap = .false.
        return
      endif
    enddo
  end function check_overlap

  logical function check_continuity(all_lbounds, all_sizes, global_dims, comm_size)
  !! Check if the local pencils cover the global space without gaps
    integer(int32), intent(in) :: all_lbounds(:,:)    !! Lower bounds of local pencils for each process
    integer(int32), intent(in) :: all_sizes(:,:)      !! Sizes of local pencils for each process
    integer(int32), intent(in) :: global_dims(:)      !! Global dimensions of the problem
    integer(int32), intent(in) :: comm_size           !! Number of processes in the communicator
    integer(int64) :: total_local_volume, global_volume
    integer :: i
    logical :: has_zero_volume

    ! 1. Check that local pencils do not exceed global grid limits
    do i = 1, comm_size
      ! Skip processes with zero volume
      has_zero_volume = any(all_sizes(:,i) == 0)
      if (has_zero_volume) cycle

      if (any(all_lbounds(:,i) < 0) .or. &
          any(all_lbounds(:,i) + all_sizes(:,i) > global_dims)) then
          check_continuity = .false.
          return
      endif
    enddo

    ! 2. Compare the sum of the local block volumes with the volume of the global grid
    total_local_volume = 0
    do i = 1, comm_size
      ! Skip processes with zero volume
      has_zero_volume = any(all_sizes(:,i) == 0)
      if (has_zero_volume) cycle

      total_local_volume = total_local_volume + product(int(all_sizes(:,i), int64))
    enddo

    global_volume = product(int(global_dims, int64))

    if (total_local_volume == global_volume) then
      check_continuity = .true.
    else
      check_continuity = .false.
    end if
  end function check_continuity

  function get_varying_dim(fixed_dims, total_dims) result(varying_dim)
    integer(int32), intent(in) :: fixed_dims(:)
    integer(int32), intent(in) :: total_dims
    integer(int32) :: varying_dim

    do varying_dim = 1, total_dims
      if (.not. any(fixed_dims == varying_dim)) exit
    enddo
  end function get_varying_dim

  subroutine sort_by_varying_dim(ranks, coords)
    integer(int32), intent(inout) :: ranks(:)
    integer(int32), intent(inout)    :: coords(:)
    integer(int32) :: i, j, tmp_rank, tmp_coord, n
    ! logical :: swapped
    ! integer :: temp_coord, temp_num

    n = size(coords)
    if (n <= 1) return

    do i = 2, n
      tmp_rank = ranks(i)
      tmp_coord = coords(i)
      j = i - 1

      do while (j >= 1)
        if (coords(j) <= tmp_coord) exit
        ranks(j + 1) = ranks(j)
        coords(j + 1) = coords(j)
        j = j - 1
      end do

      ranks(j + 1) = tmp_rank
      coords(j + 1) = tmp_coord
    end do
  end subroutine sort_by_varying_dim

  subroutine create_1d_comm(lbounds, all_lbounds, fixed_dims, comm, new_comm, all_counts)
  !! Creates a new 1D communicator based on the fixed dimensions of the current pencil
    integer(int32), intent(in) :: lbounds(:)          !! Local starts of the current pencil
    integer(int32), intent(in) :: all_lbounds(:,:)    !! Local starts of all processes
    integer(int32), intent(in) :: fixed_dims(:)       !! Indices of fixed coordinates
    TYPE_MPI_COMM, intent(in)  :: comm                !! Original MPI communicator
    TYPE_MPI_COMM, intent(out) :: new_comm            !! New 1D MPI communicator
    integer(int32), intent(in) :: all_counts(:,:)
    integer(int32) :: comm_size   !! Size of `comm`
    integer(int32) :: ierr        !! Error codes for mpi calls
    integer(int32), allocatable :: neighbors(:) !! Array to neighbors ranks
    integer(int32), allocatable :: varying_dim_coord(:)    !! Coordinates along the non-fixed dimension
    integer(int32) :: i, j                      !! Counters
    integer(int32) :: neighbor_count            !! Number of neighboring processes
    integer(int32) :: var_dim, comm_rank
    logical :: matched, count_matched

    call MPI_Comm_size(comm, comm_size, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)
    var_dim = get_varying_dim(fixed_dims, size(lbounds))
    allocate(neighbors(comm_size), varying_dim_coord(comm_size))
    neighbor_count = 0

    ! Find processes with matching fixed dimensions
    do i = 1, comm_size
      matched = .true.
      count_matched = .true.
      do j = 1, size(fixed_dims)
        if (all_lbounds(fixed_dims(j), i) /= lbounds(fixed_dims(j))) matched = .false.
        if ( all_counts(fixed_dims(j), i) /= all_counts(fixed_dims(j), comm_rank + 1) ) count_matched = .false.
      enddo
      matched = matched .and. count_matched

      if (  (matched .and. all_lbounds(var_dim, i) /= lbounds(var_dim)) & 
          .or. i - 1 == comm_rank                   &
          .or. (matched .and. (all_lbounds(var_dim, i) == lbounds(var_dim) .and. (all_counts(var_dim, i) == 0 .or. all_counts(var_dim, comm_rank + 1) == 0)))) then
        neighbor_count = neighbor_count + 1
        neighbors(neighbor_count) = i - 1
        ! Store the coordinate along the first non-fixed dimension for sorting
        varying_dim_coord(neighbor_count) = all_lbounds(var_dim, i)
      endif
    enddo

    ! Sort neighbors by their coordinate along the varying dimension
    call sort_by_varying_dim(neighbors, varying_dim_coord(1:neighbor_count))
    call create_subcomm(comm, neighbors(1:neighbor_count), new_comm)
    deallocate(neighbors, varying_dim_coord)
  end subroutine create_1d_comm
end module dtfft_pencil