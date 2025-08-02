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
module dtfft_pencil
!! This module describes private [[pencil]] and public [[dtfft_pencil_t]] classes
use iso_fortran_env
use iso_c_binding
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_private.h"
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#include "dtfft_cuda.h"
#endif
implicit none
private
public :: pencil, pencil_init
public :: dtfft_pencil_t, dtfft_pencil_c
public :: get_local_sizes
public :: get_transpose_type
public :: pencil_c2f, pencil_f2c

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
    final :: destroy_pencil_t
    procedure, pass(self),  private  :: destroy => destroy_pencil_t_private
  end type dtfft_pencil_t

  interface dtfft_pencil_t
  !! Type bound constuctor for dtfft_pencil_t
    module procedure create_pencil_t
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
  end type pencil_init

contains
  subroutine create(self, rank, aligned_dim, counts, comms, lstarts, lcounts)
  !! Creates pencil
    class(pencil),                  intent(inout) :: self             !! Pencil
    integer(int8),                  intent(in)    :: rank             !! Rank of buffer
    integer(int8),                  intent(in)    :: aligned_dim      !! Position of aligned dimension
    integer(int32),                 intent(in)    :: counts(:)        !! Global counts
    TYPE_MPI_COMM,                  intent(in)    :: comms(:)         !! Grid communicators
    integer(int32),       optional, intent(in)    :: lstarts(:)       !! Local starts
    integer(int32),       optional, intent(in)    :: lcounts(:)       !! Local counts
    integer(int8)                     :: d                !! Counter
    logical, allocatable              :: is_even(:)       !! Even distribution flag

    call self%destroy()
    allocate(self%counts(rank))
    allocate(self%starts(rank))
    allocate(is_even(rank))
    self%aligned_dim = aligned_dim
    self%rank = rank
    if ( present(lstarts) .and. present(lcounts) ) then
      if ( aligned_dim == 1 ) then
        self%starts(:) = lstarts(:)
        self%counts(:) = lcounts(:)
      else
        do d = 1, rank
          if ( aligned_dim == 2 .and. rank == 3 .and. d == 3 ) then
            call get_local_size(counts(d), comms(d), self%starts(d), self%counts(d), lstarts(3), lcounts(3))
          else if ( aligned_dim == 3 .and. rank == 3 .and. d == 2 ) then
            call get_local_size(counts(d), comms(d), self%starts(d), self%counts(d), lstarts(2), lcounts(2))
          else
            call get_local_size(counts(d), comms(d), self%starts(d), self%counts(d))
          endif
        enddo
      endif
    else
      do d = 1, rank
        call get_local_size(counts(d), comms(d), self%starts(d), self%counts(d))
      enddo
    endif
    do d = 1, rank
      is_even(d) = check_if_even(self%counts(d), comms(d))
    enddo
    self%is_even = all(is_even)
    deallocate(is_even)
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
!     type(c_ptr),                  intent(in)  :: vec               !! Device pointer to data
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

!     allocate( buf( product(self%counts) ) )

! #ifdef DTFFT_WITH_CUDA
!     if ( is_device_ptr(vec) ) then
!       CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize())
!       CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(buf), vec, int(real32, int64) * product(self%counts), cudaMemcpyDeviceToHost) )
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

  subroutine destroy_pencil_t_private(self)
  !! Destroys pencil
    class(dtfft_pencil_t), intent(inout) :: self !! Public pencil

    if ( allocated(self%counts) ) deallocate( self%counts )
    if ( allocated(self%starts) ) deallocate( self%starts )
    self%dim = -1
    self%ndims = -1
    self%size = -1
    self%is_created = .false.
  end subroutine destroy_pencil_t_private

  subroutine destroy_pencil_t(self)
  !! Destroys pencil
    type(dtfft_pencil_t), intent(inout) :: self !! Public pencil

    call self%destroy()
  end subroutine destroy_pencil_t

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

  subroutine pencil_c2f(c_pencil, pencil)
  !! Converts C pencil to Fortran pencil
    type(dtfft_pencil_c), intent(in)  :: c_pencil   !! C pencil
    type(dtfft_pencil_t), intent(out) :: pencil     !! Fortran pencil

    pencil = dtfft_pencil_t(c_pencil%starts(1:c_pencil%ndims), c_pencil%counts(1:c_pencil%ndims))
  end subroutine pencil_c2f

  subroutine get_local_sizes(pencils, in_starts, in_counts, out_starts, out_counts, alloc_size)
  !! Obtain local starts and counts in `real` and `fourier` spaces
    type(pencil),             intent(in)  :: pencils(:)             !! Array of pencils
    integer(int32), optional, intent(out) :: in_starts(:)           !! Start indexes in `real` space (0-based)
    integer(int32), optional, intent(out) :: in_counts(:)           !! Number of elements in `real` space
    integer(int32), optional, intent(out) :: out_starts(:)          !! Start indexes in `fourier` space (0-based)
    integer(int32), optional, intent(out) :: out_counts(:)          !! Number of elements in `fourier` space
    integer(int64), optional, intent(out) :: alloc_size             !! Minimal number of elements required to execute plan
    integer(int8)                         :: d                      !! Counter
    integer(int8)                         :: ndims                  !! Number of dimensions

    ndims = size(pencils, kind=int8)
    if ( present(in_starts) )  in_starts(1:ndims)   = pencils(1)%starts(1:ndims)
    if ( present(in_counts) )  in_counts(1:ndims)   = pencils(1)%counts(1:ndims)
    if ( present(out_starts) ) out_starts(1:ndims)  = pencils(ndims)%starts(1:ndims)
    if ( present(out_counts) ) out_counts(1:ndims)  = pencils(ndims)%counts(1:ndims)
    if ( present(alloc_size) ) alloc_size = maxval([(product(pencils(d)%counts), d=1,ndims)])
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

  function create_pencil_init(self, pencil, comm) result(error_code)
  !! Creates and validates pencil passed by user to plan constructors
    class(pencil_init),     intent(inout) :: self         !! Internal pencil representation based on dtfft_pencil_t
    type(dtfft_pencil_t),   intent(in)    :: pencil       !! Pencil passed by user to plan constructors
    TYPE_MPI_COMM,          intent(in)    :: comm         !! MPI Communicator passed to plan constructors
    integer(int32)                        :: error_code   !! Error code
    integer(int32)              :: comm_rank          !! Rank of current MPI process
    integer(int32)              :: comm_size          !! Size of communicator
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
      do p2 = p1 + 1, comm_size
        do d1 = 1, ndims
          do d2 = d1 + 1, ndims
            ! If two processes have the same start on two different axes,
            ! then their blocks must have the same size on these axes.
            if (all_starts(d1, p1) == all_starts(d1, p2) .and. &
                all_starts(d2, p1) == all_starts(d2, p2)) then

              if (all_counts(d1, p1) /= all_counts(d1, p2) .or. &
                  all_counts(d2, p1) /= all_counts(d2, p2)) then
                    error_code = DTFFT_ERROR_PENCIL_SHAPE_MISMATCH
              endif
            endif
          enddo
        enddo
      enddo
    enddo
    CHECK_ERROR_AND_RETURN_AGG(comm)

    ! Check intersection of pencils
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
      call create_1d_comm(self%starts, all_starts, fixed_dims, comm, self%comms(d))
    enddo
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

    ! 1. Check that local pencils do not exceed global grid limits
    do i = 1, comm_size
      if (any(all_lbounds(:,i) < 0) .or. &
          any(all_lbounds(:,i) + all_sizes(:,i) > global_dims)) then
          check_continuity = .false.
          return
      endif
    enddo

    ! 2. Compare the sum of the local block volumes with the volume of the global grid
    total_local_volume = 0
    do i = 1, comm_size
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
    integer(int32), intent(in)    :: coords(:)
    integer(int32) :: i, j, tmp_rank, tmp_coord, n

    n = size(ranks)
    if (n <= 1) return

    do i = 2, n
      tmp_rank = ranks(i)
      tmp_coord = coords(i)
      j = i - 1

      do while (j >= 1)
        if (coords(j) <= tmp_coord) exit
        ranks(j + 1) = ranks(j)
        j = j - 1
      end do

      ranks(j + 1) = tmp_rank
    end do
  end subroutine sort_by_varying_dim

  subroutine create_1d_comm(lbounds, all_lbounds, fixed_dims, comm, new_comm)
  !! Creates a new 1D communicator based on the fixed dimensions of the current pencil
    integer(int32), intent(in) :: lbounds(:)          !! Local starts of the current pencil
    integer(int32), intent(in) :: all_lbounds(:,:)    !! Local starts of all processes
    integer(int32), intent(in) :: fixed_dims(:)       !! Indices of fixed coordinates
    TYPE_MPI_COMM, intent(in)  :: comm                !! Original MPI communicator
    TYPE_MPI_COMM, intent(out) :: new_comm            !! New 1D MPI communicator
    integer(int32) :: comm_size   !! Size of `comm`
    integer(int32) :: ierr        !! Error codes for mpi calls
    integer(int32), allocatable :: neighbors(:) !! Array to neighbors ranks
    integer(int32), allocatable :: varying_dim(:)    !! Coordinates along the non-fixed dimension

    integer(int32) :: i, j                      !! Counters
    integer(int32) :: neighbor_count            !! Number of neighboring processes
    TYPE_MPI_GROUP :: group                     !! Original MPI group
    TYPE_MPI_GROUP :: new_group                 !! New MPI group for 1D communicator

    call MPI_Comm_size(comm, comm_size, ierr)

    allocate(neighbors(comm_size), varying_dim(comm_size))
    neighbor_count = 0

    ! Find processes with matching fixed dimensions
    do i = 1, comm_size
      do j = 1, size(fixed_dims)
        if (all_lbounds(fixed_dims(j), i) /= lbounds(fixed_dims(j))) exit
      enddo
      if (j > size(fixed_dims)) then
        neighbor_count = neighbor_count + 1
        neighbors(neighbor_count) = i - 1  ! MPI ranks are 0-based
        ! Store the coordinate along the first non-fixed dimension for sorting
        varying_dim(neighbor_count) = all_lbounds(get_varying_dim(fixed_dims, size(lbounds)), i)
      endif
    enddo

    ! Sort neighbors by their coordinate along the varying dimension
    call sort_by_varying_dim(neighbors(1:neighbor_count), varying_dim(1:neighbor_count))

    ! Create the new group and communicator
    call MPI_Comm_group(comm, group, ierr)
    call MPI_Group_incl(group, neighbor_count, neighbors(1:neighbor_count), new_group, ierr)
    call MPI_Comm_create(comm, new_group, new_comm, ierr)
    call MPI_Group_free(group, ierr)
    call MPI_Group_free(new_group, ierr)
    deallocate(neighbors, varying_dim)
  end subroutine create_1d_comm
end module dtfft_pencil