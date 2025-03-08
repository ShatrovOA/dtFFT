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
!! This module describes private `pencil` and public `dtfft_pencil` classes
use iso_c_binding,    only: c_int8_t, c_int32_t
use iso_fortran_env,  only: int8, int32, int64, real64, output_unit
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: pencil
public :: dtfft_pencil_t
public :: get_local_sizes
public :: get_transpose_type

  type, bind(C) :: dtfft_pencil_t
  !! Structure to hold pencil decomposition info
    integer(c_int8_t)   :: dim        !! Aligned dimension id
    integer(c_int8_t)   :: ndims      !! Number of dimensions
    integer(c_int32_t)  :: starts(3)  !! Local starts, starting from 0 for both C and Fortran
    integer(c_int32_t)  :: counts(3)  !! Local counts of data, in elements
  end type dtfft_pencil_t

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

contains
  subroutine create(self, rank, aligned_dim, counts, comms)
  !! Creates pencil
    class(pencil),      intent(inout) :: self             !! Pencil
    integer(int8),      intent(in)    :: rank             !! Rank of buffer
    integer(int8),      intent(in)    :: aligned_dim      !! Position of aligned dimension
    integer(int32),     intent(in)    :: counts(:)        !! Global counts
    TYPE_MPI_COMM,      intent(in)    :: comms(:)         !! Grid communicators
    integer(int8)                     :: d                !! Counter
    logical, allocatable              :: is_even(:)       !! Even distribution flag

    call self%destroy()
    allocate(self%counts(rank))
    allocate(self%starts(rank))
    allocate(is_even(rank))
    self%aligned_dim = aligned_dim
    self%rank = rank
    do d = 1, rank
      call get_local_size(counts(d), comms(d), self%starts(d), self%counts(d), is_even(d))
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

  subroutine get_local_size(n_global, comm, start, count, is_even)
  !! Computes local portions of data based on global count and position inside grid communicator
    integer(int32), intent(in)    :: n_global             !! Global number of points
    TYPE_MPI_COMM,  intent(in)    :: comm                 !! Grid communicator
    integer(int32), intent(out)   :: start                !! Local start
    integer(int32), intent(out)   :: count                !! Local count
    logical,        intent(out)   :: is_even              !! Is data evenly distributed across processes
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
      is_even = .true.
      return
    elseif ( comm_rank >= comm_dim - res ) then
      count = int(n_global / comm_dim, int32) + 1
    else
      count = int(n_global / comm_dim, int32)
    endif
    allocate( shift(comm_dim) )
    call MPI_Allgather(count, 1, MPI_INTEGER, shift, 1, MPI_INTEGER, comm, ierr)

    do i = 0, comm_rank - 1
      start = start + shift(i + 1)
    end do
    is_even = all(shift == shift(1))
    deallocate(shift)
  end subroutine get_local_size

!   subroutine output(self, name, vec)
!   !! Writes pencil data to stdout
!     class(pencil),                intent(in)  :: self                 !! Pencil
!     character(len=*),             intent(in)  :: name                 !! Name of pencil
!     real(real64),   DEVICE_PTR    intent(in)  :: vec(:)               !! Device pointer to data
!     integer(int32)                            :: iter                 !! Iteration counter
!     integer(int32)                            :: i,j,k,ijk            !! Counters
!     integer(int32)                            :: comm_size            !! Number of MPI processes
!     integer(int32)                            :: comm_rank            !! Rank of current MPI process
!     integer(int32)                            :: ierr                 !! Error code
! #ifdef DTFFT_WITH_CUDA
!     real(real64),                 allocatable :: buf(:)               !! Host buffer
! #endif

!     call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
!     call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

! #ifdef DTFFT_WITH_CUDA
!     allocate( buf( product(self%counts) ) )

!     CUDA_CALL( "cudaMemcpy", cudaMemcpy(buf, vec, product(self%counts), cudaMemcpyDeviceToHost) )
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
    class(pencil),  intent(in)  :: self                 !! Pencil

    make_public%dim = self%aligned_dim
    make_public%ndims = self%rank
    make_public%counts(1:self%rank) = self%counts
    make_public%starts(1:self%rank) = self%starts
  end function make_public

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
    type(dtfft_transpose_type_t)  :: transpose_type   !! Transpose ID

    transpose_type = dtfft_transpose_type_t(0)
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
end module dtfft_pencil