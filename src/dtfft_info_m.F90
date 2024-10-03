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
module dtfft_info_m
!------------------------------------------------------------------------------------------------
!< This module describes `info_t` class
!------------------------------------------------------------------------------------------------
use dtfft_precisions
#include "dtfft_mpi.h"
implicit none
private
public :: info_t

  type :: info_t
  !< Class that describes information about data layout
    integer(IP)               :: aligned_dim      !< Position of aligned dimension. For example: X pencil aligned_dim = 1, Z pencil aligned_dim = 3
    integer(IP)               :: rank             !< Rank of buffer: 2 or 3
    integer(IP), allocatable  :: starts(:)        !< Local starts, starting from 0 for both C and Fortran
    integer(IP), allocatable  :: counts(:)        !< Local counts of data
    logical                   :: is_even          !< Is data evenly distributed across processes
  contains
    procedure, pass(self) :: init                 !< Creates class
    procedure, pass(self) :: destroy              !< Destroys class
  end type info_t

contains
!------------------------------------------------------------------------------------------------
  subroutine init(self, rank, aligned_dim, counts, comms, comm_dims, comm_coords)
!------------------------------------------------------------------------------------------------
!< Creates info_t class
!------------------------------------------------------------------------------------------------
    class(info_t),      intent(inout) :: self             !< Info class
    integer(IP),        intent(in)    :: rank             !< Rank of buffer
    integer(IP),        intent(in)    :: aligned_dim      !< Position of aligned dimension
    integer(IP),        intent(in)    :: counts(:)        !< Global counts
    TYPE_MPI_COMM,      intent(in)    :: comms(:)         !< Grid communicators
    integer(IP),        intent(in)    :: comm_dims(:)     !< Grid dimensions
    integer(IP),        intent(in)    :: comm_coords(:)   !< Grid coordinates
    integer(IP)                       :: d                !< Counter
    logical, allocatable              :: is_even(:)       !< Even distribution flag

    call self%destroy()
    allocate(self%counts(rank))
    allocate(self%starts(rank))
    allocate(is_even(rank))
    self%aligned_dim = aligned_dim
    self%rank = rank
    do d = 1, rank
      call get_local_size(counts(d), comms(d), comm_dims(d), comm_coords(d), self%starts(d), self%counts(d), is_even(d))
    enddo
    self%is_even = all(is_even)
    deallocate(is_even)
  end subroutine init

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys info_t class
!------------------------------------------------------------------------------------------------
    class(info_t),  intent(inout) :: self                 !< Info class

    if ( allocated(self%counts) ) deallocate(self%counts)
    if ( allocated(self%starts) ) deallocate(self%starts)
  end subroutine destroy

!------------------------------------------------------------------------------------------------
  subroutine get_local_size(n_global, comm, comm_dim, comm_coord, start, count, is_even)
!------------------------------------------------------------------------------------------------
!< Computes local portions of data based on global count and position inside grid communicator
!------------------------------------------------------------------------------------------------
    integer(IP),    intent(in)    :: n_global             !< Global number of points
    TYPE_MPI_COMM,  intent(in)    :: comm                 !< Grid communicator
    integer(IP),    intent(in)    :: comm_dim             !< Number of MPI processes along n_global
    integer(IP),    intent(in)    :: comm_coord           !< Coordinate of current MPI process
    integer(IP),    intent(out)   :: start                !< Local start
    integer(IP),    intent(out)   :: count                !< Local count
    logical,        intent(out)   :: is_even              !< Is data evenly distributed across processes
    integer(IP)                   :: shift(comm_dim)      !< Work buffer
    integer(IP)                   :: res                  !< Residual from n_global / comm_dim
    integer(IP)                   :: i                    !< Counter
    integer(IP)                   :: ierr

    res = mod(n_global, comm_dim)
    start = 0

    if ( comm_dim == 1 ) then
      count = n_global
      is_even = .true.
      return
    elseif ( comm_coord >= comm_dim - res ) then
      count = int(n_global / comm_dim, IP) + 1
    else
      count = int(n_global / comm_dim, IP)
    endif
    call MPI_Allgather(count, 1, MPI_INTEGER, shift, 1, MPI_INTEGER, comm, ierr)

    do i = 0, comm_coord - 1
      start = start + shift(i + 1)
    end do
    is_even = all(shift == shift(1))
  end subroutine get_local_size
end module dtfft_info_m