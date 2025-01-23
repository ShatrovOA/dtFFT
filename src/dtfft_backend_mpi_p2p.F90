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
module dtfft_backend_mpi_p2p
!! This module defines MPI P2P backend: `backend_mpi_p2p`
use iso_fortran_env
use cudafor
use dtfft_abstract_backend, only: backend_helper
use dtfft_backend_mpi,      only: backend_mpi
use dtfft_parameters,       only: DTFFT_GPU_BACKEND_MPI_P2P
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_mpi_p2p

  type, extends(backend_mpi) :: backend_mpi_p2p
  !! MPI P2P Backend
  private
  contains
    procedure         :: create_selfcopy => create    !< Creates MPI P2P Backend
    procedure         :: execute_mpi => execute       !< Executes MPI P2P Backend
  end type backend_mpi_p2p

contains

  subroutine create(self, helper)
  !! Creates MPI P2P Backend
    class(backend_mpi_p2p),   intent(inout) :: self           !< MPI P2P Backend
    type(backend_helper),     intent(in)    :: helper           !< MPI communicator
    call self%create_requests(2 * (self%comm_size - 1))
  end subroutine create

  subroutine execute(self, in, out)
  !! Executes MPI P2P Backend
    class(backend_mpi_p2p),       intent(inout) :: self       !< MPI P2P Backend
    type(c_devptr),               intent(in)    :: in         !< Send pointer
    type(c_devptr),               intent(in)    :: out        !< Recv pointer
    integer(int32)                                :: request_counter  !< Counter
    integer(int32)                                :: mpi_ierr         !< MPI error code
    real(real32), DEVICE_PTR pointer, contiguous  :: pin(:)           !< Pointer to `in`
    real(real32), DEVICE_PTR pointer, contiguous  :: pout(:)          !< Pointer to `out`
    integer(int32)                                :: i                !< Counter

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])
    call c_f_pointer(out, pout, [self%send_recv_buffer_size])

    request_counter = 0
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    if ( .not. self%is_request_created ) then
      do i = 0, self%comm_size - 1
        if ( self%recv_floats(i) > 0 ) then
          request_counter = request_counter + 1
          call MPI_Recv_init(pout( self%recv_displs(i) ), int(self%recv_floats(i), CNT_KIND), MPI_REAL, i, 0,   &
                             self%comm, self%requests(request_counter), mpi_ierr)
        endif
      enddo

      do i = 0, self%comm_size - 1
        if ( self%send_floats(i) > 0 ) then
          request_counter = request_counter + 1
          call MPI_Send_init(pin( self%send_displs(i) ), int(self%send_floats(i), CNT_KIND), MPI_REAL, i, 0,    &
                             self%comm, self%requests(request_counter), mpi_ierr)
        endif
      enddo

      self%is_request_created = .true.
      self%n_requests = request_counter
    endif

    call MPI_Startall(self%n_requests, self%requests, mpi_ierr)
#else
    do i = 0, self%comm_size - 1
      if ( self%recv_floats(i) > 0 ) then
        request_counter = request_counter + 1
        call MPI_Irecv(pout( self%recv_displs(i) ), int(self%recv_floats(i), CNT_KIND), MPI_REAL, i, 0,         &
                       self%comm, self%requests(request_counter), mpi_ierr)
      endif
    enddo

    do i = 0, self%comm_size - 1
      if ( self%send_floats(i) > 0 ) then
        request_counter = request_counter + 1
        call MPI_Isend(pin( self%send_displs(i) ), int(self%send_floats(i), CNT_KIND), MPI_REAL, i, 0,          &
                       self%comm, self%requests(request_counter), mpi_ierr)
      endif
    enddo
    self%n_requests = request_counter
#endif
  end subroutine execute
end module dtfft_backend_mpi_p2p