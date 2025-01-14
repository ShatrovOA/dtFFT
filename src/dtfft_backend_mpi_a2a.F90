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
module dtfft_backend_mpi_a2a
!! This module defines MPI GPU backend that uses MPI_Ialltoallv: `backend_mpi_a2a`
use iso_fortran_env
use cudafor
use dtfft_backend_mpi,          only: backend_mpi
use dtfft_parameters
use dtfft_pencil,               only: pencil
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_mpi_a2a

  type, extends(backend_mpi) :: backend_mpi_a2a
  !! MPI GPU backend that uses MPI_Ialltoallv
  private
  contains
    procedure         :: create_selfcopy => create  !< Procedure to create a self-copy
    procedure         :: execute_mpi => execute     !< Procedure to execute MPI operations
  end type backend_mpi_a2a

contains

  subroutine create(self, comm)
  !! Creates MPI GPU backend that uses MPI_Ialltoallv
    class(backend_mpi_a2a),   intent(inout) :: self  !< Self object of type backend_mpi_a2a
    TYPE_MPI_COMM,            intent(in)    :: comm  !< MPI communicator
    self%n_requests = 1
    call self%create_mpi(comm, self%n_requests)
    self%send_displs = self%send_displs - 1
    self%recv_displs = self%recv_displs - 1
  end subroutine create

  subroutine execute(self, in, out)
  !! Executes MPI operations
    class(backend_mpi_a2a),     intent(inout)   :: self  !< Self object of type backend_mpi_a2a
    type(c_devptr),             intent(in)      :: in    !< Send pointer
    type(c_devptr),             intent(in)      :: out   !< Recv pointer
    integer(int32)                                :: mpi_ierr  !< MPI error code
    real(real32), DEVICE_PTR pointer, contiguous  :: pin(:)  !< Pointer to send buffer
    real(real32), DEVICE_PTR pointer, contiguous  :: pout(:) !< Pointer to receive buffer

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])
    call c_f_pointer(out, pout, [self%send_recv_buffer_size])

#if defined(DTFFT_ENABLE_PERSISTENT_COMM) && defined(DTFFT_HAVE_PERSISTENT_COLLECTIVES)
    if ( .not. self%is_request_created ) then
      call MPI_Alltoallv_init(pin, int(self%send_floats, CNT_KIND), int(self%send_displs, ADDR_KIND), MPI_REAL4,  &
                              pout, int(self%recv_floats, CNT_KIND), int(self%recv_displs, ADDR_KIND), MPI_REAL4, &
                              self%comm, MPI_INFO_NULL, self%requests(1), mpi_ierr)
      self%is_request_created = .true.
    endif

    call MPI_Startall(self%n_requests, self%requests, mpi_ierr)
#else
    call MPI_Ialltoallv(pin, int(self%send_floats, CNT_KIND), int(self%send_displs, ADDR_KIND), MPI_REAL4,        &
                        pout, int(self%recv_floats, CNT_KIND), int(self%recv_displs, ADDR_KIND), MPI_REAL4,       &
                        self%comm, self%requests(1), mpi_ierr)
#endif
  end subroutine execute
end module dtfft_backend_mpi_a2a