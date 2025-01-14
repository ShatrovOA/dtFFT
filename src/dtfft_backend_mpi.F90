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
module dtfft_backend_mpi
!!
use iso_fortran_env
use cudafor
use dtfft_abstract_gpu_backend_selfcopy
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_mpi

  type, abstract, extends(abstract_gpu_backend_selfcopy) :: backend_mpi
    TYPE_MPI_COMM                 :: comm
    integer(int32)                :: n_requests
    TYPE_MPI_REQUEST, allocatable :: requests(:)
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    logical                       :: is_request_created
#endif
  contains
    procedure,                        pass(self)            :: create_mpi
    procedure(execute_mpi_interface), pass(self), deferred  :: execute_mpi
    procedure, pass(self) :: execute_private => execute
    procedure :: destroy_selfcopy => destroy
  end type backend_mpi

  interface
    subroutine execute_mpi_interface(self, in, out)
    import
      class(backend_mpi),                           intent(inout) :: self
      type(c_devptr),                               intent(in)    :: in    !< Actual `in` pointer
      type(c_devptr),                               intent(in)    :: out   !< Actual `out` pointer
    end subroutine execute_mpi_interface
  endinterface

contains

  subroutine create_mpi(self, comm, max_requests)
    class(backend_mpi),   intent(inout) :: self
    TYPE_MPI_COMM,        intent(in)    :: comm
    integer(int32),       intent(in)    :: max_requests
    integer(int32) :: mpi_ierr

    call MPI_Comm_dup(comm, self%comm, mpi_ierr)
    allocate( self%requests(max_requests) )

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    self%is_request_created = .false.
#endif
  end subroutine create_mpi

  subroutine execute(self, in, out, stream)
    class(backend_mpi),           intent(inout) :: self
    type(c_devptr),               intent(in)    :: in         !< Send pointer
    type(c_devptr),               intent(in)    :: out        !< Recv pointer
    integer(cuda_stream_kind),    intent(in)    :: stream     !< Main execution CUDA stream
    integer(int32)                              :: mpi_ierr

    ! Need to sync stream since there is no way pass current stream to MPI
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
    call self%execute_mpi(in, out)
    call MPI_Waitall(self%n_requests, self%requests, MPI_STATUSES_IGNORE, mpi_ierr)
  end subroutine execute

  subroutine destroy(self)
    class(backend_mpi),  intent(inout) :: self
    integer(int32) :: i, mpi_ierr

    call MPI_Comm_free(self%comm, mpi_ierr)
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    do i = 1, self%n_requests
      call MPI_Request_free(self%requests(i), mpi_ierr)
    enddo
    self%is_request_created = .false.
#endif
    self%n_requests = 0
    deallocate( self%requests )
  end subroutine destroy
end module dtfft_backend_mpi