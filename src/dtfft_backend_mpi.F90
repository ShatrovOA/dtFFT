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
use dtfft_abstract_backend_selfcopy
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_mpi

  type, abstract, extends(abstract_backend_selfcopy) :: backend_mpi
    integer(int32)                :: n_requests
    TYPE_MPI_REQUEST, allocatable :: requests(:)
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    logical                       :: is_request_created
#endif
  contains
    procedure,                        pass(self)            :: create_requests
    procedure(executeMPIInterface),   pass(self), deferred  :: execute_mpi
    procedure, pass(self) :: execute_private => execute
    procedure :: destroy_selfcopy => destroy
  end type backend_mpi

  interface
    subroutine executeMPIInterface(self, in, out)
    import
      class(backend_mpi),                           intent(inout) :: self
      type(c_devptr),                               intent(in)    :: in    !< Actual `in` pointer
      type(c_devptr),                               intent(in)    :: out   !< Actual `out` pointer
    end subroutine executeMPIInterface


  endinterface

contains

  subroutine create_requests(self, max_requests)
    class(backend_mpi),   intent(inout) :: self
    integer(int32),       intent(in)    :: max_requests

    allocate( self%requests(max_requests) )

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    self%is_request_created = .false.
#endif
  end subroutine create_requests

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

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    if ( self%is_request_created ) then
      do i = 1, self%n_requests
        call MPI_Request_free(self%requests(i), mpi_ierr)
      enddo
    endif
    self%is_request_created = .false.
#endif
    self%n_requests = 0
    deallocate( self%requests )
  end subroutine destroy
end module dtfft_backend_mpi