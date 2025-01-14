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
module dtfft_backend_nccl
!! This module defines NCCL backend: `backend_nccl`
use iso_fortran_env
use cudafor
use nccl
use dtfft_abstract_gpu_backend,     only: abstract_gpu_backend
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_nccl

  type, extends(abstract_gpu_backend) :: backend_nccl
  !! NCCL backend
  private
    type(ncclComm) :: comm
  contains
    procedure         :: create_private => create_nccl        !< Creates NCCL backend
    procedure         :: execute => execute_nccl              !< Executes NCCL backend
    procedure         :: destroy_private => destroy           !< Destroys NCCL backend
  end type backend_nccl

contains

  subroutine create_nccl(self, comm)
  !! Creates NCCL backend
    class(backend_nccl),  intent(inout) :: self               !< NCCL backend
    TYPE_MPI_COMM,        intent(in)    :: comm               !< MPI Communicator
    integer(int32)      :: mpi_ierr     !< MPI error code
    type(ncclUniqueId)  :: id           !< NCCL unique id

    if (self%comm_rank == 0) then
      NCCL_CALL( "ncclGetUniqueId", ncclGetUniqueId(id) )
    end if
    call MPI_Bcast(id, int(sizeof(id)), MPI_BYTE, 0, comm, mpi_ierr)
    NCCL_CALL( "ncclCommInitRank", ncclCommInitRank(self%comm, self%comm_size, id, self%comm_rank) )
  end subroutine create_nccl

  subroutine execute_nccl(self, in, out, stream)
  !! Executes NCCL backend
    class(backend_nccl),          intent(inout) :: self       !< NCCL backend
    type(c_devptr),               intent(in)    :: in         !< Send pointer
    type(c_devptr),               intent(in)    :: out        !< Recv pointer
    integer(cuda_stream_kind),    intent(in)    :: stream     !< Main execution CUDA stream
    real(real32), DEVICE_PTR pointer, contiguous   :: pin(:)  !< Pointer to `in`
    real(real32), DEVICE_PTR pointer, contiguous   :: pout(:) !< Pointer to `out`
    integer(int32)                                 :: i       !< Counter

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])
    call c_f_pointer(out, pout, [self%send_recv_buffer_size])

    NCCL_CALL( "ncclGroupStart", ncclGroupStart() )
    do i = 0, self%comm_size - 1
      if ( self%send_floats(i) > 0 ) then
        NCCL_CALL( "ncclSend", ncclSend(pin( self%send_displs(i) ), self%send_floats(i), ncclFloat, i, self%comm, stream) )
      endif
      if ( self%recv_floats(i) > 0) then
        NCCL_CALL( "ncclRecv", ncclRecv(pout( self%recv_displs(i) ), self%recv_floats(i), ncclFloat, i, self%comm, stream) )
      endif
    enddo
    NCCL_CALL( "ncclGroupEnd", ncclGroupEnd() )
  end subroutine execute_nccl

  subroutine destroy(self)
  !! Destroys NCCL backend
    class(backend_nccl),  intent(inout) :: self       !< NCCL backend

    NCCL_CALL( "ncclCommDestroy", ncclCommDestroy(self%comm) )
  end subroutine destroy
end module dtfft_backend_nccl