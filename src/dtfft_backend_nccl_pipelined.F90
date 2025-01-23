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
module dtfft_backend_nccl_pipelined
use iso_fortran_env
use cudafor
use nccl
use dtfft_abstract_backend,           only: backend_helper
use dtfft_abstract_backend_selfcopy,  only: abstract_backend_pipelined
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_nccl_pipelined

  type, extends(abstract_backend_pipelined) :: backend_nccl_pipelined
  !! DTFFT_GPU_BACKEND_NCCL_PIPELINED
  private
    integer(cuda_stream_kind)     :: nccl_stream        !< Separate stream for NCCL operations
                                                        !< This allows to overlap NCCL kernels with
                                                        !< lightweiht partial unpacking kernels
    type(ncclComm)                :: nccl_comm          !< Local NCCL communicator
    type(cudaEvent), allocatable  :: nccl_events(:)     !< Events that allow to wait for completion of NCCL operations before unpacking
    type(cudaEvent)               :: main_event         !< Event used to wait for completion of main stream operations
  contains
    procedure         :: create_selfcopy => create
    procedure         :: execute_private => execute
    procedure         :: destroy_selfcopy => destroy
  end type backend_nccl_pipelined

contains

  subroutine create(self, helper)
    class(backend_nccl_pipelined),  intent(inout) :: self               !< 
    type(backend_helper),           intent(in)    :: helper             !< Backend helper
    ! integer(int32)                                :: mpi_ierr
    integer(int32)                                :: i
    ! type(ncclUniqueId)                            :: id

    ! if (self%comm_rank == 0) then
    !   NCCL_CALL( "ncclGetUniqueId", ncclGetUniqueId(id) )
    ! end if
    ! call MPI_Bcast(id, int(sizeof(id)), MPI_BYTE, 0, comm, mpi_ierr)
    ! NCCL_CALL( "ncclCommInitRank", ncclCommInitRank(self%comm, self%comm_size, id, self%comm_rank) )
    self%nccl_comm = helper%nccl_comm
    allocate( self%nccl_events( 0:self%comm_size - 1 ) )
    do i = 0, self%comm_size - 1
      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%nccl_events(i), cudaEventDisableTiming) )
    enddo
    CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%main_event, cudaEventDisableTiming) )
    CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(self%nccl_stream) )
  end subroutine create

  subroutine execute(self, in, out, stream)
    class(backend_nccl_pipelined),  intent(inout) :: self
    type(c_devptr),                 intent(in)    :: in         !< Send pointer
    type(c_devptr),                 intent(in)    :: out        !< Recv pointer
    integer(cuda_stream_kind),      intent(in)    :: stream     !< Main execution CUDA stream
    integer(int32)                                :: i
    real(real32), DEVICE_PTR pointer, contiguous  :: pin(:)
    real(real32), DEVICE_PTR pointer, contiguous  :: paux(:)
    integer(int32)                                :: rnk        !< Rank to send-recv

    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%main_event, stream) )
    ! Waiting for transpose kernel to finish execution on stream `stream` before running on `nccl_stream`
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(self%nccl_stream, self%main_event, 0) )

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])
    call c_f_pointer(self%aux, paux, [self%send_recv_buffer_size])

    do i = 0, self%comm_size - 1
      if ( i == self%comm_rank ) cycle
      rnk = self%comm_mapping(i)
      NCCL_CALL( "ncclGroupStart", ncclGroupStart() )
      ! Sending from `aux` buffer to `in`
      if ( self%send_floats(i) > 0 ) then
        NCCL_CALL( "ncclSend", ncclSend(paux( self%send_displs(i) ), self%send_floats(i), ncclFloat, rnk, self%nccl_comm, self%nccl_stream) )
      endif
      if ( self%recv_floats(i) > 0) then
        NCCL_CALL( "ncclRecv", ncclRecv(pin( self%recv_displs(i) ), self%recv_floats(i), ncclFloat, rnk, self%nccl_comm, self%nccl_stream) )
      endif

      NCCL_CALL( "ncclGroupEnd", ncclGroupEnd() )
      CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%nccl_events(i), self%nccl_stream))
    enddo

    do i = 0, self%comm_size - 1
      if ( i == self%comm_rank ) cycle
      CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(stream, self%nccl_events(i), 0) )
      ! Upacking data on default `stream`
      call self%unpack_kernel%execute(in, out, stream, i + 1)
    enddo
  end subroutine execute

  subroutine destroy(self)
    class(backend_nccl_pipelined),  intent(inout) :: self
    integer(int32) :: i

    ! NCCL_CALL( "ncclCommDestroy", ncclCommDestroy(self%comm) )
    do i = 0, self%comm_size - 1
      CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%nccl_events(i)) )
    enddo
    deallocate( self%nccl_events )
    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%main_event) )
    CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(self%nccl_stream) )
  end subroutine destroy
end module dtfft_backend_nccl_pipelined