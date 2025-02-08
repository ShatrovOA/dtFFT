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
!! This module implements NCCL backend: `backend_nccl`
use iso_fortran_env
use iso_c_binding, only: c_ptr, c_f_pointer
use cudafor
#ifdef DTFFT_WITH_CUSTOM_NCCL
use dtfft_nccl_interfaces
#else
use nccl
#endif
use dtfft_abstract_backend,         only: abstract_backend, backend_helper
use dtfft_parameters,               only: is_backend_nccl, is_backend_pipelined
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_nccl

  type, extends(abstract_backend) :: backend_nccl
  !! NCCL backend
  private
    type(ncclComm)                :: nccl_comm
    integer(cuda_stream_kind)     :: nccl_stream        !< Separate stream for NCCL operations
    !< This allows to overlap NCCL kernels with
    !< lightweiht partial unpacking kernels
    type(cudaEvent), allocatable  :: nccl_events(:)     !< Events that allow to wait for completion of NCCL operations before unpacking
    type(cudaEvent)               :: main_event         !< Event used to wait for completion of main stream operations
  contains
    procedure         :: create_private => create_nccl        !< Creates NCCL backend
    procedure         :: execute_private => execute_nccl      !< Executes NCCL backend
    procedure         :: destroy_private => destroy_nccl      !< Destroys NCCL backend
  end type backend_nccl

contains

  subroutine create_nccl(self, helper)
  !! Creates NCCL backend
    class(backend_nccl),  intent(inout) :: self               !< NCCL backend
    type(backend_helper), intent(in)    :: helper             !< Backend helper
    integer(int32)                      :: i

    if ( .not. is_backend_nccl(self%backend_id) ) error stop "dtFFT internal error: .not. is_backend_nccl"
    if ( .not. helper%is_nccl_created ) error stop "dtFFT internal error: .not. helper%is_nccl_created"
    self%nccl_comm = helper%nccl_comm

    if ( self%is_pipelined ) then
      allocate( self%nccl_events( 0:self%comm_size - 1 ) )
      do i = 0, self%comm_size - 1
        CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%nccl_events(i), cudaEventDisableTiming) )
      enddo
      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%main_event, cudaEventDisableTiming) )
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(self%nccl_stream) )
    endif
  end subroutine create_nccl

  subroutine execute_nccl(self, in, out, stream)
  !! Executes NCCL backend
    class(backend_nccl),          intent(inout) :: self       !< NCCL backend
    real(real32),   DEVICE_PTR    intent(inout) :: in(:)      !< Send pointer
    real(real32),   DEVICE_PTR    intent(inout) :: out(:)     !< Recv pointer
    integer(cuda_stream_kind),    intent(in)    :: stream     !< Main execution CUDA stream
    integer(int32)                                :: i        !< Counter
    integer(int32)                                :: rnk      !< Rank to send-recv

    if ( self%is_pipelined ) then
      CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%main_event, stream) )
      ! Waiting for transpose kernel to finish execution on stream `stream` before running on `nccl_stream`
      CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(self%nccl_stream, self%main_event, 0) )

      do i = 0, self%comm_size - 1
        if ( i == self%comm_rank ) cycle
        rnk = self%comm_mapping(i)
        NCCL_CALL( "ncclGroupStart", ncclGroupStart() )
        ! Sending from `aux` buffer to `in`
        if ( self%send_floats(i) > 0 ) then
          NCCL_CALL( "ncclSend", ncclSend(self%aux( self%send_displs(i) ), self%send_floats(i), ncclFloat, rnk, self%nccl_comm, self%nccl_stream) )
        endif
        if ( self%recv_floats(i) > 0) then
          NCCL_CALL( "ncclRecv", ncclRecv(in( self%recv_displs(i) ), self%recv_floats(i), ncclFloat, rnk, self%nccl_comm, self%nccl_stream) )
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
    else
      NCCL_CALL( "ncclGroupStart", ncclGroupStart() )
      do i = 0, self%comm_size - 1
        rnk = self%comm_mapping(i)
        if ( self%send_floats(i) > 0 ) then
          NCCL_CALL( "ncclSend", ncclSend(in( self%send_displs(i) ), self%send_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
        endif
        if ( self%recv_floats(i) > 0) then
          NCCL_CALL( "ncclRecv", ncclRecv(out( self%recv_displs(i) ), self%recv_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
        endif
      enddo
      NCCL_CALL( "ncclGroupEnd", ncclGroupEnd() )
    endif
  end subroutine execute_nccl

  subroutine destroy_nccl(self)
  !! Destroys NCCL backend
    class(backend_nccl),  intent(inout) :: self       !< NCCL backend
    integer(int32)                      :: i

    if ( self%is_pipelined ) then
      do i = 0, self%comm_size - 1
        CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%nccl_events(i)) )
      enddo
      deallocate( self%nccl_events )
      CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%main_event) )
      CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(self%nccl_stream) )
    endif
  end subroutine destroy_nccl
end module dtfft_backend_nccl