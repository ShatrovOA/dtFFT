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
module dtfft_backend_nccl_m
!! NCCL Based GPU Backends [[backend_nccl]]
use iso_fortran_env
use iso_c_binding, only: c_ptr, c_f_pointer
use dtfft_interface_cuda_runtime
use dtfft_interface_nccl
use dtfft_abstract_backend,         only: abstract_backend, backend_helper
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: backend_nccl

  type, extends(abstract_backend) :: backend_nccl
  !! NCCL backend
  private
    type(ncclComm)                :: nccl_comm
      !! NCCL Communicator
  contains
    procedure         :: create_private => create_nccl        !! Creates NCCL backend
    procedure         :: execute_private => execute_nccl      !! Executes NCCL backend
    procedure         :: destroy_private => destroy_nccl      !! Destroys NCCL backend
  end type backend_nccl

contains

  subroutine create_nccl(self, helper, tranpose_type, base_storage)
  !! Creates NCCL backend
    class(backend_nccl),      intent(inout) :: self               !! NCCL backend
    type(backend_helper),     intent(in)    :: helper             !! Backend helper
    type(dtfft_transpose_t),  intent(in)    :: tranpose_type      !! Type of transpose to create (unused)
    integer(int64),           intent(in)    :: base_storage       !! Number of bytes to store single element (unused)

    if ( .not. is_backend_nccl(self%backend) ) INTERNAL_ERROR(".not. is_backend_nccl")
    if ( .not. helper%is_nccl_created ) INTERNAL_ERROR(".not. helper%is_nccl_created")
    self%nccl_comm = helper%nccl_comm
  end subroutine create_nccl

  subroutine execute_nccl(self, in, out, stream, aux)
  !! Executes NCCL backend
    class(backend_nccl),          intent(inout) :: self       !! NCCL backend
    real(real32),   target,       intent(inout) :: in(:)      !! Send pointer
    real(real32),   target,       intent(inout) :: out(:)     !! Recv pointer
    type(dtfft_stream_t),         intent(in)    :: stream     !! Main execution CUDA stream
    real(real32),   target,       intent(inout) :: aux(:)     !! Auxiliary pointer
    integer(int32)                              :: i        !! Counter
    integer(int32)                              :: rnk      !! Rank to send-recv
    real(real32), pointer :: pin(:), pout(:)

    if ( self%is_pipelined ) then
      pin => in(:)
      pout => aux(:)
    else
      pin => in(:)
      pout => out(:)
    endif

    NCCL_CALL( "ncclGroupStart", ncclGroupStart() )
    do i = 0, self%comm_size - 1
      if ( i == self%comm_rank .and. self%is_pipelined) cycle
      rnk = self%comm_mapping(i)
      if ( self%send_floats(i) > 0 ) then
        NCCL_CALL( "ncclSend", ncclSend(pin( self%send_displs(i) ), self%send_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
      endif
      if ( self%recv_floats(i) > 0) then
        NCCL_CALL( "ncclRecv", ncclRecv(pout( self%recv_displs(i) ), self%recv_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
      endif
    enddo
    NCCL_CALL( "ncclGroupEnd", ncclGroupEnd() )

    if ( self%is_pipelined ) then
      call self%unpack_kernel2%execute(pout, out, stream)
    endif
  end subroutine execute_nccl

  subroutine destroy_nccl(self)
  !! Destroys NCCL backend
    class(backend_nccl),  intent(inout) :: self       !! NCCL backend

  end subroutine destroy_nccl
end module dtfft_backend_nccl_m