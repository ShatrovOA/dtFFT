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
module dtfft_nccl_interfaces
use iso_c_binding
use cudafor
#include "dtfft_cuda.h"
implicit none

  type, bind(c) :: ncclUniqueId
    character(c_char) :: internal(128)
  end type ncclUniqueId

  type, bind(c) :: ncclComm
    type(c_ptr) :: member
  end type ncclComm

  type, bind(c) :: ncclResult
    integer(c_int) :: member
  end type ncclResult

  type, bind(c) :: ncclDataType
    integer(c_int) :: member
  end type ncclDataType

  type(ncclResult),   parameter :: ncclSuccess = ncclResult(0)
  type(ncclDataType), parameter :: ncclFloat = ncclDataType(7)


  interface operator (.NE.)
    module procedure ncclResult_ne
  end interface

  interface
    function ncclGetErrorString_c(ncclResult_t)                                     &
      result(message)                                                               &
      bind(C, name="ncclGetErrorString")
    import
    !! Returns a human-readable string corresponding to the passed error code.
      type(ncclResult),  intent(in), value  :: ncclResult_t    !< Completion status of a NCCL function.
      type(c_ptr)                           :: message         !< Pointer to message
    end function ncclGetErrorString_c

    function ncclGetUniqueId(uniqueId)                                              &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGetUniqueId")
    import
    !! Generates an Id to be used in ncclCommInitRank. 
    !! ncclGetUniqueId should be called once when creating a communicator and the Id should be 
    !! distributed to all ranks in the communicator before calling ncclCommInitRank. 
    !! uniqueId should point to a ncclUniqueId object allocated by the user.
      type(ncclUniqueId), intent(out)       :: uniqueId       !< Unique ID
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclGetUniqueId

    function ncclCommInitRank(comm, nranks, uniqueId, rank)                         &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommInitRank")
    import
    !! Creates a new communicator (multi thread/process version).
    !!
    !! rank must be between 0 and nranks-1 and unique within a communicator clique. 
    !! Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank. 
    !!
    !! ncclCommInitRank implicitly synchronizes with other ranks, hence it must be called by different 
    !! threads/processes or used within ncclGroupStart/ncclGroupEnd.
      type(ncclComm)                        :: comm           !< Communicator
      integer(c_int),               value   :: nranks         !< Number of ranks in communicator
      type(ncclUniqueId),           value   :: uniqueId       !< Unique ID
      integer(c_int),               value   :: rank           !< Calling rank
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclCommInitRank

    function ncclSend(sendbuff, count, datatype, peer, comm, stream)                &
      result(ncclResult_t)                                                          &
      bind(c, name='ncclSend')
    import
    !! Send data from sendbuff to rank peer.
    !!
    !! Rank peer needs to call ncclRecv with the same datatype and the same count as this rank.
    !!
    !! This operation is blocking for the GPU.
    !! If multiple ncclSend() and ncclRecv() operations need to progress concurrently to complete, 
    !! they must be fused within a ncclGroupStart()/ ncclGroupEnd() section.
      real(c_float),  DEVICE_PTR intent(in) :: sendbuff(*)    !< Buffer to send data from
      integer(c_size_t),             value  :: count          !< Number of elements to send
      type(ncclDataType),            value  :: datatype       !< Datatype to send
      integer(c_int),                value  :: peer           !< Target GPU
      type(ncclComm),                value  :: comm           !< Communicator
      integer(cuda_stream_kind),     value  :: stream         !< CUDA Stream
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclSend

    function ncclRecv(recvbuff, count, datatype, peer, comm, stream)                &
      result(ncclResult_t)                                                          &
      bind(c, name='ncclRecv')
    import
    !! Receive data from rank peer into recvbuff.
    !!
    !! Rank peer needs to call ncclSend with the same datatype and the same count as this rank.

    !! This operation is blocking for the GPU.
    !! If multiple ncclSend() and ncclRecv() operations need to progress concurrently to complete, 
    !! they must be fused within a ncclGroupStart()/ ncclGroupEnd() section.
      real(c_float), DEVICE_PTR intent(in)  :: recvbuff(*)    !< Buffer to recv data into
      integer(c_size_t),             value  :: count          !< Number of elements to recv
      type(ncclDataType),            value  :: datatype       !< Datatype to recv
      integer(c_int),                value  :: peer           !< Source GPU
      type(ncclComm),                value  :: comm           !< Communicator
      integer(cuda_stream_kind),     value  :: stream         !< CUDA Stream
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclRecv

    function ncclGroupStart()                                                       &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGroupStart")
    import
    !! Start a group call.
    !!
    !! All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclGroupStart

    function ncclGroupEnd()                                                         &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGroupEnd")
    import
    !! End a group call.
    !!
    !! Returns when all operations since ncclGroupStart have been processed.
    !! This means the communication primitives have been enqueued to the provided streams, 
    !! but are not necessarily complete.
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclGroupEnd

    function ncclCommDestroy(comm)                                                  &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommDestroy")
    import
    !! Destroy a communicator object comm.
      type(ncclComm),                 value :: comm           !< Communicator
      type(ncclResult)                      :: ncclResult_t   !< Completion status
    end function ncclCommDestroy

  end interface

contains
  elemental logical function ncclResult_ne(a, b)
    type(ncclResult), intent(in) :: a, b
    ncclResult_ne = (a%member .NE. b%member)
  end function ncclResult_ne

  function ncclGetErrorString(ncclResult_t) result(string)
  !! Generates an error message.
    type(ncclResult),    intent(in)   :: ncclResult_t       !< Completion status of a function.
    character(len=:),   allocatable   :: string             !< Error message
    type(c_ptr)                       :: c_string
    character(len=256), pointer       :: f_string

    c_string = ncclGetErrorString_c(ncclResult_t)
    call c_f_pointer(c_string, f_string)
    allocate( string, source=f_string(1:index(f_string, c_null_char) - 1) )
  end function ncclGetErrorString
end module dtfft_nccl_interfaces