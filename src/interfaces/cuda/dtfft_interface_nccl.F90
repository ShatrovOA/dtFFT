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
module dtfft_interface_nccl
!! NCCL Interfaces
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
use dtfft_utils,      only: string_c2f
implicit none
private
public :: ncclGetErrorString

public :: ncclUniqueId
  type, bind(c) :: ncclUniqueId
    character(c_char) :: internal(128)
  end type ncclUniqueId

public :: ncclComm
  type, bind(c) :: ncclComm
    type(c_ptr) :: member
  end type ncclComm

public :: ncclDataType
  type, bind(c) :: ncclDataType
    integer(c_int) :: member
  end type ncclDataType

  type(ncclDataType), parameter, public :: ncclFloat = ncclDataType(7)


  interface
  !! Returns a human-readable string corresponding to the passed error code.
    function ncclGetErrorString_c(ncclResult_t)                                     &
      result(message)                                                               &
      bind(C, name="ncclGetErrorString")
    import
      integer(c_int32_t), intent(in), value :: ncclResult_t    !! Completion status of a NCCL function.
      type(c_ptr)                           :: message         !! Pointer to message
    end function ncclGetErrorString_c
  endinterface

public :: ncclGetUniqueId
  interface
  !! Generates an Id to be used in ncclCommInitRank. 
  !! ncclGetUniqueId should be called once when creating a communicator and the Id should be 
  !! distributed to all ranks in the communicator before calling ncclCommInitRank. 
  !! uniqueId should point to a ncclUniqueId object allocated by the user.
    function ncclGetUniqueId(uniqueId)                                              &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGetUniqueId")
    import
      type(ncclUniqueId), intent(out)       :: uniqueId       !! Unique ID
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclGetUniqueId
  end interface
  
public :: ncclMemAlloc
  interface
  !! Allocate a GPU buffer with size.
  !! Allocated buffer head address will be returned by ptr, and the actual allocated size can be larger 
  !! than requested because of the buffer granularity requirements from all types of NCCL optimizations.
    function ncclMemAlloc(ptr, alloc_bytes)                                         &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclMemAlloc")
    import
      type(c_ptr),        intent(out)       :: ptr            !! Buffer address
      integer(c_size_t),  intent(in), value :: alloc_bytes    !! Number of bytes to allocate
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclMemAlloc
  end interface

public :: ncclMemFree
  interface
  !! Free memory allocated by ncclMemAlloc().
    function ncclMemFree(ptr)                                                       &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclMemFree")
    import
      type(c_ptr),        intent(in), value :: ptr            !! Buffer address
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclMemFree
  end interface

public :: ncclCommInitRank
  interface
  !! Creates a new communicator (multi thread/process version).
  !!
  !! rank must be between 0 and nranks-1 and unique within a communicator clique.
  !! Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank.
  !!
  !! ncclCommInitRank implicitly synchronizes with other ranks, hence it must be called by different
  !! threads/processes or used within ncclGroupStart/ncclGroupEnd.
    function ncclCommInitRank(comm, nranks, uniqueId, rank)                         &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommInitRank")
    import
      type(ncclComm)                        :: comm           !! Communicator
      integer(c_int),               value   :: nranks         !! Number of ranks in communicator
      type(ncclUniqueId),           value   :: uniqueId       !! Unique ID
      integer(c_int),               value   :: rank           !! Calling rank
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclCommInitRank
  end interface

public :: ncclSend
  interface
  !! Send data from sendbuff to rank peer.
  !!
  !! Rank peer needs to call ncclRecv with the same datatype and the same count as this rank.
  !!
  !! This operation is blocking for the GPU.
  !! If multiple ncclSend() and ncclRecv() operations need to progress concurrently to complete, 
  !! they must be fused within a ncclGroupStart()/ ncclGroupEnd() section.
    function ncclSend(sendbuff, count, datatype, peer, comm, stream)                &
      result(ncclResult_t)                                                          &
      bind(c, name='ncclSend')
    import
      real(c_float)                         :: sendbuff       !! Buffer to send data from
      integer(c_size_t),            value   :: count          !! Number of elements to send
      type(ncclDataType),           value   :: datatype       !! Datatype to send
      integer(c_int),               value   :: peer           !! Target GPU
      type(ncclComm),               value   :: comm           !! Communicator
      type(dtfft_stream_t),         value   :: stream         !! CUDA Stream
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclSend
  end interface

public :: ncclRecv
  interface
  !! Receive data from rank peer into recvbuff.
  !!
  !! Rank peer needs to call ncclSend with the same datatype and the same count as this rank.

  !! This operation is blocking for the GPU.
  !! If multiple ncclSend() and ncclRecv() operations need to progress concurrently to complete, 
  !! they must be fused within a ncclGroupStart()/ ncclGroupEnd() section.
    function ncclRecv(recvbuff, count, datatype, peer, comm, stream)                &
      result(ncclResult_t)                                                          &
      bind(c, name='ncclRecv')
    import
      real(c_float)                         :: recvbuff       !! Buffer to recv data into
      integer(c_size_t),            value   :: count          !! Number of elements to recv
      type(ncclDataType),           value   :: datatype       !! Datatype to recv
      integer(c_int),               value   :: peer           !! Source GPU
      type(ncclComm),               value   :: comm           !! Communicator
      type(dtfft_stream_t),         value   :: stream         !! CUDA Stream
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclRecv
  end interface

public :: ncclGroupStart
  interface
  !! Start a group call.
  !!
  !! All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.
    function ncclGroupStart()                                                       &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGroupStart")
    import
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclGroupStart
  end interface

public :: ncclGroupEnd
  interface
  !! End a group call.
  !!
  !! Returns when all operations since ncclGroupStart have been processed.
  !! This means the communication primitives have been enqueued to the provided streams, 
  !! but are not necessarily complete.
    function ncclGroupEnd()                                                         &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclGroupEnd")
    import
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclGroupEnd
  end interface

public :: ncclCommDestroy
  interface
  !! Destroy a communicator object comm.
    function ncclCommDestroy(comm)                                                  &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommDestroy")
    import
      type(ncclComm),                 value :: comm           !! Communicator
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclCommDestroy
  end interface

public :: ncclCommRegister
  interface
  !! Register a buffer for collective communication.
    function ncclCommRegister(comm, buff, size, handle)                             &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommRegister")
    import
      type(ncclComm),                 value :: comm           !! Communicator
      type(c_ptr),                    value :: buff           !! Buffer to register
      integer(c_size_t),              value :: size           !! Size of the buffer in bytes
      type(c_ptr)                           :: handle         !! Handle to the registered buffer
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclCommRegister
  end interface

public :: ncclCommDeregister
  interface
  !! Deregister a buffer for collective communication.
    function ncclCommDeregister(comm, handle)                                       &
      result(ncclResult_t)                                                          &
      bind(C, name="ncclCommDeregister")
    import
      type(ncclComm),                 value :: comm           !! Communicator
      type(c_ptr),                    value :: handle         !! Handle to the registered buffer
      integer(c_int32_t)                    :: ncclResult_t   !! Completion status
    end function ncclCommDeregister
  end interface

contains

  function ncclGetErrorString(ncclResult_t) result(string)
  !! Generates an error message.
    integer(c_int32_t), intent(in)    :: ncclResult_t       !! Completion status of a function.
    character(len=:),   allocatable   :: string             !! Error message

    call string_c2f(ncclGetErrorString_c(ncclResult_t), string)
  end function ncclGetErrorString
end module dtfft_interface_nccl