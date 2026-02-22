!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
module dtfft_interface_nccl
!! NCCL Interfaces
use iso_c_binding
use iso_fortran_env
use dtfft_parameters, only: dtfft_stream_t
use dtfft_utils
#ifdef DTFFT_WITH_MOCK_ENABLED
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#endif
implicit none
private
public :: ncclGetErrorString
public :: ncclGetUniqueId, ncclMemAlloc, ncclMemFree
public :: ncclCommInitRank, ncclSend, ncclRecv
public :: ncclGroupStart, ncclGroupEnd
public :: ncclCommDestroy, ncclCommRegister, ncclCommDeregister

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

#ifndef DTFFT_WITH_MOCK_ENABLED
! Real NCCL interfaces

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

#else
  ! Storage for MPI requests in mock mode
    TYPE_MPI_REQUEST, allocatable  :: requests(:)
    integer(int32)    :: num_requests
#endif

contains

#ifdef DTFFT_WITH_MOCK_ENABLED
  ! Mock implementations for CPU testing

  function ncclGetUniqueId(uniqueId) result(ncclResult_t)
  !! Mock: Generates dummy unique ID
    type(ncclUniqueId), intent(out)       :: uniqueId
    integer(c_int32_t)                    :: ncclResult_t
    uniqueId%internal = achar(0)
    ncclResult_t = 0  ! ncclSuccess
  end function ncclGetUniqueId

  function ncclMemAlloc(ptr, alloc_bytes) result(ncclResult_t)
  !! Mock: Allocates memory on CPU
    type(c_ptr),        intent(out)       :: ptr
    integer(c_size_t),  intent(in), value :: alloc_bytes
    integer(c_int32_t)                    :: ncclResult_t

    ptr = mem_alloc_host(alloc_bytes)
    if( is_null_ptr(ptr) ) then
      ncclResult_t = 1  ! ncclUnhandledCudaError
    else
      ncclResult_t = 0  ! ncclSuccess
    end if
  end function ncclMemAlloc

  function ncclMemFree(ptr) result(ncclResult_t)
  !! Mock: Frees memory allocated on CPU
    type(c_ptr),        intent(in)   :: ptr
    integer(c_int32_t)                  :: ncclResult_t

    if ( is_null_ptr(ptr) ) then
      ncclResult_t = 1  ! ncclUnhandledCudaError
      return
    end if
    call mem_free_host(ptr)
    ncclResult_t = 0  ! ncclSuccess
  end function ncclMemFree

  function ncclCommInitRank(comm, nranks, uniqueId, rank) result(ncclResult_t)
  !! Mock: Creates dummy communicator
    type(ncclComm)                   :: comm
    integer(c_int),       intent(in) :: nranks
    type(ncclUniqueId),   intent(in) :: uniqueId
    integer(c_int),       intent(in) :: rank
    integer(c_int32_t)               :: ncclResult_t
    comm%member = c_null_ptr
    ncclResult_t = 0  ! ncclSuccess
  end function ncclCommInitRank

  function ncclSend(sendbuff, count, datatype, peer, comm, stream) result(ncclResult_t)
  !! Mock: Uses MPI_Isend for CPU communication
    real(c_float),                intent(in) :: sendbuff
    integer(c_size_t),            intent(in) :: count
    type(ncclDataType),           intent(in) :: datatype
    integer(c_int),               intent(in) :: peer
    type(ncclComm),               intent(in) :: comm
    type(dtfft_stream_t),         intent(in) :: stream
    integer(c_int32_t)                       :: ncclResult_t
    TYPE_MPI_REQUEST                      :: mpi_request
    integer                               :: mpi_ierr
    integer                               :: mpi_count

    mpi_count = int(count, kind=int32)
    call MPI_Isend(sendbuff, mpi_count, MPI_REAL, peer, 0, MPI_COMM_WORLD, mpi_request, mpi_ierr)

    if (mpi_ierr == MPI_SUCCESS) then
      ! Store request for later completion
      num_requests = num_requests + 1
      requests(num_requests) = mpi_request
      ncclResult_t = 0  ! ncclSuccess
    else
      ncclResult_t = 1  ! ncclError
    end if
  end function ncclSend

  function ncclRecv(recvbuff, count, datatype, peer, comm, stream) result(ncclResult_t)
  !! Mock: Uses MPI_Irecv for CPU communication
    real(c_float),                intent(out) :: recvbuff
    integer(c_size_t),            intent(in)  :: count
    type(ncclDataType),           intent(in)  :: datatype
    integer(c_int),               intent(in)  :: peer
    type(ncclComm),               intent(in)  :: comm
    type(dtfft_stream_t),         intent(in)  :: stream
    integer(c_int32_t)                        :: ncclResult_t
    TYPE_MPI_REQUEST  :: mpi_request
    integer(int32)    :: mpi_ierr
    integer(int32)    :: mpi_count

    mpi_count = int(count, kind=int32)
    call MPI_Irecv(recvbuff, mpi_count, MPI_REAL, peer, 0, MPI_COMM_WORLD, mpi_request, mpi_ierr)

    if (mpi_ierr == MPI_SUCCESS) then
      ! Store request for later completion
      num_requests = num_requests + 1
      requests(num_requests) = mpi_request
      ncclResult_t = 0  ! ncclSuccess
    else
      ncclResult_t = 1  ! ncclError
    end if
  end function ncclRecv

  function ncclGroupStart() result(ncclResult_t)
  !! Mock: Resets request counter
    integer(c_int32_t)                    :: ncclResult_t
    integer(int32) :: comm_size, mpi_ierr

    if ( num_requests /= 0 ) then
      INTERNAL_ERROR("ncclGroupStart: num_requests is not zero")
    end if
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, mpi_ierr)
    allocate(requests(comm_size * 2))  ! Allocate enough space for all sends and receives
    num_requests = 0
    ncclResult_t = 0  ! ncclSuccess
  end function ncclGroupStart

  function ncclGroupEnd() result(ncclResult_t)
  !! Mock: Completes all pending MPI requests
    integer(c_int32_t)                    :: ncclResult_t
    integer(int32)                        :: i, mpi_ierr

    if ( .not. allocated(requests) ) then
      INTERNAL_ERROR("ncclGroupEnd: mpi_requests should be allocated at this point")
    end if
    if ( num_requests == 0 ) then
      INTERNAL_ERROR("ncclGroupEnd: num_requests should be greater than 0 at this point")
    end if

    ! call MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE, mpi_ierr)
    ! MPI_Waitall is somehow messing with internal memory
    ! `get_conf_fourier_reshape_enabled` returns different value after MPI_Waitall
    ! Waiting for each request individually
    do i = 1, num_requests
      call MPI_Wait(requests(i), MPI_STATUS_IGNORE, mpi_ierr)
    enddo

    if (mpi_ierr == MPI_SUCCESS) then
      ncclResult_t = 0  ! ncclSuccess
    else
      ncclResult_t = 1  ! ncclError
    end if
    deallocate(requests)
    num_requests = 0
  end function ncclGroupEnd

  function ncclCommDestroy(comm) result(ncclResult_t)
  !! Mock: Does nothing
    type(ncclComm),           intent(in) :: comm
    integer(c_int32_t)                   :: ncclResult_t
    ncclResult_t = 0  ! ncclSuccess
  end function ncclCommDestroy

  function ncclCommRegister(comm, buff, size, handle) result(ncclResult_t)
  !! Mock: Returns dummy handle
    type(ncclComm),           intent(in)  :: comm
    type(c_ptr),              intent(in)  :: buff
    integer(c_size_t),        intent(in)  :: size
    type(c_ptr),              intent(out) :: handle
    integer(c_int32_t)                    :: ncclResult_t
    handle = buff  ! Just return the same pointer
    ncclResult_t = 0  ! ncclSuccess
  end function ncclCommRegister

  function ncclCommDeregister(comm, handle) result(ncclResult_t)
  !! Mock: Does nothing
    type(ncclComm),        intent(in) :: comm
    type(c_ptr),           intent(in) :: handle
    integer(c_int32_t)                :: ncclResult_t
    ncclResult_t = 0  ! ncclSuccess
  end function ncclCommDeregister

#endif

  function ncclGetErrorString(ncclResult_t) result(string)
  !! Generates an error message.
    integer(c_int32_t), intent(in)    :: ncclResult_t       !! Completion status of a function.
    character(len=:),   allocatable   :: string             !! Error message

#ifndef DTFFT_WITH_MOCK_ENABLED
    call string_c2f(ncclGetErrorString_c(ncclResult_t), string)
#else
    if (ncclResult_t == 0) then
      allocate(string, source="ncclSuccess (mock)")
    else
      allocate(string, source="ncclError (mock)")
    end if
#endif
  end function ncclGetErrorString
end module dtfft_interface_nccl