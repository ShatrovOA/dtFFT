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
module dtfft_abstract_backend
!! This module defines most Abstract GPU Backend: `abstract_backend`
use iso_c_binding
use iso_fortran_env
use cudafor
#ifdef DTFFT_WITH_CUSTOM_NCCL
use dtfft_nccl_interfaces
#else
use nccl
#endif
use dtfft_nvrtc_kernel,   only: nvrtc_kernel
use dtfft_parameters
use dtfft_utils,          only: int_to_str
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: abstract_backend, backend_helper

  type :: backend_helper
  !! Helper with nccl, mpi and nvshmem communicators
    logical                     :: is_nccl_created = .false.    !< Flag is `nccl_comm` has been created
    type(ncclComm)              :: nccl_comm                    !< NCCL communicator
    TYPE_MPI_COMM,  allocatable :: comms(:)                     !< MPI communicators
    integer(int32), allocatable :: comm_mappings(:,:)           !< Mapping of 1d comm ranks to global comm
  contains
    procedure,  pass(self) :: create => create_helper
    procedure,  pass(self) :: destroy => destroy_helper
  end type backend_helper

  type, abstract :: abstract_backend
  !! The most Abstract GPU Backend
    type(dtfft_gpu_backend_t)         :: gpu_backend
    logical                           :: is_selfcopy
    logical                           :: is_pipelined
    real(real32), DEVICE_PTR  pointer :: aux(:)                 !< Auxiliary buffer used in pipelined algorithm
    integer(int64)                    :: aux_size               !< Number of bytes required by aux buffer
    integer(int64)                    :: send_recv_buffer_size  !< Number of float elements used in ``c_f_pointer``
    TYPE_MPI_COMM                     :: comm                   !< MPI Communicator
    integer(int32),       allocatable :: comm_mapping(:)        !< Mapping of 1d comm ranks to global comm
    integer(int32)                    :: comm_size              !< Size of MPI Comm
    integer(int32)                    :: comm_rank              !< Rank in MPI Comm
    integer(int64),       allocatable :: send_displs(:)         !< Send data displacements, in float elements
    integer(int64),       allocatable :: send_floats(:)         !< Send data elements, in float elements
    integer(int64),       allocatable :: recv_displs(:)         !< Recv data displacements, in float elements
    integer(int64),       allocatable :: recv_floats(:)         !< Recv data elements, in float elements
    ! Self copy params
    type(cudaEvent)                   :: execution_event        !< Event for main execution stream
    type(cudaEvent)                   :: copy_event             !< Event for copy stream
    integer(cuda_stream_kind)         :: copy_stream            !< Stream for copy operations
    integer(int64)                    :: self_copy_elements     !< Number of elements to copy
    integer(int64)                    :: self_send_displ        !< Displacement for send buffer
    integer(int64)                    :: self_recv_displ        !< Displacement for recv buffer
    ! Pipelined params
    type(nvrtc_kernel),       pointer :: unpack_kernel          !< Kernel for unpacking data
  contains
    procedure,                                    pass(self)  :: create           !< Creates Abstract GPU Backend
    procedure,                                    pass(self)  :: execute          !< Executes GPU Backend
    procedure,                                    pass(self)  :: destroy          !< Destroys Abstract GPU Backend
    procedure,                                    pass(self)  :: get_aux_size     !< Returns number of bytes required by aux buffer
    procedure,                                    pass(self)  :: set_aux          !< Sets Auxiliary buffer
    procedure,                                    pass(self)  :: set_unpack_kernel
    procedure(createInterface),       deferred,   pass(self)  :: create_private   !< Creates overring class
    procedure(executeInterface),      deferred,   pass(self)  :: execute_private  !< Executes GPU Backend
    procedure(destroyInterface),      deferred,   pass(self)  :: destroy_private  !< Destroys overring class
  end type abstract_backend

  interface
  subroutine createInterface(self, helper)
  !! Creates overring class
  import
    class(abstract_backend),    intent(inout) :: self       !< Abstract GPU Backend
    type(backend_helper),       intent(in)    :: helper     !< Backend helper
  end subroutine createInterface

  subroutine executeInterface(self, in, out, stream)
  !! Executes GPU Backend
  import
    class(abstract_backend),    intent(inout) :: self       !< Abstract GPU Backend
    real(real32),   DEVICE_PTR  intent(inout) :: in(:)      !< Send pointer
    real(real32),   DEVICE_PTR  intent(inout) :: out(:)     !< Recv pointer
    integer(cuda_stream_kind),  intent(in)    :: stream     !< Main execution CUDA stream
  end subroutine executeInterface

  subroutine destroyInterface(self)
  !! Destroys overring class
  import
    class(abstract_backend),    intent(inout) :: self       !< Abstract GPU Backend
  end subroutine destroyInterface
end interface

contains

  subroutine create(self, gpu_backend, helper, comm_id, send_displs, send_counts, recv_displs, recv_counts, base_storage)
  !! Creates Abstract GPU Backend
    class(abstract_backend),    intent(inout) :: self           !< Abstract GPU Backend
    type(dtfft_gpu_backend_t),  intent(in)    :: gpu_backend
    type(backend_helper),       intent(in)    :: helper         !< Backend helper
    integer(int8),              intent(in)    :: comm_id        !< Id of communicator to use
    integer(int32),             intent(in)    :: send_displs(:) !< Send data displacements, in original elements
    integer(int32),             intent(in)    :: send_counts(:) !< Send data elements, in float elements
    integer(int32),             intent(in)    :: recv_displs(:) !< Recv data displacements, in float elements
    integer(int32),             intent(in)    :: recv_counts(:) !< Recv data elements, in float elements
    integer(int8),              intent(in)    :: base_storage   !< Number of bytes to store single element
    integer(int64)                            :: send_size      !< Total number of floats to send
    integer(int64)                            :: recv_size      !< Total number of floats to recv
    integer(int32)                            :: ierr           !< MPI Error code
    integer(int64)                            :: scaler         !< Scaling data amount to float size

    scaler = int(base_storage, int64) / int(FLOAT_STORAGE_SIZE, int64)

    send_size = sum(send_counts) * scaler
    recv_size = sum(recv_counts) * scaler
    self%send_recv_buffer_size = max(send_size, recv_size)

    self%comm = helper%comms(comm_id)

    call MPI_Comm_size(self%comm, self%comm_size, ierr)
    call MPI_Comm_rank(self%comm, self%comm_rank, ierr)

    if ( allocated(helper%comm_mappings) ) then
      allocate( self%comm_mapping( 0:self%comm_size - 1 ), source=helper%comm_mappings(0:self%comm_size - 1, comm_id) )
    endif

    allocate( self%send_displs(0:self%comm_size - 1) )
    allocate( self%send_floats(0:self%comm_size - 1) )
    self%send_displs = int(send_displs, int64) * scaler
    self%send_displs = self%send_displs + 1
    self%send_floats = int(send_counts, int64) * scaler

    allocate( self%recv_displs(0:self%comm_size - 1) )
    allocate( self%recv_floats(0:self%comm_size - 1) )
    self%recv_displs = int(recv_displs, int64) * scaler
    self%recv_displs = self%recv_displs + 1
    self%recv_floats = int(recv_counts, int64) * scaler

    self%gpu_backend = gpu_backend
    self%is_pipelined = is_backend_pipelined(gpu_backend)
    self%is_selfcopy = self%is_pipelined .or. is_backend_mpi(gpu_backend)

    self%aux_size = 0_int64
    if ( self%is_pipelined ) then
      self%aux_size = self%send_recv_buffer_size * int(FLOAT_STORAGE_SIZE, int64)
    endif

    if ( self%is_selfcopy ) then
      self%self_send_displ = self%send_displs(self%comm_rank)
      self%self_recv_displ = self%recv_displs(self%comm_rank)
      self%self_copy_elements = self%send_floats(self%comm_rank)
      self%send_floats(self%comm_rank) = 0
      self%recv_floats(self%comm_rank) = 0

      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%execution_event, cudaEventDisableTiming) )
      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%copy_event, cudaEventDisableTiming) )
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(self%copy_stream) )
    endif

    call self%create_private(helper)
  end subroutine create

  subroutine execute(self, in, out, stream)
  !! Executes self-copying backend
    class(abstract_backend),    intent(inout) :: self     !< Self-copying backend
    real(real32),   DEVICE_PTR  intent(inout) :: in(:)    !< Send pointer
    real(real32),   DEVICE_PTR  intent(inout) :: out(:)   !< Recv pointer
    integer(cuda_stream_kind),  intent(in)    :: stream   !< CUDA stream

    if ( .not. self%is_selfcopy ) then
      call self%execute_private(in, out, stream)
      return
    endif

    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%execution_event, stream) )
    ! Waiting for transpose kernel to finish execution on stream `stream`
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(self%copy_stream, self%execution_event, 0) )

    if( self%self_copy_elements > 0 ) then
      if ( self%is_pipelined ) then
    ! Tranposed data is actually located in aux buffer for pipelined algorithm
        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(in( self%self_recv_displ ),
                                                      self%aux( self%self_send_displ ),
                                                      self%self_copy_elements,
                                                      cudaMemcpyDeviceToDevice, self%copy_stream) )
#ifdef __DEBUG
        CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
        ! Data can be unpacked in same stream as `cudaMemcpyAsync`
        call self%unpack_kernel%execute(in, out, self%copy_stream, self%comm_rank + 1)
      else
        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(out( self%self_recv_displ ),
                                                      in( self%self_send_displ ),
                                                      self%self_copy_elements,
                                                      cudaMemcpyDeviceToDevice, self%copy_stream) )
      endif
    endif
#ifdef __DEBUG
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(self%copy_stream) )
#endif
    call self%execute_private(in, out, stream)
#ifndef __DEBUG
    ! Making `stream` wait for finish of `cudaMemcpyAsync`
    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%copy_event, self%copy_stream) )
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(stream, self%copy_event, 0) )
#endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys Abstract GPU Backend
    class(abstract_backend),    intent(inout) :: self     !< Abstract GPU backend

    if ( allocated( self%send_displs ) ) deallocate( self%send_displs )
    if ( allocated( self%send_floats ) ) deallocate( self%send_floats )
    if ( allocated( self%recv_displs ) ) deallocate( self%recv_displs )
    if ( allocated( self%recv_floats ) ) deallocate( self%recv_floats )
    if ( allocated( self%comm_mapping) ) deallocate( self%comm_mapping)
    self%comm = MPI_COMM_NULL
    if ( self%is_selfcopy ) then
      CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%execution_event) )
      CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%copy_event) )
      CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(self%copy_stream) )
    endif
    if ( self%is_pipelined ) then
      nullify( self%unpack_kernel )
    endif
    self%is_pipelined = .false.
    self%is_selfcopy = .false.
    call self%destroy_private()
  end subroutine destroy

  integer(int64) function get_aux_size(self)
  !! Returns number of bytes required by aux buffer
    class(abstract_backend),    intent(in)    :: self     !< Abstract GPU backend
    get_aux_size = self%aux_size
  end function get_aux_size

  subroutine set_aux(self, aux)
  !! Sets aux buffer that can be used by various implementations
    class(abstract_backend),          intent(inout) :: self     !< Abstract GPU backend
    real(real32), DEVICE_PTR  target, intent(in)    :: aux(:)   !< Aux pointer
    self%aux => aux
  end subroutine set_aux

  subroutine set_unpack_kernel(self, unpack_kernel)
  !! Sets unpack kernel for pipelined backend
    class(abstract_backend),    intent(inout)   :: self           !< Pipelined backend
    type(nvrtc_kernel), target, intent(in)      :: unpack_kernel  !< Kernel for unpacking data

    self%unpack_kernel => unpack_kernel
  end subroutine set_unpack_kernel

  subroutine create_helper(self, base_comm, comms, is_nccl_needed)
    class(backend_helper),  intent(inout) :: self                 !< Backend helper
    TYPE_MPI_COMM,          intent(in)    :: base_comm            !< MPI communicator
    TYPE_MPI_COMM,          intent(in)    :: comms(:)             !< 1D Communicators
    logical,                intent(in)    :: is_nccl_needed       !< If nccl communicator will be needed
    integer :: i, n_comms, max_size, comm_size, comm_rank, ierr
    type(ncclUniqueId)  :: id           !< NCCL unique id

    call self%destroy()

    n_comms = size(comms)
    allocate( self%comms(n_comms) )

    self%comms(1) = base_comm
    do i = 2, n_comms
      self%comms(i) = comms(i)
    enddo
    self%is_nccl_created = .false.
    if ( .not.is_nccl_needed ) return

    max_size = -1
    do i = 1, n_comms
      call MPI_Comm_size(self%comms(i), comm_size, ierr)
      max_size = max(max_size, comm_size)
    enddo
    call MPI_Comm_rank(base_comm, comm_rank, ierr)

    allocate( self%comm_mappings(0:max_size - 1, n_comms), source=-1 )
    do i = 1, n_comms
      call MPI_Allgather(comm_rank, 1, MPI_INTEGER, self%comm_mappings(:, i), 1, MPI_INTEGER, self%comms(i), ierr)
    enddo

    if (comm_rank == 0) then
      NCCL_CALL( "ncclGetUniqueId", ncclGetUniqueId(id) )
    end if
    call MPI_Bcast(id, int(sizeof(id)), MPI_BYTE, 0, base_comm, ierr)
    NCCL_CALL( "ncclCommInitRank", ncclCommInitRank(self%nccl_comm, max_size, id, comm_rank) )
    self%is_nccl_created = .true.
  end subroutine create_helper

  subroutine destroy_helper(self)
    class(backend_helper),  intent(inout) :: self                 !< Backend helper

    if ( allocated( self%comms ) )          deallocate(self%comms)
    if ( allocated( self%comm_mappings ) )  deallocate(self%comm_mappings)
    if ( self%is_nccl_created ) then
      NCCL_CALL( "ncclCommDestroy", ncclCommDestroy(self%nccl_comm) )
    endif
    self%is_nccl_created = .false.
  end subroutine destroy_helper

end module dtfft_abstract_backend