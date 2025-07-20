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
!! This module describes Abstraction for all GPU Backends: [[abstract_backend]]
use iso_c_binding
use iso_fortran_env
use dtfft_interface_cuda_runtime
#ifdef DTFFT_WITH_NCCL
use dtfft_interface_nccl
#endif
use dtfft_nvrtc_kernel,   only: nvrtc_kernel
use dtfft_parameters
use dtfft_pencil,         only: pencil
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: abstract_backend, backend_helper

#ifdef NCCL_HAVE_COMMREGISTER
  integer(int32), parameter, public :: NCCL_REGISTER_PREALLOC_SIZE = 8
#endif

  type :: backend_helper
  !! Helper with nccl, mpi and nvshmem communicators
    logical                     :: is_nccl_created = .false.    !! Flag is `nccl_comm` has been created
#ifdef DTFFT_WITH_NCCL
    type(ncclComm)              :: nccl_comm                    !! NCCL communicator
#endif
#ifdef NCCL_HAVE_COMMREGISTER
    logical                     :: should_register              !! If NCCL buffer should be registered
    type(c_ptr),    allocatable :: nccl_register(:,:)           !! NCCL register cache
    integer(int32)              :: nccl_register_size           !! Number of elements in `nccl_register`
#endif
    TYPE_MPI_COMM,  allocatable :: comms(:)                     !! MPI communicators
    integer(int32), allocatable :: comm_mappings(:,:)           !! Mapping of 1d comm ranks to global comm
    type(dtfft_transpose_t)     :: tranpose_type                !! Type of transpose to create
    type(pencil),   pointer     :: pencils(:)                   !! Pencils
  contains
    procedure,  pass(self) :: create => create_helper           !! Creates helper
    procedure,  pass(self) :: destroy => destroy_helper         !! Destroys helper
  end type backend_helper

  type, abstract :: abstract_backend
  !! The most Abstract GPU Backend
    type(dtfft_backend_t)             :: backend                !! Backend type
    logical                           :: is_selfcopy            !! If backend is self-copying
    logical                           :: is_pipelined           !! If backend is pipelined
    integer(int64)                    :: aux_size               !! Number of bytes required by aux buffer
    integer(int64)                    :: send_recv_buffer_size  !! Number of float elements used in ``c_f_pointer``
    TYPE_MPI_COMM                     :: comm                   !! MPI Communicator
    integer(int32),       allocatable :: comm_mapping(:)        !! Mapping of 1d comm ranks to global comm
    integer(int32)                    :: comm_size              !! Size of MPI Comm
    integer(int32)                    :: comm_rank              !! Rank in MPI Comm
    integer(int64),       allocatable :: send_displs(:)         !! Send data displacements, in float elements
    integer(int64),       allocatable :: send_floats(:)         !! Send data elements, in float elements
    integer(int64),       allocatable :: recv_displs(:)         !! Recv data displacements, in float elements
    integer(int64),       allocatable :: recv_floats(:)         !! Recv data elements, in float elements
    ! Self copy params
    type(cudaEvent)                   :: execution_event        !! Event for main execution stream
    type(cudaEvent)                   :: copy_event             !! Event for copy stream
    type(dtfft_stream_t)              :: copy_stream            !! Stream for copy operations
    integer(int64)                    :: self_copy_bytes        !! Number of bytes to copy it itself
    integer(int64)                    :: self_send_displ        !! Displacement for send buffer
    integer(int64)                    :: self_recv_displ        !! Displacement for recv buffer
    ! Pipelined params
    type(nvrtc_kernel),       pointer :: unpack_kernel          !! Kernel for unpacking data
    type(nvrtc_kernel),       pointer :: unpack_kernel2         !! Kernel for unpacking data
  contains
    procedure,            non_overridable,  pass(self)  :: create           !! Creates Abstract GPU Backend
    procedure,            non_overridable,  pass(self)  :: execute          !! Executes GPU Backend
    procedure,            non_overridable,  pass(self)  :: destroy          !! Destroys Abstract GPU Backend
    procedure,            non_overridable,  pass(self)  :: get_aux_size     !! Returns number of bytes required by aux buffer
    procedure,            non_overridable,  pass(self)  :: set_unpack_kernel!! Sets unpack kernel for pipelined backend
    procedure(create_interface),  deferred, pass(self)  :: create_private   !! Creates overring class
    procedure(execute_interface), deferred, pass(self)  :: execute_private  !! Executes GPU Backend
    procedure(destroy_interface), deferred, pass(self)  :: destroy_private  !! Destroys overring class
  end type abstract_backend

  abstract interface
    subroutine create_interface(self, helper, tranpose_type, base_storage)
    !! Creates overring class
    import
      class(abstract_backend),  intent(inout) :: self           !! Abstract GPU Backend
      type(backend_helper),     intent(in)    :: helper         !! Backend helper
      type(dtfft_transpose_t),  intent(in)    :: tranpose_type  !! Type of transpose to create
      integer(int64),           intent(in)    :: base_storage   !! Number of bytes to store single element
    end subroutine create_interface

    subroutine execute_interface(self, in, out, stream, aux)
    !! Executes GPU Backend
    import
      class(abstract_backend),  intent(inout) :: self       !! Abstract GPU Backend
      real(real32),   target,   intent(inout) :: in(:)      !! Send pointer
      real(real32),   target,   intent(inout) :: out(:)     !! Recv pointer
      type(dtfft_stream_t),     intent(in)    :: stream     !! Main execution CUDA stream
      real(real32),   target,   intent(inout) :: aux(:)     !! Aux pointer
    end subroutine execute_interface

    subroutine destroy_interface(self)
    !! Destroys overring class
    import
      class(abstract_backend),    intent(inout) :: self       !! Abstract GPU Backend
    end subroutine destroy_interface
  end interface

contains

  subroutine create(self, backend, tranpose_type, helper, comm_id, send_displs, send_counts, recv_displs, recv_counts, base_storage)
  !! Creates Abstract GPU Backend
    class(abstract_backend),      intent(inout) :: self           !! Abstract GPU Backend
    type(dtfft_backend_t),        intent(in)    :: backend        !! GPU Backend type
    type(dtfft_transpose_t),      intent(in)    :: tranpose_type  !! Type of transpose to create
    type(backend_helper),         intent(in)    :: helper         !! Backend helper
    integer(int8),                intent(in)    :: comm_id        !! Id of communicator to use
    integer(int32),               intent(in)    :: send_displs(:) !! Send data displacements, in original elements
    integer(int32),               intent(in)    :: send_counts(:) !! Send data elements, in float elements
    integer(int32),               intent(in)    :: recv_displs(:) !! Recv data displacements, in float elements
    integer(int32),               intent(in)    :: recv_counts(:) !! Recv data elements, in float elements
    integer(int64),               intent(in)    :: base_storage   !! Number of bytes to store single element
    integer(int64)                            :: send_size      !! Total number of floats to send
    integer(int64)                            :: recv_size      !! Total number of floats to recv
    integer(int32)                            :: ierr           !! MPI Error code
    integer(int64)                            :: scaler         !! Scaling data amount to float size

    scaler = base_storage / FLOAT_STORAGE_SIZE

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
    self%send_displs(:) = int(send_displs(:), int64) * scaler
    self%send_displs(:) = self%send_displs(:) + 1
    self%send_floats(:) = int(send_counts(:), int64) * scaler

    allocate( self%recv_displs(0:self%comm_size - 1) )
    allocate( self%recv_floats(0:self%comm_size - 1) )
    self%recv_displs(:) = int(recv_displs(:), int64) * scaler
    self%recv_displs(:) = self%recv_displs(:) + 1
    self%recv_floats(:) = int(recv_counts(:), int64) * scaler

    self%backend = backend
    self%is_pipelined = is_backend_pipelined(backend)
    self%is_selfcopy = self%is_pipelined .or. is_backend_mpi(backend)

    self%aux_size = 0_int64
    if ( self%is_pipelined ) then
      self%aux_size = self%send_recv_buffer_size * FLOAT_STORAGE_SIZE
    endif

    if ( self%is_selfcopy ) then
      self%self_send_displ = self%send_displs(self%comm_rank)
      self%self_recv_displ = self%recv_displs(self%comm_rank)
      self%self_copy_bytes = self%send_floats(self%comm_rank) * FLOAT_STORAGE_SIZE
      self%send_floats(self%comm_rank) = 0
      self%recv_floats(self%comm_rank) = 0

      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%execution_event, cudaEventDisableTiming) )
      CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%copy_event, cudaEventDisableTiming) )
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(self%copy_stream) )
    endif

    call self%create_private(helper, tranpose_type, base_storage)
  end subroutine create

  subroutine execute(self, in, out, stream, aux)
  !! Executes GPU Backend
    class(abstract_backend),    intent(inout) :: self     !! Self-copying backend
    real(real32),               intent(inout) :: in(:)    !! Send pointer
    real(real32),               intent(inout) :: out(:)   !! Recv pointer
    type(dtfft_stream_t),       intent(in)    :: stream   !! CUDA stream
    real(real32),               intent(inout) :: aux(:)   !! Aux pointer

    if ( .not. self%is_selfcopy ) then
      call self%execute_private(in, out, stream, aux)
      return
    endif

    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%execution_event, stream) )
    ! Waiting for transpose kernel to finish execution on stream `stream`
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(self%copy_stream, self%execution_event, 0) )

    if( self%self_copy_bytes > 0 ) then
      if ( self%is_pipelined ) then
        ! Tranposed data is actually located in aux buffer for pipelined algorithm
        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(aux( self%self_recv_displ ), in( self%self_send_displ ), self%self_copy_bytes, cudaMemcpyDeviceToDevice, self%copy_stream) )
        ! Data can be unpacked in same stream as `cudaMemcpyAsync`
        call self%unpack_kernel%execute(aux, out, self%copy_stream, self%comm_rank + 1)
      else
        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(out( self%self_recv_displ ), in( self%self_send_displ ), self%self_copy_bytes, cudaMemcpyDeviceToDevice, self%copy_stream) )
      endif
    endif
    call self%execute_private(in, out, stream, aux)
    ! Making future events, like FFT, on `stream` to wait for `copy_event`
    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%copy_event, self%copy_stream) )
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(stream, self%copy_event, 0) )
  end subroutine execute

  subroutine destroy(self)
  !! Destroys Abstract GPU Backend
    class(abstract_backend),    intent(inout) :: self     !! Abstract GPU backend

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
      if ( associated(self%unpack_kernel2) ) nullify( self%unpack_kernel2 )
    endif
    self%is_pipelined = .false.
    self%is_selfcopy = .false.
    call self%destroy_private()
  end subroutine destroy

  integer(int64) function get_aux_size(self)
  !! Returns number of bytes required by aux buffer
    class(abstract_backend),    intent(in)    :: self     !! Abstract GPU backend
    get_aux_size = self%aux_size
  end function get_aux_size

  subroutine set_unpack_kernel(self, unpack_kernel, unpack_kernel2)
  !! Sets unpack kernel for pipelined backend
    class(abstract_backend),    intent(inout)             :: self           !! Pipelined backend
    type(nvrtc_kernel), target, intent(in)                :: unpack_kernel  !! Kernel for unpacking data
    type(nvrtc_kernel), target, intent(in), optional      :: unpack_kernel2  !! Kernel for unpacking data

    self%unpack_kernel => unpack_kernel
    if ( present( unpack_kernel2 ) ) self%unpack_kernel2 => unpack_kernel2
  end subroutine set_unpack_kernel

  subroutine create_helper(self, base_comm, comms, is_nccl_needed, pencils)
  !! Creates helper
    class(backend_helper),  intent(inout) :: self                 !! Backend helper
    TYPE_MPI_COMM,          intent(in)    :: base_comm            !! MPI communicator
    TYPE_MPI_COMM,          intent(in)    :: comms(:)             !! 1D Communicators
    logical,                intent(in)    :: is_nccl_needed       !! If nccl communicator will be needed
    type(pencil), target,   intent(in)    :: pencils(:)           !! Pencils
    integer :: i, n_comms

    call self%destroy()

    self%pencils => pencils(:)

    n_comms = size(comms)
    allocate( self%comms(n_comms) )

    self%comms(1) = base_comm
    do i = 2, n_comms
      self%comms(i) = comms(i)
    enddo
    self%is_nccl_created = .false.
    if ( .not.is_nccl_needed ) return

#ifdef DTFFT_WITH_NCCL
    block
      type(ncclUniqueId)  :: id           ! NCCL unique id
      integer(int32) :: max_size, comm_size, comm_rank, ierr

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
      call MPI_Bcast(id, int(c_sizeof(id)), MPI_BYTE, 0, base_comm, ierr)
      NCCL_CALL( "ncclCommInitRank", ncclCommInitRank(self%nccl_comm, max_size, id, comm_rank) )
      self%is_nccl_created = .true.
    endblock

# ifdef NCCL_HAVE_COMMREGISTER
    self%should_register = get_env("NCCL_BUFFER_REGISTER", .true.)
    if ( self%should_register ) then
      self%nccl_register_size = 0
      allocate( self%nccl_register(2, NCCL_REGISTER_PREALLOC_SIZE) )
      do i = 1, NCCL_REGISTER_PREALLOC_SIZE
        self%nccl_register(1, i) = c_null_ptr
        self%nccl_register(2, i) = c_null_ptr
      enddo
    endif
# endif
#endif
  end subroutine create_helper

  subroutine destroy_helper(self)
  !! Destroys helper
    class(backend_helper),  intent(inout) :: self                 !! Backend helper

    if ( allocated( self%comms ) )          deallocate(self%comms)
    if ( allocated( self%comm_mappings ) )  deallocate(self%comm_mappings)
    nullify( self%pencils )
#ifdef DTFFT_WITH_NCCL
    if ( self%is_nccl_created ) then
      NCCL_CALL( "ncclCommDestroy", ncclCommDestroy(self%nccl_comm) )
    endif
    self%is_nccl_created = .false.
#endif
#ifdef NCCL_HAVE_COMMREGISTER
    if ( self%nccl_register_size > 0 ) then
      WRITE_ERROR("NCCL register is not empty")
    endif
    if ( allocated( self%nccl_register ) ) deallocate(self%nccl_register)
    self%nccl_register_size = 0
#endif
  end subroutine destroy_helper

end module dtfft_abstract_backend