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
module dtfft_abstract_backend
!! This module describes Abstraction for all Backends: [[abstract_backend]]
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel, only: abstract_kernel
use dtfft_config, only: get_env
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
#ifdef DTFFT_WITH_NCCL
use dtfft_interface_nccl
#endif
use dtfft_parameters
use dtfft_pencil, only: pencil
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: abstract_backend, backend_helper

#ifdef NCCL_HAVE_COMMREGISTER
integer(int32), parameter, public :: NCCL_REGISTER_PREALLOC_SIZE = 8
!! Number of register elements to preallocate
#endif

type :: backend_helper
!! Helper with nccl, mpi and nvshmem communicators
    logical                     :: is_nccl_created = .false.   !! Flag is `nccl_comm` has been created
#ifdef DTFFT_WITH_NCCL
    type(ncclComm)              :: nccl_comm                   !! NCCL communicator
#endif
#ifdef NCCL_HAVE_COMMREGISTER
    logical                     :: should_register             !! If NCCL buffer should be registered
    type(c_ptr),    allocatable :: nccl_register(:, :)         !! NCCL register cache
    integer(int32)              :: nccl_register_size          !! Number of elements in `nccl_register`
#endif
    TYPE_MPI_COMM,  allocatable :: comms(:)                    !! MPI communicators
    integer(int32), allocatable :: comm_mappings(:, :)         !! Mapping of 1d comm ranks to global comm
    type(dtfft_transpose_t)     :: transpose_type              !! Type of transpose to create
    type(dtfft_reshape_t)       :: reshape_type                !! Type of reshape to create
    type(pencil),   pointer     :: pencils(:)                  !! Pencils
contains
    procedure, pass(self) :: create => create_helper           !! Creates helper
    procedure, pass(self) :: destroy => destroy_helper         !! Destroys helper
end type backend_helper

type, abstract :: abstract_backend
!! The most Abstract Backend
    type(dtfft_backend_t)           :: backend                  !! Backend type
    type(dtfft_platform_t)          :: platform                 !! Platform to use
    logical                         :: is_selfcopy              !! If backend is self-copying
    logical                         :: is_pipelined             !! If backend is pipelined
    logical                         :: is_even                  !! If all processes send/recv same amount of memory
    integer(int64)                  :: aux_bytes                !! Number of bytes required by aux buffer
    TYPE_MPI_COMM                   :: comm                     !! MPI Communicator
    integer(int32),     allocatable :: comm_mapping(:)          !! Mapping of 1d comm ranks to global comm
    integer(int32)                  :: comm_size                !! Size of MPI Comm
    integer(int32)                  :: comm_rank                !! Rank in MPI Comm
    integer(int64),     allocatable :: send_displs(:)           !! Send data displacements, in float elements
    integer(int64),     allocatable :: send_floats(:)           !! Send data elements, in float elements
    integer(int64),     allocatable :: recv_displs(:)           !! Recv data displacements, in float elements
    integer(int64),     allocatable :: recv_floats(:)           !! Recv data elements, in float elements
    ! Self copy params
#ifdef DTFFT_WITH_CUDA
    type(cudaEvent)                 :: execution_event          !! Event for main execution stream
    type(cudaEvent)                 :: copy_event               !! Event for copy stream
#endif
    type(dtfft_stream_t)            :: copy_stream              !! Stream for copy operations
    integer(int64)                  :: self_copy_bytes          !! Number of bytes to copy it itself
    integer(int64)                  :: self_send_displ          !! Displacement for send buffer
    integer(int64)                  :: self_recv_displ          !! Displacement for recv buffer
    class(abstract_kernel), pointer :: unpack_kernel            !! Kernel for unpacking data
contains
    procedure,  non_overridable,                pass(self) :: create           !! Creates Abstract Backend
    procedure,  non_overridable,                pass(self) :: execute          !! Executes Backend
    procedure,  non_overridable,                pass(self) :: destroy          !! Destroys Abstract Backend
    procedure,  non_overridable,                pass(self) :: get_aux_bytes     !! Returns number of bytes required by aux buffer
    procedure,  non_overridable,                pass(self) :: set_unpack_kernel!! Sets unpack kernel for pipelined backend
    procedure,                                  pass(self) :: execute_end      !! Ends execution of Backend
    procedure,                                  pass(self) :: execute_self_copy
    procedure,                                  pass(self) :: get_async_active !! Returns if async execution is active
    procedure(create_interface),    deferred,   pass(self) :: create_private   !! Creates overriding class
    procedure(execute_interface),   deferred,   pass(self) :: execute_private  !! Executes Backend
    procedure(destroy_interface),   deferred,   pass(self) :: destroy_private  !! Destroys overriding class
end type abstract_backend

abstract interface
    subroutine create_interface(self, helper, base_storage)
    !! Creates overriding class
        import
        class(abstract_backend),    intent(inout)   :: self           !! Abstract Backend
        type(backend_helper),       intent(in)      :: helper         !! Backend helper
        integer(int64),             intent(in)      :: base_storage   !! Number of bytes to store single element
    end subroutine create_interface

    subroutine execute_interface(self, in, out, stream, aux, error_code)
    !! Executes Backend
        import
        class(abstract_backend),    intent(inout)   :: self       !! Abstract Backend
        real(real32),   target,     intent(inout)   :: in(:)      !! Send pointer
        real(real32),   target,     intent(inout)   :: out(:)     !! Recv pointer
        type(dtfft_stream_t),       intent(in)      :: stream     !! Main execution CUDA stream
        real(real32),   target,     intent(inout)   :: aux(:)     !! Aux pointer
        integer(int32),             intent(out)     :: error_code !! Error code
    end subroutine execute_interface

    subroutine destroy_interface(self)
    !! Destroys overriding class
        import
        class(abstract_backend), intent(inout) :: self       !! Abstract Backend
    end subroutine destroy_interface
end interface

contains

subroutine create(self, backend, helper, platform, comm_id, send_displs, send_counts, recv_displs, recv_counts, base_storage)
!! Creates Abstract Backend
class(abstract_backend),    intent(inout)   :: self           !! Abstract Backend
type(dtfft_backend_t),      intent(in)      :: backend        !! Backend type
type(backend_helper),       intent(in)      :: helper         !! Backend helper
type(dtfft_platform_t),     intent(in)      :: platform       !! Platform to use
integer(int8),              intent(in)      :: comm_id        !! Id of communicator to use
integer(int32),             intent(in)      :: send_displs(:) !! Send data displacements, in original elements
integer(int32),             intent(in)      :: send_counts(:) !! Send data elements, in float elements
integer(int32),             intent(in)      :: recv_displs(:) !! Recv data displacements, in float elements
integer(int32),             intent(in)      :: recv_counts(:) !! Recv data elements, in float elements
integer(int64),             intent(in)      :: base_storage   !! Number of bytes to store single element
integer(int64) :: send_size      !! Total number of floats to send
integer(int64) :: recv_size      !! Total number of floats to recv
integer(int32) :: ierr           !! MPI Error code
integer(int64) :: scaler         !! Scaling data amount to float size

    scaler = base_storage / FLOAT_STORAGE_SIZE

    self%platform = platform

    self%comm = helper%comms(comm_id)

    call MPI_Comm_size(self%comm, self%comm_size, ierr)
    call MPI_Comm_rank(self%comm, self%comm_rank, ierr)

    if (allocated(helper%comm_mappings)) then
        allocate (self%comm_mapping(0:self%comm_size - 1), source=helper%comm_mappings(0:self%comm_size - 1, comm_id))
    end if

    allocate (self%send_displs(0:self%comm_size - 1))
    allocate (self%send_floats(0:self%comm_size - 1))
    self%send_displs(:) = int(send_displs(:), int64) * scaler
    self%send_displs(:) = self%send_displs(:) + 1
    self%send_floats(:) = int(send_counts(:), int64) * scaler

    allocate (self%recv_displs(0:self%comm_size - 1))
    allocate (self%recv_floats(0:self%comm_size - 1))
    self%recv_displs(:) = int(recv_displs(:), int64) * scaler
    self%recv_displs(:) = self%recv_displs(:) + 1
    self%recv_floats(:) = int(recv_counts(:), int64) * scaler

    self%backend = backend
    self%is_pipelined = is_backend_pipelined(backend)
    self%is_selfcopy = (self%is_pipelined .or. is_backend_mpi(backend)) .and. backend /= DTFFT_BACKEND_CUFFTMP_PIPELINED
    self%is_even = all(send_counts(1) == send_counts(:)) .and. all(recv_counts(1) == recv_counts(:))
    if (self%is_selfcopy .and. self%is_even .and. backend == DTFFT_BACKEND_MPI_A2A .and. platform == DTFFT_PLATFORM_HOST) then
        self%is_selfcopy = .false.
    endif

    self%aux_bytes = 0_int64
    if (self%is_pipelined) then
        send_size = sum(send_counts) * scaler
        recv_size = sum(recv_counts) * scaler
        self%aux_bytes = max(send_size, recv_size) * FLOAT_STORAGE_SIZE
    end if

    self%self_copy_bytes = 0_int64
    if (self%is_selfcopy) then
        self%self_send_displ = self%send_displs(self%comm_rank)
        self%self_recv_displ = self%recv_displs(self%comm_rank)
        self%self_copy_bytes = self%send_floats(self%comm_rank) * FLOAT_STORAGE_SIZE
        self%send_floats(self%comm_rank) = 0
        self%recv_floats(self%comm_rank) = 0
#ifdef DTFFT_WITH_CUDA
        if (platform == DTFFT_PLATFORM_CUDA) then
            CUDA_CALL( cudaEventCreateWithFlags(self%execution_event, cudaEventDisableTiming) )
            CUDA_CALL( cudaEventCreateWithFlags(self%copy_event, cudaEventDisableTiming) )
            CUDA_CALL( cudaStreamCreate(self%copy_stream) )
        end if
#endif
    end if

    call self%create_private(helper, base_storage)
end subroutine create

subroutine execute(self, in, out, stream, aux, exec_type, error_code)
!! Executes Backend
class(abstract_backend),    intent(inout)   :: self       !! Self-copying backend
real(real32),               intent(inout)   :: in(:)      !! Send pointer
real(real32),               intent(inout)   :: out(:)     !! Recv pointer
type(dtfft_stream_t),       intent(in)      :: stream     !! CUDA stream
real(real32),               intent(inout)   :: aux(:)     !! Aux pointer
type(async_exec_t),         intent(in)      :: exec_type  !! Type of async execution
integer(int32),             intent(out)     :: error_code !! Error code
integer(int32) :: dummy

    REGION_BEGIN("dtfft_backend_execute", COLOR_EXECUTE)
    if (.not. self%is_selfcopy) then
        call self%execute_private(in, out, stream, aux, error_code)
        if ( error_code /= DTFFT_SUCCESS ) return

        if ( exec_type == EXEC_BLOCKING .or. self%platform == DTFFT_PLATFORM_CUDA) then
            call self%execute_end(error_code)
        endif
#if defined(DTFFT_DEBUG) && defined(DTFFT_WITH_CUDA)
        if (self%platform == DTFFT_PLATFORM_CUDA) then
            CUDA_CALL( cudaStreamSynchronize(stream) )
        end if
#endif
        REGION_END("dtfft_backend_execute")
        return
    end if

#ifdef DTFFT_WITH_CUDA
    if ( self%self_copy_bytes > 0 .and. self%platform == DTFFT_PLATFORM_CUDA ) then
        if ( self%is_pipelined ) then
            call self%execute_self_copy(in, aux, stream)
            call self%unpack_kernel%execute(aux, out, self%copy_stream, self%comm_rank + 1)
        else
            call self%execute_self_copy(in, out, stream)
        endif
    endif
#endif

    call self%execute_private(in, out, stream, aux, error_code)
    if ( error_code /= DTFFT_SUCCESS ) return

    if ( exec_type == EXEC_BLOCKING .or. self%platform == DTFFT_PLATFORM_CUDA) then
        ! Do not want to return error in case execution is not active, like in mpi scheduled
        call self%execute_end(dummy)
    endif
#ifdef DTFFT_WITH_CUDA
    if ( self%platform == DTFFT_PLATFORM_CUDA ) then
        ! Making future events, like FFT, on `stream` to wait for `copy_event`
        CUDA_CALL( cudaEventRecord(self%copy_event, self%copy_stream) )
        CUDA_CALL( cudaStreamWaitEvent(stream, self%copy_event, 0) )
    end if
#endif
#if defined(DTFFT_DEBUG) && defined(DTFFT_WITH_CUDA)
    if ( self%platform == DTFFT_PLATFORM_CUDA ) then
        CUDA_CALL( cudaStreamSynchronize(stream) )
    end if
#endif
    REGION_END("dtfft_backend_execute")
end subroutine execute

subroutine execute_end(self, error_code)
!! Ends execution of Backend
class(abstract_backend),    intent(inout)   :: self       !! Abstract backend
integer(int32),             intent(out)     :: error_code !! Error code
    error_code = DTFFT_SUCCESS
    if (self%platform == DTFFT_PLATFORM_HOST) return
end subroutine execute_end

subroutine execute_self_copy(self, in, out, stream)
class(abstract_backend),    intent(in)      :: self       !! Abstract backend
real(real32),               intent(in)      :: in(*)      !! Send pointer
real(real32),               intent(inout)   :: out(*)     !! Recv pointer
type(dtfft_stream_t),       intent(in)      :: stream     !! CUDA stream
integer(int64) :: float_count

    if (self%self_copy_bytes == 0) return

    if (self%platform == DTFFT_PLATFORM_HOST) then
        float_count = self%self_copy_bytes / FLOAT_STORAGE_SIZE
        out(self%self_recv_displ:self%self_recv_displ + float_count - 1) = in(self%self_send_displ:self%self_send_displ + float_count - 1)
#ifdef DTFFT_WITH_CUDA
    else
        CUDA_CALL( cudaEventRecord(self%execution_event, stream) )
        ! Waiting for transpose kernel to finish execution on stream `stream`
        CUDA_CALL( cudaStreamWaitEvent(self%copy_stream, self%execution_event, 0) )

        CUDA_CALL( cudaMemcpyAsync(out( self%self_recv_displ ), in( self%self_send_displ ), self%self_copy_bytes, cudaMemcpyDeviceToDevice, self%copy_stream) )
#endif
    endif
end subroutine execute_self_copy

elemental logical function get_async_active(self)
!! Returns if async execution is active
class(abstract_backend),    intent(in)      :: self       !! Abstract backend
    get_async_active = .false.
end function get_async_active

subroutine destroy(self)
!! Destroys Abstract Backend
class(abstract_backend),    intent(inout)   :: self       !! Abstract backend

    if (allocated(self%send_displs))    deallocate (self%send_displs)
    if (allocated(self%send_floats))    deallocate (self%send_floats)
    if (allocated(self%recv_displs))    deallocate (self%recv_displs)
    if (allocated(self%recv_floats))    deallocate (self%recv_floats)
    if (allocated(self%comm_mapping))   deallocate (self%comm_mapping)
    self%comm = MPI_COMM_NULL
#ifdef DTFFT_WITH_CUDA
    if (self%is_selfcopy .and. self%platform == DTFFT_PLATFORM_CUDA) then
        CUDA_CALL( cudaEventDestroy(self%execution_event) )
        CUDA_CALL( cudaEventDestroy(self%copy_event) )
        CUDA_CALL( cudaStreamDestroy(self%copy_stream) )
    end if
#endif
    nullify (self%unpack_kernel)
    self%is_pipelined = .false.
    self%is_selfcopy = .false.
    call self%destroy_private()
end subroutine destroy

pure integer(int64) function get_aux_bytes(self)
!! Returns number of bytes required by aux buffer
class(abstract_backend),    intent(in)      :: self     !! Abstract backend
    get_aux_bytes = self%aux_bytes
end function get_aux_bytes

subroutine set_unpack_kernel(self, unpack_kernel)
!! Sets unpack kernel for pipelined backend
class(abstract_backend),        intent(inout)   :: self           !! Pipelined backend
class(abstract_kernel), target, intent(in)      :: unpack_kernel  !! Kernel for unpacking data
    self%unpack_kernel => unpack_kernel
end subroutine set_unpack_kernel

subroutine create_helper(self, platform, base_comm, comms, is_nccl_needed, pencils)
!! Creates helper
class(backend_helper),  intent(inout)   :: self                 !! Backend helper
type(dtfft_platform_t), intent(in)      :: platform             !! Platform to use
TYPE_MPI_COMM,          intent(in)      :: base_comm            !! MPI communicator
TYPE_MPI_COMM,          intent(in)      :: comms(:)             !! 1D Communicators
logical,                intent(in)      :: is_nccl_needed       !! If nccl communicator will be needed
type(pencil), target,   intent(in)      :: pencils(:)           !! Pencils
integer(int32) :: i, n_comms

    call self%destroy()

    self%pencils => pencils(:)

    n_comms = size(comms)
    allocate (self%comms(n_comms))

    self%comms(1) = base_comm
    do i = 2, n_comms
        self%comms(i) = comms(i)
    end do
    self%is_nccl_created = .false.
    if (.not. is_nccl_needed) return
    if (platform == DTFFT_PLATFORM_HOST) return

#ifdef DTFFT_WITH_NCCL
    block
        type(ncclUniqueId) :: id           ! NCCL unique id
        integer(int32) :: max_size, comm_size, comm_rank, ierr

        max_size = -1
        do i = 1, n_comms
            call MPI_Comm_size(self%comms(i), comm_size, ierr)
            max_size = max(max_size, comm_size)
        end do
        call MPI_Comm_rank(base_comm, comm_rank, ierr)

        allocate (self%comm_mappings(0:max_size - 1, n_comms), source=-1)
        do i = 1, n_comms
            call MPI_Allgather(comm_rank, 1, MPI_INTEGER, self%comm_mappings(:, i), 1, MPI_INTEGER, self%comms(i), ierr)
        end do

        if (comm_rank == 0) then
            NCCL_CALL( ncclGetUniqueId(id) )
        end if
        call MPI_Bcast(id, int(c_sizeof(id)), MPI_BYTE, 0, base_comm, ierr)
        NCCL_CALL( ncclCommInitRank(self%nccl_comm, max_size, id, comm_rank) )
        self%is_nccl_created = .true.
    end block

# ifdef NCCL_HAVE_COMMREGISTER
    self%should_register = get_env("NCCL_BUFFER_REGISTER", .true.)
    if (self%should_register) then
        self%nccl_register_size = 0
        allocate (self%nccl_register(2, NCCL_REGISTER_PREALLOC_SIZE))
        do i = 1, NCCL_REGISTER_PREALLOC_SIZE
            self%nccl_register(1, i) = c_null_ptr
            self%nccl_register(2, i) = c_null_ptr
        end do
    end if
# endif
#endif
end subroutine create_helper

subroutine destroy_helper(self)
!! Destroys helper
    class(backend_helper),  intent(inout)   :: self         !! Backend helper

    if (allocated(self%comms)) deallocate (self%comms)
    if (allocated(self%comm_mappings)) deallocate (self%comm_mappings)
    nullify (self%pencils)
#ifdef DTFFT_WITH_NCCL
    if (self%is_nccl_created) then
        NCCL_CALL( ncclCommDestroy(self%nccl_comm) )
    end if
    self%is_nccl_created = .false.
#endif
#ifdef NCCL_HAVE_COMMREGISTER
    if (self%nccl_register_size > 0) then
        WRITE_ERROR("NCCL register is not empty")
    end if
    if (allocated(self%nccl_register)) deallocate (self%nccl_register)
    self%nccl_register_size = 0
#endif
end subroutine destroy_helper

end module dtfft_abstract_backend
