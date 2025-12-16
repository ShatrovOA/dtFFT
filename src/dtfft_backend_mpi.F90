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
module dtfft_backend_mpi
!! MPI Based Backends [[backend_mpi]]
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: backend_mpi

type :: mpi_backend_helper
!! MPI Helper
    integer(CNT_KIND),  allocatable :: counts(:)        !! Counts of data to send or recv
    integer(ADDR_KIND), allocatable :: displs(:)        !! Displacements of data to send or recv
    TYPE_MPI_REQUEST,   allocatable :: requests(:)      !! MPI Requests
    integer(int32),     allocatable :: process_map(:)   !! Process map for pipelined communication
    integer(int32)                  :: n_requests       !! Number of requests
contains
    procedure, pass(self) :: create => create_helper    !! Creates MPI helper
    procedure, pass(self) :: destroy => destroy_helper  !! Destroys MPI helper
end type mpi_backend_helper

type, extends(abstract_backend) :: backend_mpi
!! MPI Backend
private
    logical                     :: is_active            !! If async transpose is active
    type(mpi_backend_helper)    :: send                 !! MPI Helper for send data
    type(mpi_backend_helper)    :: recv                 !! MPI Helper for recv data
    logical                     :: is_rma               !! Using RMA backend
    TYPE_MPI_WIN                :: win                  !! MPI Window for RMA backend
    logical                     :: is_request_created   !! Request created flag. Used for persistent functions
    integer(int32), allocatable :: schedule(:)          !! Communication schedule for all-to-all
contains
    procedure :: create_private => create_mpi   !! Creates MPI backend
    procedure :: execute_private => execute_mpi !! Executes MPI backend
    procedure :: destroy_private => destroy_mpi !! Destroys MPI backend
    procedure :: execute_end => execute_end_mpi !! Finalizes async transpose
    procedure :: get_async_active               !! Overrides abstract method and returns if async transpose is active
    procedure :: execute_p2p
    procedure :: execute_a2a
    procedure :: execute_p2p_scheduled
    procedure :: compute_alltoall_schedule
#ifdef DTFFT_WITH_RMA
    procedure :: execute_rma
#endif
end type backend_mpi

contains

subroutine create_helper(self, counts, displs, max_requests)
  !! Creates MPI helper
    class(mpi_backend_helper), intent(inout) :: self         !! MPI Helper
    integer(int64), intent(in) :: counts(:)    !! Counts of data to send or recv
    integer(int64), intent(in) :: displs(:)    !! Displacements of data to send or recv
    integer(int32), intent(in) :: max_requests !! Maximum number of requests required
    integer(int32) :: i, n_counts

    n_counts = size(counts)
    allocate (self%counts(0:n_counts - 1), self%displs(0:n_counts - 1))
    self%counts(0:) = int(counts(:), CNT_KIND)
    self%displs(0:) = int(displs(:), ADDR_KIND)
    if (max_requests > 0) then
        allocate (self%requests(max_requests))
        do i = 1, max_requests
            self%requests(i) = MPI_REQUEST_NULL
        end do
    end if
    self%n_requests = 0
end subroutine create_helper

subroutine destroy_helper(self, is_request_created)
!! Destroys MPI helper
class(mpi_backend_helper), intent(inout) :: self !! MPI Helper
logical, intent(in) :: is_request_created

    if (allocated(self%counts)) deallocate (self%counts)
    if (allocated(self%displs)) deallocate (self%displs)
#if defined(ENABLE_PERSISTENT_COMM) || defined(DTFFT_WITH_RMA)
    block
        integer(int32) :: mpi_ierr, i

        if (is_request_created) then
            do i = 1, self%n_requests
                if (self%requests(i) /= MPI_REQUEST_NULL) &
                    call MPI_Request_free(self%requests(i), mpi_ierr)
            end do
        end if
    end block
#endif
    if (allocated(self%requests)) deallocate (self%requests)
    if (allocated(self%process_map)) deallocate (self%process_map)
    self%n_requests = 0
end subroutine destroy_helper

subroutine create_mpi(self, helper, base_storage)
!! Creates MPI backend
class(backend_mpi),     intent(inout) :: self           !! MPI Backend
type(backend_helper),   intent(in) :: helper         !! Backend helper (unused)
integer(int64),         intent(in) :: base_storage   !! Number of bytes to store single element (unused)
integer(int32) :: mpi_err

#ifdef DTFFT_DEBUG
    if (.not. is_backend_mpi(self%backend)) INTERNAL_ERROR(".not. is_backend_mpi")
#endif
    if (self%backend == DTFFT_BACKEND_MPI_A2A) then
        call self%send%create(self%send_floats, self%send_displs - 1, 1)
        call self%recv%create(self%recv_floats, self%recv_displs - 1, 0)
    else if ( self%backend == DTFFT_BACKEND_MPI_P2P_SCHEDULED ) then
        call self%send%create(self%send_floats, self%send_displs, 0)
        call self%recv%create(self%recv_floats, self%recv_displs, 0)
        call self%compute_alltoall_schedule()
    else
        call self%send%create(self%send_floats, self%send_displs, self%comm_size)
        call self%recv%create(self%recv_floats, self%recv_displs, self%comm_size)
    end if

    self%is_rma = .false.
#ifdef DTFFT_WITH_RMA
    if (any(self%backend == [DTFFT_BACKEND_MPI_RMA, DTFFT_BACKEND_MPI_RMA_PIPELINED])) then
        block
            integer(int32), allocatable :: all_displs(:, :)

            self%is_rma = .true.
            self%win = MPI_WIN_NULL
            allocate (all_displs(0:self%comm_size - 1, 0:self%comm_size - 1))
            call MPI_Allgather(self%send%displs, self%comm_size, MPI_INTEGER, all_displs, self%comm_size, MPI_INTEGER, self%comm, mpi_err)
            self%send%displs(:) = all_displs(self%comm_rank, :) - 1
            deallocate (all_displs)
        end block
    end if
#endif
    self%is_request_created = .false.
    self%is_active = .false.
end subroutine create_mpi

subroutine destroy_mpi(self)
  !! Destroys MPI backend
    class(backend_mpi), intent(inout) :: self           !! MPI Backend
    integer(int32) :: mpi_ierr

    call self%send%destroy(self%is_request_created)
    call self%recv%destroy(self%is_request_created)
    if ( allocated(self%schedule) ) deallocate (self%schedule)
#ifdef DTFFT_WITH_RMA
    if (self%is_rma) then
        if (self%win /= MPI_WIN_NULL) &
            call MPI_Win_free(self%win, mpi_ierr)
    end if
#endif
    self%is_request_created = .false.
    self%is_active = .false.
end subroutine destroy_mpi

elemental logical function get_async_active(self)
  !! Returns if async transpose is active
    class(backend_mpi), intent(in) :: self              !!  MPI Backend
    get_async_active = self%is_active
end function get_async_active

subroutine execute_mpi(self, in, out, stream, aux, error_code)
!! Executes MPI backend
class(backend_mpi),     intent(inout)   :: self       !! MPI Backend
real(real32), target,   intent(inout)   :: in(:)      !! Send pointer
real(real32), target,   intent(inout)   :: out(:)     !! Recv pointer
type(dtfft_stream_t),   intent(in)      :: stream     !! Main execution CUDA stream
real(real32), target,   intent(inout)   :: aux(:)     !! Aux pointer
integer(int32),         intent(out)     :: error_code !! Error code
integer(int32)              :: mpi_ierr             !! MPI error code
integer(int32), allocatable :: indices(:)
integer(int32)              :: total_completed, n_completed      !! Request counter
integer(int32)              :: need_completed
integer(int32)              :: i                    !! Loop index

    error_code = DTFFT_SUCCESS
    if (self%is_active) then
        error_code = DTFFT_ERROR_TRANSPOSE_ACTIVE
        return
    end if
#ifdef DTFFT_WITH_CUDA
    if (self%platform == DTFFT_PLATFORM_CUDA) then
        ! Need to sync stream since there is no way pass current stream to MPI
        CUDA_CALL(cudaStreamSynchronize(stream))
    end if
#endif

    if ( .not. self%is_pipelined ) then
        if (self%backend == DTFFT_BACKEND_MPI_A2A) then
            call self%execute_a2a(in, out)
        else if (self%backend == DTFFT_BACKEND_MPI_P2P) then
            call self%execute_p2p(in, out)
        else if (self%backend == DTFFT_BACKEND_MPI_P2P_SCHEDULED) then
            call self%execute_p2p_scheduled(in, out)
#ifdef DTFFT_WITH_RMA
        else
            call self%execute_rma(in, out)
#endif
        endif
        if ( self%platform == DTFFT_PLATFORM_HOST .and. self%backend /= DTFFT_BACKEND_MPI_P2P_SCHEDULED ) then
            call self%execute_self_copy(in, out, stream)
        endif
        self%is_active = .true.
        if ( self%backend == DTFFT_BACKEND_MPI_P2P_SCHEDULED ) self%is_active = .false.
        return
    endif

    if (self%backend == DTFFT_BACKEND_MPI_P2P_PIPELINED) then
        call self%execute_p2p(in, aux)
#ifdef DTFFT_WITH_RMA
    else
        call self%execute_rma(in, aux)
#endif
    end if

    if ( self%platform == DTFFT_PLATFORM_HOST ) then
        call self%execute_self_copy(in, aux, stream)
        call self%unpack_kernel%execute(aux, out, stream, self%comm_rank + 1)
    endif

    allocate (indices(self%recv%n_requests))
    need_completed = self%recv%n_requests
    total_completed = 0
    do while (.true.)
        ! Testing that all data has been recieved so we can unpack it
        call MPI_Waitsome(self%recv%n_requests, self%recv%requests, n_completed, indices, MPI_STATUSES_IGNORE, mpi_ierr)
        if (n_completed == MPI_UNDEFINED .or. need_completed == 0) exit

        do i = 1, n_completed
#ifdef MPICH_FIX_REQUIRED
            call self%unpack_kernel%execute(aux, out, stream, self%recv%process_map(indices(i) + 1) + 1)
#else
            call self%unpack_kernel%execute(aux, out, stream, self%recv%process_map(indices(i)) + 1)
#endif
        end do
        total_completed = total_completed + n_completed
        if (total_completed == need_completed) exit
    end do
    deallocate (indices)
    if (self%send%n_requests > 0) then
        call MPI_Waitall(self%send%n_requests, self%send%requests, MPI_STATUSES_IGNORE, mpi_ierr)
    end if
    if (self%is_rma) then
        call MPI_Win_fence(MPI_MODE_NOSUCCEED, self%win, error_code)
    end if
end subroutine execute_mpi

subroutine execute_end_mpi(self, error_code)
    class(backend_mpi), intent(inout)   :: self       !! MPI Backend
    integer(int32),     intent(out)     :: error_code !! Error code

    error_code = DTFFT_SUCCESS
    if (self%is_pipelined) return
    if (.not. self%is_active) then
        error_code = DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE
        return
    end if

    if (self%recv%n_requests > 0) then
        call MPI_Waitall(self%recv%n_requests, self%recv%requests, MPI_STATUSES_IGNORE, error_code)
    end if
    if (self%send%n_requests > 0) then
        call MPI_Waitall(self%send%n_requests, self%send%requests, MPI_STATUSES_IGNORE, error_code)
    end if
    if (self%is_rma) then
        call MPI_Win_fence(MPI_MODE_NOSUCCEED, self%win, error_code)
    end if
    self%is_active = .false.
end subroutine execute_end_mpi

subroutine execute_p2p(self, in, out)
    class(backend_mpi), intent(inout) :: self   !! MPI Backend
    real(real32),       intent(in)    :: in(:)  !! Data to be sent
    real(real32),       intent(inout) :: out(:) !! Data to be received
    integer(int32) :: send_request_counter, recv_request_counter
    integer(int32) :: i, mpi_ierr

    associate( recv => self%recv, send => self%send )

    if (.not. allocated(recv%process_map)) then
        allocate (recv%process_map(self%comm_size))
    end if

#ifdef ENABLE_PERSISTENT_COMM
    if (.not. self%is_request_created) then
        recv_request_counter = 0
        do i = 0, self%comm_size - 1
            if (recv%counts(i) > 0) then
                recv_request_counter = recv_request_counter + 1
                recv%process_map(recv_request_counter) = i
                call MPI_Recv_init(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, 0, &
                                   self%comm, recv%requests(recv_request_counter), mpi_ierr)
            end if
        end do
        recv%n_requests = recv_request_counter

        send_request_counter = 0
        do i = 0, self%comm_size - 1
            if (send%counts(i) > 0) then
                send_request_counter = send_request_counter + 1
                call MPI_Send_init(in(send%displs(i)), send%counts(i), MPI_REAL, i, 0, &
                                   self%comm, send%requests(send_request_counter), mpi_ierr)
            end if
        end do
        send%n_requests = send_request_counter
        self%is_request_created = .true.
    end if
    call MPI_Startall(recv%n_requests, recv%requests, mpi_ierr)
    call MPI_Startall(send%n_requests, send%requests, mpi_ierr)
#else
    send_request_counter = 0
    recv_request_counter = 0
    do i = 0, self%comm_size - 1
        if (recv%counts(i) > 0) then
            recv_request_counter = recv_request_counter + 1
            recv%process_map(recv_request_counter) = i
            call MPI_Irecv(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, 0, &
                           self%comm, recv%requests(recv_request_counter), mpi_ierr)
        end if
    end do
    recv%n_requests = recv_request_counter

    do i = 0, self%comm_size - 1
        if (send%counts(i) > 0) then
            send_request_counter = send_request_counter + 1
            call MPI_Isend(in(send%displs(i)), send%counts(i), MPI_REAL, i, 0, &
                           self%comm, send%requests(send_request_counter), mpi_ierr)
        end if
    end do
    send%n_requests = send_request_counter
#endif

    endassociate
end subroutine execute_p2p

subroutine execute_p2p_scheduled(self, in, out)
    class(backend_mpi), intent(inout) :: self   !! MPI Backend
    real(real32),       intent(in)    :: in(:)  !! Data to be sent
    real(real32),       intent(inout) :: out(:) !! Data to be received
    integer(int32) :: i, tgt, mpi_ierr

    associate( recv => self%recv, send => self%send )
    do i = 0, self%comm_size - 1
        tgt = self%schedule(i)
        if ( tgt == self%comm_rank .and. self%platform == DTFFT_PLATFORM_HOST) then
            call self%execute_self_copy(in, out, NULL_STREAM)
        else
            call MPI_Sendrecv(in(send%displs(tgt)), send%counts(tgt), MPI_REAL, tgt, 0, &
                              out(recv%displs(tgt)), recv%counts(tgt), MPI_REAL, tgt, 0, &
                              self%comm, MPI_STATUS_IGNORE, mpi_ierr)
        endif
    enddo

    endassociate
end subroutine execute_p2p_scheduled

subroutine execute_a2a(self, in, out)
class(backend_mpi), intent(inout) :: self   !! MPI Backend
real(real32),       intent(in)    :: in(:)  !! Data to be sent
real(real32),       intent(inout) :: out(:) !! Data to be received
integer(int32) :: mpi_ierr

    associate( recv => self%recv, send => self%send )

    if ( self%is_even .and. self%platform /= DTFFT_PLATFORM_CUDA ) then
#if defined(ENABLE_PERSISTENT_COLLECTIVES)
        if ( .not. self%is_request_created ) then
            call MPI_Alltoall_init(in, send%counts(0), MPI_REAL, &
                                   out, recv%counts(0), MPI_REAL, &
                                   self%comm, MPI_INFO_NULL, send%requests(1), mpi_ierr)
            self%is_request_created = .true.
        endif
        call MPI_Start(send%requests(1), mpi_ierr)
#else
        call MPI_Ialltoall(in, send%counts(0), MPI_REAL, &
                    out, recv%counts(0), MPI_REAL, &
                    self%comm, send%requests(1), mpi_ierr)
#endif
    else
#if defined(ENABLE_PERSISTENT_COLLECTIVES)
        if (.not. self%is_request_created) then
            call MPI_Alltoallv_init(in, send%counts, send%displs, MPI_REAL, &
                                    out, recv%counts, recv%displs, MPI_REAL, &
                                    self%comm, MPI_INFO_NULL, send%requests(1), mpi_ierr)
            self%is_request_created = .true.
        end if
        call MPI_Start(send%requests(1), mpi_ierr)
#else
    call MPI_Ialltoallv(in, send%counts, send%displs, MPI_REAL, &
                        out, recv%counts, recv%displs, MPI_REAL, &
                        self%comm, send%requests(1), mpi_ierr)
#endif
    endif

    send%n_requests = 1
    endassociate
end subroutine execute_a2a

#ifdef DTFFT_WITH_RMA
subroutine execute_rma(self, in, out)
class(backend_mpi), intent(inout) :: self   !! MPI Backend
real(real32), target,      intent(in)    :: in(:)  !! Data to be sent
real(real32),       intent(inout) :: out(:) !! Data to be received
integer(int32) :: i, mpi_ierr
integer(int32) :: recv_request_counter

    associate( recv => self%recv, send => self%send )

    if (.not. allocated(recv%process_map)) then
        allocate (recv%process_map(self%comm_size))
    end if
    if (.not. self%is_request_created) then
        call MPI_Win_create(in, int( size(in) * FLOAT_STORAGE_SIZE, MPI_ADDRESS_KIND ), int(FLOAT_STORAGE_SIZE, int32), MPI_INFO_NULL, self%comm, self%win, mpi_ierr)
        self%is_request_created = .true.
    end if

    call MPI_Win_fence(MPI_MODE_NOPRECEDE, self%win, mpi_ierr)

    recv_request_counter = 0
    do i = 0, self%comm_size - 1
        if (recv%counts(i) > 0) then
            recv_request_counter = recv_request_counter + 1
            recv%process_map(recv_request_counter) = i
            call MPI_RGet(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, int(send%displs(i), MPI_ADDRESS_KIND), &
                recv%counts(i), MPI_REAL, self%win, recv%requests(recv_request_counter), mpi_ierr)
        end if
    end do
    recv%n_requests = recv_request_counter

    endassociate
end subroutine execute_rma
#endif

subroutine compute_alltoall_schedule(self)
!! Generate optimal round-robin communication schedule for all-to-all pattern
class(backend_mpi), intent(inout) :: self           !! MPI Backend
integer(int32) :: round, offset, num_rounds, idx, peer

    idx = 0
    allocate( self%schedule(0:self%comm_size - 1) )
    ! Handle even/odd number of processes differently
    if (mod(self%comm_size, 2) == 0) then
        num_rounds = self%comm_size
        ! Self-communication first
        self%schedule(idx) = self%comm_rank; idx = idx + 1
    else
        num_rounds = self%comm_size + 1
    end if

    ! Generate round-robin schedule using circular shift pattern
    do round = 0, num_rounds - 2
        if (mod(self%comm_size, 2) == 0) then
            ! Special handling for last process in even case
            if (round == self%comm_rank) then
                self%schedule(idx) = self%comm_size - 1; idx = idx + 1
            else if (self%comm_size - 1 == self%comm_rank) then
                self%schedule(idx) = round; idx = idx + 1
            end if
        else if (round == self%comm_rank) then
            self%schedule(idx) = round; idx = idx + 1
        end if
        
        if (round /= self%comm_rank .and. self%comm_rank < num_rounds - 1) then
            ! Forward pairing
            offset = mod(round - self%comm_rank + (num_rounds - 1), num_rounds - 1)
            if (offset < num_rounds/2) then
                peer = mod(round + offset, num_rounds - 1)
                self%schedule(idx) = peer; idx = idx + 1
            end if
            
            ! Backward pairing
            offset = mod(self%comm_rank - round + (num_rounds - 1), num_rounds - 1)
            if (offset < num_rounds/2) then
                peer = mod(round - offset + (num_rounds - 1), num_rounds - 1)
                self%schedule(idx) = peer; idx = idx + 1
            end if
        end if
    end do
end subroutine compute_alltoall_schedule
end module dtfft_backend_mpi
