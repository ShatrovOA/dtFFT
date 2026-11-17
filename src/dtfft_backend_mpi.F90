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
use, intrinsic :: iso_fortran_env
use, intrinsic :: iso_c_binding
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
use dtfft_abstract_backend, only: abstract_backend, backend_helper
use dtfft_abstract_kernel
use dtfft_parameters
use dtfft_errors
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
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
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        logical                     :: staging_allocated = .false.  !! If staging buffers are allocated
        type(c_ptr)                 :: host_send = c_null_ptr       !! Host send buffer for staging
        type(c_ptr)                 :: host_recv = c_null_ptr       !! Host recv buffer for staging
        type(c_ptr)                 :: staging_output = c_null_ptr  !! Device output buffer
        integer(int64)              :: staging_bytes = 0_int64      !! Size of staging buffers in bytes
        type(dtfft_stream_t)        :: staging_stream               !! Stream for staging operations
#endif

    contains
        procedure :: create_private => create_mpi   !! Creates MPI backend
        procedure :: execute_private => execute_mpi !! Executes MPI backend
        procedure :: destroy_private => destroy_mpi !! Destroys MPI backend
        procedure :: execute_end => execute_end_mpi !! Finalizes async transpose
        procedure :: get_async_active               !! Overrides abstract method and returns if async transpose is active
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        procedure :: memcpy_h2d     !! Copies data from host to device
#endif
        procedure :: execute_p2p        !! Executes point-to-point communication
        procedure :: execute_a2a        !! Executes all-to-all communication
        procedure :: execute_p2p_scheduled  !! Executes scheduled point-to-point communication
        procedure :: compute_alltoall_schedule  !! Computes the communication schedule for all-to-all
        procedure :: execute_fused  !! Executes fused communication
#ifdef DTFFT_WITH_RMA
        procedure :: execute_rma    !! Executes RMA communication
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
        if (.not. is_backend_mpi(self%backend)) then
            INTERNAL_ERROR(".not. is_backend_mpi")
        endif
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
            if ( self%is_fused ) call self%compute_alltoall_schedule()
        end if

        self%is_rma = .false.
#ifdef DTFFT_WITH_RMA
        self%win = MPI_WIN_NULL
        if ( is_backend_rma(self%backend) ) then
            block
                integer(int32), allocatable :: all_displs(:, :)

                self%is_rma = .true.

                allocate (all_displs(0:self%comm_size - 1, 0:self%comm_size - 1))
                call MPI_Allgather(self%send%displs, self%comm_size, MPI_INTEGER, all_displs, self%comm_size, MPI_INTEGER, self%comm, mpi_err)
                self%send%displs(:) = all_displs(self%comm_rank, :) - 1
                deallocate (all_displs)
            end block
        endif
#endif
        self%is_request_created = .false.
        self%is_active = .false.
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        self%host_send = c_null_ptr
        self%host_recv = c_null_ptr
        self%staging_output = c_null_ptr
        self%staging_bytes = max(self%buffer_size, 1_int64) * FLOAT_STORAGE_SIZE
        self%staging_allocated = .false.
#endif
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
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        if ( self%staging_allocated ) then
            CUDA_CALL(cudaFreeHost(self%host_send))
            CUDA_CALL(cudaFreeHost(self%host_recv))
        end if
        self%host_send = c_null_ptr
        self%host_recv = c_null_ptr
        self%staging_output = c_null_ptr
        self%staging_bytes = 0_int64
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
        class(backend_mpi),                 intent(inout)   :: self       !! MPI Backend
        real(real32),   target, contiguous, intent(inout)   :: in(:)      !! Send pointer
        real(real32),   target, contiguous, intent(inout)   :: out(:)     !! Recv pointer
        type(dtfft_stream_t),               intent(in)      :: stream     !! Main execution CUDA stream
        real(real32),   target, contiguous, intent(inout)   :: aux(:)     !! Aux pointer
        integer(int32),                     intent(out)     :: error_code !! Error code
        real(real32), pointer, contiguous :: mpi_in(:), mpi_out(:), mpi_aux(:)
        integer(int32), allocatable :: indices(:)
        integer(int32)              :: total_completed, n_completed      !! Request counter
        integer(int32)              :: need_completed
        integer(int32)              :: i                    !! Loop index
        integer(int32) :: source

        error_code = DTFFT_SUCCESS
        mpi_in => in
        mpi_out => out
        mpi_aux => aux
        if (self%is_active) then
            error_code = DTFFT_ERROR_TRANSPOSE_ACTIVE
            return
        end if
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        if ( .not. self%staging_allocated .and. self%platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL(cudaHostAlloc(self%host_send, int(self%staging_bytes, c_size_t), cudaHostAllocDefault))
            CUDA_CALL(cudaHostAlloc(self%host_recv, int(self%staging_bytes, c_size_t), cudaHostAllocDefault))
            self%staging_allocated = .true.
        endif
#endif
        if ( self%is_fused ) then
            call self%execute_fused(in, out, stream, aux)
            return
        endif
#ifdef DTFFT_WITH_CUDA
        if ( self%platform == DTFFT_PLATFORM_CUDA ) then
# ifdef WITHOUT_DEVICE_AWARE
            CUDA_CALL(cudaMemcpyAsync(self%host_send, c_loc(in), int(self%staging_bytes, c_size_t), cudaMemcpyDeviceToHost, stream))
            self%staging_stream = stream

            call c_f_pointer(self%host_send, mpi_in, [size(in)])
            if ( self%is_pipelined ) then
                call c_f_pointer(self%host_recv, mpi_aux, [size(aux)])
                self%staging_output = c_loc(aux)
            else
                call c_f_pointer(self%host_recv, mpi_out, [size(out)])
                self%staging_output = c_loc(out)
            endif
# endif
            ! Need to sync stream since there is no way pass current stream to MPI
            CUDA_CALL(cudaStreamSynchronize(stream))
        end if
#endif

        if ( .not. self%is_pipelined ) then
            select case (self%backend%val)
            case (DTFFT_BACKEND_MPI_P2P%val)
                call self%execute_p2p(mpi_in, mpi_out)
            case (DTFFT_BACKEND_MPI_A2A%val)
                call self%execute_a2a(mpi_in, mpi_out)
            case (DTFFT_BACKEND_MPI_P2P_SCHEDULED%val)
                call self%execute_p2p_scheduled(mpi_in, mpi_out)
#ifdef DTFFT_WITH_RMA
            case (DTFFT_BACKEND_MPI_RMA%val)
                call self%execute_rma(mpi_in, mpi_out)
#endif
            end select
            if ( self%platform == DTFFT_PLATFORM_HOST .and. self%backend /= DTFFT_BACKEND_MPI_P2P_SCHEDULED ) then
                call self%execute_self_copy(in, out, stream)
            endif
            self%is_active = self%backend /= DTFFT_BACKEND_MPI_P2P_SCHEDULED
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
            if ( self%backend == DTFFT_BACKEND_MPI_P2P_SCHEDULED .and. self%platform == DTFFT_PLATFORM_CUDA ) then
                call self%memcpy_h2d(stream)
                self%staging_output = c_null_ptr
            end if
#endif
            return
        endif

        if (self%backend == DTFFT_BACKEND_MPI_P2P_PIPELINED) then
            call self%execute_p2p(mpi_in, mpi_aux)
#ifdef DTFFT_WITH_RMA
        else
            call self%execute_rma(mpi_in, mpi_aux)
#endif
        end if

        if ( self%platform == DTFFT_PLATFORM_HOST ) then
            call self%execute_self_copy(mpi_in, mpi_aux, stream)
            call self%unpack_kernel%execute(c_loc(mpi_aux), c_loc(mpi_out), stream, self%comm_rank + 1)
        endif

        allocate (indices(self%recv%n_requests))
        need_completed = self%recv%n_requests
        total_completed = 0
        do while (.true.)
            ! Testing that all data has been recieved so we can unpack it
            call waitsome_requests(self%recv%n_requests, self%recv%requests, n_completed, indices)
            if (n_completed == MPI_UNDEFINED .or. need_completed == 0) exit

            do i = 1, n_completed
                source = indices(i)
#ifdef MPICH_FIX_REQUIRED
                source = source + 1
#endif
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
                if ( self%platform == DTFFT_PLATFORM_CUDA ) then
                    call self%memcpy_h2d(self%copy_stream, self%recv%process_map(source))
                    call self%wait(stream)
                endif
#endif
                call self%unpack_kernel%execute(c_loc(aux), c_loc(out), stream, self%recv%process_map(source) + 1)
            end do
            total_completed = total_completed + n_completed
            if (total_completed == need_completed) exit
        end do
        deallocate (indices)
        if (self%send%n_requests > 0) then
            call waitall_requests(self%send%n_requests, self%send%requests)
        end if
        if (self%is_rma) then
            call MPI_Win_fence(MPI_MODE_NOSUCCEED, self%win, error_code)
        end if
    end subroutine execute_mpi

    subroutine execute_end_mpi(self, error_code)
    !! Finalizes async backend execution
        class(backend_mpi), intent(inout)   :: self       !! MPI Backend
        integer(int32),     intent(out)     :: error_code !! Error code
        error_code = DTFFT_SUCCESS
        if (self%is_pipelined .or. self%is_fused) return
        if (.not. self%is_active) then
            error_code = DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE
            return
        end if

        if (self%recv%n_requests > 0) then
            call waitall_requests(self%recv%n_requests, self%recv%requests)
        end if
        if (self%send%n_requests > 0) then
            call waitall_requests(self%send%n_requests, self%send%requests)
        end if
        if (self%is_rma) then
            call MPI_Win_fence(MPI_MODE_NOSUCCEED, self%win, error_code)
        end if
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        if ( self%platform == DTFFT_PLATFORM_CUDA ) then
            call self%memcpy_h2d(self%staging_stream)
            self%staging_output = c_null_ptr
        endif
#endif
        self%is_active = .false.
    end subroutine execute_end_mpi

#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
    subroutine memcpy_h2d(self, stream, source_rank)
    !! Copies received data from the pinned host buffer to the CUDA output buffer.
        class(backend_mpi),       intent(inout) :: self
        type(dtfft_stream_t),     intent(in)    :: stream
        integer(int32), optional, intent(in)    :: source_rank
        integer(int32) :: i, source_rank_

        OPTVAL(source_rank_, VARIABLE_NOT_SET, source_rank)

        if ( self%backend == DTFFT_BACKEND_MPI_A2A .and. self%is_even) then
            CUDA_CALL(cudaMemcpyAsync(self%staging_output, self%host_recv, int(self%staging_bytes, c_size_t), cudaMemcpyHostToDevice, stream))
            return
        endif

        if ( source_rank_ == VARIABLE_NOT_SET ) then
            do i = 0, self%comm_size - 1
                if (self%recv%counts(i) == 0) cycle
                call launch_memcpy(i)
            end do
            return
        endif
        if (self%recv%counts(source_rank_) == 0) return
        call launch_memcpy(source_rank_)
    contains

        subroutine launch_memcpy(source)
            integer(int32), intent(in) :: source
            integer(int64) :: displacement, count_bytes
            type(c_ptr) :: dst, src

            displacement = int(self%recv%displs(source), int64)
            if (self%backend /= DTFFT_BACKEND_MPI_A2A) displacement = displacement - 1_int64
            count_bytes = int(self%recv%counts(source), int64) * FLOAT_STORAGE_SIZE
            dst = ptr_offset(self%staging_output, displacement * FLOAT_STORAGE_SIZE)
            src = ptr_offset(self%host_recv, displacement * FLOAT_STORAGE_SIZE)
            CUDA_CALL(cudaMemcpyAsync(dst, src, int(count_bytes, c_size_t), cudaMemcpyHostToDevice, stream))
        end subroutine launch_memcpy
    end subroutine memcpy_h2d
#endif

    subroutine execute_p2p(self, in, out)
    !! Executes Point-to-Point MPI backend
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
                    call MPI_Recv_init(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, 0, self%comm, recv%requests(recv_request_counter), mpi_ierr)
                end if
            end do
            recv%n_requests = recv_request_counter

            send_request_counter = 0
            do i = 0, self%comm_size - 1
                if (send%counts(i) > 0) then
                    send_request_counter = send_request_counter + 1
                    call MPI_Send_init(in(send%displs(i)), send%counts(i), MPI_REAL, i, 0, self%comm, send%requests(send_request_counter), mpi_ierr)
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
                call MPI_Irecv(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, 0, self%comm, recv%requests(recv_request_counter), mpi_ierr)
            end if
        end do
        recv%n_requests = recv_request_counter

        do i = 0, self%comm_size - 1
            if (send%counts(i) > 0) then
                send_request_counter = send_request_counter + 1
                call MPI_Isend(in(send%displs(i)), send%counts(i), MPI_REAL, i, 0, self%comm, send%requests(send_request_counter), mpi_ierr)
            end if
        end do
        send%n_requests = send_request_counter
#endif

        endassociate
    end subroutine execute_p2p

    subroutine execute_p2p_scheduled(self, in, out)
    !! Executes Scheduled Point-to-Point MPI backend
        class(backend_mpi), intent(inout) :: self   !! MPI Backend
        real(real32),       intent(in)    :: in(:)  !! Data to be sent
        real(real32),       intent(inout) :: out(:) !! Data to be received
        integer(int32) :: i, tgt, mpi_ierr
        associate( recv => self%recv, send => self%send )
        do i = 0, self%comm_size - 1
            tgt = self%schedule(i)
            if ( tgt == self%comm_rank .and. self%platform == DTFFT_PLATFORM_HOST) then
                call self%execute_self_copy(in, out, NULL_STREAM)
            else if ( send%counts(tgt) > 0 .and. recv%counts(tgt) > 0 ) then
                call MPI_Sendrecv(in(send%displs(tgt)), send%counts(tgt), MPI_REAL, tgt, 0,                             &
                    out(recv%displs(tgt)), recv%counts(tgt), MPI_REAL, tgt, 0, self%comm, MPI_STATUS_IGNORE, mpi_ierr)
            else if ( send%counts(tgt) > 0 ) then
                call MPI_Send(in(send%displs(tgt)), send%counts(tgt), MPI_REAL, tgt, 0, self%comm, mpi_ierr)
            else if ( recv%counts(tgt) > 0 ) then
                call MPI_Recv(out(recv%displs(tgt)), recv%counts(tgt), MPI_REAL, tgt, 0, self%comm, MPI_STATUS_IGNORE, mpi_ierr)
            end if
            ! else
            !     call MPI_Sendrecv(in(send%displs(tgt)), send%counts(tgt), MPI_REAL, tgt, 0,                             &
            !         out(recv%displs(tgt)), recv%counts(tgt), MPI_REAL, tgt, 0, self%comm, MPI_STATUS_IGNORE, mpi_ierr)
            ! endif
        enddo

        endassociate
    end subroutine execute_p2p_scheduled

    subroutine execute_a2a(self, in, out)
    !! Executes All-to-All MPI backend
        class(backend_mpi), intent(inout) :: self   !! MPI Backend
        real(real32),       intent(in)    :: in(:)  !! Data to be sent
        real(real32),       intent(inout) :: out(:) !! Data to be received
        integer(int32) :: mpi_ierr
        logical :: use_alltoall

        associate( recv => self%recv, send => self%send )

        use_alltoall = self%is_even .and. self%platform /= DTFFT_PLATFORM_CUDA
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        if (self%platform == DTFFT_PLATFORM_CUDA) use_alltoall = self%is_even
#endif
        if ( use_alltoall ) then
#if defined(ENABLE_PERSISTENT_COLLECTIVES)
            if ( .not. self%is_request_created ) then
                call MPI_Alltoall_init(in, send%counts(0), MPI_REAL, out, recv%counts(0), MPI_REAL, self%comm, MPI_INFO_NULL, send%requests(1), mpi_ierr)
                self%is_request_created = .true.
            endif
            call MPI_Start(send%requests(1), mpi_ierr)
#else
            call MPI_Ialltoall(in, send%counts(0), MPI_REAL, out, recv%counts(0), MPI_REAL, self%comm, send%requests(1), mpi_ierr)
#endif
        else
#if defined(ENABLE_PERSISTENT_COLLECTIVES)
            if (.not. self%is_request_created) then
                call MPI_Alltoallv_init(in, send%counts, send%displs, MPI_REAL, out, recv%counts, recv%displs, MPI_REAL, self%comm, MPI_INFO_NULL, send%requests(1), mpi_ierr)
                self%is_request_created = .true.
            end if
            call MPI_Start(send%requests(1), mpi_ierr)
#else
        call MPI_Ialltoallv(in, send%counts, send%displs, MPI_REAL, out, recv%counts, recv%displs, MPI_REAL, self%comm, send%requests(1), mpi_ierr)
#endif
        endif

        send%n_requests = 1
        endassociate
    end subroutine execute_a2a

#ifdef DTFFT_WITH_RMA
    subroutine execute_rma(self, in, out)
    !! Executes RMA backend
        class(backend_mpi),     intent(inout) :: self   !! MPI Backend
        real(real32), target,   intent(in)    :: in(:)  !! Data to be sent
        real(real32),           intent(inout) :: out(:) !! Data to be received
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
                call MPI_RGet(out(recv%displs(i)), recv%counts(i), MPI_REAL, i, int(send%displs(i), MPI_ADDRESS_KIND), recv%counts(i), MPI_REAL, self%win, recv%requests(recv_request_counter), mpi_ierr)
            end if
        end do
        recv%n_requests = recv_request_counter

        endassociate
    end subroutine execute_rma
#endif

    subroutine execute_fused(self, in, out, stream, aux)
    !! Executes FUSED backends (p2p/rma)
        class(backend_mpi),                 intent(inout)   :: self       !! MPI Backend
        real(real32), target,   contiguous, intent(inout)   :: in(:)      !! Send pointer
        real(real32), target,   contiguous, intent(inout)   :: out(:)     !! Recv pointer
        type(dtfft_stream_t),               intent(in)      :: stream     !! Main execution CUDA stream
        real(real32), target,   contiguous, intent(inout)   :: aux(:)     !! Aux pointer
        integer(int64) :: aux_size, send_comp_bytes, recv_comp_bytes, copy_bytes
        real(real32), pointer, contiguous :: aux2(:)
        integer(int32) :: i, k, tgt, mpi_ierr, comm_rank
        integer(int32) :: send_request_counter, recv_request_counter
        integer(int32), allocatable :: indices(:)
        integer(int32)              :: total_completed, n_completed      !! Request counter
        real(real32), pointer, contiguous :: send_buffer(:), recv_buffer(:)
        real(real32), pointer, contiguous :: mpi_send_buffer(:), mpi_recv_buffer(:)
        integer(int64), allocatable :: comp_sizes(:)
        integer(int32) :: source_rank, send_count, recv_count
        integer(int32) :: active_recvs
        logical :: use_persistent_p2p, sync
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        integer(int64) :: send_displacement, recv_displacement
        type(c_ptr) :: source_ptr, staging_ptr
#endif

        call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, mpi_ierr)

        aux_size = size(aux, kind=int64) / 2
        call c_f_pointer(ptr_offset(c_loc(aux), aux_size * FLOAT_STORAGE_SIZE), aux2, [aux_size])

        allocate( comp_sizes(0:self%comm_size - 1) )
        comp_sizes(:) = 0_int64

        if ( self%pack_kernel%is_dummy_kernel ) then
            send_buffer => in
            recv_buffer => aux
        else if ( self%pack_kernel%is_dummy_compressed ) then
            send_buffer => aux
            recv_buffer => aux2
        else if ( self%unpack_kernel%is_dummy_kernel ) then
            send_buffer => aux
            recv_buffer => out
        else
            ! if compressed:
            !   pack: in -> aux2
            !   compress: aux2 -> aux
            !   send: aux -> aux2
            !   decompress: aux2 -> in
            !   unpack: in -> out
            ! else:
            !   pack: in -> aux
            !   send: aux -> aux2
            !   unpack aux2 -> out

            ! Self copy
            ! if compressed or not:
            !   pack: in -> aux
            !   copy: aux -> aux2
            !   unpack: aux2 -> out
            send_buffer => aux
            recv_buffer => aux2
        endif
        mpi_send_buffer => send_buffer
        mpi_recv_buffer => recv_buffer
        sync = .true.
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
        if (self%platform == DTFFT_PLATFORM_CUDA) then
            call c_f_pointer(self%host_send, mpi_send_buffer, [self%buffer_size])
            call c_f_pointer(self%host_recv, mpi_recv_buffer, [self%buffer_size])
        endif
        sync = .false.
#endif

        recv_request_counter = 0
        send_request_counter = 0
        active_recvs = 0
        use_persistent_p2p = .false.
#ifdef ENABLE_PERSISTENT_COMM
        use_persistent_p2p = .not. self%is_rma .and. &
            .not. self%pack_kernel%is_compress .and. .not. self%unpack_kernel%is_decompress
#endif
        associate( recv => self%recv, send => self%send )

            if (.not. allocated(recv%process_map)) then
                allocate (recv%process_map(self%comm_size))
            end if

#ifdef DTFFT_WITH_RMA
            if ( self%is_rma ) then
                if ( .not. self%is_request_created ) then
                    call MPI_Win_create(mpi_send_buffer, int( size(in) * FLOAT_STORAGE_SIZE, MPI_ADDRESS_KIND ), int(1, int32), MPI_INFO_NULL, self%comm, self%win, mpi_ierr)
                    self%is_request_created = .true.
                endif
                call MPI_Win_fence(MPI_MODE_NOPRECEDE, self%win, mpi_ierr)
            endif
#endif

            do i = 0, self%comm_size - 1
                tgt = self%schedule(i)
                if ( tgt == self%comm_rank ) then
                    call self%pack_kernel%execute(c_loc(in), c_loc(send_buffer), stream, tgt + 1, skip_compression=.true.)
                    cycle
                endif
                call self%pack_kernel%execute(c_loc(in), c_loc(send_buffer), stream, tgt + 1, aux=c_loc(out), csize=send_comp_bytes, sync=sync)
                if ( self%pack_kernel%is_dummy .or. self%pack_kernel%is_dummy_kernel ) send_comp_bytes = send%counts(tgt) * FLOAT_STORAGE_SIZE

#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
                if ( self%platform == DTFFT_PLATFORM_CUDA ) then
                    send_displacement = int(self%send_displs(tgt), int64) - 1_int64
                    source_ptr = ptr_offset(c_loc(send_buffer), send_displacement * FLOAT_STORAGE_SIZE)
                    staging_ptr = ptr_offset(self%host_send, send_displacement * FLOAT_STORAGE_SIZE)
                    CUDA_CALL(cudaMemcpyAsync(staging_ptr, source_ptr, int(send_comp_bytes, c_size_t), cudaMemcpyDeviceToHost, stream))
                    CUDA_CALL(cudaStreamSynchronize(stream))
                endif
#endif

                ! This is synchronization point between two processes
                ! `recv_comp_bytes` might be known if compression is not used, however
                ! we still use Sendrecv to be sure both sides are synchronized
                call MPI_Sendrecv(send_comp_bytes, 1, MPI_INTEGER8, tgt, 0, recv_comp_bytes, 1, MPI_INTEGER8, tgt, 0, self%comm, MPI_STATUS_IGNORE, mpi_ierr)

                comp_sizes(tgt) = recv_comp_bytes
                send_count = bytes_to_floats(send_comp_bytes)
                recv_count = bytes_to_floats(recv_comp_bytes)

                if ( self%is_rma ) then
#ifdef DTFFT_WITH_RMA
                    if (recv_count > 0) then
                        active_recvs = active_recvs + 1
                        recv_request_counter = recv_request_counter + 1
                        recv%process_map(recv_request_counter) = tgt
                        ! call MPI_Win_lock(MPI_LOCK_EXCLUSIVE, tgt, 0, self%win, mpi_ierr)
                        call MPI_Rget(mpi_recv_buffer(recv%displs(tgt)), recv_count, MPI_REAL, tgt, int(send%displs(tgt) * FLOAT_STORAGE_SIZE, MPI_ADDRESS_KIND), recv_count, MPI_REAL, self%win, recv%requests(recv_request_counter), mpi_ierr)
                        ! call MPI_Win_unlock(tgt, self%win, mpi_ierr)
                    end if
#endif
                else    ! P2P
                    if (recv_count > 0) then
                        active_recvs = active_recvs + 1
                        recv_request_counter = recv_request_counter + 1
                        recv%process_map(recv_request_counter) = tgt
                        if (use_persistent_p2p) then
                            if (recv%requests(recv_request_counter) == MPI_REQUEST_NULL) then
                                call MPI_Recv_init(mpi_recv_buffer(recv%displs(tgt)), recv_count, MPI_REAL, tgt, 0, self%comm, recv%requests(recv_request_counter), mpi_ierr)
                            end if
                            call MPI_Start(recv%requests(recv_request_counter), mpi_ierr)
                        else
                            call MPI_Irecv(mpi_recv_buffer(recv%displs(tgt)), recv_count, MPI_REAL, tgt, 0, self%comm, recv%requests(recv_request_counter), mpi_ierr)
                        end if
                    end if
                    if ( send_count > 0 ) then
                        send_request_counter = send_request_counter + 1
                        if (use_persistent_p2p) then
                            if (send%requests(send_request_counter) == MPI_REQUEST_NULL) then
                                call MPI_Send_init(mpi_send_buffer(send%displs(tgt)), send_count, MPI_REAL, tgt, 0, self%comm, send%requests(send_request_counter), mpi_ierr)
                            end if
                            call MPI_Start(send%requests(send_request_counter), mpi_ierr)
                        else
                            call MPI_Isend(mpi_send_buffer(send%displs(tgt)), send_count, MPI_REAL, tgt, 0, self%comm, send%requests(send_request_counter), mpi_ierr)
                        end if
                    endif
                endif
            enddo

            if ( self%unpack_kernel%is_dummy .or. self%unpack_kernel%is_dummy_compressed ) then
                call self%execute_self_copy(send_buffer, out, stream)
            else if ( self%pack_kernel%is_dummy_compressed ) then
                call self%execute_self_copy(in, recv_buffer, stream)
                call self%unpack_kernel%execute(c_loc(recv_buffer), c_loc(out), self%copy_stream, self%comm_rank + 1, skip_compression=.true.)
            else
                call self%execute_self_copy(send_buffer, recv_buffer, stream)
                call self%unpack_kernel%execute(c_loc(recv_buffer), c_loc(out), self%copy_stream, self%comm_rank + 1, skip_compression=.true.)
            endif
            call self%wait(stream)

            total_completed = 0
            allocate( indices(recv_request_counter) )

            do while (active_recvs > 0)
                call waitsome_requests(recv_request_counter, recv%requests, n_completed, indices)
                if (n_completed /= MPI_UNDEFINED .and. n_completed > 0) then
                    do k = 1, n_completed
#ifdef MPICH_FIX_REQUIRED
                        source_rank = recv%process_map(indices(k) + 1)
#else
                        source_rank = recv%process_map(indices(k))
#endif
                        copy_bytes = comp_sizes(source_rank)
#if defined(DTFFT_WITH_CUDA) && defined(WITHOUT_DEVICE_AWARE)
                        if (self%platform == DTFFT_PLATFORM_CUDA .and. copy_bytes > 0) then
                            recv_displacement = int(recv%displs(source_rank) - 1, int64)
                            source_ptr = ptr_offset(self%host_recv, recv_displacement * FLOAT_STORAGE_SIZE)
                            staging_ptr = ptr_offset(c_loc(recv_buffer), recv_displacement * FLOAT_STORAGE_SIZE)
                            CUDA_CALL(cudaMemcpyAsync(staging_ptr, source_ptr, int(copy_bytes, c_size_t), cudaMemcpyHostToDevice, self%copy_stream))
                            call self%wait(stream)
                        endif
#endif
                        call self%unpack_kernel%execute(c_loc(recv_buffer), c_loc(out), stream, source_rank + 1, aux=c_loc(in), csize=copy_bytes)
                        active_recvs = active_recvs - 1
                    end do
                end if
            end do
            deallocate (indices)

            if (send_request_counter > 0) then
                call waitall_requests(send_request_counter, send%requests)
            end if
        endassociate

#ifdef DTFFT_WITH_RMA
        if ( self%is_rma ) then
            call MPI_Win_fence(MPI_MODE_NOSUCCEED, self%win, mpi_ierr)
        endif
#endif

        if ( use_persistent_p2p .and. .not. self%is_request_created ) then
            self%is_request_created = .true.
            self%send%n_requests = send_request_counter
            self%recv%n_requests = recv_request_counter
        endif

    deallocate( comp_sizes )
    end subroutine execute_fused

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