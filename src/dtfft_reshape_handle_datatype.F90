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
module dtfft_reshape_handle_datatype
!! This module describes [[reshape_handle_datatype]] class
!! It executes either transposition or reshaping from bricks to pencils and vice versa using MPI_Ialltoall(w) with custom MPI datatypes
!! For the end user this is `DTFFT_BACKEND_MPI_DATATYPE` - backend.
!! But since it does not perform sequence: transpose -> exchange -> unpack, it is internally treated as reshape_handle.
use iso_fortran_env
use iso_c_binding,                  only: c_ptr, c_f_pointer
use dtfft_abstract_reshape_handle,  only: abstract_reshape_handle, create_args, execute_args
use dtfft_errors
use dtfft_parameters
use dtfft_pencil,     only: pencil, get_float_buffer_size
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
#include "_dtfft_cuda.h"
implicit none
private
public :: reshape_handle_datatype

    integer(MPI_ADDRESS_KIND), parameter :: LB = 0
    !! Lower bound for all derived datatypes

#if defined (ENABLE_PERSISTENT_COMM)
# if defined(ENABLE_PERSISTENT_COLLECTIVES)
#   if defined(OMPI_FIX_REQUIRED)
      logical, parameter :: IS_P2P_ENABLED = .true.
#   else
      logical, parameter :: IS_P2P_ENABLED = .false.
#   endif
# else
      logical, parameter :: IS_P2P_ENABLED = .true.
# endif
#else
# if defined(OMPI_FIX_REQUIRED)
      logical, parameter :: IS_P2P_ENABLED = .true.
# else
      logical, parameter :: IS_P2P_ENABLED = .false.
# endif
#endif
  !! Is point-to-point communication enabled

    type :: handle_t
    !! Transposition handle class
        TYPE_MPI_DATATYPE,     allocatable :: dtypes(:)           !! Datatypes buffer
        integer(int32),        allocatable :: counts(:)           !! Number of datatypes (always equals 1)
        integer(int32),        allocatable :: displs(:)           !! Displacements is bytes
    contains
        procedure, pass(self) :: create => create_handle          !! Creates transposition handle
        procedure, pass(self) :: destroy => destroy_handle        !! Destroys transposition handle
    end type handle_t


    type, extends(abstract_reshape_handle) :: reshape_handle_datatype
    !! Transpose backend that uses MPI_Ialltoall(w) with custom MPI datatypes
    ! private
        TYPE_MPI_COMM                   :: comm                   !! 1d communicator
        logical                         :: is_even = .false.      !! Is decomposition even
        logical                         :: is_active = .false.    !! Is async transposition active
        type(handle_t)                  :: send                   !! Handle to send data
        type(handle_t)                  :: recv                   !! Handle to recieve data
        TYPE_MPI_REQUEST, allocatable   :: requests(:)            !! Requests for communication
        integer(int32)                  :: n_requests             !! Actual number of requests, can be less than size(requests)
        integer(int64)                  :: min_buffer_size
#if defined(ENABLE_PERSISTENT_COMM)
        logical                         :: is_request_created = .false.     !! Is request created
#endif
    contains
    private
        procedure, pass(self),  public  :: create_private => create !! Initializes class
        procedure, pass(self),  public  :: execute                !! Performs MPI_Ialltoall(w)
        procedure, pass(self),  public  :: execute_end            !! Waits for MPI_Ialltoall(w) to complete
        procedure, pass(self),  public  :: destroy                !! Destroys class
        procedure, pass(self),  public  :: get_async_active       !! Returns .true. if async transposition is active
        procedure, pass(self),  public  :: get_backend
    end type reshape_handle_datatype

contains

    subroutine create_handle(self, n)
    !! Creates reshape handle
        class(handle_t),  intent(inout) :: self   !! Reshape handle
        integer(int32),   intent(in)    :: n      !! Number of datatypes to be created

        call self%destroy()
        allocate(self%dtypes(n), source = MPI_DATATYPE_NULL)
        allocate(self%counts(n), source = 1_int32)
        allocate(self%displs(n), source = 0_int32)
    end subroutine create_handle

    subroutine destroy_handle(self)
    !! Destroys reshape handle
        class(handle_t),  intent(inout)   :: self   !! Reshape handle
        integer(int32)                    :: i      !! Counter
        integer(int32)                    :: ierr   !! Error code

        if ( allocated(self%dtypes) ) then
            do i = 1, size(self%dtypes)
                call MPI_Type_free(self%dtypes(i), ierr)
            enddo
            deallocate(self%dtypes)
        endif
        if ( allocated(self%displs) ) deallocate(self%displs)
        if ( allocated(self%counts) ) deallocate(self%counts)
    end subroutine destroy_handle

    subroutine create(self, comm, send, recv, kwargs)
    !! Creates `reshape_handle_datatype` class
        class(reshape_handle_datatype),   intent(inout) :: self           !! Reshape handle
        TYPE_MPI_COMM,                    intent(in)    :: comm           !! MPI Communicator
        type(pencil),                     intent(in)    :: send           !! Send pencil
        type(pencil),                     intent(in)    :: recv           !! Recv pencil
        type(create_args),                intent(in)    :: kwargs         !! Additional arguments
        integer(int32)                              :: comm_size          !! Size of 1d communicator
        integer(int32)                              :: n_neighbors        !! Number of datatypes to be created
        integer(int32),               allocatable   :: recv_counts(:,:)   !! Each processor should know how much data each processor recieves
        integer(int32),               allocatable   :: send_counts(:,:)   !! Each processor should know how much data each processor sends
        integer(int32),               allocatable   :: recv_starts(:,:)
        integer(int32),               allocatable   :: send_starts(:,:)
        integer(int32)                              :: i                  !! Counter
        integer(int32)                              :: ierr               !! Error code
        integer(int32) :: send_displ, recv_displ
        type(dtfft_transpose_t) :: transpose_type
        type(dtfft_reshape_t)   :: reshape_type
        type(dtfft_transpose_mode_t) :: tmode
        TYPE_MPI_DATATYPE :: base_type
        integer(int64) :: base_storage
        integer(int8) :: reshape_strat
        logical :: zslab, yslab

        call self%destroy()
        self%comm = comm
        call MPI_Comm_size(comm, comm_size, ierr)
        self%is_even = send%is_even .and. recv%is_even
        n_neighbors = comm_size;  if ( self%is_even ) n_neighbors = 1
        self%is_active = .false.

        allocate(self%requests(2 * comm_size))

#if defined(ENABLE_PERSISTENT_COMM)
        self%is_request_created = .false.
#endif

        call self%send%create(n_neighbors)
        call self%recv%create(n_neighbors)

        allocate(recv_counts(recv%rank, comm_size), source = 0_int32)
        allocate(send_counts, source = recv_counts)
        allocate(recv_starts, source = recv_counts)
        allocate(send_starts, source = recv_counts)
        call MPI_Allgather(recv%counts, int(recv%rank, int32), MPI_INTEGER4, recv_counts, int(recv%rank, int32), MPI_INTEGER4, comm, ierr)
        call MPI_Allgather(send%counts, int(send%rank, int32), MPI_INTEGER4, send_counts, int(send%rank, int32), MPI_INTEGER4, comm, ierr)
        call MPI_Allgather(recv%starts, int(recv%rank, int32), MPI_INTEGER4, recv_starts, int(send%rank, int32), MPI_INTEGER4, comm, ierr)
        call MPI_Allgather(send%starts, int(send%rank, int32), MPI_INTEGER4, send_starts, int(send%rank, int32), MPI_INTEGER4, comm, ierr)

        transpose_type = kwargs%helper%transpose_type
        reshape_type = kwargs%helper%reshape_type
        tmode = kwargs%transpose_mode
        base_type = kwargs%base_type
        base_storage = kwargs%base_storage

        if ( .not. self%is_transpose .and. send%rank == 3 ) then
            zslab = .true.
            yslab = .true.
            if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                do i = 1, n_neighbors
                    zslab = zslab .and. send%counts(2) == recv_counts(2, i)
                    yslab = yslab .and. send%counts(3) == recv_counts(3, i)
                enddo
            else
                do i = 1, n_neighbors
                    zslab = zslab .and. send_counts(2, i) == recv%counts(2)
                    yslab = yslab .and. send_counts(3, i) == recv%counts(3)
                enddo
            endif
            if( zslab ) then
                reshape_strat = 1
            else if ( yslab ) then
                reshape_strat = 2
            else
                reshape_strat = 3
            endif
        endif

        self%min_buffer_size = get_float_buffer_size(send, recv, base_storage)

        recv_displ = 0
        do i = 1, n_neighbors
            if ( self%is_transpose ) then
                if ( send%rank == 2 ) then
                    call create_transpose_2d(send, send_counts(:,i), recv, recv_counts(:,i), tmode, base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                else if ( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z]) ) then
                    call create_forw_permutation(send, send_counts(:,i), recv, recv_counts(:,i), tmode, base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                else if ( any( transpose_type == [DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Z_TO_Y]) ) then
                    call create_back_permutation(send, send_counts(:,i), recv, recv_counts(:,i), tmode, base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                else if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
                    call create_transpose_XZ(send, send_counts(:,i), recv, recv_counts(:,i), tmode, base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                else
                    call create_transpose_ZX(send, send_counts(:,i), recv, recv_counts(:,i), tmode, base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                endif
            else
                if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                    if ( send%rank == 2 ) then
                        call create_reshape_21(send, send_counts(:,i), recv, recv_counts(:,i), base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                    else
                        call create_reshape_32(send, send_counts(:,i), recv, recv_counts(:,i), recv_starts(:,i), base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ, reshape_strat)
                    endif
                else
                    if ( send%rank == 2 ) then
                        call create_reshape_12(send, send_counts(:,i), recv, recv_counts(:,i), base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ)
                    else
                        call create_reshape_23(send, send_counts(:,i), send_starts(:,i), recv, recv_counts(:,i), base_type, base_storage, self%send%dtypes(i), send_displ, self%recv%dtypes(i), recv_displ, reshape_strat)
                    endif
                endif
            endif
            if ( self%is_even ) then
                self%send%displs(i) = send_displ
                self%recv%displs(i) = recv_displ
            else
                if ( i < n_neighbors ) then
                    if ( any(send%counts == 0) ) then
                        self%send%displs(i + 1) = self%send%displs(i)
                    else
                        self%send%displs(i + 1) = self%send%displs(i) + send_displ
                    endif
                    if (  any(recv%counts == 0)) then
                        self%recv%displs(i + 1) = self%recv%displs(i)
                    else
                        self%recv%displs(i + 1) = self%recv%displs(i) + recv_displ
                    endif
                    recv_displ = self%recv%displs(i + 1)
                endif
            endif
        enddo

        if ( IS_P2P_ENABLED ) then
            self%send%displs(:) = self%send%displs(:) / int(FLOAT_STORAGE_SIZE, int32)
            self%recv%displs(:) = self%recv%displs(:) / int(FLOAT_STORAGE_SIZE, int32)
            if ( .not. self%is_even ) then
                if ( any(send%counts == 0) ) self%send%counts(:) = 0
                if ( any(recv%counts == 0) ) self%recv%counts(:) = 0
                do i = 1, n_neighbors
                    if ( any(recv_counts(:,i) == 0) ) then
                        self%send%counts(i) = 0
                    endif
                    if ( any(send_counts(:,i) == 0) ) then
                        self%recv%counts(i) = 0
                    endif
                enddo
            endif
        endif
        deallocate(recv_counts, send_counts, recv_starts, send_starts)
    end subroutine create

    elemental type(dtfft_backend_t) function get_backend(self)
        class(reshape_handle_datatype),   intent(in)    :: self       !! Abstract reshape Handle
        get_backend = DTFFT_BACKEND_MPI_DATATYPE
    end function get_backend

    subroutine execute(self, in, out, kwargs, error_code)
    !! Executes transposition/reshaping
        class(reshape_handle_datatype), intent(inout)   :: self         !! Datatype handle
        type(c_ptr),                    intent(in)      :: in           !! Send pointer
        type(c_ptr),                    intent(in)      :: out          !! Recv pointer
        type(execute_args),             intent(inout)   :: kwargs       !! Additional arguments
        integer(int32),                 intent(out)     :: error_code   !! Result of execution
        integer(int32) :: i, comm_size, ierr
        real(real32), pointer, contiguous :: pin(:), pout(:)

        call c_f_pointer(in, pin, [self%min_buffer_size])
        call c_f_pointer(out, pout, [self%min_buffer_size])


        error_code = DTFFT_SUCCESS
        if ( self%is_active ) then
            error_code = DTFFT_ERROR_TRANSPOSE_ACTIVE
            return
        endif

        call MPI_Comm_size(self%comm, comm_size, ierr)
#if defined (ENABLE_PERSISTENT_COMM)
        if ( .not. self%is_request_created ) then
# if defined(ENABLE_PERSISTENT_COLLECTIVES)
            if ( self%is_even ) then
                self%n_requests = 1
                call MPI_Alltoall_init(pin, 1, self%send%dtypes(1), pout, 1, self%recv%dtypes(1), self%comm, MPI_INFO_NULL, self%requests(1), ierr)
            else
#   if defined(OMPI_FIX_REQUIRED)
                self%n_requests = 0
                do i = 1, comm_size
                    if ( self%recv%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Recv_init(pout(self%recv%displs(i)), 1, self%recv%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo

                do i = 1, comm_size
                    if ( self%send%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Send_init(pin(self%send%displs(i)), 1, self%send%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo
#   else
                self%n_requests = 1
                call MPI_Alltoallw_init(pin, self%send%counts, self%send%displs, self%send%dtypes, pout, self%recv%counts, self%recv%displs, self%recv%dtypes, self%comm, MPI_INFO_NULL, self%requests(1), ierr)
            endif
#   endif
# else
            self%n_requests = 0
            if ( self%is_even ) then
                do i = 1, comm_size
                    self%n_requests = self%n_requests + 1
                    call MPI_Recv_init(pout((i - 1) * self%recv%displs(1) + 1), 1, self%recv%dtypes(1), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                enddo

                do i = 1, comm_size
                    self%n_requests = self%n_requests + 1
                    call MPI_Send_init(pin((i - 1) * self%send%displs(1) + 1), 1, self%send%dtypes(1), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                enddo
            else
                do i = 1, comm_size
                    if ( self%recv%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Recv_init(pout(self%recv%displs(i) + 1), 1, self%recv%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo

                do i = 1, comm_size
                    if ( self%send%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Send_init(pin(self%send%displs(i) + 1), 1, self%send%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo
            endif
# endif
            self%is_request_created = .true.
        endif

        call MPI_Startall(self%n_requests, self%requests, ierr)
#else
        if ( self%is_even ) then
            self%n_requests = 1
            call MPI_Ialltoall(pin, 1, self%send%dtypes(1), pout, 1, self%recv%dtypes(1), self%comm, self%requests(1), ierr)
        else
# if defined(OMPI_FIX_REQUIRED)
            block
                integer(int32) :: i, comm_size
                call MPI_Comm_size(self%comm, comm_size, ierr)
                self%n_requests = 0

                do i = 1, comm_size
                    if ( self%recv%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Irecv(pout(self%recv%displs(i) + 1), 1, self%recv%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo
                do i = 1, comm_size
                    if ( self%send%counts(i) > 0 ) then
                        self%n_requests = self%n_requests + 1
                        call MPI_Isend(pin(self%send%displs(i) + 1), 1, self%send%dtypes(i), i - 1, 0_int32, self%comm, self%requests(self%n_requests), ierr)
                    endif
                enddo
            endblock
# else
            self%n_requests = 1
            call MPI_Ialltoallw(pin, self%send%counts, self%send%displs, self%send%dtypes, pout, self%recv%counts, self%recv%displs, self%recv%dtypes, self%comm, self%requests(1), ierr)
# endif
        endif
#endif
        self%is_active = .true.
        if ( kwargs%exec_type == EXEC_BLOCKING ) call self%execute_end(kwargs, error_code)
    end subroutine execute

    subroutine execute_end(self, kwargs, error_code)
    !! Ends execution of transposition/reshaping
        class(reshape_handle_datatype),   intent(inout) :: self       !! Datatype handle
        type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
        integer(int32),                   intent(out)   :: error_code !! Error code
        integer(int32)  :: ierr         !! Error code

        error_code = DTFFT_SUCCESS
        if ( .not. self%is_active ) then
            error_code = DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE
            return
        endif
        call MPI_Waitall(self%n_requests, self%requests, MPI_STATUSES_IGNORE, ierr)
        self%is_active = .false.
    end subroutine execute_end

    elemental logical function get_async_active(self)
    !! Returns if async transpose/reshape is active
        class(reshape_handle_datatype),   intent(in)    :: self         !! Datatype handle
        get_async_active = self%is_active
    end function get_async_active

    subroutine destroy(self)
    !! Destroys `reshape_handle_datatype` class
        class(reshape_handle_datatype),   intent(inout) :: self         !! Datatype handle

        call self%send%destroy()
        call self%recv%destroy()
#if defined(ENABLE_PERSISTENT_COLLECTIVES)
        block
        integer(int32) :: i, ierr
            if( self%is_request_created ) then
                do i = 1, self%n_requests
                call MPI_Request_free(self%requests(i), ierr)
                enddo
                self%is_request_created = .false.
            endif
        endblock
#endif
        if( allocated(self%requests) ) deallocate( self%requests )
        self%is_active = .false.
        self%is_even = .false.
    end subroutine destroy

    subroutine create_transpose_2d(send, send_counts, recv, recv_counts, transpose_mode, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates two-dimensional transposition datatypes
        class(pencil),                  intent(in)    :: send               !! Information about send buffer
        integer(int32),                 intent(in)    :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                  intent(in)    :: recv               !! Information about send buffer
        integer(int32),                 intent(in)    :: recv_counts(:)     !! Rank i is recieving this counts
        type(dtfft_transpose_mode_t),   intent(in)    :: transpose_mode     !! Transpose mode to use
        TYPE_MPI_DATATYPE,              intent(in)    :: base_type          !! Base MPI_Datatype
        integer(int64),                 intent(in)    :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,              intent(out)   :: send_dtype         !! Datatype used to send data
        integer(int32),                 intent(out)   :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,              intent(out)   :: recv_dtype         !! Datatype used to recv data
        integer(int32),                 intent(out)   :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1              !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp2              !! Temporary datatype
        integer(int32)      :: ierr               !! Error code

        send_displ = recv_counts(2) * int(base_storage, int32)
        recv_displ = send_counts(2) * int(base_storage, int32)
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK ) then
            call MPI_Type_vector(send%counts(2), recv_counts(2), send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)

            call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(send_counts(2), temp2, recv_dtype, ierr)
            call free_datatypes(temp1, temp2)
        else
            call MPI_Type_vector(send%counts(2), 1, send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv_counts(2), temp2, send_dtype, ierr)
            call free_datatypes(temp1, temp2)

            call MPI_Type_vector(recv%counts(2), send_counts(2), recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
        endif

        call MPI_Type_commit(send_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_transpose_2d

    subroutine create_forw_permutation(send, send_counts, recv, recv_counts, transpose_mode, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates three-dimensional X --> Y and Y -> Z transposition datatypes
        class(pencil),                  intent(in)  :: send               !! Information about send buffer
        integer(int32),                 intent(in)  :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                  intent(in)  :: recv               !! Information about send buffer
        integer(int32),                 intent(in)  :: recv_counts(:)     !! Rank i is recieving this counts
        type(dtfft_transpose_mode_t),   intent(in)  :: transpose_mode     !! Transpose mode to use
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type          !! Base MPI_Datatype
        integer(int64),                 intent(in)  :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,              intent(out) :: send_dtype         !! Datatype used to send data
        integer(int32),                 intent(out) :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,              intent(out) :: recv_dtype         !! Datatype used to recv data
        integer(int32),                 intent(out) :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp2                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp3                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp4                !! Temporary datatype
        integer(int32)      :: ierr                 !! Error code

        send_displ = recv_counts(3) * int(base_storage, int32)
        recv_displ = send_counts(2) * int(base_storage, int32)
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK ) then
            call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(3), send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)

            call MPI_Type_vector(recv%counts(3), 1, recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(send_counts(2), temp2, temp3, ierr)
            call MPI_Type_create_hvector(recv%counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
            call MPI_Type_create_resized(temp4, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3, temp4)
        else
            call MPI_Type_vector(send%counts(2) * send%counts(3), 1, send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv_counts(3), temp2, send_dtype, ierr)
            call free_datatypes(temp1, temp2)

            call MPI_Type_vector(recv%counts(2), send_counts(2), recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
            call MPI_Type_create_resized(temp3, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3)
        endif

        call MPI_Type_commit(send_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_forw_permutation

    subroutine create_back_permutation(send, send_counts, recv, recv_counts, transpose_mode, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates three-dimensional Y --> X and Z --> Y transposition datatypes
        class(pencil),                  intent(in)  :: send               !! Information about send buffer
        integer(int32),                 intent(in)  :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                  intent(in)  :: recv               !! Information about send buffer
        integer(int32),                 intent(in)  :: recv_counts(:)     !! Rank i is recieving this counts
        type(dtfft_transpose_mode_t),   intent(in)  :: transpose_mode     !! Transpose mode to use
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type          !! Base MPI_Datatype
        integer(int64),                 intent(in)  :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,              intent(out) :: send_dtype         !! Datatype used to send data
        integer(int32),                 intent(out) :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,              intent(out) :: recv_dtype         !! Datatype used to recv data
        integer(int32),                 intent(out) :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp2                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp3                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp4                !! Temporary datatype
        integer(int32)      :: ierr                 !! Error code

        send_displ = recv_counts(2) * int(base_storage, int32)
        recv_displ = send_counts(3) * int(base_storage, int32)
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK ) then
            call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(2), send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)

            call MPI_Type_vector(recv%counts(2) * recv%counts(3), 1, recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(send_counts(3), temp2, temp3, ierr)
            call MPI_Type_create_resized(temp3, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3)
        else
            call MPI_Type_vector(send%counts(3), 1, send%counts(1) * send%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv_counts(2), temp2, temp3, ierr)
            call MPI_Type_create_hvector(send%counts(2), 1, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
            call MPI_Type_create_resized(temp4, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3, temp4)

            call MPI_Type_vector(recv%counts(2) * recv%counts(3), send_counts(3), recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
        endif

        call MPI_Type_commit(send_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_back_permutation

    subroutine create_transpose_XZ(send, send_counts, recv, recv_counts, transpose_mode, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates three-dimensional X --> Z transposition datatypes
    !! Can only be used with 3D slab decomposition when slabs are distributed in Z direction
        class(pencil),                  intent(in)  :: send               !! Information about send buffer
        integer(int32),                 intent(in)  :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                  intent(in)  :: recv               !! Information about send buffer
        integer(int32),                 intent(in)  :: recv_counts(:)     !! Rank i is recieving this counts
        type(dtfft_transpose_mode_t),   intent(in)  :: transpose_mode     !! Transpose mode to use
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type          !! Base MPI_Datatype
        integer(int64),                 intent(in)  :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,              intent(out) :: send_dtype         !! Datatype used to send data
        integer(int32),                 intent(out) :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,              intent(out) :: recv_dtype         !! Datatype used to recv data
        integer(int32),                 intent(out) :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp2                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp3                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp4                !! Temporary datatype
        integer(int32)      :: ierr                 !! Error code

        send_displ = send%counts(1) * recv_counts(3) * int(base_storage, int32)
        recv_displ = send_counts(3) * int(base_storage, int32)
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK ) then
            call MPI_Type_vector(send%counts(3), send%counts(1), send%counts(1) * send%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv_counts(3), temp2, send_dtype, ierr)
            call free_datatypes(temp1, temp2)

            call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(send_counts(3), temp2, temp3, ierr)
            call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
            call MPI_Type_create_resized(temp4, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3, temp4)
        else
            call MPI_Type_vector(send%counts(3), 1, send%counts(1) * send%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(send%counts(1), temp2, temp3, ierr)
            call MPI_Type_create_hvector(recv_counts(3), 1, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
            call MPI_Type_create_resized(temp4, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3, temp4)

            call MPI_Type_vector(recv%counts(2) * recv%counts(3), send_counts(3), recv%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
        endif

        call MPI_Type_commit(send_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_transpose_XZ

    subroutine create_transpose_ZX(send, send_counts, recv, recv_counts, transpose_mode, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates three-dimensional Z --> X transposition datatypes
    !! Can only be used with 3D slab decomposition when slabs are distributed in Z direction
        class(pencil),                  intent(in)  :: send               !! Information about send buffer
        integer(int32),                 intent(in)  :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                  intent(in)  :: recv               !! Information about send buffer
        integer(int32),                 intent(in)  :: recv_counts(:)     !! Rank i is recieving this counts
        type(dtfft_transpose_mode_t),   intent(in)  :: transpose_mode     !! Transpose mode to use
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type          !! Base MPI_Datatype
        integer(int64),                 intent(in)  :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,              intent(out) :: send_dtype         !! Datatype used to send data
        integer(int32),                 intent(out) :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,              intent(out) :: recv_dtype         !! Datatype used to recv data
        integer(int32),                 intent(out) :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp2                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp3                !! Temporary datatype
        TYPE_MPI_DATATYPE   :: temp4                !! Temporary datatype
        integer(int32)      :: ierr                 !! Error code

        send_displ = recv_counts(3) * int(base_storage, int32)
        recv_displ = recv%counts(1) * send_counts(3) * int(base_storage, int32)
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK ) then
            call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(3), send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)

            call MPI_Type_vector(recv%counts(3), 1, recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv%counts(1), temp2, temp3, ierr)
            call MPI_Type_create_hvector(send_counts(3), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
            call MPI_Type_create_resized(temp4, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1, temp2, temp3, temp4)
        else
            call MPI_Type_vector(send%counts(2) * send%counts(3), 1, send%counts(1), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
            call MPI_Type_contiguous(recv_counts(3), temp2, send_dtype, ierr)
            call free_datatypes(temp1, temp2)

            call MPI_Type_vector(recv%counts(3), recv%counts(1) * send_counts(3), recv%counts(1) * recv%counts(2),  base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
        endif

        call MPI_Type_commit(send_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_transpose_ZX

    subroutine create_reshape_32(send, send_counts, recv, recv_counts, recv_starts, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ, reshape_strat)
    !! Creates reshape datatypes from 3d bricks to 2d pencils
        class(pencil),                intent(in)    :: send               !! Information about send buffer
        integer(int32),               intent(in)    :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                intent(in)    :: recv               !! Information about send buffer
        integer(int32),               intent(in)    :: recv_counts(:)     !! Rank i is recieving this counts
        integer(int32),               intent(in)    :: recv_starts(:)     !! Rank i is recieving to this starts
        TYPE_MPI_DATATYPE,            intent(in)    :: base_type          !! Base MPI_Datatype
        integer(int64),               intent(in)    :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,            intent(out)   :: send_dtype         !! Datatype used to send data
        integer(int32),               intent(out)   :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,            intent(out)   :: recv_dtype         !! Datatype used to recv data
        integer(int32),               intent(out)   :: recv_displ         !! Recv displacement in bytes
        integer(int8),                intent(in)    :: reshape_strat
        TYPE_MPI_DATATYPE   :: temp1
        integer(int32)      :: i, j, k, dsp, dsp2, count, ierr                 !! Error code
        integer(int32), allocatable :: displs(:)

        select case ( reshape_strat )
        case ( 1_int8 )
        ! Z - slab
            send_displ = send%counts(1) * send%counts(2) * recv_counts(3) * int(base_storage, int32)
            call MPI_Type_contiguous(send%counts(1) * send%counts(2) * recv_counts(3), base_type, send_dtype, ierr)
        case ( 2_int8 )
        ! Y - slab
            send_displ = send%counts(1) * recv_counts(2) * int(base_storage, int32)
            call MPI_Type_vector(recv_counts(3), send%counts(1) * recv_counts(2), send%counts(1) * send%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)
        case default
        ! Pencil
            send_displ = 0 
            ! send%counts(1) * recv_counts(2) * recv_counts(3) * int(base_storage, int32)
            count = recv_counts(2) * recv_counts(3)
            allocate(displs(count))
            i = 1
            dsp = (recv_starts(3) - send%starts(3)) * send%counts(1) * send%counts(2) + (recv_starts(2) - send%starts(2)) * send%counts(1)
            dsp2 = dsp
            do k = 1, recv_counts(3)
                do j = 1, recv_counts(2)
                    displs(i) = dsp
                    i = i + 1
                    dsp = dsp + send%counts(1)
                enddo
                dsp = dsp2 + k * send%counts(1) * send%counts(2)
            enddo
            call MPI_Type_create_indexed_block(count, send%counts(1), displs, base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(0, MPI_ADDRESS_KIND), send_dtype, ierr)
            call free_datatypes(temp1)
            deallocate(displs)
        endselect
        call MPI_Type_commit(send_dtype, ierr)
        if ( any(recv_counts== 0) ) send_displ = 0

        recv_displ = send_counts(1) * int(base_storage, int32)
        call MPI_Type_vector(recv%counts(2) * recv%counts(3), send_counts(1), recv%counts(1), base_type, temp1, ierr)
        call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
        if ( any(send_counts == 0) ) recv_displ = 0
        call free_datatypes(temp1)
    end subroutine create_reshape_32

    subroutine create_reshape_23(send, send_counts, send_starts, recv, recv_counts, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ, reshape_strat)
    !! Creates reshape datatypes from 2d pencils to 3d bricks
        class(pencil),                intent(in)    :: send               !! Information about send buffer
        integer(int32),               intent(in)    :: send_counts(:)     !! Rank i is sending this counts
        integer(int32),               intent(in)    :: send_starts(:)     !! Rank i is recieving to this starts
        class(pencil),                intent(in)    :: recv               !! Information about send buffer
        integer(int32),               intent(in)    :: recv_counts(:)     !! Rank i is recieving this counts
        TYPE_MPI_DATATYPE,            intent(in)    :: base_type          !! Base MPI_Datatype
        integer(int64),               intent(in)    :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,            intent(out)   :: send_dtype         !! Datatype used to send data
        integer(int32),               intent(out)   :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,            intent(out)   :: recv_dtype         !! Datatype used to recv data
        integer(int32),               intent(inout) :: recv_displ         !! Recv displacement in bytes
        integer(int8),                intent(in)    :: reshape_strat
        TYPE_MPI_DATATYPE   :: temp1
        integer(int32)      :: i, j, k, dsp, dsp2, count, ierr                 !! Error code
        integer(int32), allocatable :: displs(:)

        send_displ = recv_counts(1) * int(base_storage, int32)
        call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(1), send%counts(1), base_type, temp1, ierr)
        call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
        call free_datatypes(temp1)
        call MPI_Type_commit(send_dtype, ierr)

        select case ( reshape_strat )
        case ( 1_int8 )
            recv_displ = recv%counts(1) * recv%counts(2) * send_counts(3) * int(base_storage, int32)
            call MPI_Type_contiguous(recv%counts(1) * recv%counts(2) * send_counts(3), base_type, recv_dtype, ierr)
        case ( 2_int8 )
            recv_displ = recv%counts(1) * send_counts(2) * int(base_storage, int32)
            call MPI_Type_vector(send_counts(3), recv%counts(1) * send_counts(2), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
        case default
        ! Pencil
            count = send_counts(2) * send_counts(3)
            allocate(displs(count))
            i = 1
            dsp =  abs(recv%starts(3) - send_starts(3))  * recv%counts(1) * recv%counts(2) + abs(recv%starts(2) - send_starts(2)) * recv%counts(1) - recv_displ / int(base_storage, int32)
            dsp2 = dsp
            do k = 1, send_counts(3)
                do j = 1, send_counts(2)
                    displs(i) = dsp
                    i = i + 1
                    dsp = dsp + recv%counts(1)
                enddo
                dsp = dsp2 + k * recv%counts(1) * recv%counts(2)
            enddo
            call MPI_Type_create_indexed_block(count, recv%counts(1), displs, base_type, temp1, ierr)
            call MPI_Type_create_resized(temp1, LB, int(0, MPI_ADDRESS_KIND), recv_dtype, ierr)
            call free_datatypes(temp1)
            deallocate(displs)
            recv_displ = recv%counts(1) * send_counts(2) * send_counts(3) * int(base_storage, int32)
            ! recv_displ = recv%counts(1) * send_counts(2) * send_counts(3) * int(base_storage, int32)
            ! call MPI_Type_contiguous(recv%counts(1) * recv%counts(2) * send_counts(3), base_type, recv_dtype, ierr)
        endselect
        call MPI_Type_commit(recv_dtype, ierr)
        if ( any(recv%counts == 0) ) send_displ = 0
    end subroutine create_reshape_23

    subroutine create_reshape_21(send, send_counts, recv, recv_counts, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates reshape datatypes for 2D data: from 2d bricks to 1d slabs
        class(pencil),                intent(in)    :: send               !! Information about send buffer
        integer(int32),               intent(in)    :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                intent(in)    :: recv               !! Information about send buffer
        integer(int32),               intent(in)    :: recv_counts(:)     !! Rank i is recieving this counts
        TYPE_MPI_DATATYPE,            intent(in)    :: base_type          !! Base MPI_Datatype
        integer(int64),               intent(in)    :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,            intent(out)   :: send_dtype         !! Datatype used to send data
        integer(int32),               intent(out)   :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,            intent(out)   :: recv_dtype         !! Datatype used to recv data
        integer(int32),               intent(out)   :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1
        integer(int32)      :: ierr                 !! Error code

        send_displ = send%counts(1) * recv_counts(2) * int(base_storage, int32)
        call MPI_Type_contiguous(send%counts(1) * recv_counts(2), base_type, send_dtype, ierr)
        call MPI_Type_commit(send_dtype, ierr)

        recv_displ = send_counts(1) * int(base_storage, int32)
        call MPI_Type_vector(recv%counts(2), send_counts(1), recv%counts(1), base_type, temp1, ierr)
        call MPI_Type_create_resized(temp1, LB, int(recv_displ, MPI_ADDRESS_KIND), recv_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
        call free_datatypes(temp1)
    end subroutine create_reshape_21

    subroutine create_reshape_12(send, send_counts, recv, recv_counts, base_type, base_storage, send_dtype, send_displ, recv_dtype, recv_displ)
    !! Creates reshape datatypes for 2D data: from 1d slabs to 2d bricks
        class(pencil),                intent(in)    :: send               !! Information about send buffer
        integer(int32),               intent(in)    :: send_counts(:)     !! Rank i is sending this counts
        class(pencil),                intent(in)    :: recv               !! Information about send buffer
        integer(int32),               intent(in)    :: recv_counts(:)     !! Rank i is recieving this counts
        TYPE_MPI_DATATYPE,            intent(in)    :: base_type          !! Base MPI_Datatype
        integer(int64),               intent(in)    :: base_storage       !! Number of bytes needed to store single element
        TYPE_MPI_DATATYPE,            intent(out)   :: send_dtype         !! Datatype used to send data
        integer(int32),               intent(out)   :: send_displ         !! Send displacement in bytes
        TYPE_MPI_DATATYPE,            intent(out)   :: recv_dtype         !! Datatype used to recv data
        integer(int32),               intent(out)   :: recv_displ         !! Recv displacement in bytes
        TYPE_MPI_DATATYPE   :: temp1
        integer(int32)      :: ierr                 !! Error code

        send_displ = recv_counts(1) * int(base_storage, int32)
        call MPI_Type_vector(send%counts(2), recv_counts(1), send%counts(1), base_type, temp1, ierr)
        call MPI_Type_create_resized(temp1, LB, int(send_displ, MPI_ADDRESS_KIND), send_dtype, ierr)
        call MPI_Type_commit(send_dtype, ierr)
        call free_datatypes(temp1)

        recv_displ = recv%counts(1) * send_counts(2) * int(base_storage, int32)
        call MPI_Type_contiguous(recv%counts(1) * send_counts(2), base_type, recv_dtype, ierr)
        call MPI_Type_commit(recv_dtype, ierr)
    end subroutine create_reshape_12

    subroutine free_datatypes(t1, t2, t3, t4)
    !! Frees temporary datatypes
        TYPE_MPI_DATATYPE,  intent(inout), optional :: t1     !! Temporary datatype
        TYPE_MPI_DATATYPE,  intent(inout), optional :: t2     !! Temporary datatype
        TYPE_MPI_DATATYPE,  intent(inout), optional :: t3     !! Temporary datatype
        TYPE_MPI_DATATYPE,  intent(inout), optional :: t4     !! Temporary datatype
        integer(int32)                              :: ierr   !! Error code

        if ( present(t1) ) call MPI_Type_free(t1, ierr)
        if ( present(t2) ) call MPI_Type_free(t2, ierr)
        if ( present(t3) ) call MPI_Type_free(t3, ierr)
        if ( present(t4) ) call MPI_Type_free(t4, ierr)
    end subroutine free_datatypes
end module dtfft_reshape_handle_datatype
