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
module dtfft_backend_mpi
!! MPI Based GPU Backends [[backend_mpi]]
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend
use dtfft_interface_cuda_runtime
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: backend_mpi

  type :: mpi_backend_helper
  !! MPI Helper
    integer(CNT_KIND),  allocatable :: counts(:)    !! Counts of data to send or recv
    integer(ADDR_KIND), allocatable :: displs(:)    !! Displacements of data to send or recv
    TYPE_MPI_REQUEST,   allocatable :: requests(:)  !! MPI Requests
    integer(int32)                  :: n_requests   !! Number of requests
#ifdef ENABLE_PERSISTENT_COMM
    logical                         :: is_request_created = .false. !! Request created flag. Used for persistent functions
#endif
  contains
    procedure,  pass(self) :: create => create_helper   !! Creates MPI helper
    procedure,  pass(self) :: destroy => destoy_helper  !! Destroys MPI helper
  endtype mpi_backend_helper


  type, extends(abstract_backend) :: backend_mpi
  !! MPI Backend
  private
    type(mpi_backend_helper)        :: send     !! MPI Helper for send data
    type(mpi_backend_helper)        :: recv     !! MPI Helper for recv data
  contains
    procedure :: create_private => create_mpi   !! Creates MPI backend
    procedure :: execute_private => execute_mpi !! Executes MPI backend
    procedure :: destroy_private => destroy_mpi !! Destroys MPI backend
  end type backend_mpi

contains

  subroutine create_helper(self, counts, displs, max_requests)
  !! Creates MPI helper
    class(mpi_backend_helper),  intent(inout) :: self         !! MPI Helper
    integer(int64),             intent(in)    :: counts(:)    !! Counts of data to send or recv
    integer(int64),             intent(in)    :: displs(:)    !! Displacements of data to send or recv
    integer(int32),             intent(in)    :: max_requests !! Maximum number of requests required
    integer(int32)  :: n_counts

    n_counts = size(counts)
    allocate( self%counts(0:n_counts - 1), self%displs(0:n_counts - 1) )
    self%counts(0:) = int(counts(:), CNT_KIND)
    self%displs(0:) = int(displs(:), ADDR_KIND)
    if ( max_requests > 0 ) then
      allocate( self%requests(max_requests) )
#ifdef ENABLE_PERSISTENT_COMM
      self%is_request_created = .false.
#endif
    endif
  end subroutine create_helper

  subroutine destoy_helper(self)
  !! Destroys MPI helper
    class(mpi_backend_helper),  intent(inout) :: self !! MPI Helper

    if ( allocated(self%counts) ) deallocate( self%counts )
    if ( allocated(self%displs) ) deallocate( self%displs )
#ifdef ENABLE_PERSISTENT_COMM
    block
      integer(int32)  :: mpi_ierr, i

      if ( self%is_request_created ) then
        do i = 1, self%n_requests
          call MPI_Request_free(self%requests(i), mpi_ierr)
        enddo
      endif
      self%is_request_created = .false.
    endblock
#endif
    if ( allocated(self%requests) ) deallocate(self%requests)
    self%n_requests = 0
  end subroutine destoy_helper

  subroutine create_mpi(self, helper, tranpose_type, base_storage)
  !! Creates MPI backend
    class(backend_mpi),       intent(inout) :: self           !! MPI GPU Backend
    type(backend_helper),     intent(in)    :: helper         !! Backend helper (unused)
    type(dtfft_transpose_t),  intent(in)    :: tranpose_type  !! Type of transpose to create (unused)
    integer(int8),            intent(in)    :: base_storage   !! Number of bytes to store single element (unused)

    if ( .not. is_backend_mpi(self%backend) ) INTERNAL_ERROR(".not. is_backend_mpi")

    if ( self%backend == DTFFT_BACKEND_MPI_A2A ) then
      call self%send%create(self%send_floats, self%send_displs - 1, 1)
      call self%recv%create(self%recv_floats, self%recv_displs - 1, 0)
    else
      call self%send%create(self%send_floats, self%send_displs, self%comm_size)
      call self%recv%create(self%recv_floats, self%recv_displs, self%comm_size)
    endif
  end subroutine create_mpi

  subroutine destroy_mpi(self)
  !! Destroys MPI backend
    class(backend_mpi),  intent(inout) :: self  !! MPI GPU Backend

    call self%send%destroy()
    call self%recv%destroy()
  end subroutine destroy_mpi

  subroutine execute_mpi(self, in, out, stream)
  !! Executes MPI backend
    class(backend_mpi),           intent(inout) :: self       !! MPI GPU Backend
    real(real32),   target,       intent(inout) :: in(:)      !! Send pointer
    real(real32),   target,       intent(inout) :: out(:)     !! Recv pointer
    type(dtfft_stream_t),         intent(in)    :: stream     !! Main execution CUDA stream
    integer(int32)          :: mpi_ierr             !! MPI error code
    logical,  allocatable   :: is_complete_comm(:)  !! Testing for request completion
    integer(int32)          :: request_counter      !! Request counter
    integer(int32)          :: i                    !! Loop index

    ! Need to sync stream since there is no way pass current stream to MPI
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )

    select case ( self%backend%val )
    case ( DTFFT_BACKEND_MPI_A2A%val )
      call run_mpi_a2a(self%comm, self%send, self%recv, in, out)
      ! All-to-all request is stored in `send%requests(1)`, so no need to wait for recv requests
    case ( DTFFT_BACKEND_MPI_P2P%val )
      call run_mpi_p2p(self%comm, self%send, self%recv, in, out)
      ! Waiting for all recv requests to finish
      call MPI_Waitall(self%recv%n_requests, self%recv%requests, MPI_STATUSES_IGNORE, mpi_ierr)
    case ( DTFFT_BACKEND_MPI_P2P_PIPELINED%val )
      call run_mpi_p2p(self%comm, self%send, self%recv, self%aux, in)

      allocate( is_complete_comm(self%recv%n_requests), source=.false. )
      do while (.true.)
        ! Testing that all data has been recieved so we can unpack it
        request_counter = 0
        do i = 0, self%comm_size - 1
          if ( self%recv_floats(i) == 0 ) cycle

          request_counter = request_counter + 1

          if ( is_complete_comm( request_counter ) ) cycle
          call MPI_Test(self%recv%requests(request_counter), is_complete_comm( request_counter ), MPI_STATUS_IGNORE, mpi_ierr)
          if ( is_complete_comm( request_counter ) ) then
            call self%unpack_kernel%execute(in, out, stream, i + 1)
          endif
        enddo
        if ( all( is_complete_comm ) ) exit
      enddo
    endselect
    call MPI_Waitall(self%send%n_requests, self%send%requests, MPI_STATUSES_IGNORE, mpi_ierr)
  end subroutine execute_mpi

  subroutine run_mpi_p2p(comm, send, recv, in, out)
  !! Executes MPI point-to-point communication
    TYPE_MPI_COMM,                intent(in)    :: comm   !! MPI communicator
    type(mpi_backend_helper),     intent(inout) :: send   !! MPI Helper for send data
    type(mpi_backend_helper),     intent(inout) :: recv   !! MPI Helper for recv data
    real(real32),                 intent(in)    :: in(:)  !! Data to be sent
    real(real32),                 intent(inout) :: out(:) !! Data to be received
    integer(int32) :: send_request_counter, recv_request_counter
    integer(int32) :: i, comm_size, mpi_ierr

    send_request_counter = 0
    recv_request_counter = 0
    call MPI_Comm_size(comm, comm_size, mpi_ierr)

#ifdef ENABLE_PERSISTENT_COMM
    if ( .not. send%is_request_created ) then
      do i = 0, comm_size - 1
        if ( recv%counts(i) > 0 ) then
          recv_request_counter = recv_request_counter + 1
          call MPI_Recv_init(out( recv%displs(i) ), recv%counts(i), MPI_REAL, i, 0,   &
                             comm, recv%requests(recv_request_counter), mpi_ierr)
        endif
      enddo
      recv%n_requests = recv_request_counter; recv%is_request_created = .true.

      do i = 0, comm_size - 1
        if ( send%counts(i) > 0 ) then
          send_request_counter = send_request_counter + 1
          call MPI_Send_init(in( send%displs(i) ), send%counts(i), MPI_REAL, i, 0,    &
                             comm, send%requests(send_request_counter), mpi_ierr)
        endif
      enddo
      send%n_requests = send_request_counter; send%is_request_created = .true.
    endif

    call MPI_Startall(recv%n_requests, recv%requests, mpi_ierr)
    call MPI_Startall(send%n_requests, send%requests, mpi_ierr)
#else
    do i = 0, comm_size - 1
      if ( recv%counts(i) > 0 ) then
        recv_request_counter = recv_request_counter + 1
        call MPI_Irecv(out( recv%displs(i) ), recv%counts(i), MPI_REAL, i, 0,   &
                       comm, recv%requests(recv_request_counter), mpi_ierr)
      endif
    enddo
    recv%n_requests = recv_request_counter

    do i = 0, comm_size - 1
      if ( send%counts(i) > 0 ) then
        send_request_counter = send_request_counter + 1
        call MPI_Isend(in( send%displs(i) ), send%counts(i), MPI_REAL, i, 0,          &
                       comm, send%requests(send_request_counter), mpi_ierr)
      endif
    enddo
    send%n_requests = send_request_counter
#endif
  end subroutine run_mpi_p2p

  subroutine run_mpi_a2a(comm, send, recv, in, out)
  !! Executes MPI all-to-all communication
    TYPE_MPI_COMM,                intent(in)    :: comm   !! MPI communicator
    type(mpi_backend_helper),     intent(inout) :: send   !! MPI Helper for send data
    type(mpi_backend_helper),     intent(inout) :: recv   !! MPI Helper for recv data
    real(real32),                 intent(in)    :: in(:)  !! Data to be sent
    real(real32),                 intent(inout) :: out(:) !! Data to be received
    integer(int32) :: mpi_ierr

#if defined(ENABLE_PERSISTENT_COLLECTIVES)
    if ( .not. send%is_request_created ) then
      call MPI_Alltoallv_init(in, send%counts, send%displs, MPI_REAL,        &
                              out, recv%counts, recv%displs, MPI_REAL,       &
                              comm, MPI_INFO_NULL, send%requests(1), mpi_ierr)
      send%is_request_created = .true.
    endif
    call MPI_Start(send%requests(1), mpi_ierr)
#else
    call MPI_Ialltoallv(in, send%counts, send%displs, MPI_REAL,        &
                        out, recv%counts, recv%displs, MPI_REAL,       &
                        comm, send%requests(1), mpi_ierr)
#endif
    send%n_requests = 1
  end subroutine run_mpi_a2a
end module dtfft_backend_mpi