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
module dtfft_backend_mpi_p2p_pipelined
!! This module defines MPI P2P Pipelined backend: `backend_mpi_p2p_pipelined`
use iso_fortran_env
use cudafor
use dtfft_abstract_backend,           only: backend_helper
use dtfft_abstract_backend_selfcopy,  only: abstract_backend_pipelined
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: backend_mpi_p2p_pipelined


  type, extends(abstract_backend_pipelined) :: backend_mpi_p2p_pipelined
  !! MPI P2P Pipelined backend
  private
    TYPE_MPI_REQUEST, allocatable :: send_requests(:)
    TYPE_MPI_REQUEST, allocatable :: recv_requests(:)
    integer(int32)                :: n_send_requests, n_recv_requests
    ! TYPE_MPI_COMM                 :: comm
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    logical                       :: is_request_created
#endif
  contains
    procedure         :: create_selfcopy => create    !< Creates MPI Pipelined backend
    procedure         :: execute_private => execute   !< Executes MPI Pipelined backend
    procedure         :: destroy_selfcopy => destroy  !< Destroys MPI Pipelined backend
  end type backend_mpi_p2p_pipelined

contains

  subroutine create(self, helper)
  !! Creates MPI Pipelined backend
    class(backend_mpi_p2p_pipelined), intent(inout) :: self
    type(backend_helper),             intent(in)    :: helper           !< MPI communicator
    ! integer(int32) :: mpi_ierr

    ! call MPI_Comm_dup(comm, self%comm, mpi_ierr)
    allocate( self%send_requests(self%comm_size - 1) )
    allocate( self%recv_requests(self%comm_size - 1) )

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    self%is_request_created = .false.
#endif
  end subroutine create

  subroutine execute(self, in, out, stream)
  !! Executes MPI Pipelined backend
    class(backend_mpi_p2p_pipelined),     intent(inout) :: self       !< MPI P2P Pipelined backend
    type(c_devptr),                       intent(in)    :: in         !< Send pointer
    type(c_devptr),                       intent(in)    :: out        !< Recv pointer
    integer(cuda_stream_kind),            intent(in)    :: stream     !< Main execution CUDA stream
    integer(int32)                                :: i                    !< Counter
    integer(int32)                                :: request_counter      !< MPI Requests counter
    integer(int32)                                :: mpi_ierr             !< MPI Error code
    logical,                          allocatable :: is_complete_comm(:)  !< Testing for request completion
    real(real32), DEVICE_PTR pointer, contiguous  :: pin(:)               !< Pointer to in
    real(real32), DEVICE_PTR pointer, contiguous  :: paux(:)              !< Pointer to aux

    ! Need to sync stream since there is no way pass current stream to MPI
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])
    call c_f_pointer(self%aux, paux, [self%send_recv_buffer_size])

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    if ( .not. self%is_request_created ) then
      request_counter = 0
      do i = 0, self%comm_size - 1
        if ( self%recv_floats(i) > 0 ) then
          request_counter = request_counter + 1
          call MPI_Recv_init(pin( self%recv_displs(i) ), int(self%recv_floats(i), CNT_KIND), MPI_REAL4, i, 0,   &
                             self%comm, self%recv_requests(request_counter), mpi_ierr)
        endif
      enddo
      self%n_recv_requests = request_counter

      request_counter = 0
      do i = 0, self%comm_size - 1
        if ( self%send_floats(i) > 0 ) then
          request_counter = request_counter + 1
          call MPI_Send_init(paux( self%send_displs(i) ), int(self%send_floats(i), CNT_KIND), MPI_REAL4, i, 0,  &
                             self%comm, self%send_requests(request_counter), mpi_ierr)
        endif
      enddo
      self%n_send_requests = request_counter

      self%is_request_created = .true.
    endif

    call MPI_Startall(self%n_recv_requests, self%recv_requests, mpi_ierr)
    call MPI_Startall(self%n_send_requests, self%send_requests, mpi_ierr)
#else
    request_counter = 0
    do i = 0, self%comm_size - 1
      if ( self%recv_floats(i) > 0 ) then
        request_counter = request_counter + 1
        call MPI_Irecv(pin( self%recv_displs(i) ), int(self%recv_floats(i), CNT_KIND), MPI_REAL4, i, 0,         &
                       self%comm, self%recv_requests(request_counter), mpi_ierr)
      endif
    enddo
    self%n_recv_requests = request_counter

    request_counter = 0
    do i = 0, self%comm_size - 1
      if ( self%send_floats(i) > 0 ) then
        request_counter = request_counter + 1
        call MPI_Isend(paux( self%send_displs(i) ), int(self%send_floats(i), CNT_KIND), MPI_REAL4, i, 0,        &
                       self%comm, self%send_requests(request_counter), mpi_ierr)
      endif
    enddo
    self%n_send_requests = request_counter
#endif

    allocate( is_complete_comm(self%n_recv_requests), source=.false. )
    do while (.true.)
      ! Testing that all data has been recieved so we can unpack it
      request_counter = 0
      do i = 0, self%comm_size - 1
        if ( self%recv_floats(i) == 0 ) cycle

        request_counter = request_counter + 1

        if ( is_complete_comm( request_counter ) ) cycle
        call MPI_Test(self%recv_requests(request_counter), is_complete_comm( request_counter ), MPI_STATUS_IGNORE, mpi_ierr)
        if ( is_complete_comm( request_counter ) ) then
          call self%unpack_kernel%execute(in, out, stream, i + 1)
        endif
      enddo
      if ( all( is_complete_comm ) ) exit
    enddo
    call MPI_Waitall(self%n_send_requests, self%send_requests, MPI_STATUSES_IGNORE, mpi_ierr)

    deallocate( is_complete_comm )
  end subroutine execute

  subroutine destroy(self)
  !! Destroys MPI Pipelined backend
    class(backend_mpi_p2p_pipelined),   intent(inout) :: self
    integer(int32) :: i, mpi_ierr

    call MPI_Comm_free(self%comm, mpi_ierr)
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    do i = 1, self%n_recv_requests
      call MPI_Request_free(self%recv_requests(i), mpi_ierr)
    enddo
    do i = 1, self%n_send_requests
      call MPI_Request_free(self%send_requests(i), mpi_ierr)
    enddo

    self%is_request_created = .false.
#endif
    deallocate( self%recv_requests, self%send_requests )
  end subroutine destroy
end module dtfft_backend_mpi_p2p_pipelined