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
module dtfft_abstract_gpu_backend
!! This module defines most Abstract GPU Backend: `abstract_gpu_backend`
use iso_fortran_env
use cudafor
use dtfft_parameters, only: FLOAT_STORAGE_SIZE
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: abstract_gpu_backend

  type, abstract :: abstract_gpu_backend
  !! The most Abstract GPU Backend
    type(c_devptr)              :: aux                    !< Auxiliary buffer used in pipelined algorithm
    integer(int64)              :: aux_size               !< Number of bytes required by aux buffer
    integer(int64)              :: send_recv_buffer_size  !< Number of float elements used in ``c_f_pointer``
    integer(int32)              :: comm_size              !< Size of MPI Comm
    integer(int32)              :: comm_rank              !< Rank in MPI Comm
    integer(int64), allocatable :: send_displs(:)         !< Send data displacements, in float elements
    integer(int64), allocatable :: send_floats(:)         !< Send data elements, in float elements
    integer(int64), allocatable :: recv_displs(:)         !< Recv data displacements, in float elements
    integer(int64), allocatable :: recv_floats(:)         !< Recv data elements, in float elements
  contains
    procedure,                                    pass(self)  :: create           !< Creates Abstract GPU Backend
    procedure,                                    pass(self)  :: destroy          !< Destroys Abstract GPU Backend
    procedure,                                    pass(self)  :: get_aux_size     !< Returns number of bytes required by aux buffer
    procedure,                                    pass(self)  :: set_aux          !< Sets Auxiliary buffer
    procedure(createInterface),       deferred,   pass(self)  :: create_private   !< Creates overring class
    procedure(executeInterface),      deferred,   pass(self)  :: execute          !< Executes GPU Backend
    procedure(destroyInterface),      deferred,   pass(self)  :: destroy_private  !< Destroys overring class
  end type abstract_gpu_backend

  interface
    subroutine createInterface(self, comm)
    !! Creates overring class
    import
      class(abstract_gpu_backend),  intent(inout) :: self       !< Abstract GPU Backend
      TYPE_MPI_COMM,                intent(in)    :: comm       !< MPI communicator
    end subroutine createInterface

    subroutine executeInterface(self, in, out, stream)
    !! Executes GPU Backend
    import
      class(abstract_gpu_backend),  intent(inout) :: self       !< Abstract GPU Backend
      type(c_devptr),               intent(in)    :: in         !< Send pointer
      type(c_devptr),               intent(in)    :: out        !< Recv pointer
      integer(cuda_stream_kind),    intent(in)    :: stream     !< Main execution CUDA stream
    end subroutine executeInterface

    subroutine destroyInterface(self)
    !! Destroys overring class
    import
      class(abstract_gpu_backend),  intent(inout) :: self       !< Abstract GPU Backend
    end subroutine destroyInterface
  end interface

contains

  subroutine create(self, comm, send_displs, send_counts, recv_displs, recv_counts, base_storage)
  !! Creates Abstract GPU Backend
    class(abstract_gpu_backend),    intent(inout) :: self           !< Abstract GPU Backend
    TYPE_MPI_COMM,                  intent(in)    :: comm           !< MPI communicator
    integer(int32),                 intent(in)    :: send_displs(:) !< Send data displacements, in original elements
    integer(int32),                 intent(in)    :: send_counts(:) !< Send data elements, in float elements
    integer(int32),                 intent(in)    :: recv_displs(:) !< Recv data displacements, in float elements
    integer(int32),                 intent(in)    :: recv_counts(:) !< Recv data elements, in float elements
    integer(int8),                  intent(in)    :: base_storage   !< Number of bytes to store single element
    integer(int64)                                :: send_size      !< Total number of floats to send
    integer(int64)                                :: recv_size      !< Total number of floats to recv
    integer(int32)                                :: ierr           !< MPI Error code
    integer(int64)                                :: scaler         !< Scaling data amount to float size

    scaler = int(base_storage, int64) / int(FLOAT_STORAGE_SIZE, int64)

    send_size = sum(send_counts) * scaler
    recv_size = sum(recv_counts) * scaler
    self%send_recv_buffer_size = max(send_size, recv_size)

    call MPI_Comm_size(comm, self%comm_size, ierr)
    call MPI_Comm_rank(comm, self%comm_rank, ierr)

    allocate( self%send_displs(0:self%comm_size - 1) )
    allocate( self%send_floats(0:self%comm_size - 1) )
    self%send_displs = int(send_displs, int64) * scaler
    self%send_displs = self%send_displs + 1
    self%send_floats = int(send_counts, int64) * scaler

    self%aux_size = 0_int64

    allocate( self%recv_displs(0:self%comm_size - 1) )
    allocate( self%recv_floats(0:self%comm_size - 1) )
    self%recv_displs = int(recv_displs, int64) * scaler
    self%recv_displs = self%recv_displs + 1
    self%recv_floats = int(recv_counts, int64) * scaler

    call self%create_private(comm)
  end subroutine create

  subroutine destroy(self)
  !! Destroys Abstract GPU Backend
    class(abstract_gpu_backend),    intent(inout) :: self     !< Abstract GPU backend

    if ( allocated( self%send_displs ) ) deallocate( self%send_displs )
    if ( allocated( self%send_floats ) ) deallocate( self%send_floats )
    if ( allocated( self%recv_displs ) ) deallocate( self%recv_displs )
    if ( allocated( self%recv_floats ) ) deallocate( self%recv_floats )
    call self%destroy_private()
  end subroutine destroy

  integer(int64) function get_aux_size(self)
  !! Returns number of bytes required by aux buffer
    class(abstract_gpu_backend),    intent(in)    :: self     !< Abstract GPU backend
    get_aux_size = self%aux_size
  end function get_aux_size

  subroutine set_aux(self, aux)
  !! Sets aux buffer that can be used by various implementations
    class(abstract_gpu_backend),    intent(inout) :: self     !< Abstract GPU backend
    type(c_devptr),                 intent(in)    :: aux      !< Aux pointer
    self%aux = aux
  end subroutine set_aux
end module dtfft_abstract_gpu_backend