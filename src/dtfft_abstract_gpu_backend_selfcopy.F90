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
module dtfft_abstract_gpu_backend_selfcopy
!! This module defines Abstract GPU Backend with self-copying `abstract_gpu_backend_selfcopy` 
!! and its pipelined version `abstract_gpu_backend_pipelined`
use cudafor
use iso_fortran_env
use dtfft_abstract_gpu_backend, only: abstract_gpu_backend
use dtfft_nvrtc_kernel,         only: nvrtc_kernel
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
! Separate module is used for these two classes due to a bug in nvfortran
public :: abstract_gpu_backend_selfcopy
public :: abstract_gpu_backend_pipelined

  type, abstract, extends(abstract_gpu_backend) :: abstract_gpu_backend_selfcopy
  !! Abstract class for GPU backend with self-copying
    type(cudaEvent)             :: execution_event      !< Event for main execution stream
    type(cudaEvent)             :: copy_event           !< Event for copy stream
    integer(cuda_stream_kind)   :: copy_stream          !< Stream for copy operations
    integer(int64)              :: self_copy_elements   !< Number of elements to copy
    integer(int64)              :: self_send_displ      !< Displacement for send buffer
    integer(int64)              :: self_recv_displ      !< Displacement for recv buffer
  contains
    procedure                                     :: create_private => create     !< Creates self-copying backend
    procedure                                     :: execute => execute_selfcopy  !< Executes self-copying backend
    procedure                                     :: destroy_private              !< Destroys self-copying backend
    procedure(createPrivateInterface),  deferred  :: create_selfcopy              !< Create self-copying backend
    procedure(executePrivateInterface), deferred  :: execute_private              !< Execute self-copying backend
    procedure(destroyPrivateInterface), deferred  :: destroy_selfcopy             !< Destroy self-copying backend
  end type abstract_gpu_backend_selfcopy

  type, abstract, extends(abstract_gpu_backend_selfcopy) :: abstract_gpu_backend_pipelined
  !! Abstract class for GPU backend with self-copying and pipelined execution
    type(nvrtc_kernel), pointer :: unpack_kernel        !< Kernel for unpacking data
  contains
    procedure,  pass(self)  :: set_unpack_kernel
  end type abstract_gpu_backend_pipelined

  interface
    subroutine createPrivateInterface(self, comm)
    !! Creates private interface for self-copying backend
    import
      class(abstract_gpu_backend_selfcopy), intent(inout) :: self       !< Self-copying backend
      TYPE_MPI_COMM,                        intent(in)    :: comm       !< MPI communicator
    end subroutine createPrivateInterface

    subroutine executePrivateInterface(self, in, out, stream)
    !! Executes private interface for self-copying backend
    import
      class(abstract_gpu_backend_selfcopy), intent(inout) :: self       !< Self-copying backend
      type(c_devptr),                       intent(in)    :: in         !< Send pointer
      type(c_devptr),                       intent(in)    :: out        !< Recv pointer
      integer(cuda_stream_kind),            intent(in)    :: stream     !< Main execution CUDA stream
    end subroutine executePrivateInterface

    subroutine destroyPrivateInterface(self)
    !! Destroys private interface for self-copying backend
    import
      class(abstract_gpu_backend_selfcopy), intent(inout) :: self       !< Self-copying backend
    end subroutine destroyPrivateInterface
  endinterface

contains

  subroutine create(self, comm)
  !! Creates self-copying backend
    class(abstract_gpu_backend_selfcopy),   intent(inout) :: self       !< Self-copying backend
    TYPE_MPI_COMM,                          intent(in)    :: comm       !< MPI communicator

    select type (self)
    class is (abstract_gpu_backend_pipelined)
      self%aux_size = self%send_recv_buffer_size * int(FLOAT_STORAGE_SIZE, int64)
    endselect

    self%self_send_displ = self%send_displs(self%comm_rank)
    self%self_recv_displ = self%recv_displs(self%comm_rank)
    self%self_copy_elements = self%send_floats(self%comm_rank)
    self%send_floats(self%comm_rank) = 0
    self%recv_floats(self%comm_rank) = 0

    CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%execution_event, cudaEventDisableTiming) )
    CUDA_CALL( "cudaEventCreateWithFlags", cudaEventCreateWithFlags(self%copy_event, cudaEventDisableTiming) )
    CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(self%copy_stream) )

    call self%create_selfcopy(comm)
  end subroutine create

  subroutine execute_selfcopy(self, in, out, stream)
  !! Executes self-copying backend
    class(abstract_gpu_backend_selfcopy), intent(inout) :: self     !< Self-copying backend
    type(c_devptr),                       intent(in)    :: in       !< Send pointer
    type(c_devptr),                       intent(in)    :: out      !< Recv pointer
    integer(cuda_stream_kind),            intent(in)    :: stream   !< CUDA stream
    real(real32), DEVICE_PTR pointer, contiguous :: pin(:)  !< Send buffer
    real(real32), DEVICE_PTR pointer, contiguous :: pout(:) !< Recv buffer
    real(real32), DEVICE_PTR pointer, contiguous :: paux(:) !< Aux buffer

    call c_f_pointer(in, pin, [self%send_recv_buffer_size])

    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%execution_event, stream) )
    ! Waiting for transpose kernel to finish execution on stream `stream`
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(self%copy_stream, self%execution_event, 0) )

    select type( self )
    class is ( abstract_gpu_backend_pipelined )
    ! Tranposed data is actually located in aux buffer for pipelined algorithm
      if( self%self_copy_elements > 0 ) then
        call c_f_pointer(self%aux, paux, [self%send_recv_buffer_size])

        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(pin( self%self_recv_displ ),
                                                      paux( self%self_send_displ ),
                                                      self%self_copy_elements,
                                                      cudaMemcpyDeviceToDevice, self%copy_stream) )
#ifdef __DEBUG
        CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
        ! Data can be unpacked in same stream as `cudaMemcpyAsync`
        call self%unpack_kernel%execute(in, out, self%copy_stream, self%comm_rank + 1)
      endif
    class default
      if( self%self_copy_elements > 0 ) then
        call c_f_pointer(out, pout, [self%send_recv_buffer_size])

        CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(pout( self%self_recv_displ ),
                                                      pin( self%self_send_displ ),
                                                      self%self_copy_elements,
                                                      cudaMemcpyDeviceToDevice, self%copy_stream) )
      endif
    endselect
#ifdef __DEBUG
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    call self%execute_private(in, out, stream)
#ifndef __DEBUG
    ! Making `stream` wait for finish of `cudaMemcpyAsync`
    CUDA_CALL( "cudaEventRecord", cudaEventRecord(self%copy_event, self%copy_stream) )
    CUDA_CALL( "cudaStreamWaitEvent", cudaStreamWaitEvent(stream, self%copy_event, 0) )
#endif
  end subroutine execute_selfcopy

  subroutine destroy_private(self)
  !! Destroys self-copying backend
    class(abstract_gpu_backend_selfcopy), intent(inout) :: self             !< Self-copying backend

    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%execution_event) )
    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(self%copy_event) )
    CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(self%copy_stream) )
    call self%destroy_selfcopy()
  end subroutine destroy_private

  subroutine set_unpack_kernel(self, unpack_kernel)
  !! Sets unpack kernel for pipelined backend
    class(abstract_gpu_backend_pipelined),  intent(inout)   :: self           !< Pipelined backend
    type(nvrtc_kernel),         target,     intent(in)      :: unpack_kernel  !< Kernel for unpacking data

    self%unpack_kernel => unpack_kernel
  end subroutine set_unpack_kernel
end module dtfft_abstract_gpu_backend_selfcopy