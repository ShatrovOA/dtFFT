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
module dtfft_transpose_handle_cuda
!! This module describes [[transpose_handle_cuda]] class
use iso_c_binding,                        only: c_ptr
use iso_fortran_env,                      only: int8, int32, int64, real32
use dtfft_interface_cuda_runtime
use dtfft_abstract_backend,               only: abstract_backend, backend_helper
#ifdef DTFFT_WITH_NVSHMEM
use dtfft_backend_cufftmp_m,              only: backend_cufftmp
#endif
#ifdef DTFFT_WITH_NCCL
use dtfft_backend_nccl_m,                 only: backend_nccl
#endif
use dtfft_backend_mpi,                    only: backend_mpi
use dtfft_nvrtc_kernel
use dtfft_pencil,                         only: pencil, get_transpose_type
use dtfft_parameters
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: transpose_handle_cuda

  type :: data_handle
  !! Helper class used to obtain displacements and 
  !! counts needed to send to other processes
    integer(int32),        allocatable  :: ls(:,:)                  !! Starts of my data that I should send or recv
                                                                    !! while communicating with other processes
    integer(int32),        allocatable  :: ln(:,:)                  !! Counts of my data that I should send or recv
                                                                    !! while communicating with other processes
    integer(int32),        allocatable  :: sizes(:,:)               !! Counts of every rank in a comm
    integer(int32),        allocatable  :: starts(:,:)              !! Starts of every rank in a comm
    integer(int32),        allocatable  :: displs(:)                !! Local buffer displacement
    integer(int32),        allocatable  :: counts(:)                !! Number of elements to send or recv
  contains
    procedure,  pass(self)  :: create => create_data_handle         !! Creates handle
    procedure,  pass(self)  :: destroy => destroy_data_handle       !! Destroys handle
  end type data_handle

  type :: transpose_handle_cuda
  !! CUDA Transpose Handle
  private
    type(dtfft_transpose_t)                   :: transpose_type
    logical                                   :: has_exchange = .false.   !! If current handle has exchanges between GPUs
    logical                                   :: is_pipelined = .false.   !! If underlying exchanges are pipelined
    type(nvrtc_kernel)                        :: transpose_kernel              !! Transposes data
    type(nvrtc_kernel)                        :: unpack_kernel            !! Unpacks data
    type(nvrtc_kernel)                        :: unpack_kernel2
    class(abstract_backend),  allocatable     :: comm_handle              !! Communication handle
  contains
    procedure, pass(self) :: create           !! Creates CUDA Transpose Handle
    procedure, pass(self) :: execute          !! Executes transpose - exchange - unpack
    procedure, pass(self) :: destroy          !! Destroys CUDA Transpose Handle
    procedure, pass(self) :: get_aux_size     !! Returns number of bytes required by aux buffer
    procedure, pass(self) :: get_tranpose_type!! Returns transpose_type, associated with handle
  end type transpose_handle_cuda

contains

  subroutine create_data_handle(self, info, comm, comm_size)
  !! Creates handle
    class(data_handle),   intent(inout) :: self       !! Helper class
    type(pencil),         intent(in)    :: info       !! Pencil info
    TYPE_MPI_COMM,        intent(in)    :: comm       !! MPI communicator
    integer(int32),       intent(in)    :: comm_size  !! Size of ``comm``
    integer(int32)                      :: ierr       !! MPI error flag

    allocate( self%ls( info%rank, 0:comm_size - 1 ), source=0_int32 )
    allocate( self%ln( info%rank, 0:comm_size - 1 ), source=0_int32 )
    allocate( self%sizes ( info%rank, 0:comm_size - 1 ) )
    allocate( self%starts( info%rank, 0:comm_size - 1 ) )
    allocate( self%displs( 0:comm_size - 1 ), source=0_int32 )
    allocate( self%counts( 0:comm_size - 1 ), source=0_int32 )

    call MPI_Allgather(info%counts, int(info%rank, int32), MPI_INTEGER, self%sizes,  int(info%rank, int32), MPI_INTEGER, comm, ierr)
    call MPI_Allgather(info%starts, int(info%rank, int32), MPI_INTEGER, self%starts, int(info%rank, int32), MPI_INTEGER, comm, ierr)
  end subroutine create_data_handle

  subroutine destroy_data_handle(self)
  !! Destroys handle
    class(data_handle),   intent(inout) :: self       !! Helper class

    if(allocated(self%ls))        deallocate(self%ls)
    if(allocated(self%ln))        deallocate(self%ln)
    if(allocated(self%sizes))     deallocate(self%sizes)
    if(allocated(self%starts))    deallocate(self%starts)
    if(allocated(self%displs))    deallocate(self%displs)
    if(allocated(self%counts))    deallocate(self%counts)
  end subroutine destroy_data_handle

  subroutine create(self, helper, send, recv, base_storage, backend)
  !! Creates CUDA Transpose Handle
    class(transpose_handle_cuda),   intent(inout) :: self               !! CUDA Transpose Handle
    type(backend_helper),           intent(in)    :: helper             !! Backend helper
    ! TYPE_MPI_COMM,                  intent(in)    :: comm               !! 1d communicator
    type(pencil),                   intent(in)    :: send               !! Send pencil
    type(pencil),                   intent(in)    :: recv               !! Recv pencil
    integer(int64),                 intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(dtfft_backend_t),          intent(in)    :: backend            !! Backend type
    integer(int8)                                 :: ndims              !! Number of dimensions
    type(dtfft_transpose_t)                       :: transpose_type     !! Type of transpose based on ``send`` and ``recv``
    integer(int32)                                :: comm_size          !! Size of ``comm``
    integer(int32)                                :: comm_rank          !! Rank in ``comm``
    integer(int32)                                :: ierr               !! MPI error flag
    logical                                       :: packing_required   !! If transpose kernel requires packing. X-Y 3d only
    integer(int32)                                :: sdispl             !! Send displacement
    integer(int32)                                :: rdispl             !! Recv displacement
    integer(int32)                                :: sendsize           !! Number of elements to send
    integer(int32)                                :: recvsize           !! Number of elements to recieve
    integer(int32)                                :: i                  !! Counter
    TYPE_MPI_REQUEST                              :: sr                 !! Send request
    TYPE_MPI_REQUEST                              :: rr                 !! Recv request
    integer(int8)                                 :: kernel_type        !! Type of kernel
    integer(int32),                   allocatable :: k1(:,:)            !! Pack kernel arguments
    integer(int32),                   allocatable :: k2(:,:)            !! Unpack kernel arguments
    type(data_handle)                             :: in                 !! Send helper
    type(data_handle)                             :: out                !! Recv helper
    integer(int8)                                 :: comm_id
    TYPE_MPI_COMM :: comm

    transpose_type = get_transpose_type(send, recv)

    select case ( abs(transpose_type%val) )
    case ( DTFFT_TRANSPOSE_X_TO_Y%val )
      comm_id = 2
    case ( DTFFT_TRANSPOSE_Y_TO_Z%val )
      comm_id = 3
    case ( DTFFT_TRANSPOSE_X_TO_Z%val )
      comm_id = 1
    case default
      INTERNAL_ERROR("unknown `abs(transpose_type)`")
    endselect

    comm = helper%comms(comm_id)

    call MPI_Comm_size(comm, comm_size, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)
    self%has_exchange = comm_size > 1

    self%transpose_type = transpose_type
    ndims = send%rank

    packing_required = (abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val)  &
                        .and. self%has_exchange                                 &
                        .and. ndims == 3                                        &
                        .and. (.not. backend==DTFFT_BACKEND_CUFFTMP)

    kernel_type = KERNEL_TRANSPOSE
    if ( packing_required ) kernel_type = KERNEL_TRANSPOSE_PACKED

    if ( .not. self%has_exchange ) then
      call self%transpose_kernel%create(comm, send%counts, base_storage, transpose_type, kernel_type)
      return
    endif

    allocate( k1(3, comm_size), source=0_int32 )
    allocate( k2(5, comm_size), source=0_int32 )

    call in%create(send, comm, comm_size)
    call out%create(recv, comm, comm_size)

    sdispl = 0
    do i = 0, comm_size - 1
      select case ( transpose_type%val )
      case ( DTFFT_TRANSPOSE_X_TO_Y%val, DTFFT_TRANSPOSE_Y_TO_X%val )
        in%ln(1, i) = out%sizes(2, i)
        in%ln(2, i) = in%sizes(2, comm_rank)

        in%ls(1, i) = out%starts(2, i)
        in%ls(2, i) = in%starts(2, comm_rank)
        if ( ndims == 3 ) then
          in%ln(3, i) = in%sizes(3, comm_rank)

          in%ls(3, i) = in%starts(3, comm_rank)
        endif
      case ( DTFFT_TRANSPOSE_Y_TO_Z%val, DTFFT_TRANSPOSE_Z_TO_Y%val )
        in%ln(1, i) = out%sizes(3, i)
        in%ln(2, i) = in%sizes(2, comm_rank)
        in%ln(3, i) = in%sizes(3, comm_rank)

        in%ls(1, i) = out%starts(3, i)
        in%ls(2, i) = in%starts(2, comm_rank)
        in%ls(3, i) = in%starts(3, comm_rank)
      case ( DTFFT_TRANSPOSE_X_TO_Z%val )
        in%ln(1, i) = in%sizes(1, comm_rank)
        in%ln(2, i) = out%sizes(3, i)
        in%ln(3, i) = in%sizes(3, comm_rank)

        in%ls(1, i) = in%starts(1, comm_rank)
        in%ls(2, i) = out%starts(3, i)
        in%ls(3, i) = in%starts(3, comm_rank)
      case ( DTFFT_TRANSPOSE_Z_TO_X%val )
        in%ln(1, i) = out%sizes(3, i)
        in%ln(2, i) = in%sizes(2, comm_rank)
        in%ln(3, i) = in%sizes(3, comm_rank)

        in%ls(1, i) = out%starts(3, i)
        in%ls(2, i) = in%starts(2, comm_rank)
        in%ls(3, i) = in%starts(3, comm_rank)
      endselect

      if ( packing_required ) then
        k1(1, i + 1) = in%ls(1, i)
        k1(2, i + 1) = in%ln(1, i)
        k1(3, i + 1) = sdispl
        if ( sdispl > 0 ) then
          k1(3, i + 1) =  k1(3, i + 1) - in%ls(1, i) * in%ln(2, i)
        endif
      endif

      sendsize = product( in%ln(:, i) )

      in%counts(i) = sendsize
      in%displs(i) = sdispl
      sdispl = sdispl + sendsize
      ! Sending with tag = me to rank i
      call MPI_Isend(sendsize, 1, MPI_INTEGER4, i, comm_rank, comm, sr, ierr)
      call MPI_Wait(sr, MPI_STATUS_IGNORE, ierr)
    enddo

    rdispl = 0
    do i = 0, comm_size - 1
      ! Recieving from i with tag i
      call MPI_Irecv(recvsize, 1, MPI_INTEGER4, i, i, comm, rr, ierr)
      call MPI_Wait(rr, MPI_STATUS_IGNORE, ierr)
      if ( recvsize > 0 ) then
        select case ( transpose_type%val )
        case ( DTFFT_TRANSPOSE_X_TO_Y%val, DTFFT_TRANSPOSE_Y_TO_X%val )
          out%ln(1, i) = in%sizes(2, i)
          out%ln(2, i) = out%sizes(2, comm_rank)

          out%ls(1, i) = in%starts(2, i)
          out%ls(2, i) = out%starts(2, comm_rank)
          if ( ndims == 3 ) then
            out%ln(3, i) = in%sizes(3, comm_rank)

            out%ls(3, i) = in%starts(3, comm_rank)
          endif
        case ( DTFFT_TRANSPOSE_Y_TO_Z%val, DTFFT_TRANSPOSE_Z_TO_Y%val )
          out%ln(1, i) = in%sizes(3, i)
          out%ln(2, i) = out%sizes(2, comm_rank)
          out%ln(3, i) = in%sizes(3, comm_rank)

          out%ls(1, i) = in%starts(3, i)
          out%ls(2, i) = out%starts(2, comm_rank)
          out%ls(3, i) = in%starts(3, comm_rank)
        case ( DTFFT_TRANSPOSE_X_TO_Z%val )
          out%ln(1, i) = in%sizes(3, i)
          out%ln(2, i) = out%sizes(2, comm_rank)
          out%ln(3, i) = out%sizes(3, comm_rank)

          out%ls(1, i) = in%starts(3, i)
          out%ls(2, i) = out%starts(2, comm_rank)
          out%ls(3, i) = out%starts(3, comm_rank)
        case ( DTFFT_TRANSPOSE_Z_TO_X%val )
          out%ln(1, i) = out%sizes(1, comm_rank)
          out%ln(2, i) = in%sizes(3, i)
          out%ln(3, i) = out%sizes(3, comm_rank)

          out%ls(1, i) = out%starts(1, comm_rank)
          out%ls(2, i) = in%starts(3, i)
          out%ls(3, i) = out%starts(3, comm_rank)
        endselect
      endif

      k2(1, i + 1) = rdispl
      if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        k2(2, i + 1) = out%ln(1, i) * out%ls(2, i)
        k2(3, i + 1) = out%ln(1, i) * out%ln(2, i)
        k2(5, i + 1) = out%sizes(1, comm_rank) * out%sizes(2, comm_rank)
      else
        k2(2, i + 1) = out%ls(1, i)
        k2(3, i + 1) = out%ln(1, i)
        k2(5, i + 1) = out%sizes(1, comm_rank)
      endif
      k2(4, i + 1) = recvsize

      out%counts(i) = recvsize
      out%displs(i) = rdispl
      rdispl = rdispl + recvsize
    enddo

    call self%transpose_kernel%create(comm, send%counts, base_storage, transpose_type, kernel_type, k1)

    self%is_pipelined = is_backend_pipelined(backend)
    kernel_type = KERNEL_UNPACK
    if ( self%is_pipelined ) kernel_type = KERNEL_UNPACK_PIPELINED
    if ( backend == DTFFT_BACKEND_CUFFTMP ) kernel_type = KERNEL_UNPACK_SIMPLE_COPY
    if ( backend == DTFFT_BACKEND_CUFFTMP_PIPELINED ) kernel_type = KERNEL_DUMMY
    call self%unpack_kernel%create(comm, recv%counts, base_storage, transpose_type, kernel_type, k2)

    if ( backend == DTFFT_BACKEND_NCCL_PIPELINED ) then
      call self%unpack_kernel2%create(comm, recv%counts, base_storage, transpose_type, KERNEL_UNPACK_PARTIAL, k2)
    endif

    if ( is_backend_mpi(backend) ) then
      allocate( backend_mpi :: self%comm_handle )
    else if ( is_backend_nccl(backend) ) then
#ifdef DTFFT_WITH_NCCL
      allocate( backend_nccl :: self%comm_handle )
#else
      INTERNAL_ERROR("not DTFFT_WITH_NCCL")
#endif
    else if ( is_backend_cufftmp(backend) ) then
#ifdef DTFFT_WITH_NVSHMEM
      allocate( backend_cufftmp :: self%comm_handle )
#else
      INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
#endif
    else
      INTERNAL_ERROR("Unknown backend")
    endif

    call self%comm_handle%create(backend, transpose_type, helper, comm_id, in%displs, in%counts, out%displs, out%counts, base_storage)

    if ( self%is_pipelined ) then
      if( backend == DTFFT_BACKEND_NCCL_PIPELINED ) then
        call self%comm_handle%set_unpack_kernel(self%unpack_kernel, self%unpack_kernel2)
      else
        call self%comm_handle%set_unpack_kernel(self%unpack_kernel)
      endif
    endif

    call in%destroy()
    call out%destroy()
    deallocate( k1, k2 )
  end subroutine create

  subroutine execute(self, in, out, stream, aux)
  !! Executes transpose - exchange - unpack
    class(transpose_handle_cuda),   intent(inout) :: self       !! CUDA Transpose Handle
    real(real32),                   intent(inout) :: in(:)      !! Send pointer
    real(real32),                   intent(inout) :: out(:)     !! Recv pointer
    type(dtfft_stream_t),           intent(in)    :: stream     !! Main execution CUDA stream
    real(real32),                   intent(inout) :: aux(:)     !! Aux pointer

    if ( self%is_pipelined ) then
      call self%transpose_kernel%execute(in, aux, stream)
#ifdef __DEBUG
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
      call self%comm_handle%execute(aux, out, stream, in)
#ifdef __DEBUG
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
      return
    endif

    call self%transpose_kernel%execute(in, out, stream)
#ifdef __DEBUG
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    if ( .not. self%has_exchange ) return
    call self%comm_handle%execute(out, in, stream, aux)
#ifdef __DEBUG
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    call self%unpack_kernel%execute(in, out, stream)
#ifdef __DEBUG
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys CUDA Transpose Handle
    class(transpose_handle_cuda),   intent(inout) :: self       !! CUDA Transpose Handle

    call self%transpose_kernel%destroy()
    if ( .not. self%has_exchange ) return
    call self%comm_handle%destroy()
    deallocate( self%comm_handle )
    call self%unpack_kernel%destroy()
    call self%unpack_kernel2%destroy()
  end subroutine destroy

  integer(int64) function get_aux_size(self)
  !! Returns number of bytes required by aux buffer
    class(transpose_handle_cuda),   intent(in)    :: self       !! CUDA Transpose Handle

    if ( .not. self%has_exchange ) then
      get_aux_size = 0
      return
    endif
    get_aux_size = self%comm_handle%get_aux_size()
  end function get_aux_size

  function get_tranpose_type(self) result(tranpose_type)
  !! Returns transpose_type, associated with handle
    class(transpose_handle_cuda),   intent(in)    :: self       !! CUDA Transpose Handle
    type(dtfft_transpose_t)         :: tranpose_type
    tranpose_type = self%transpose_type
  end function get_tranpose_type
end module dtfft_transpose_handle_cuda