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
module dtfft_transpose_handle_generic
!! This module describes [[transpose_handle_generic]] class
!! It is responsible for managing both Host and CUDA-based transposition operations
!! It executes transpose kernels, memory transfers between GPUs/Hosts, and data unpacking if required
use iso_c_binding,                        only: c_ptr, c_f_pointer
use iso_fortran_env,                      only: int8, int32, int64, real32
use dtfft_abstract_backend,               only: abstract_backend, backend_helper
use dtfft_abstract_kernel
use dtfft_abstract_transpose_handle,      only: abstract_transpose_handle, create_args, execute_args
#ifdef DTFFT_WITH_NVSHMEM
use dtfft_backend_cufftmp_m,              only: backend_cufftmp
#endif
#ifdef DTFFT_WITH_NCCL
use dtfft_backend_nccl_m,                 only: backend_nccl
#endif
use dtfft_backend_mpi,                    only: backend_mpi
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_kernel_device,                  only: kernel_device
#endif
use dtfft_kernel_host,                    only: kernel_host
use dtfft_pencil,                         only: pencil, get_transpose_type
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: transpose_handle_generic

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

  type, extends(abstract_transpose_handle) :: transpose_handle_generic
  !! Generic Transpose Handle
  !! Executes transposition in 3 steps:
  !! 
  !! - Transpose kernel execution
  !! - Data exchange between processes
  !! - Unpacking kernel execution
  private
    logical                                   :: has_exchange = .false.   !! If current handle has exchanges between GPUs
    logical                                   :: is_pipelined = .false.   !! If underlying exchanges are pipelined
    logical                                   :: is_async_supported = .false. !! If underlying backend support async execution(execute/execute_end)
    class(abstract_kernel),   allocatable     :: transpose_kernel         !! Kernel for data transposition
    class(abstract_kernel),   allocatable     :: unpack_kernel            !! Kernel for unpacking data
    class(abstract_backend),  allocatable     :: comm_handle              !! Communication handle
  contains
    procedure, pass(self) :: create_private => create           !! Creates Generic Transpose Handle
    procedure, pass(self) :: execute          !! Executes transpose - exchange - unpack
    procedure, pass(self) :: execute_end      !! Finalizes async transpose
    procedure, pass(self) :: get_async_active !! Returns if async transpose is active
    procedure, pass(self) :: destroy          !! Destroys Generic Transpose Handle
    procedure, pass(self) :: get_aux_size     !! Returns number of bytes required by aux buffer
  end type transpose_handle_generic

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

  subroutine check_if_overflow(sizes)
  !! Checks if product of sizes fits into integer(int32)
    integer(int32), intent(in) :: sizes(:)  !! Sizes to check
    integer(int64)  :: sizes_prod !! Product of sizes
    integer(int32)  :: i          !! Counter

    sizes_prod = 1_int64
    do i = 1, size(sizes)
      sizes_prod = sizes_prod * int(sizes(i), int64)
    enddo
    if ( sizes_prod /= product(sizes) ) then
      INTERNAL_ERROR("integer(int64) indexing currently not supported")
    endif
  end subroutine check_if_overflow

  subroutine create(self, comm, send, recv, transpose_type, base_storage, kwargs)
  !! Creates Generic Transpose Handle
    class(transpose_handle_generic),  intent(inout) :: self           !! Generic Transpose Handle
    TYPE_MPI_COMM,                    intent(in)    :: comm           !! MPI Communicator
    type(pencil),                     intent(in)    :: send           !! Send pencil
    type(pencil),                     intent(in)    :: recv           !! Recv pencil
    type(dtfft_transpose_t),          intent(in)    :: transpose_type !! Type of transpose to create
    integer(int64),                   intent(in)    :: base_storage   !! Base storage
    type(create_args),                intent(in)    :: kwargs         !! Additional arguments
    integer(int8)                     :: ndims              !! Number of dimensions
    integer(int32)                    :: comm_size          !! Size of ``comm``
    integer(int32)                    :: comm_rank          !! Rank in ``comm``
    integer(int32)                    :: ierr               !! MPI error flag
    integer(int32)                    :: sdispl             !! Send displacement
    integer(int32)                    :: rdispl             !! Recv displacement
    integer(int32)                    :: sendsize           !! Number of elements to send
    integer(int32)                    :: recvsize           !! Number of elements to recieve
    integer(int32)                    :: i                  !! Counter
    TYPE_MPI_REQUEST                  :: sr                 !! Send request
    TYPE_MPI_REQUEST                  :: rr                 !! Recv request
    type(kernel_type_t)               :: kernel_type        !! Type of kernel
    integer(int32),       allocatable :: neighbor_data(:,:) !! Aggregated neighbor data
    type(data_handle)                 :: in                 !! Send helper
    type(data_handle)                 :: out                !! Recv helper
    logical                           :: is_two_step_permute!! If transpose is two-step permute

    call self%destroy()
    call check_if_overflow(send%counts)
    call check_if_overflow(recv%counts)

    if ( kwargs%platform == DTFFT_PLATFORM_HOST ) then
      allocate( kernel_host :: self%transpose_kernel )
      allocate( kernel_host :: self%unpack_kernel )
#ifdef DTFFT_WITH_CUDA
    else
      allocate( kernel_device :: self%transpose_kernel )
      allocate( kernel_device :: self%unpack_kernel )
#endif
    endif

    call MPI_Comm_size(comm, comm_size, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)
    self%has_exchange = comm_size > 1

    if( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_X] ) ) then
      kernel_type = KERNEL_PERMUTE_FORWARD
    else
      kernel_type = KERNEL_PERMUTE_BACKWARD
    endif

    if ( .not. self%has_exchange ) then
      self%is_pipelined = .false.
      self%is_async_supported = .false.
      call self%transpose_kernel%create(send%counts, kwargs%effort, base_storage, kernel_type, force_effort=kwargs%force_effort)
      return
    endif

    allocate( neighbor_data(5, comm_size), source=0_int32 )

    call in%create(send, comm, comm_size)
    call out%create(recv, comm, comm_size)

    ndims = send%rank
    sdispl = 0
    do i = 0, comm_size - 1
      ! Finding amount of data to send to rank i
      if ( ndims == 2 ) then
        in%ln(1, i) = out%sizes(2, i)
        in%ln(2, i) = in%sizes(2, comm_rank)

        in%ls(1, i) = out%starts(2, i)
        in%ls(2, i) = in%starts(2, comm_rank)
      else
        if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
          in%ln(1, i) = in%sizes(1, comm_rank)
          in%ln(2, i) = out%sizes(3, i)
          in%ln(3, i) = in%sizes(3, comm_rank)

          in%ls(1, i) = in%starts(1, comm_rank)
          in%ls(2, i) = out%starts(3, i)
          in%ls(3, i) = in%starts(3, comm_rank)
        else if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
          in%ln(1, i) = out%sizes(3, i)
          in%ln(2, i) = in%sizes(2, comm_rank)
          in%ln(3, i) = in%sizes(3, comm_rank)

          in%ls(1, i) = out%starts(3, i)
          in%ls(2, i) = in%starts(2, comm_rank)
          in%ls(3, i) = in%starts(3, comm_rank)
        else if( kernel_type == KERNEL_PERMUTE_FORWARD ) then
          in%ln(1, i) = out%sizes(3, i)
          in%ln(2, i) = in%sizes(2, comm_rank)
          in%ln(3, i) = in%sizes(3, comm_rank)

          in%ls(1, i) = out%starts(3, i)
          in%ls(2, i) = in%starts(2, comm_rank)
          in%ls(3, i) = in%starts(3, comm_rank)
        else
          in%ln(1, i) = out%sizes(2, i)
          in%ln(2, i) = in%sizes(2, comm_rank)
          in%ln(3, i) = in%sizes(3, comm_rank)

          in%ls(1, i) = out%starts(2, i)
          in%ls(2, i) = in%starts(2, comm_rank)
          in%ls(3, i) = in%starts(3, comm_rank)
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
        if ( ndims == 2 ) then
          out%ln(1, i) = in%sizes(2, i)
          out%ln(2, i) = out%sizes(2, comm_rank)

          out%ls(1, i) = in%starts(2, i)
          out%ls(2, i) = out%starts(2, comm_rank)
        else
          if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
            out%ln(1, i) = in%sizes(3, i)
            out%ln(2, i) = out%sizes(2, comm_rank)
            out%ln(3, i) = out%sizes(3, comm_rank)

            out%ls(1, i) = in%starts(3, i)
            out%ls(2, i) = out%starts(2, comm_rank)
            out%ls(3, i) = out%starts(3, comm_rank)
          else if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
            out%ln(1, i) = out%sizes(1, comm_rank)
            out%ln(2, i) = in%sizes(3, i)
            out%ln(3, i) = out%sizes(3, comm_rank)

            out%ls(1, i) = out%starts(1, comm_rank)
            out%ls(2, i) = in%starts(3, i)
            out%ls(3, i) = out%starts(3, comm_rank)
          else if ( kernel_type == KERNEL_PERMUTE_FORWARD ) then
            out%ln(1, i) = in%sizes(2, i)
            out%ln(2, i) = out%sizes(2, comm_rank)
            out%ln(3, i) = out%sizes(3, comm_rank)

            out%ls(1, i) = in%starts(2, i)
            out%ls(2, i) = in%starts(2, comm_rank)
            out%ls(3, i) = out%starts(3, comm_rank)
          else
            ! ZXY -> YZX
            ! YZX -> XYZ
            out%ln(1, i) = in%sizes(3, i)
            out%ln(2, i) = out%sizes(2, comm_rank)
            out%ln(3, i) = out%sizes(3, comm_rank)

            out%ls(1, i) = in%starts(3, i)
            out%ls(2, i) = in%starts(2, comm_rank)
            out%ls(3, i) = out%starts(3, comm_rank)
          endif
        endif
      endif

      neighbor_data(1, i + 1) = out%ln(1, i)
      neighbor_data(2, i + 1) = out%ln(2, i)
      if ( ndims == 3 ) then
        neighbor_data(3, i + 1) = out%ln(3, i)
      else
        neighbor_data(3, i + 1) = 1
      endif
      neighbor_data(4, i + 1) = rdispl
      if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        neighbor_data(5, i + 1) = out%ln(1, i) * out%ls(2, i)! out%sizes(1, comm_rank) * out%sizes(2, comm_rank)
      else
        neighbor_data(5, i + 1) = out%ls(1, i)
      endif

      out%counts(i) = recvsize
      out%displs(i) = rdispl
      rdispl = rdispl + recvsize
    enddo

    is_two_step_permute = any(transpose_type == [DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Z_TO_Y])           &
                      .and. ndims == 3                                                                      &
                      .and. (.not. any(kwargs%backend == [DTFFT_BACKEND_CUFFTMP, DTFFT_BACKEND_CUFFTMP_PIPELINED]))


    if ( is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_START

    call self%transpose_kernel%create(send%counts, kwargs%effort, base_storage, kernel_type, force_effort=kwargs%force_effort)

    self%is_pipelined = is_backend_pipelined(kwargs%backend)
    kernel_type = KERNEL_UNPACK
    if ( self%is_pipelined ) kernel_type = KERNEL_UNPACK_PIPELINED
    if ( is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_END
    if ( self%is_pipelined .and. is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_END_PIPELINED
    if ( kwargs%backend == DTFFT_BACKEND_CUFFTMP ) kernel_type = KERNEL_UNPACK_SIMPLE_COPY
    if ( kwargs%backend == DTFFT_BACKEND_CUFFTMP_PIPELINED ) kernel_type = KERNEL_DUMMY

    call self%unpack_kernel%create(recv%counts, kwargs%effort, base_storage, kernel_type, neighbor_data, kwargs%force_effort)

    if ( is_backend_mpi(kwargs%backend) ) then
      allocate( backend_mpi :: self%comm_handle )
#ifdef DTFFT_WITH_CUDA
    else if ( is_backend_nccl(kwargs%backend) ) then
# ifdef DTFFT_WITH_NCCL
      allocate( backend_nccl :: self%comm_handle )
# else
      INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
    else if ( is_backend_cufftmp(kwargs%backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
      allocate( backend_cufftmp :: self%comm_handle )
# else
      INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
#endif
    else
      INTERNAL_ERROR("Unknown backend")
    endif

    self%is_async_supported = is_backend_mpi(kwargs%backend)                                                &
                              .and. .not. self%is_pipelined                                                 &
                              .and. kwargs%platform == DTFFT_PLATFORM_HOST
    call self%comm_handle%create(kwargs%backend, kwargs%helper, kwargs%platform, kwargs%comm_id, in%displs, in%counts, out%displs, out%counts, base_storage)
    if ( self%is_pipelined ) then
      call self%comm_handle%set_unpack_kernel(self%unpack_kernel)
    endif

    call in%destroy()
    call out%destroy()
    deallocate( neighbor_data )
  end subroutine create

  subroutine execute(self, in, out, kwargs, error_code)
  !! Executes transpose - exchange - unpack
    class(transpose_handle_generic),  intent(inout) :: self       !! Generic Transpose Handle
    real(real32),                     intent(inout) :: in(:)      !! Send pointer
    real(real32),                     intent(inout) :: out(:)     !! Recv pointer
    type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
    integer(int32),                   intent(out)   :: error_code !! Error code

    error_code = DTFFT_SUCCESS
    if ( self%is_pipelined ) then
      call self%transpose_kernel%execute(in, kwargs%p1, kwargs%stream)
      call self%comm_handle%execute(kwargs%p1, out, kwargs%stream, in, kwargs%exec_type, error_code)
      return
    endif

    call self%transpose_kernel%execute(in, out, kwargs%stream)
    if ( .not. self%has_exchange ) return
    call self%comm_handle%execute(out, in, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
    if ( error_code /= DTFFT_SUCCESS ) return
    if ( self%is_async_supported .and. kwargs%exec_type == EXEC_NONBLOCKING ) return
    call self%unpack_kernel%execute(in, out, kwargs%stream)
  end subroutine execute

  subroutine execute_end(self, kwargs, error_code)
  !! Ends execution of transposition
    class(transpose_handle_generic),  intent(inout) :: self       !! Generic Transpose Handle
    type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
    integer(int32),                   intent(out)   :: error_code !! Error code

    error_code = DTFFT_SUCCESS
    if( .not. self%is_async_supported ) return

    call self%comm_handle%execute_end(error_code)
    if( error_code /= DTFFT_SUCCESS ) return
    call self%unpack_kernel%execute(kwargs%p1, kwargs%p2, kwargs%stream)
  end subroutine execute_end

  elemental logical function get_async_active(self)
    class(transpose_handle_generic),  intent(in)    :: self       !! Generic Transpose Handle

    get_async_active = .false.
    if( .not. self%is_async_supported ) return
    get_async_active = self%comm_handle%get_async_active()
  end function get_async_active

  subroutine destroy(self)
  !! Destroys Generic Transpose Handle
    class(transpose_handle_generic),   intent(inout) :: self      !! Generic Transpose Handle

    if ( allocated( self%transpose_kernel ) ) then
      call self%transpose_kernel%destroy()
      deallocate( self%transpose_kernel )
    endif
    if ( allocated( self%comm_handle ) ) then
      call self%comm_handle%destroy()
      deallocate( self%comm_handle )
    endif
    if ( allocated( self%unpack_kernel ) ) then
      call self%unpack_kernel%destroy()
      deallocate( self%unpack_kernel )
    endif
  end subroutine destroy

  pure integer(int64) function get_aux_size(self)
  !! Returns number of bytes required by aux buffer
    class(transpose_handle_generic),   intent(in)    :: self      !! Generic Transpose Handle

    if ( .not. self%has_exchange ) then
      get_aux_size = 0
      return
    endif
    get_aux_size = self%comm_handle%get_aux_size()
  end function get_aux_size
end module dtfft_transpose_handle_generic