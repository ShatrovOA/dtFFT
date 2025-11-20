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
module dtfft_backend_cufftmp_m
!! cuFFTMp GPU Backend [[backend_cufftmp]]
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend, only: abstract_backend, backend_helper
use dtfft_interface_nvshmem
use dtfft_interface_cuda_runtime
use dtfft_interface_cufft
use dtfft_errors
use dtfft_parameters
use dtfft_pencil, only: pencil
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: backend_cufftmp

type :: Box3D
!! cuFFTMp Box
    integer(c_long_long) :: lower(3)   !! Lower box boundaries
    integer(c_long_long) :: upper(3)   !! Upper box boundaries
    integer(c_long_long) :: strides(3) !! Strides in memory
end type Box3D

type, extends(abstract_backend) :: backend_cufftmp
!! cuFFTMp GPU Backend
private
    type(cufftReshapeHandle) :: plan
contains
    procedure :: create_private => create
    procedure :: execute_private => execute
    procedure :: destroy_private => destroy
end type backend_cufftmp

contains

subroutine create(self, helper, base_storage)
!! Creates cuFFTMp GPU Backend
class(backend_cufftmp), intent(inout)   :: self               !! cuFFTMp GPU Backend
type(backend_helper),   intent(in)      :: helper             !! Backend helper
integer(int64),         intent(in)      :: base_storage       !! Number of bytes to store single element
type(Box3D) :: inbox, outbox  !! Reshape boxes
type(pencil), pointer :: in, out
type(c_ptr) :: c_comm
integer(int64) :: aux_size
type(dtfft_transpose_t) :: tranpose_type

    tranpose_type = helper%tranpose_type

    select case (tranpose_type%val)
    case (DTFFT_TRANSPOSE_X_TO_Y%val)
        in => helper%pencils(1)
        out => helper%pencils(2)
    case (DTFFT_TRANSPOSE_Y_TO_X%val)
        in => helper%pencils(2)
        out => helper%pencils(1)
    case (DTFFT_TRANSPOSE_Y_TO_Z%val)
        in => helper%pencils(2)
        out => helper%pencils(3)
    case (DTFFT_TRANSPOSE_Z_TO_Y%val)
        in => helper%pencils(3)
        out => helper%pencils(2)
    case (DTFFT_TRANSPOSE_X_TO_Z%val)
        in => helper%pencils(1)
        out => helper%pencils(3)
    case (DTFFT_TRANSPOSE_Z_TO_X%val)
        in => helper%pencils(3)
        out => helper%pencils(1)
    case default
        INTERNAL_ERROR("unknown `tranpose_type`")
    end select

    if (in%rank == 3) then
        if (any(tranpose_type == [DTFFT_TRANSPOSE_X_TO_Z, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Z_TO_Y])) then
            inbox%lower = [in%starts(2), in%starts(1), in%starts(3)]
            inbox%upper = [in%starts(2) + in%counts(2), in%starts(1) + in%counts(1), in%starts(3) + in%counts(3)]
            inbox%strides = [in%counts(1) * in%counts(3), in%counts(3), 1]
        else
            inbox%lower = [in%starts(1), in%starts(3), in%starts(2)]
            inbox%upper = [in%starts(1) + in%counts(1), in%starts(3) + in%counts(3), in%starts(2) + in%counts(2)]
            inbox%strides = [in%counts(2) * in%counts(3), in%counts(2), 1]
        end if

        outbox%lower = [out%starts(3), out%starts(2), out%starts(1)]
        outbox%upper = [out%starts(3) + out%counts(3), out%starts(2) + out%counts(2), out%starts(1) + out%counts(1)]
        outbox%strides = [out%counts(1) * out%counts(2), out%counts(1), 1]
    else
        inbox%lower = [0, in%starts(1), in%starts(2)]
        inbox%upper = [1, in%starts(1) + in%counts(1), in%starts(2) + in%counts(2)]
        inbox%strides = [in%counts(1) * in%counts(2), in%counts(2), 1]

        outbox%lower = [0, out%starts(2), out%starts(1)]
        outbox%upper = [1, out%starts(2) + out%counts(2), out%starts(1) + out%counts(1)]
        outbox%strides = [out%counts(1) * out%counts(2), out%counts(1), 1]
    end if

    CUFFT_CALL( cufftMpCreateReshape(self%plan) )
    c_comm = Comm_f2c(GET_MPI_VALUE(helper%comms(1)))
    CUFFT_CALL( cufftMpAttachReshapeComm(self%plan, CUFFT_COMM_MPI, c_comm) )
    CUFFT_CALL( cufftMpMakeReshape(self%plan, base_storage, 3, inbox%lower, inbox%upper, outbox%lower, outbox%upper, inbox%strides, outbox%strides) )
    CUFFT_CALL( cufftMpGetReshapeSize(self%plan, aux_size) )
    self%aux_size = max(aux_size, self%aux_size)
end subroutine create

subroutine execute(self, in, out, stream, aux, error_code)
!! Executes cuFFTMp GPU Backend
class(backend_cufftmp), intent(inout)   :: self       !! cuFFTMp GPU Backend
real(real32),   target, intent(inout)   :: in(:)      !! Send pointer
real(real32),   target, intent(inout)   :: out(:)     !! Recv pointer
type(dtfft_stream_t),   intent(in)      :: stream     !! Main execution CUDA stream
real(real32),   target, intent(inout)   :: aux(:)     !! Aux pointer
integer(int32),         intent(out)     :: error_code !! Error code
integer(int32) :: ierr

    ! call nvshmemx_sync_all_on_stream(stream)
    CUDA_CALL( cudaStreamSynchronize(stream) )
    call MPI_Barrier(self%comm, ierr)
    CUFFT_CALL( cufftMpExecReshapeAsync(self%plan, c_loc(out), c_loc(in), c_loc(aux), stream) )
    error_code = DTFFT_SUCCESS
end subroutine execute

subroutine destroy(self)
!! Destroys cuFFTMp GPU Backend
class(backend_cufftmp), intent(inout) :: self        !! cuFFTMp GPU Backend

    CUFFT_CALL( cufftMpDestroyReshape(self%plan) )
end subroutine destroy
end module dtfft_backend_cufftmp_m
