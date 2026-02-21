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
module dtfft_backend_nccl_m
!! NCCL Based GPU Backends [[backend_nccl]]
use iso_fortran_env
use iso_c_binding, only: c_ptr, c_f_pointer, c_loc
use dtfft_abstract_backend, only: abstract_backend, backend_helper
use dtfft_interface_cuda_runtime
use dtfft_interface_nccl
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: backend_nccl

    type, extends(abstract_backend) :: backend_nccl
    !! NCCL backend
    private
        type(ncclComm) :: nccl_comm     !! NCCL Communicator
    contains
        procedure :: create_private => create_nccl        !! Creates NCCL backend
        procedure :: execute_private => execute_nccl      !! Executes NCCL backend
        procedure :: destroy_private => destroy_nccl      !! Destroys NCCL backend
    end type backend_nccl

contains

    subroutine create_nccl(self, helper, base_storage)
    !! Creates NCCL backend
        class(backend_nccl),    intent(inout)   :: self               !! NCCL backend
        type(backend_helper),   intent(in)      :: helper             !! Backend helper
        integer(int64),         intent(in)      :: base_storage       !! Number of bytes to store single element (unused)
#ifdef DTFFT_DEBUG
        if (.not. is_backend_nccl(self%backend)) then
            INTERNAL_ERROR(".not. is_backend_nccl")
        endif
        if (.not. helper%is_nccl_created) then
            INTERNAL_ERROR(".not. helper%is_nccl_created")
        endif
#endif
        self%nccl_comm = helper%nccl_comm
    end subroutine create_nccl

    subroutine execute_nccl(self, in, out, stream, aux, error_code)
    !! Executes NCCL backend
        class(backend_nccl),                intent(inout)   :: self       !! NCCL backend
        real(real32),   target, contiguous, intent(inout)   :: in(:)      !! Send pointer
        real(real32),   target, contiguous, intent(inout)   :: out(:)     !! Recv pointer
        type(dtfft_stream_t),               intent(in)      :: stream     !! Main execution CUDA stream
        real(real32),   target, contiguous, intent(inout)   :: aux(:)     !! Aux pointer
        integer(int32),         intent(out)                 :: error_code !! Error code
        integer(int32) :: i        !! Counter
        integer(int32) :: rnk      !! Rank to send-recv
        integer(int32) :: ierr
        real(real32), pointer :: pin(:), pout(:)

        if (self%is_pipelined) then
            pin => in(:)
            pout => aux(:)
        else
            pin => in(:)
            pout => out(:)
        end if

#ifdef DTFFT_WITH_COMPRESSION
        if ( self%backend == DTFFT_BACKEND_NCCL_COMPRESSED ) then
        ! pack: in -> out: in is free
        ! compress: out -> in
        ! send: in -> aux
        ! decompress: aux -> in
        ! unpack: in -> out
        block
            integer(int32), allocatable :: csends(:), crecvs(:)

            allocate(csends(self%comm_size), crecvs(self%comm_size), source=0_int32)
            call self%pack_kernel%execute(c_loc(in), c_loc(pin), stream, aux=c_loc(out), csizes=csends, skip_rank=self%comm_rank + 1)
            call self%execute_self_copy(out, pout, stream)
            call self%unpack_kernel%execute(c_loc(pout), c_loc(out), self%copy_stream, self%comm_rank + 1, aux=c_loc(in), skip_compression=.true.)
            call MPI_Alltoall(csends, 1, MPI_INTEGER, crecvs, 1, MPI_INTEGER, self%comm, ierr)
            self%send_floats(:) = int(csends, int64)
            self%recv_floats(:) = int(crecvs, int64)
            deallocate(csends, crecvs)
        endblock
        endif
#endif

        NCCL_CALL( ncclGroupStart() )
        do i = 0, self%comm_size - 1
            if (i == self%comm_rank .and. self%is_pipelined) cycle
            rnk = self%comm_mapping(i)
            if (self%send_floats(i) > 0) then
                NCCL_CALL( ncclSend(pin(self%send_displs(i)), self%send_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
            end if
            if (self%recv_floats(i) > 0) then
                NCCL_CALL( ncclRecv(pout(self%recv_displs(i)), self%recv_floats(i), ncclFloat, rnk, self%nccl_comm, stream) )
            end if
        end do
        NCCL_CALL( ncclGroupEnd() )

#ifdef DTFFT_WITH_COMPRESSION
        if ( self%backend == DTFFT_BACKEND_NCCL_COMPRESSED ) then
            call self%wait(stream)
        endif
#endif
        if (self%is_pipelined) then
            do i = 0, self%comm_size - 1
                if (self%recv_floats(i) > 0) then
                    call self%unpack_kernel%execute(c_loc(pout), c_loc(out), stream, i + 1, aux=c_loc(in))
                end if
            end do
        end if
        error_code = DTFFT_SUCCESS
    end subroutine execute_nccl

    subroutine destroy_nccl(self)
    !! Destroys NCCL backend
        class(backend_nccl), intent(inout) :: self       !! NCCL backend

    end subroutine destroy_nccl
end module dtfft_backend_nccl_m
