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
module dtfft_backend_nvshmem
! use iso_fortran_env
! use cudafor
! use dtfft_abstract_backend,     only: abstract_backend, backend_helper
! use dtfft_interface_cuda_runtime
! use dtfft_interface_nvshmem
! use dtfft_parameters
! use dtfft_utils
! #include "dtfft_mpi.h"
! #include "dtfft_cuda.h"
! implicit none
! private
! public :: backend_nvshmem

!   type, extends(abstract_backend) :: backend_nvshmem
!   private
!     logical :: is_even
!     type(nvshmem_team_t) :: team
!   contains
!     procedure :: create_private => create
!     procedure :: execute_private => execute
!     procedure :: destroy_private => destroy
!   endtype backend_nvshmem

! contains

!   subroutine create(self, helper, tranpose_type, base_storage)
!     class(backend_nvshmem),       intent(inout) :: self       !! CufftMp GPU Backend
!     type(backend_helper),         intent(in)    :: helper     !! Backend helper
!     type(dtfft_transpose_t), intent(in)    :: tranpose_type
!     integer(int8),                intent(in)    :: base_storage       !! Number of bytes to store single element
!     type(c_ptr) :: c_comm

!     self%is_even = all(self%recv_floats == self%recv_floats(1))
!     self%is_even = self%is_even .and. all(self%send_floats == self%send_floats(1))
!     c_comm = Comm_f2c(self%comm)

!   end subroutine create

!   subroutine execute(self, in, out, stream)
!     class(backend_nvshmem),       intent(inout) :: self       !! CufftMp GPU Backend
!     real(real32),   target,       intent(inout) :: in(:)      !! Send pointer
!     real(real32),   target,       intent(inout) :: out(:)     !! Recv pointer
!     integer(cuda_stream_kind),    intent(in)    :: stream     !! Main execution CUDA stream
!     type(cudaStream_t) :: stream_

!     if ( self%is_pipelined ) then

!     else
!       if ( self%is_even ) then

!       else
!         NVSHMEM_CALL( "nvshmemx_float_alltoall_on_stream",
!           nvshmemx_float_alltoall_on_stream(self%team, c_loc(out), c_loc(in), self%send_floats(1), stream_) )
!       endif
!     endif
!   end subroutine execute

!   subroutine destroy(self)
!     class(backend_nvshmem),      intent(inout) :: self        !! CufftMp GPU Backend


!   end subroutine destroy
end module dtfft_backend_nvshmem