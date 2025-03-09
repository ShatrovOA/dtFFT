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
module dtfft_interface_nvshmem
!! NVSHMEM Interfaces
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
implicit none
private
public :: nvshmem_team_t


  type, bind(C) :: nvshmem_team_t
  !! NVSHMEM team.
    integer(c_int32_t) :: val  !! Internal representation of the NVSHMEM team.
  end type nvshmem_team_t

  type(nvshmem_team_t), parameter, public :: NVSHMEM_TEAM_WORLD = nvshmem_team_t(0)

public :: nvshmem_malloc
  interface nvshmem_malloc
  !! Allocates symmetric memory in the NVSHMEM heap.
    function nvshmem_malloc(size) bind(C)
      import
      type(c_ptr)               :: nvshmem_malloc  !! Pointer to the allocated memory.
      integer(c_size_t), value :: size            !! Size of the allocation in bytes.
    end function nvshmem_malloc
  end interface

public :: nvshmem_free
  interface nvshmem_free
  !! Frees symmetric memory allocated by nvshmem_malloc.
    subroutine nvshmem_free(ptr) bind(C)
      import
      type(c_ptr), value :: ptr  !! Pointer to the memory to free.
    end subroutine nvshmem_free
  end interface

public :: nvshmemx_sync_all_on_stream
  interface nvshmemx_sync_all_on_stream
  !! Synchronizes all PEs (Processing Elements) on the specified stream.
    subroutine nvshmemx_sync_all_on_stream(stream) bind(C)
      import
      type(dtfft_stream_t), intent(in), value :: stream  !! CUDA stream for synchronization.
    end subroutine nvshmemx_sync_all_on_stream
  end interface

public :: nvshmemx_float_alltoall_on_stream
  interface nvshmemx_float_alltoall_on_stream
  !! Performs an all-to-all exchange of floating-point data on the specified stream.
    function nvshmemx_float_alltoall_on_stream(team, dest, source, nelems, stream) bind(C)
      import
      integer(c_int)                        :: nvshmemx_float_alltoall_on_stream  !! Completion status.
      type(nvshmem_team_t), intent(in), value :: team  !! NVSHMEM team.
      type(c_ptr),                      value :: dest   !! Destination buffer.
      type(c_ptr),                      value :: source !! Source buffer.
      integer(c_size_t),    intent(in), value :: nelems !! Number of elements to exchange.
      type(dtfft_stream_t), intent(in), value :: stream !! CUDA stream for the operation.
    end function nvshmemx_float_alltoall_on_stream
  end interface

public :: nvshmem_ptr
  interface nvshmem_ptr
  !! Returns a pointer to a symmetric memory location on a specified PE.
    function nvshmem_ptr(ptr, pe) bind(C)
      import
      type(c_ptr)               :: nvshmem_ptr  !! Pointer to the symmetric memory on the specified PE.
      type(c_ptr),    value     :: ptr          !! Local pointer to the symmetric memory.
      integer(c_int), value     :: pe           !! PE (Processing Element) number.
    end function nvshmem_ptr
  end interface

public :: nvshmem_my_pe
  interface nvshmem_my_pe
  !! Returns the PE (Processing Element) number of the calling thread.
    function nvshmem_my_pe() bind(C)
      import
      integer(c_int) :: nvshmem_my_pe  !! PE number of the calling thread.
    end function nvshmem_my_pe
  end interface
end module dtfft_interface_nvshmem