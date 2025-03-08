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
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
implicit none
private
public :: nvshmem_malloc, nvshmem_free, nvshmemx_sync_all_on_stream
public :: nvshmem_team_t
public :: nvshmemx_float_alltoall_on_stream
public :: nvshmem_ptr, nvshmem_my_pe


  type, bind(C) :: nvshmem_team_t
    integer(c_int32_t) :: val
  end type nvshmem_team_t

  type(nvshmem_team_t), parameter, public :: NVSHMEM_TEAM_WORLD = nvshmem_team_t(0)

  interface
    type(c_ptr) function nvshmem_malloc(size) bind(C)
    import
      integer(c_size_t), value :: size  ! Size is in bytes 
    end function nvshmem_malloc

    subroutine nvshmem_free(ptr) bind(C)
    import
      type(c_ptr), value :: ptr
    end subroutine nvshmem_free

    subroutine nvshmemx_sync_all_on_stream(stream) bind(C)
    import
      type(dtfft_stream_t), intent(in), value  :: stream
    end subroutine nvshmemx_sync_all_on_stream

    integer(c_int) function nvshmemx_float_alltoall_on_stream(team, dest, source, nelems, stream) bind(C)
    import
      type(nvshmem_team_t), intent(in), value :: team
      type(c_ptr),                      value :: dest
      type(c_ptr),                      value :: source
      integer(c_size_t),    intent(in), value :: nelems
      type(dtfft_stream_t), intent(in), value :: stream
    end function nvshmemx_float_alltoall_on_stream

    type(c_ptr) function nvshmem_ptr(ptr, pe) bind(C)
    import
      type(c_ptr),    value :: ptr
      integer(c_int), value :: pe
    end function nvshmem_ptr

    integer(c_int) function nvshmem_my_pe() bind(C)
    import
    end function nvshmem_my_pe
  end interface
end module dtfft_interface_nvshmem