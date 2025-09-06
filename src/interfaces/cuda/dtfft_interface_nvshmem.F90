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
module dtfft_interface_nvshmem
!! NVSHMEM Interfaces
use iso_c_binding
use iso_fortran_env
use dtfft_parameters
use dtfft_utils
implicit none
private
public :: nvshmem_team_t
public :: is_nvshmem_ptr
! public :: load_nvshmem


  type, bind(C) :: nvshmem_team_t
  !! NVSHMEM team.
    integer(c_int32_t) :: val  !! Internal representation of the NVSHMEM team.
  end type nvshmem_team_t

  type(nvshmem_team_t), parameter, public :: NVSHMEM_TEAM_WORLD = nvshmem_team_t(0)
  !! Global NVSHMEM team.

public :: nvshmem_malloc
public :: nvshmem_free
public :: nvshmemx_sync_all_on_stream
public :: nvshmemx_float_alltoall_on_stream
public :: nvshmem_ptr
public :: nvshmem_my_pe
public :: nvshmemx_init_status

  interface
    function nvshmem_malloc(size) result(ptr) bind(C)
    !! Allocates symmetric memory in the NVSHMEM heap.
      import
      type(c_ptr)               :: ptr  !! Pointer to the allocated memory.
      integer(c_size_t), value  :: size            !! Size of the allocation in bytes.
    end function nvshmem_malloc
  end interface

  interface nvshmem_free
    subroutine nvshmem_free(ptr) bind(C)
    !! Frees symmetric memory allocated by nvshmem_malloc.
      import
      type(c_ptr), value :: ptr  !! Pointer to the memory to free.
    end subroutine nvshmem_free
  end interface

  interface
    subroutine nvshmemx_sync_all_on_stream(stream) bind(C)
    !! Synchronizes all PEs (Processing Elements) on the specified stream.
      import
      type(dtfft_stream_t), intent(in), value :: stream  !! CUDA stream for synchronization.
    end subroutine nvshmemx_sync_all_on_stream
  end interface

  interface
    function nvshmemx_float_alltoall_on_stream(team, dest, source, nelems, stream) result(ierr) bind(C)
    !! Performs an all-to-all exchange of floating-point data on the specified stream.
      import
      integer(c_int)                          :: ierr  !! Completion status.
      type(nvshmem_team_t), intent(in), value :: team   !! NVSHMEM team.
      type(c_ptr),                      value :: dest   !! Destination buffer.
      type(c_ptr),                      value :: source !! Source buffer.
      integer(c_size_t),    intent(in), value :: nelems !! Number of elements to exchange.
      type(dtfft_stream_t), intent(in), value :: stream !! CUDA stream for the operation.
    end function nvshmemx_float_alltoall_on_stream
  end interface

  interface
    function nvshmem_ptr(ptr, pe) result(pe_ptr) bind(C)
    !! Returns a pointer to a symmetric memory location on a specified PE.
      import
      type(c_ptr)               :: pe_ptr  !! Pointer to the symmetric memory on the specified PE.
      type(c_ptr),    value     :: ptr          !! Local pointer to the symmetric memory.
      integer(c_int), value     :: pe           !! PE (Processing Element) number.
    end function nvshmem_ptr
  end interface

  interface
    function nvshmem_my_pe() result(pe) bind(C)
    !! Returns the PE (Processing Element) number of the calling thread.
      import
      integer(c_int) :: pe  !! PE number of the calling thread.
    end function nvshmem_my_pe
  end interface

  interface
    function nvshmemx_init_status() result(status) bind(C)
      import
      integer(c_int)                          :: status  !! Completion status.
    end function nvshmemx_init_status
  end interface

  ! logical, save :: is_loaded = .false.
  !   !! Flag indicating whether the library is loaded
  ! type(c_ptr), save :: libnvshmem
  !   !! Handle to the loaded library
  ! type(c_funptr), save :: nvshmemFunctions(7)
  !   !! Array of pointers to the NVSHMEM functions

  ! procedure(nvshmem_malloc_interface),              pointer, public :: nvshmem_malloc
  !   !! Fortran pointer to the nvshmem_malloc function
  ! procedure(nvshmem_free_interface),                pointer, public :: nvshmem_free
  !   !! Fortran pointer to the nvshmem_free function
  ! procedure(nvshmemx_sync_all_on_stream_interface), pointer, public :: nvshmemx_sync_all_on_stream
  !   !! Fortran pointer to the nvshmemx_sync_all_on_stream function
  ! procedure(nvshmemx_float_alltoall_on_stream_interface), pointer, public :: nvshmemx_float_alltoall_on_stream
  !   !! Fortran pointer to the nvshmemx_float_alltoall_on_stream function
  ! procedure(nvshmem_ptr_interface),                 pointer, public :: nvshmem_ptr
  !   !! Fortran pointer to the nvshmem_ptr function
  ! procedure(nvshmem_my_pe_interface),               pointer, public :: nvshmem_my_pe
  !   !! Fortran pointer to the nvshmem_my_pe function
  ! procedure(nvshmemx_init_status_interface),                pointer,  public  :: nvshmemx_init_status

contains

  ! function load_nvshmem(cufftmp_handle) result(error_code)
  ! !! Loads the NVSHMEM library and needed symbols
  !   type(c_ptr),  intent(in)  :: cufftmp_handle
  !   integer(int32)  :: error_code !! Error code
  !   type(string), allocatable :: func_names(:)
  !   integer(int32) :: i

  !   error_code = DTFFT_SUCCESS
    ! if ( is_loaded ) return

    ! allocate(func_names(7))
    ! func_names(1) = string("nvshmem_malloc")
    ! func_names(2) = string("nvshmem_free")
    ! func_names(3) = string("nvshmemx_sync_all_on_stream")
    ! func_names(4) = string("nvshmemx_float_alltoall_on_stream")
    ! func_names(5) = string("nvshmem_ptr")
    ! func_names(6) = string("nvshmem_my_pe")
    ! func_names(7) = string("nvshmemx_init_status")

    ! ! error_code = dynamic_load(NVSHMEM_HOST_LIB, func_names, libnvshmem, nvshmemFunctions)
    ! ! call destroy_strings(func_names)
    ! ! if ( error_code /= DTFFT_SUCCESS ) return
    ! do i = 1, size(func_names)
    !   nvshmemFunctions(i) = load_symbol(cufftmp_handle, func_names(i)%raw)
    ! enddo

    ! call c_f_procpointer(nvshmemFunctions(1), nvshmem_malloc)
    ! call c_f_procpointer(nvshmemFunctions(2), nvshmem_free)
    ! call c_f_procpointer(nvshmemFunctions(3), nvshmemx_sync_all_on_stream)
    ! call c_f_procpointer(nvshmemFunctions(4), nvshmemx_float_alltoall_on_stream)
    ! call c_f_procpointer(nvshmemFunctions(5), nvshmem_ptr)
    ! call c_f_procpointer(nvshmemFunctions(6), nvshmem_my_pe)
    ! call c_f_procpointer(nvshmemFunctions(7), nvshmemx_init_status)

    ! print*, 'nvshmemx_init_status after load = ', nvshmemx_init_status()

    ! is_loaded = .true.
  ! end function load_nvshmem

  function is_nvshmem_ptr(ptr) result(bool)
  !! Checks if pointer is a symmetric nvshmem allocated pointer
    type(c_ptr)   :: ptr    !! Device pointer
    logical       :: bool   !! Result

    bool = .not.is_null_ptr( nvshmem_ptr(ptr, nvshmem_my_pe()) )
  end function is_nvshmem_ptr
end module dtfft_interface_nvshmem