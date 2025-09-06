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
module dtfft_nvrtc_kernel_cache
use iso_fortran_env
use iso_c_binding,        only: c_null_ptr
use dtfft_config,         only: get_conf_log_enabled
use dtfft_interface_cuda, only: CUmodule, CUfunction, cuModuleUnload
use dtfft_interface_cuda_runtime
use dtfft_parameters
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private

  integer(int32), parameter         :: CACHE_PREALLOC_SIZE = 10
  !! Number of preallocated cache entries

  type :: nvrtc_cache_entry
  !! Cache entry for a compiled kernel
  private
    integer(int32)            :: ref_count = 0                        !! Number of references to this kernel
    type(CUmodule)            :: cuda_module = CUmodule(c_null_ptr)   !! Pointer to CUDA Module.
    type(CUfunction)          :: cuda_kernel = CUfunction(c_null_ptr) !! Pointer to CUDA kernel.
    type(kernel_type_t)       :: kernel_type                          !! Type of kernel to execute.
    type(dtfft_transpose_t)   :: transpose_type                       !! Type of transpose
    integer(int32)            :: tile_size                            !! Tile size of transpose kernel
    integer(int32)            :: padding                              !! Padding size of transpose kernel
    integer(int64)            :: base_storage                         !! Number of bytes needed to store single element
  end type nvrtc_cache_entry

  type :: nvrtc_cache
  !! Cache for compiled kernels
  private
    logical                               :: is_created = .false.     !! Flag indicating if cache is created
    type(nvrtc_cache_entry), allocatable  :: cache(:)                 !! Cache entries
    integer(int32)                        :: size                     !! Number of entries in cache
  contains
  private
    procedure,  pass(self), public :: create    !! Creates cache
    procedure,  pass(self), public :: add       !! Adds new entry to cache
    procedure,  pass(self), public :: get       !! Gets entry from cache
    procedure,  pass(self), public :: remove    !! Removes entry from cache
    procedure,  pass(self), public :: cleanup   !! Cleans up cache
  end type nvrtc_cache


  type(nvrtc_cache),  public, save :: cache
    !! Cache of compiled kernels

contains

  subroutine create(self)
  !! Creates cache
    class(nvrtc_cache),       intent(inout) :: self           !! Cache instance

    if (self%is_created) return
    allocate(self%cache(CACHE_PREALLOC_SIZE))
    self%size = 0
    self%is_created = .true.
  end subroutine create

  subroutine add(self, cuda_module, cuda_kernel, kernel_type, transpose_type, tile_size, padding, base_storage)
  !! Adds new entry to cache
    class(nvrtc_cache),       intent(inout) :: self           !! Cache instance
    type(CUmodule),           intent(in)    :: cuda_module    !! Compiled CUDA module
    type(CUfunction),         intent(in)    :: cuda_kernel    !! Extracted CUDA kernel
    type(kernel_type_t),      intent(in)    :: kernel_type    !! Kernel type
    type(dtfft_transpose_t),  intent(in)    :: transpose_type !! Transpose type
    integer(int32),           intent(in)    :: tile_size      !! Tile size
    integer(int32),           intent(in)    :: padding        !! Padding
    integer(int64),           intent(in)    :: base_storage   !! Base storage
    type(nvrtc_cache_entry),  allocatable   :: temp(:)        !! Temporary cache
    type(dtfft_transpose_t)   :: transpose_type_    !! Fixed id of transposition

    call self%create()

    ! Need more cache
    if ( self%size == size(self%cache) ) then
      allocate( temp(self%size + CACHE_PREALLOC_SIZE) )
      temp(1:self%size) = self%cache(1:self%size)
      deallocate( self%cache )
      call move_alloc(temp, self%cache)
    endif
    transpose_type_ = get_true_transpose_type(transpose_type)

    self%size = self%size + 1
    self%cache(self%size)%cuda_module = cuda_module
    self%cache(self%size)%cuda_kernel = cuda_kernel
    self%cache(self%size)%kernel_type = kernel_type
    self%cache(self%size)%transpose_type = transpose_type_
    self%cache(self%size)%tile_size = tile_size
    self%cache(self%size)%padding = padding
    self%cache(self%size)%base_storage = base_storage
    self%cache(self%size)%ref_count = 1
  end subroutine add

  function get(self, transpose_type, kernel_type, base_storage, tile_size, padding) result(kernel)
  !! Returns cached kernel if it exists.
  !! If not returns null pointer.
    class(nvrtc_cache),       intent(inout) :: self               !! Cache instance
    type(dtfft_transpose_t),  intent(in)    :: transpose_type     !! Type of transposition to perform
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel to build
    integer(int64),           intent(in)    :: base_storage       !! Number of bytes needed to store single element
    integer(int32),           intent(in)    :: tile_size          !! Tile size
    integer(int32),           intent(in)    :: padding            !! Padding
    type(CUfunction)          :: kernel             !! Cached kernel
    type(dtfft_transpose_t)   :: transpose_type_    !! Fixed id of transposition
    integer(int32)            :: i                  !! Counter

    kernel = CUfunction(c_null_ptr)
    if ( .not. self%is_created ) return
    transpose_type_ = get_true_transpose_type(transpose_type)
    do i = 1, self%size
      if ( self%cache(i)%transpose_type == transpose_type_                                    &
        .and. self%cache(i)%kernel_type == kernel_type                                        &
        .and. self%cache(i)%base_storage == base_storage                                      &
        .and. self%cache(i)%tile_size == tile_size                                            &
        .and. self%cache(i)%tile_size /= VARIABLE_NOT_SET                                     &
        .and. self%cache(i)%padding == padding                                                &
        .or. ( self%cache(i)%kernel_type == kernel_type .and. is_unpack_kernel(kernel_type) ) &
    ) then
      kernel = self%cache(i)%cuda_kernel
      self%cache(i)%ref_count = self%cache(i)%ref_count + 1
      return
     endif
    end do
  end function get

  subroutine cleanup(self)
  !! Removes unused modules from cuda context
    class(nvrtc_cache),       intent(inout) :: self   !! Cache instance
    integer(int32)  :: i  !! Counter

    if ( .not. self%is_created ) return
    do i = 1, self%size
      if ( self%cache(i)%ref_count == 0 .and. .not.is_null_ptr(self%cache(i)%cuda_module%ptr) ) then
        CUDA_CALL( "cuModuleUnload", cuModuleUnload(self%cache(i)%cuda_module) )
        self%cache(i)%cuda_module = CUmodule(c_null_ptr)
        self%cache(i)%cuda_kernel = CUfunction(c_null_ptr)
        self%cache(i)%base_storage = 0
        self%cache(i)%kernel_type = kernel_type_t(0)
        self%cache(i)%tile_size = -1
        self%cache(i)%padding = -1
        self%cache(i)%transpose_type%val = 0
      endif
    enddo
    if ( all( self%cache(:)%ref_count == 0 ) ) then
      deallocate( self%cache )
      self%size = 0
      self%is_created = .false.
      WRITE_DEBUG("nvrtc_cache.cleanup: Cleared all cache")
    else
      WRITE_DEBUG("nvrtc_cache.cleanup: Some of entries are still in use...")
    endif
  end subroutine cleanup

  subroutine remove(self, kernel)
  !! Takes CUDA kernel as an argument and searches for it in cache
  !! If kernel is found than reduces `ref_count` of such entry and kernel becomes a null pointer
    class(nvrtc_cache), intent(inout) :: self     !! Cache instance
    type(CUfunction),   intent(inout) :: kernel   !! CUDA kernel to search for
    integer(int32)              :: i        !! Counter

    if ( .not. self%is_created ) return
    if ( is_null_ptr(kernel%ptr) ) return
    do i = 1, self%size
      if ( is_same_ptr(self%cache(i)%cuda_kernel%ptr, kernel%ptr) ) then
        kernel = CUfunction(c_null_ptr)
        self%cache(i)%ref_count = self%cache(i)%ref_count - 1
        return
      endif
    end do
  end subroutine remove

  function get_true_transpose_type(transpose_type) result(transpose_type_)
  !! Returns generic transpose id.
  !! Since X-Y and Y-Z transpositions are symmectric, it returns only one of them.
  !! X-Z and Z-X are not symmetric
    type(dtfft_transpose_t), intent(in)    :: transpose_type       !! Type of transposition to perform
    type(dtfft_transpose_t)                :: transpose_type_      !! Fixed id of transposition

    if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
      transpose_type_ = transpose_type
    else
      transpose_type_%val = abs(transpose_type%val)
    endif
  end function get_true_transpose_type

end module dtfft_nvrtc_kernel_cache