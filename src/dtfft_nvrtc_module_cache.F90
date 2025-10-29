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
module dtfft_nvrtc_module_cache
!! Module that implements a cache for nvrtc modules
!! Cache is used to avoid recompilation of kernels with the same parameters
use iso_fortran_env
use iso_c_binding,                only: c_null_ptr
use dtfft_abstract_kernel,        only: kernel_type_t, get_kernel_string
use dtfft_config
use dtfft_interface_cuda,         only: CUfunction
use dtfft_interface_cuda_runtime, only: device_props
use dtfft_nvrtc_block_optimizer,  only: kernel_config
use dtfft_nvrtc_module
use dtfft_utils
#include "_dtfft_private.h"
implicit none
private
public :: create_nvrtc_module, get_kernel_instance


  integer(int32), parameter         :: CACHE_PREALLOC_SIZE = 10
  !! Number of preallocated cache entries

  type :: nvrtc_module_cache
  !! Cache for nvrtc modules
  !!
  !! This type is used internally by the module and is not exposed to the user
  !! It maintains a list of compiled nvrtc modules and provides methods to add new modules
  !! and retrieve existing ones
  !!
  !! The cache automatically grows as needed
  private
    logical                         :: is_created = .false.     !! Flag that indicates if cache is created
    type(nvrtc_module), allocatable :: cache(:)                 !! Array of cached modules
    integer(int32)                  :: size                     !! Number of entries in cache
  contains
  private
    procedure,  pass(self) :: create    !! Creates cache
    procedure,  pass(self) :: add       !! Adds new entry to cache
  end type nvrtc_module_cache

  type(nvrtc_module_cache), save :: cache
  !! Cache instance

contains

  subroutine create(self)
  !! Creates cache
    class(nvrtc_module_cache),  intent(inout) :: self   !! Cache instance

    if (self%is_created) return
    allocate(self%cache(CACHE_PREALLOC_SIZE))
    self%size = 0
    self%is_created = .true.
  end subroutine create

  subroutine add(self, m)
  !! Adds new entry to cache
    class(nvrtc_module_cache),  intent(inout) :: self   !! Cache instance
    type(nvrtc_module),         intent(in)    :: m      !! Module to add
    type(nvrtc_module),  allocatable  :: temp(:)        !! Temporary cache

    call cache%create()

    if ( self%size == size(self%cache) ) then
      allocate( temp(self%size + CACHE_PREALLOC_SIZE) )
      temp(1:self%size) = self%cache(1:cache%size)
      deallocate( self%cache )
      call move_alloc(temp, self%cache)
    endif

    self%size = self%size + 1
    self%cache(self%size) = m
    WRITE_DEBUG('Added to cache: size = '//to_str(self%size))
  end subroutine add

  subroutine create_nvrtc_module(ndims, kernel_type, base_storage, configs, props)
  !! Creates and adds a new nvrtc module to the cache if it does not already exist
    integer(int32),       intent(in)    :: ndims              !! Number of dimensions (2 or 3)
    type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
    integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(kernel_config),  intent(in)    :: configs(:)         !! Array of kernel configurations to build
    type(device_props),   intent(in)    :: props              !! GPU architecture properties
    integer(int32)                    :: i                  !! Loop counter
    integer(int32)                    :: j                  !! Loop counter
    integer(int32)                    :: k                  !! Loop counter for number of new configs to add
    logical                           :: is_found           !! Flag that indicates if module with required kernel_type was found
    logical                           :: is_instance_found  !! Flag that indicates if specific kernel instance was found
    type(nvrtc_module)                :: m                  !! New module to create
    type(kernel_config), allocatable  :: configs_to_add(:)  !! Configurations that need to be added

    is_found = .false.
    do i = 1, cache%size
      is_found = cache%cache(i)%check(ndims, kernel_type, base_storage)
      if ( is_found ) exit
    enddo

    k = 0
    if ( is_found ) then
      ! Found at least one module with required kernel_type
      ! Now need to loop over all modules once again and check if required configs are present in any of them
      allocate( configs_to_add(size( configs )) )

      do j = 1, size(configs)
        is_instance_found = .false.
        do i = 1, cache%size
          is_instance_found = cache%cache(i)%check(ndims, kernel_type, base_storage, configs(j)%tile_size, configs(j)%block_rows)
          if ( is_instance_found ) exit
        enddo
        if ( .not. is_instance_found ) then
          k = k + 1
          configs_to_add(k)%tile_size = configs(j)%tile_size
          configs_to_add(k)%block_rows = configs(j)%block_rows
          configs_to_add(k)%padding = configs(j)%padding
          WRITE_DEBUG("Adding new: "//to_str(configs(j)%tile_size)//"x"//to_str(configs(j)%block_rows)//": "//to_str(configs(j)%padding))
        endif
      enddo
      if ( k > 0 ) then
        WRITE_DEBUG("Adding "//to_str(k)//" new kernel configurations for module '"//get_kernel_string(kernel_type)//"'")
        call m%create(ndims, kernel_type, base_storage, configs_to_add(1:k), props)
      endif
      deallocate( configs_to_add )
    else
      WRITE_DEBUG("Adding new module with following configs for '"//get_kernel_string(kernel_type)//"'")
      do j = 1, size(configs)
        WRITE_DEBUG(to_str(configs(j)%tile_size)//"x"//to_str(configs(j)%block_rows)//": "//to_str(configs(j)%padding))
      enddo
      call m%create(ndims, kernel_type, base_storage, configs, props)
    endif

    if ( k == 0 .and. is_found ) return
    call cache%add(m)
  end subroutine create_nvrtc_module

  function get_kernel_instance(ndims, kernel_type, base_storage, tile_size, block_rows) result(fun)
  !! Retrieves a kernel instance from the cache
  !! If the instance is not found, an error is raised
    integer(int32),       intent(in)    :: ndims              !! Number of dimensions (2 or 3)
    type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
    integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element
    integer(int32),       intent(in)    :: tile_size          !! Tile size (number of columns)
    integer(int32),       intent(in)    :: block_rows         !! Block rows
    type(CUfunction)                    :: fun                !! Retrieved kernel instance
    logical         :: is_found !! Flag that indicates if instance was found
    integer(int32)  :: i        !! Loop counter

    fun = CUfunction(c_null_ptr)
    if ( .not. cache%is_created ) INTERNAL_ERROR("get_kernel_instance: cache not created")
    is_found = .false.
    do i = 1, cache%size
      fun = cache%cache(i)%get(ndims, kernel_type, base_storage, tile_size, block_rows)
      if ( .not. is_null_ptr(fun%ptr) ) return
    enddo
    WRITE_DEBUG("Kernel = "//get_kernel_string(kernel_type)//": "//to_str(tile_size)//"x"//to_str(block_rows))
    INTERNAL_ERROR("get_kernel_instance: unable to retrive function from cache")
  end function get_kernel_instance
end module dtfft_nvrtc_module_cache