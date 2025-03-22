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
module dtfft_interface_vkfft_m
!! This module creates interface with VkFFT library
!!
!! VkFFT is loaded at runtime via dynamic loading.
use iso_c_binding
use iso_fortran_env
use dtfft_parameters
use dtfft_utils
implicit none
private
#include "dtfft_private.h"
public :: load_vkfft

  abstract interface
    subroutine vkfft_create_interface(rank, dims, double_precision, how_many, r2c, c2r, dct, dst, stream, app_handle) bind(C)
    !! Creates FFT plan via vkFFT Interface
    import
      integer(c_int8_t),          value :: rank             !! Rank of fft: 1 or 2
      integer(c_int)                    :: dims(*)          !! Dimensions of transform
      integer(c_int),             value :: double_precision !! Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
      integer(c_int),             value :: how_many         !! Number of transforms to create
      integer(c_int8_t),          value :: r2c              !! Is R2C transform required
      integer(c_int8_t),          value :: c2r              !! Is C2R transform required
      integer(c_int8_t),          value :: dct              !! Is DCT transform required
      integer(c_int8_t),          value :: dst              !! Is DST transform required
      type(dtfft_stream_t),       value :: stream           !! CUDA stream
      type(c_ptr)                       :: app_handle       !! vkFFT application handle
    end subroutine vkfft_create_interface

    subroutine vkfft_execute_interface(app_handle, in, out, sign) bind(C)
    !! Executes vkFFT plan
    import
      type(c_ptr),        value :: app_handle           !! vkFFT application handle
      type(c_ptr),        value :: in                   !! Input data
      type(c_ptr),        value :: out                  !! Output data
      integer(c_int8_t),  value :: sign                 !! Sign of FFT
    end subroutine vkfft_execute_interface

    subroutine vkfft_destroy_interface(app_handle) bind(C)
    !! Destroys vkFFT plan
    import
      type(c_ptr),    value :: app_handle               !! vkFFT application handle
    end subroutine vkfft_destroy_interface
  end interface

public :: vkfft_wrapper
  type :: vkfft_wrapper
  !! VkFFT Wrapper
  private
    logical         :: is_loaded = .false.
      !! Is VkFFT library loaded
    type(c_ptr)     :: lib_handle
      !! Handle to the loaded library
    type(c_funptr)  :: vkfft_functions(3)
      !! Array of VkFFT functions
    procedure(vkfft_create_interface),  pointer, public, nopass :: create
      !! Fortran Pointer to vkFFT create function
    procedure(vkfft_execute_interface), pointer, public, nopass :: execute
      !! Fortran Pointer to vkFFT execute function
    procedure(vkfft_destroy_interface), pointer, public, nopass :: destroy
      !! Fortran Pointer to vkFFT destroy function
  end type vkfft_wrapper

  type(vkfft_wrapper), public, save, target :: cuda_wrapper
    !! VkFFT Wrapper for CUDA platform
contains

  integer(int32) function load_vkfft(platform)
  !! Loads VkFFT library based on the platform
    type(dtfft_platform_t), intent(in) :: platform
      !! Platform to load VkFFT library for

    if ( platform == DTFFT_PLATFORM_CUDA ) then
      load_vkfft = load(cuda_wrapper, "cuda")
    endif
  end function load_vkfft

  function load(wrapper, suffix) result(error_code)
  !! Loads VkFFT library
    class(vkfft_wrapper), intent(inout) :: wrapper  !! VkFFT Wrapper
    character(len=*),     intent(in)    :: suffix   !! Suffix for the library name
    type(string), allocatable :: func_names(:)
    integer(int32)  :: error_code

    error_code = DTFFT_SUCCESS
    if ( wrapper%is_loaded ) return

    allocate(func_names(3))
    func_names(1) = string("vkfft_create")
    func_names(2) = string("vkfft_execute")
    func_names(3) = string("vkfft_destroy")

    error_code = dynamic_load("libdtfft_vkfft_"//suffix//".so", func_names, wrapper%lib_handle, wrapper%vkfft_functions)
    call destroy_strings(func_names)
    if ( error_code /= DTFFT_SUCCESS ) return

    call c_f_procpointer(wrapper%vkfft_functions(1), wrapper%create)
    call c_f_procpointer(wrapper%vkfft_functions(2), wrapper%execute)
    call c_f_procpointer(wrapper%vkfft_functions(3), wrapper%destroy)

    wrapper%is_loaded = .true.
  end function load

end module dtfft_interface_vkfft_m