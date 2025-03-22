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
module dtfft_interface_cuda
!! CUDA Driver Interfaces
!!
!! CUDA Driver is loaded at runtime via dynamic loading.
use iso_c_binding
use iso_fortran_env,              only: int32
use dtfft_parameters
use dtfft_interface_cuda_runtime, only: dim3
use dtfft_utils
implicit none
private
#include "dtfft_private.h"
public :: load_cuda
public :: cuLaunchKernel

public :: kernelArgs
  type, bind(C) :: kernelArgs
  !! Arguments passed to nvrtc-compiled kernels
    integer(c_int)  :: n_ints = 0   !! Number of integers provided
    integer(c_int)  :: ints(5)      !! Integer array
    integer(c_int)  :: n_ptrs = 0   !! Number of pointers provided
    type(c_ptr)     :: ptrs(3)      !! Pointer array
  end type kernelArgs

  abstract interface
    function cuModuleLoadDataEx_interface(mod, image, numOptions, options, optionValues)              &
      result(cuResult)
    !! Load a module's data with options.
    !!
    !! Takes a pointer image and loads the corresponding module module into the current context. 
    !! The image may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output by nvcc or hand-written.
    import
      type(c_ptr)           :: mod          !! Returned module
      character(c_char)     :: image(*)     !! Module data to load
      integer(c_int), value :: numOptions   !! Number of options
      type(c_ptr),    value :: options      !! Options for JIT
      type(c_ptr)           :: optionValues !! Option values for JIT
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleLoadDataEx_interface

    function cuModuleUnload_interface(hmod)                                                           &
      result(cuResult)
    !! Unloads a module.
    !!
    !! Unloads a module ``hmod`` from the current context. 
    !! Attempting to unload a module which was obtained from the Library Management API 
    !! such as ``cuLibraryGetModule`` will return ``CUDA_ERROR_NOT_PERMITTED``.
    import
      type(c_ptr), value    :: hmod         !! Module to unload
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleUnload_interface

    function cuModuleGetFunction_interface(hfunc, hmod, name)                                          &
      result(cuResult)
    !! Returns a function handle.
    !!
    !! Returns in ``hfunc`` the handle of the function of name name located in module hmod.
    !! If no function of that name exists, ``cuModuleGetFunction`` returns ``CUDA_ERROR_NOT_FOUND``.
    import
      type(c_ptr)           :: hfunc        !! Returns a function handle.
      type(c_ptr),    value :: hmod         !! Module to retrieve function from
      character(c_char)     :: name(*)      !! Name of function to retrieve
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleGetFunction_interface
  end interface

  interface run_cuda_kernel
  !! Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
    function run_cuda_kernel(func, in, out, blocks, threads, stream, args, funptr)          &
      result(cuResult)                                                                      &
      bind(C, name="run_cuda_kernel")
    !! Wrapper around ``cuLaunchKernel``, since I have to idea how to pass array of pointers to ``cuLaunchKernel``.
    !!
    !! Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
    import
      type(c_ptr),                  value :: func         !! Function CUfunction or Kernel CUkernel to launch
      type(c_ptr),                  value :: in           !! Input pointer
      type(c_ptr),                  value :: out          !! Output pointer
      type(dim3)                          :: blocks       !! Grid in blocks
      type(dim3)                          :: threads      !! Thread block
      type(dtfft_stream_t),         value :: stream       !! Stream identifier
      type(kernelArgs)                    :: args         !! Kernel parameters
      type(c_funptr),               value :: funptr       !! Pointer to ``cuLaunchKernel``
      integer(c_int)                      :: cuResult     !! Driver result code
    end function run_cuda_kernel
  end interface

  logical,        save :: is_loaded = .false.
    !! Flag indicating whether the library is loaded
  type(c_ptr),    save :: libcuda
    !! Handle to the loaded library
  type(c_funptr), save :: cuFunctions(4)
    !! Array of pointers to the CUDA functions

  procedure(cuModuleLoadDataEx_interface),   pointer, public :: cuModuleLoadDataEx
    !! Fortran pointer to the cuModuleLoadDataEx function
  procedure(cuModuleUnload_interface),       pointer, public :: cuModuleUnload
    !! Fortran pointer to the cuModuleUnload function
  procedure(cuModuleGetFunction_interface),  pointer, public :: cuModuleGetFunction
    !! Fortran pointer to the cuModuleGetFunction function

contains

  function load_cuda() result(error_code)
  !! Loads the CUDA Driver library and needed symbols
    integer(int32)  :: error_code !! Error code
    type(string), allocatable :: func_names(:)

    error_code = DTFFT_SUCCESS
    if ( is_loaded ) return

    allocate(func_names(4))
    func_names(1) = string("cuModuleLoadDataEx")
    func_names(2) = string("cuModuleUnload")
    func_names(3) = string("cuModuleGetFunction")
    func_names(4) = string("cuLaunchKernel")

    error_code = dynamic_load("libcuda.so", func_names, libcuda, cuFunctions)
    call destroy_strings(func_names)
    if ( error_code /= DTFFT_SUCCESS ) return

    call c_f_procpointer(cuFunctions(1), cuModuleLoadDataEx)
    call c_f_procpointer(cuFunctions(2), cuModuleUnload)
    call c_f_procpointer(cuFunctions(3), cuModuleGetFunction)

    is_loaded = .true.
  end function load_cuda

  function cuLaunchKernel(func, in, out, blocks, threads, stream, args) result(cuResult)
  !! Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
    type(c_ptr)           :: func         !! Function CUfunction or Kernel CUkernel to launch
    type(c_ptr)           :: in           !! Input pointer
    type(c_ptr)           :: out          !! Output pointer
    type(dim3)            :: blocks       !! Grid in blocks
    type(dim3)            :: threads      !! Thread block
    type(dtfft_stream_t)  :: stream       !! Stream identifier
    type(kernelArgs)      :: args         !! Kernel parameters
    integer(c_int)        :: cuResult     !! Driver result code
    cuResult = run_cuda_kernel(func, in, out, blocks, threads, stream, args, cuFunctions(4))
  end function cuLaunchKernel
end module dtfft_interface_cuda