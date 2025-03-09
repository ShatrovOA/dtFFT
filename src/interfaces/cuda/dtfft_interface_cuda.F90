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
use iso_c_binding
use dtfft_parameters,             only: dtfft_stream_t
use dtfft_interface_cuda_runtime, only: dim3
implicit none
private

public :: kernelArgs
  type, bind(C) :: kernelArgs
    integer(c_int)  :: n_ints = 0
    integer(c_int)  :: ints(5)
    integer(c_int)  :: n_ptrs = 0
    type(c_ptr)     :: ptrs(3)
  end type kernelArgs

public :: cuModuleLoadDataEx
  interface cuModuleLoadDataEx
  !! Load a module's data with options.
  !!
  !! Takes a pointer image and loads the corresponding module module into the current context. 
  !! The image may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output by nvcc or hand-written.
    function cuModuleLoadDataEx(mod, image, numOptions, options, optionValues)              &
      result(cuResult)                                                                      &
      bind(C, name="cuModuleLoadDataEx")
    import
      type(c_ptr)           :: mod          !! Returned module
      character(c_char)     :: image(*)     !! Module data to load
      integer(c_int), value :: numOptions   !! Number of options
      type(c_ptr),    value :: options      !! Options for JIT
      type(c_ptr)           :: optionValues !! Option values for JIT
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleLoadDataEx
  end interface

public :: cuModuleUnload
  interface cuModuleUnload
  !! Unloads a module.
  !!
  !! Unloads a module ``hmod`` from the current context. 
  !! Attempting to unload a module which was obtained from the Library Management API 
  !! such as ``cuLibraryGetModule`` will return ``CUDA_ERROR_NOT_PERMITTED``.
    function cuModuleUnload(hmod)                                                           &
      result(cuResult)                                                                      &
      bind(C, name="cuModuleUnload")
    import
      type(c_ptr), value    :: hmod         !! Module to unload
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleUnload
  end interface

public :: cuModuleGetFunction
  interface 
  ! cuModuleGetFunction
  !! Returns a function handle.
  !!
  !! Returns in ``hfunc`` the handle of the function of name name located in module hmod.
  !! If no function of that name exists, ``cuModuleGetFunction`` returns ``CUDA_ERROR_NOT_FOUND``.
    function cuModuleGetFunction(hfunc, hmod, name)                                          &
      result(cuResult)                                                                      &
      bind(C, name="cuModuleGetFunction")
    import
      type(c_ptr)           :: hfunc        !! Returns a function handle.
      type(c_ptr),    value :: hmod         !! Module to retrieve function from
      character(c_char)     :: name(*)      !! Name of function to retrieve
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleGetFunction
  end interface

  ! interface cuLaunchKernel
  ! !! Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
  !   function cuLaunchKernel(func, gridDimX, gridDimY, gridDimZ,                             &
  !     blockDimX, blockDimY, blockDimZ, sharedMemBytes,                                      &
  !     stream, kernelParams, extra)                                                          &
  !     result(cuResult)                                                                      &
  !     bind(C, name="cuLaunchKernel")
  !   import
  !     type(c_ptr),                  value :: func         !! Function CUfunction or Kernel CUkernel to launch
  !     integer(c_int),               value :: gridDimX
  !     integer(c_int),               value :: gridDimY
  !     integer(c_int),               value :: gridDimZ
  !     integer(c_int),               value :: blockDimX
  !     integer(c_int),               value :: blockDimY
  !     integer(c_int),               value :: blockDimZ
  !     integer(c_int),               value :: sharedMemBytes
  !     type(dtfft_stream_t),         value :: stream       !! Stream identifier
  !     type(c_ptr)                         :: kernelParams(*)
  !     type(c_ptr),                  value :: extra
  !     integer(c_int)                      :: cuResult     !! Driver result code
  !   end function cuLaunchKernel
  ! end interface

public :: cuLaunchKernel
  interface cuLaunchKernel
  !! Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
    function cuLaunchKernel(func, in, out, blocks, threads, stream, args)                 &
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
      integer(c_int)                      :: cuResult     !! Driver result code
    end function cuLaunchKernel
  end interface
end module dtfft_interface_cuda