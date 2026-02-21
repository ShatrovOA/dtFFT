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
module dtfft_interface_cuda
!! CUDA Driver Interfaces
!!
!! CUDA Driver is loaded at runtime via dynamic loading.
use iso_c_binding
use iso_fortran_env
use dtfft_errors,                 only: DTFFT_SUCCESS
use dtfft_parameters,             only: dtfft_stream_t
use dtfft_utils
#include "_dtfft_mpi.h"
implicit none
private
#include "_dtfft_private.h"
public :: load_cuda
public :: cuLaunchKernel

public :: dim3
  type, bind(C) :: dim3
  !! Dimension specification type
    integer(c_int) :: x,y,z
  end type

  integer(int32), parameter, public :: MAX_KERNEL_ARGS = 9

public :: CUmodule
  type, bind(C) :: CUmodule
  !! CUDA module
    type(c_ptr) :: ptr  !! Actual pointer
  end type CUmodule

public :: CUfunction
#ifdef DTFFT_WITH_MOCK_ENABLED
  abstract interface
    pure subroutine simple_interface_r32(in, out, dims)
    import
      real(real32),     intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      real(real32),     intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
    end subroutine simple_interface_r32

    pure subroutine simple_interface_r64(in, out, dims)
    import
      real(real64),     intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      real(real64),     intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
    end subroutine simple_interface_r64

    pure subroutine simple_interface_r128(in, out, dims)
    import
      complex(real64),  intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      complex(real64),  intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
    end subroutine simple_interface_r128

    pure subroutine pipe_interface_r32(in, out, dims, locals)
    import
      real(real32),     intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      real(real32),     intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
      integer(int32),   intent(in)    :: locals(:)                  !! Local memory size specification
    end subroutine pipe_interface_r32

    pure subroutine pipe_interface_r64(in, out, dims, locals)
    import
      real(real64),     intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      real(real64),     intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
      integer(int32),   intent(in)    :: locals(:)                  !! Local memory size specification
    end subroutine pipe_interface_r64

    pure subroutine pipe_interface_r128(in, out, dims, locals)
    import
      complex(real64),  intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
      complex(real64),  intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
      integer(int32),   intent(in)    :: dims(:)                    !! Dimensions of the array
      integer(int32),   intent(in)    :: locals(:)                  !! Local memory size specification
    end subroutine pipe_interface_r128
  end interface

  type :: CUfunction
  !! CUDA function (mock)
    procedure(simple_interface_r32), pointer,  nopass :: sfun_r32 => null()  !! Pointer to the Fortran subroutine implementing the kernel
    procedure(simple_interface_r64), pointer,  nopass :: sfun_r64 => null()  !! Pointer to the Fortran subroutine implementing the kernel
    procedure(simple_interface_r128), pointer, nopass :: sfun_r128 => null()  !! Pointer to the Fortran subroutine implementing the kernel
    procedure(pipe_interface_r32), pointer,  nopass :: pfun_r32 => null()  !! Pointer to the Fortran subroutine implementing the pipelined kernel
    procedure(pipe_interface_r64), pointer,  nopass :: pfun_r64 => null()  !! Pointer to the Fortran subroutine implementing the pipelined kernel
    procedure(pipe_interface_r128), pointer, nopass :: pfun_r128 => null()  !! Pointer to the Fortran subroutine implementing the pipelined kernel
  end type CUfunction
#else
  type, bind(C) :: CUfunction
  !! CUDA function
    type(c_ptr) :: ptr  !! Actual pointer
  end type CUfunction
#endif

  logical,        save :: is_loaded = .false.
    !! Flag indicating whether the library is loaded
  type(c_ptr),    save :: libcuda
    !! Handle to the loaded library
  type(c_funptr), save :: cuFunctions(4)
    !! Array of pointers to the CUDA functions

#ifndef DTFFT_WITH_MOCK_ENABLED
! Real CUDA Driver interfaces with dynamic loading

  abstract interface
    function cuModuleLoadData_interface(mod, image)                                                   &
      result(cuResult)
    !! Load a module's data with options.
    !!
    !! Takes a pointer image and loads the corresponding module module into the current context.
    !! The image may be a cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output by nvcc or hand-written.
    import
      type(CUmodule)        :: mod          !! Returned module
      type(c_ptr),    value :: image        !! Module data to load
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleLoadData_interface

    function cuModuleUnload_interface(hmod)                                                           &
      result(cuResult)
    !! Unloads a module.
    !!
    !! Unloads a module ``hmod`` from the current context.
    !! Attempting to unload a module which was obtained from the Library Management API
    !! such as ``cuLibraryGetModule`` will return ``CUDA_ERROR_NOT_PERMITTED``.
    import
      type(CUmodule), value :: hmod         !! Module to unload
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleUnload_interface

    function cuModuleGetFunction_interface(hfunc, hmod, name)                                          &
      result(cuResult)
    !! Returns a function handle.
    !!
    !! Returns in ``hfunc`` the handle of the function of name name located in module hmod.
    !! If no function of that name exists, ``cuModuleGetFunction`` returns ``CUDA_ERROR_NOT_FOUND``.
    import
      type(CUfunction)      :: hfunc        !! Returns a function handle.
      type(CUmodule), value :: hmod         !! Module to retrieve function from
      type(c_ptr),    value :: name         !! Name of function to retrieve
      integer(c_int)        :: cuResult     !! Driver result code
    end function cuModuleGetFunction_interface

    function cuLaunchKernel_interface(func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra)                                          &
      result(cuResult)
    !! Launches a CUDA function CUfunction.
    import
      type(CUfunction),     value :: func               !! CUDA function to launch
      integer(c_int),       value :: gridDimX           !! Grid dimensions in X
      integer(c_int),       value :: gridDimY           !! Grid dimensions in Y
      integer(c_int),       value :: gridDimZ           !! Grid dimensions in Z
      integer(c_int),       value :: blockDimX          !! Block dimensions in X
      integer(c_int),       value :: blockDimY          !! Block dimensions in Y
      integer(c_int),       value :: blockDimZ          !! Block dimensions in Z
      integer(c_int),       value :: sharedMemBytes     !! Dynamic shared memory size
      type(dtfft_stream_t), value :: stream             !! Stream identifier
      type(c_ptr)                 :: kernelParams(*)    !! Array of pointers to kernel parameters
      type(c_ptr)                 :: extra              !! Dynamic shared-memory size per thread block in bytes
      integer(c_int)              :: cuResult           !! Driver result code
    end function cuLaunchKernel_interface
  end interface

  procedure(cuModuleLoadData_interface),     pointer, public  :: cuModuleLoadData
    !! Fortran pointer to the cuModuleLoadData function
  procedure(cuModuleUnload_interface),       pointer, public  :: cuModuleUnload
    !! Fortran pointer to the cuModuleUnload function
  procedure(cuModuleGetFunction_interface),  pointer, public  :: cuModuleGetFunction
    !! Fortran pointer to the cuModuleGetFunction function
  procedure(cuLaunchKernel_interface),       pointer          :: cuLaunchKernel_
    !! Fortran pointer to the cuLaunchKernel function

#else
! Mock CUDA Driver interfaces for CPU testing
public :: cuModuleLoadData, cuModuleUnload, cuModuleGetFunction
#endif

contains

#ifdef DTFFT_WITH_MOCK_ENABLED
  ! Mock implementations for CPU testing

  function cuModuleLoadData(mod, image) result(cuResult)
  !! Mock: Creates dummy module
    type(CUmodule)         :: mod
    type(c_ptr), intent(in) :: image
    integer(c_int)         :: cuResult
    mod%ptr = image
    cuResult = 0  ! CUDA_SUCCESS
  end function cuModuleLoadData

  function cuModuleUnload(hmod) result(cuResult)
  !! Mock: Does nothing
    type(CUmodule), intent(in) :: hmod
    integer(c_int)             :: cuResult
    cuResult = 0  ! CUDA_SUCCESS
  end function cuModuleUnload

  function cuModuleGetFunction(hfunc, hmod, name) result(cuResult)
  !! Mock: Returns dummy function handle
    type(CUfunction)           :: hfunc
    type(CUmodule),  intent(in) :: hmod
    type(c_ptr),     intent(in) :: name
    integer(c_int)             :: cuResult
    ! hfunc%ptr = name
    cuResult = 0  ! CUDA_SUCCESS
  end function cuModuleGetFunction

  function load_cuda() result(error_code)
  !! Mock: Does nothing, always returns success
    integer(int32)  :: error_code
    type(string), allocatable :: func_names(:)
    integer(int32)  :: ierr

    error_code = DTFFT_SUCCESS
    if ( is_loaded ) return
    allocate(func_names(1))
    func_names(1) = string("dtfft_execute")

    ! Just try loading both libs
    ierr = dynamic_load("libdtfft.so", func_names, libcuda, cuFunctions)
    if ( ierr /= DTFFT_SUCCESS ) then
      ierr = dynamic_load("libdtfft.dylib", func_names, libcuda, cuFunctions)
    endif

    call destroy_strings(func_names)
    if ( ierr == DTFFT_SUCCESS ) then
      call unload_library(libcuda)
    endif
    is_loaded = .true.
  end function load_cuda

#else
  ! Real CUDA Driver implementation with dynamic loading

  function load_cuda() result(error_code)
  !! Loads the CUDA Driver library and needed symbols
    integer(int32)  :: error_code !! Error code
    type(string), allocatable :: func_names(:)

    error_code = DTFFT_SUCCESS
    if ( is_loaded ) return

    allocate(func_names(4))
    func_names(1) = string("cuModuleLoadData")
    func_names(2) = string("cuModuleUnload")
    func_names(3) = string("cuModuleGetFunction")
    func_names(4) = string("cuLaunchKernel")

    error_code = dynamic_load("libcuda.so", func_names, libcuda, cuFunctions)
    call destroy_strings(func_names)
    if ( error_code /= DTFFT_SUCCESS ) return

    call c_f_procpointer(cuFunctions(1), cuModuleLoadData)
    call c_f_procpointer(cuFunctions(2), cuModuleUnload)
    call c_f_procpointer(cuFunctions(3), cuModuleGetFunction)
    call c_f_procpointer(cuFunctions(4), cuLaunchKernel_)

    is_loaded = .true.
  end function load_cuda

#endif

  function cuLaunchKernel(func, in, out, blocks, threads, stream, nargs, args) result(cuResult)
  !! Launches a CUDA kernel
    type(CUfunction),         intent(in)  :: func             !! Function CUfunction or Kernel CUkernel to launch
    type(c_ptr),      target, intent(in)  :: in               !! Input pointer
    type(c_ptr),      target, intent(in)  :: out              !! Output pointer
    type(dim3),               intent(in)  :: blocks           !! Grid in blocks
    type(dim3),               intent(in)  :: threads          !! Thread block
    type(dtfft_stream_t),     intent(in)  :: stream           !! Stream identifier
    integer(int32),           intent(in)  :: nargs
    integer(int32),   target, intent(in)  :: args(MAX_KERNEL_ARGS)     !! Input parameters of kernel `func`
    integer(c_int)                        :: cuResult         !! Driver result code
    type(c_ptr)                           :: params(15)
    integer(int32) :: i, temp
    integer(int32) :: dims(3), locals(5)

#ifndef DTFFT_WITH_MOCK_ENABLED
    params(:) = c_null_ptr
    ! Addresses of pointers are required, not the pointers themselves
    params(1) = c_loc(out)
    params(2) = c_loc(in)

    temp = 2
    do i = 1, nargs
      params(temp + i) = c_loc(args(i))
    enddo
    cuResult = cuLaunchKernel_(func, blocks%x, blocks%y, blocks%z, threads%x, threads%y, threads%z, 0, stream, params, c_null_ptr)
#else
    dims(:) = 1
    locals(:) = 0

    if ( associated(func%sfun_r32) .or. associated(func%sfun_r64) .or. associated(func%sfun_r128)) then
      do i = 1, nargs
        dims(i) = args(i)
      enddo

      if ( associated(func%sfun_r32) ) then
        block
          real(real32), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          call func%sfun_r32(in_ptr, out_ptr, dims)
        endblock
      else if ( associated(func%sfun_r64) ) then
        block
          real(real64), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          call func%sfun_r64(in_ptr, out_ptr, dims)
        endblock
      else if ( associated(func%sfun_r128) ) then
        block
          complex(real64), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          call func%sfun_r128(in_ptr, out_ptr, dims)
        endblock
      endif
    else if ( associated(func%pfun_r32) .or. associated(func%pfun_r64) .or. associated(func%pfun_r128)) then
      if ( args(3) == -1 )then
        dims(1:2) = args(1:2)
        locals(1:5) = args(3:7)
      else
        dims(1:3) = args(1:3)
        locals(1:5) = args(4:8)
      endif
      if ( associated(func%pfun_r32) ) then
        block
          real(real32), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          
          call func%pfun_r32(in_ptr, out_ptr, dims, locals)
        endblock
      else if ( associated(func%pfun_r64) ) then
        block
          real(real64), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          call func%pfun_r64(in_ptr, out_ptr, dims, locals)
        endblock
      else if ( associated(func%pfun_r128) ) then
        block
          complex(real64), pointer, contiguous :: in_ptr(:), out_ptr(:)

          call c_f_pointer(in, in_ptr, [product(dims)])
          call c_f_pointer(out, out_ptr, [product(dims)])
          call func%pfun_r128(in_ptr, out_ptr, dims, locals)
        endblock
      endif
    else
      INTERNAL_ERROR("cuLaunchKernel: invalid function handle")
    endif

    cuResult = DTFFT_SUCCESS
#endif
  end function cuLaunchKernel
end module dtfft_interface_cuda