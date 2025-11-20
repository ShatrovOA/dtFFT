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
module dtfft_kernel_host
!! This module defines `kernel_host` type and its type bound procedures.
!! The host kernel is an implementation of the `abstract_kernel` type
!! that runs on the host CPU.
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_config
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: kernel_host

! Exporting internal kernels for testing purposes
public :: unpack_pipelined_f32
public :: permute_forward_write_f32, permute_backward_write_f32
public :: permute_backward_start_write_f32
public :: permute_forward_write_f32_block_16, permute_backward_write_f32_block_16
public :: permute_backward_start_write_f32_block_16
public :: permute_forward_write_f32_block_32, permute_backward_write_f32_block_32
public :: permute_backward_start_write_f32_block_32
public :: permute_forward_write_f32_block_64, permute_backward_write_f32_block_64
public :: permute_backward_start_write_f32_block_64
public :: permute_forward_read_f32, permute_backward_read_f32
public :: permute_backward_start_read_f32
public :: permute_forward_read_f32_block_16, permute_backward_read_f32_block_16
public :: permute_backward_start_read_f32_block_16
public :: permute_forward_read_f32_block_32, permute_backward_read_f32_block_32
public :: permute_backward_start_read_f32_block_32
public :: permute_forward_read_f32_block_64, permute_backward_read_f32_block_64
public :: permute_backward_start_read_f32_block_64
public :: permute_backward_end_pipelined_write_f32, permute_backward_end_pipelined_read_f32
public :: permute_backward_end_pipelined_write_f32_block_16
public :: permute_backward_end_pipelined_read_f32_block_16
public :: permute_backward_end_pipelined_write_f32_block_32
public :: permute_backward_end_pipelined_read_f32_block_32
public :: permute_backward_end_pipelined_write_f32_block_64
public :: permute_backward_end_pipelined_read_f32_block_64
public :: unpack_pipelined_f32_block_16
public :: unpack_pipelined_f32_block_32
public :: unpack_pipelined_f32_block_64

integer(int8), parameter :: ACCESS_MODE_WRITE = -1_int8
    !! Aligned writing
integer(int8), parameter :: ACCESS_MODE_READ  = +1_int8
    !! Aligned reading

integer(int8), parameter :: DEFAULT_ACCESS_MODE = ACCESS_MODE_WRITE
    !! Assuming that aligned writing is more important then aligned reading

type :: host_kernel_t
    integer(int8) :: val
end type host_kernel_t

type(host_kernel_t), parameter :: HOST_KERNEL_BASE = host_kernel_t(1_int8)
    !! Base host kernel type
type(host_kernel_t), parameter :: HOST_KERNEL_BLOCK_16 = host_kernel_t(2_int8)
    !! Host kernel with block size of 16
type(host_kernel_t), parameter :: HOST_KERNEL_BLOCK_32 = host_kernel_t(3_int8)
    !! Host kernel with block size of 32
type(host_kernel_t), parameter :: HOST_KERNEL_BLOCK_64 = host_kernel_t(4_int8)
    !! Host kernel with block size of 64

interface operator(==)
    module procedure host_kernel_eq
end interface

type, extends(abstract_kernel) :: kernel_host
  !! Host kernel implementation
    integer(int8) :: access_mode
      !! Access mode for kernel execution
    procedure(execute_host_interface), pointer :: execute_impl => null()
      !! Pointer to the execute implementation
contains
    procedure :: create_private => create_host    !! Creates kernel
    procedure :: execute_private => execute_host  !! Executes kernel
    procedure :: destroy_private => destroy_host  !! Destroys kernel
    procedure :: execute_benchmark
    procedure :: select_access_mode_f32
    procedure :: select_access_mode_f64
    procedure :: select_access_mode_f128
end type kernel_host

abstract interface
    subroutine execute_host_interface(self, in, out, neighbor)
    !! Executes the given kernel on host
        import
        class(kernel_host),         intent(in)      :: self     !! Host kernel class
        real(real32),     target,   intent(in)    :: in(:)      !! Source host-allocated buffer
        real(real32),     target,   intent(inout) :: out(:)     !! Target host-allocated buffer
        integer(int32), optional,   intent(in)      :: neighbor !! Source rank for pipelined unpacking
    end subroutine execute_host_interface
end interface

contains

subroutine create_host(self, effort, base_storage, force_effort)
!! Creates host kernel
    class(kernel_host),     intent(inout)   :: self         !! Host kernel class
    type(dtfft_effort_t),   intent(in)      :: effort       !! Effort level for generating transpose kernels
    integer(int64),         intent(in)      :: base_storage !! Number of bytes needed to store single element
    logical,    optional,   intent(in)      :: force_effort !! Should effort be forced or not
    logical                         :: force_effort_    !! Local copy of force_effort
    integer(int32)                  :: n_iters          !! Number of iterations to perform when testing kernel
    integer(int32)                  :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
    real(real32)                    :: best_time        !! Best execution time
    real(real64)                    :: execution_time   !! Execution time
    integer(int8)                   :: test_id          !! Current test configuration id
    integer(int8)                   :: max_tests        !! Maximum number of tests to perform
    type(host_kernel_t)             :: current_kernel   !! Current test configuration
    type(host_kernel_t)             :: best_kernel      !! Best kernel type
    character(len=:),   allocatable :: global_phase     !! Global phase name for profiling
    character(len=:),   allocatable :: local_phase      !! Local phase name for profiling
    real(real32)                    :: bandwidth        !! Bandwidth for kernel execution
    integer(int32)                  :: ndims            !! Number of dimensions
    integer(int32),     allocatable :: fixed_dims(:)    !! Fixed dimensions for bandwidth calculation
    real(real32),       allocatable :: in(:), out(:)    !! Host buffers for benchmarking
    type(kernel_type_t)             :: temp_kernel_type !! Temporary storage for kernel type

    self%access_mode = DEFAULT_ACCESS_MODE

    force_effort_ = .false.; if (present(force_effort)) force_effort_ = force_effort
    if ((effort == DTFFT_ESTIMATE .and. force_effort_) .or. &
          .not. ( (effort == DTFFT_PATIENT .and. get_conf_kernel_optimization_enabled()) .or. get_conf_forced_kernel_optimization()) ) then
        self%execute_impl => select_kernel(HOST_KERNEL_BASE, base_storage)
        return
    end if

    n_warmup_iters = get_conf_measure_warmup_iters()
    n_iters = get_conf_measure_iters()
    best_time = MAX_REAL32

    ndims = size(self%dims)
    allocate (fixed_dims(ndims))
    fixed_dims(1:ndims) = self%dims(1:ndims)
    if (is_unpack_kernel(self%kernel_type)) fixed_dims(1:ndims) = self%neighbor_data(1:ndims, 1)

    allocate (in(base_storage * product(self%dims) / FLOAT_STORAGE_SIZE))
    allocate (out(base_storage * product(self%dims) / FLOAT_STORAGE_SIZE))

    temp_kernel_type = self%kernel_type
    if (self%kernel_type == KERNEL_PERMUTE_BACKWARD_END) then
        self%kernel_type = KERNEL_PERMUTE_BACKWARD_END_PIPELINED
    else if (self%kernel_type == KERNEL_UNPACK) then
        self%kernel_type = KERNEL_UNPACK_PIPELINED
    end if

    global_phase = "Benchmarking kernel: '"//self%kernel_string//"'"
    PHASE_BEGIN(global_phase, COLOR_AUTOTUNE)
    WRITE_INFO(global_phase)

    max_tests = int(get_conf_configs_to_test(), int8)

    do test_id = 1_int8, min(max_tests, 4_int8)
        current_kernel = host_kernel_t(test_id)

        self%execute_impl => select_kernel(current_kernel, base_storage)

        if (current_kernel == HOST_KERNEL_BASE .and. .not. (any(self%kernel_type == [KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED]))) then
            local_phase = "Selecting access mode"
            REGION_BEGIN(local_phase, COLOR_AUTOTUNE2)
            WRITE_INFO("    "//local_phase)

            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                call self%select_access_mode_f32(in, out, n_warmup_iters, n_iters, execution_time)
            case (DOUBLE_STORAGE_SIZE)
                call self%select_access_mode_f64(in, out, n_warmup_iters, n_iters, execution_time)
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                call self%select_access_mode_f128(in, out, n_warmup_iters, n_iters, execution_time)
            end select
        else
            local_phase = "Testing kernel "//get_host_kernel_string(current_kernel)
            REGION_BEGIN(local_phase, COLOR_AUTOTUNE2)
            WRITE_INFO("    "//local_phase)

            call self%execute_benchmark(in, out, n_warmup_iters, n_iters, execution_time)
        end if

        WRITE_INFO("        Average execution time = "//to_str(execution_time)//" [ms]")
        if (execution_time > 0._real64) then
        bandwidth = 2._real32 * 1000._real32 * real(base_storage * product(fixed_dims), real32) / real(1024 * 1024 * 1024, real32) / real(execution_time, real32)
            WRITE_INFO("        Bandwidth = "//to_str(bandwidth)//" [GB/s]")
        end if

        if (execution_time < best_time) then
            best_time = real(execution_time, real32)
            best_kernel = current_kernel
        end if

        REGION_END(local_phase)
    end do
    WRITE_INFO("  Selected kernel: "//get_host_kernel_string(best_kernel))

    self%kernel_type = temp_kernel_type
    self%execute_impl => select_kernel(best_kernel, base_storage)

    PHASE_END(global_phase)
    deallocate (fixed_dims)
    deallocate (in, out)
    deallocate (global_phase, local_phase)
end subroutine create_host

subroutine execute_benchmark(self, in, out, n_warmup_iters, n_iters, execution_time)
!! Executes benchmark for the given kernel
    class(kernel_host), intent(inout)   :: self           !! Host kernel class
    real(real32),       intent(in)      :: in(:)          !! Source host-allocated buffer
    real(real32),       intent(inout)   :: out(:)         !! Target host-allocated buffer
    integer(int32),     intent(in)      :: n_warmup_iters !! Number of warmup iterations to perform before testing kernel
    integer(int32),     intent(in)      :: n_iters        !! Number of iterations to perform when testing kernel
    real(real64),       intent(out)     :: execution_time !! Execution time of the selected access
    integer(int32) :: iter
    real(real64) :: start_time, end_time

#ifdef DTFFT_DEBUG
    if (.not. associated(self%execute_impl)) then
        INTERNAL_ERROR("kernel_host%execute_benchmark: Kernel execute implementation is not associated!")
    end if
#endif

    REGION_BEGIN("Warmup", COLOR_TRANSPOSE)
    do iter = 1, n_warmup_iters
        call self%execute_impl(in, out, 1)
    end do
    REGION_END("Warmup")

    REGION_BEGIN("Measure", COLOR_EXECUTE)
    call cpu_time(start_time)
    do iter = 1, n_iters
        call self%execute_impl(in, out, 1)
    end do
    call cpu_time(end_time)
    execution_time = 1000._real64 * (end_time - start_time) / real(n_iters, real64)
    REGION_END("Measure")
end subroutine execute_benchmark

subroutine execute_host(self, in, out, stream, neighbor)
!! Executes host kernel
    class(kernel_host),         intent(inout)   :: self       !! Host kernel class
    real(real32),   target,     intent(in)      :: in(:)      !! Source host-allocated buffer
    real(real32),   target,     intent(inout)   :: out(:)     !! Target host-allocated buffer
    type(dtfft_stream_t),       intent(in)      :: stream     !! Stream to execute on, unused here
    integer(int32), optional,   intent(in)      :: neighbor   !! Source rank for pipelined unpacking

#ifdef DTFFT_DEBUG
    if (.not. associated(self%execute_impl)) then
        INTERNAL_ERROR("kernel_host%execute_host: Kernel execute implementation is not associated!")
    end if
#endif

    call self%execute_impl(in, out, neighbor)
end subroutine execute_host

subroutine destroy_host(self)
!! Destroys host kernel
    class(kernel_host), intent(inout) :: self !! Host kernel class

    nullify (self%execute_impl)
end subroutine destroy_host

function select_kernel(kernel, base_storage) result(fun)
!! Selects the kernel implementation based on the given id and base storage size
    type(host_kernel_t), intent(in) :: kernel           !! Kernel id
    integer(int64),      intent(in) :: base_storage     !! Size of single element in bytes
    procedure(execute_host_interface), pointer :: fun   !! Selected kernel implementation

    select case (kernel%val)
    case (HOST_KERNEL_BASE%val)
        select case (base_storage)
        case (FLOAT_STORAGE_SIZE)
            fun => execute_f32
        case (DOUBLE_STORAGE_SIZE)
            fun => execute_f64
        case (DOUBLE_COMPLEX_STORAGE_SIZE)
            fun => execute_f128
        end select
    case (HOST_KERNEL_BLOCK_16%val)
        select case (base_storage)
        case (FLOAT_STORAGE_SIZE)
            fun => execute_f32_block_16
        case (DOUBLE_STORAGE_SIZE)
            fun => execute_f64_block_16
        case (DOUBLE_COMPLEX_STORAGE_SIZE)
            fun => execute_f128_block_16
        end select
    case (HOST_KERNEL_BLOCK_32%val)
        select case (base_storage)
        case (FLOAT_STORAGE_SIZE)
            fun => execute_f32_block_32
        case (DOUBLE_STORAGE_SIZE)
            fun => execute_f64_block_32
        case (DOUBLE_COMPLEX_STORAGE_SIZE)
            fun => execute_f128_block_32
        end select
    case (HOST_KERNEL_BLOCK_64%val)
        select case (base_storage)
        case (FLOAT_STORAGE_SIZE)
            fun => execute_f32_block_64
        case (DOUBLE_STORAGE_SIZE)
            fun => execute_f64_block_64
        case (DOUBLE_COMPLEX_STORAGE_SIZE)
            fun => execute_f128_block_64
        end select
    end select
end function select_kernel

function get_host_kernel_string(kernel) result(kernel_string)
!! Returns string representation of the given host kernel type
    type(host_kernel_t), intent(in) :: kernel       !! Host kernel type
    character(len=:),   allocatable :: kernel_string !! String representation of the kernel

    select case (kernel%val)
    case (HOST_KERNEL_BASE%val)
        kernel_string = "BASE"
    case (HOST_KERNEL_BLOCK_16%val)
        kernel_string = "BLOCK_16"
    case (HOST_KERNEL_BLOCK_32%val)
        kernel_string = "BLOCK_32"
    case (HOST_KERNEL_BLOCK_64%val)
        kernel_string = "BLOCK_64"
    case default
        kernel_string = "UNKNOWN"
    end select
end function get_host_kernel_string

MAKE_EQ_FUN(host_kernel_t, host_kernel_eq)

#define PREC _f128
#define BUFFER_TYPE complex(real64)
#define STORAGE_BYTES DOUBLE_COMPLEX_STORAGE_SIZE
#include "_dtfft_kernel_host_routines.inc"

#define PREC _f64
#define BUFFER_TYPE real(real64)
#define STORAGE_BYTES DOUBLE_STORAGE_SIZE
#include "_dtfft_kernel_host_routines.inc"

#define PREC _f32
#define BUFFER_TYPE real(real32)
#define STORAGE_BYTES FLOAT_STORAGE_SIZE
#include "_dtfft_kernel_host_routines.inc"
end module dtfft_kernel_host
