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
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_parameters
use dtfft_abstract_kernel
use dtfft_config
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: kernel_host, host_kernel_t, execute_host_interface

    type :: host_kernel_t
        integer(int8) :: val
    end type host_kernel_t

    type(host_kernel_t), parameter, public :: HOST_KERNEL_UNSET = host_kernel_t(-1_int8)

    type(host_kernel_t), parameter, public :: HOST_KERNEL_BASE = host_kernel_t(1_int8)
    !! Base host kernel type
    type(host_kernel_t), parameter, public :: HOST_KERNEL_BLOCK_4 = host_kernel_t(2_int8)
    !! Host kernel with block size of 4
    type(host_kernel_t), parameter, public :: HOST_KERNEL_BLOCK_8 = host_kernel_t(3_int8)
    !! Host kernel with block size of 8
    type(host_kernel_t), parameter, public :: HOST_KERNEL_BLOCK_16 = host_kernel_t(4_int8)
    !! Host kernel with block size of 16
    type(host_kernel_t), parameter, public :: HOST_KERNEL_BLOCK_32 = host_kernel_t(5_int8)
    !! Host kernel with block size of 32
    type(host_kernel_t), parameter, public :: HOST_KERNEL_BLOCK_64 = host_kernel_t(6_int8)
    !! Host kernel with block size of 64
    type(host_kernel_t), parameter, public :: HOST_KERNEL_CACHE_OBLIVIOUS = host_kernel_t(7_int8)
    !! Cache oblivios kernel type

    integer(int32), parameter :: CO_THRESHOLDS(*) = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    interface operator(==)
        module procedure :: host_kernel_eq
    end interface

    type, extends(abstract_kernel) :: kernel_host
    !! Host kernel implementation
    private
        type(dtfft_access_mode_t) :: access_mode = ACCESS_MODE_NOT_SET
        !! Access mode for kernel execution
        procedure(execute_host_interface), pointer :: execute_impl => null()
        !! Pointer to the execute implementation
        integer(int32)  :: co_threshold = 0
        !! Cache oblivious threshold for kernel selection
        type(host_kernel_t) :: selected_kernel = HOST_KERNEL_BASE
        !! Kernel variant `execute_impl` points at.
        !!
        !! `execute_impl` itself cannot be saved to a plan file, so the choice behind it
        !! is kept as a plain value that `get_decision` can hand back.
    contains
        procedure :: create_private => create_host      !! Creates kernel
        ! procedure :: get_decision                       !! Returns tuning results of this kernel
        procedure :: execute_private => execute_host    !! Executes kernel
        procedure :: destroy_private => destroy_host    !! Destroys kernel
        procedure :: execute_benchmark                  !! Executes benchmark for the given kernel
        procedure :: select_access_mode_f32             !! Selects the best access mode for host kernels, real32 version
        procedure :: select_access_mode_f64             !! Selects the best access mode for host kernels, real64 version
        procedure :: select_access_mode_f128            !! Selects the best access mode for host kernels, complex64 version
        procedure :: get_kernel_options
    end type kernel_host

    abstract interface
        subroutine execute_host_interface(self, in, out, neighbor)
        !! Executes the given kernel on host
        import
            class(kernel_host),         intent(in)      :: self     !! Host kernel class
            type(c_ptr),                intent(in)      :: in       !! Source host-allocated buffer
            type(c_ptr),                intent(in)      :: out      !! Target host-allocated buffer
            integer(int32), optional,   intent(in)      :: neighbor !! Source rank for pipelined unpacking
        end subroutine execute_host_interface
    end interface

interface
    module subroutine select_kernel(kernel, base_storage, fun)
    !! Selects the kernel implementation based on the given id and base storage size
        type(host_kernel_t), intent(in) :: kernel           !! Kernel id
        integer(int64),      intent(in) :: base_storage     !! Size of single element in bytes
        procedure(execute_host_interface), pointer, intent(out) :: fun   !! Selected kernel implementation
    end subroutine select_kernel

    module subroutine select_access_mode_f32(self, in, out, n_warmup_iters, n_iters, execution_time)
    !! Selects the best access mode for host kernels, real32 version
        class(kernel_host), intent(inout)   :: self             !! Host kernel class
        type(c_ptr),        intent(in)      :: in               !! Source host-allocated buffer
        type(c_ptr),        intent(in)      :: out              !! Target host-allocated buffer
        integer(int32),     intent(in)      :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
        integer(int32),     intent(in)      :: n_iters          !! Number of iterations to perform when testing kernel
        real(real64),       intent(out)     :: execution_time   !! Execution time of the selected access
    end subroutine select_access_mode_f32

    module subroutine select_access_mode_f64(self, in, out, n_warmup_iters, n_iters, execution_time)
    !! Selects the best access mode for host kernels, real64 version
        class(kernel_host), intent(inout)   :: self             !! Host kernel class
        type(c_ptr),        intent(in)      :: in               !! Source host-allocated buffer
        type(c_ptr),        intent(in)      :: out              !! Target host-allocated buffer
        integer(int32),     intent(in)      :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
        integer(int32),     intent(in)      :: n_iters          !! Number of iterations to perform when testing kernel
        real(real64),       intent(out)     :: execution_time   !! Execution time of the selected access
    end subroutine select_access_mode_f64

    module subroutine select_access_mode_f128(self, in, out, n_warmup_iters, n_iters, execution_time)
    !! Selects the best access mode for host kernels, complex64 version
        class(kernel_host), intent(inout)   :: self             !! Host kernel class
        type(c_ptr),        intent(in)      :: in               !! Source host-allocated buffer
        type(c_ptr),        intent(in)      :: out              !! Target host-allocated buffer
        integer(int32),     intent(in)      :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
        integer(int32),     intent(in)      :: n_iters          !! Number of iterations to perform when testing kernel
        real(real64),       intent(out)     :: execution_time   !! Execution time of the selected access
    end subroutine select_access_mode_f128
end interface

contains

    function get_host_kernel_string(kernel) result(kernel_string)
    !! Returns string representation of the given host kernel type
        type(host_kernel_t), intent(in) :: kernel               !! Host kernel type
        character(len=:),   allocatable :: kernel_string        !! String representation of the kernel

        select case (kernel%val)
        case (HOST_KERNEL_BASE%val)
            allocate( kernel_string, source="BASE" )
        case (HOST_KERNEL_BLOCK_4%val)
            allocate( kernel_string, source="BLOCK_4" )
        case (HOST_KERNEL_BLOCK_8%val)
            allocate( kernel_string, source="BLOCK_8" )
        case (HOST_KERNEL_BLOCK_16%val)
            allocate( kernel_string, source="BLOCK_16" )
        case (HOST_KERNEL_BLOCK_32%val)
            allocate( kernel_string, source="BLOCK_32" )
        case (HOST_KERNEL_BLOCK_64%val)
            allocate( kernel_string, source="BLOCK_64" )
        case (HOST_KERNEL_CACHE_OBLIVIOUS%val)
            allocate( kernel_string, source="CACHE_OBLIVIOUS" )
        case default
            allocate( kernel_string, source="UNKNOWN" )
        end select
    end function get_host_kernel_string
    ! module type(kernel_host_options) function create_kernel_host_options(kernel_type, access_mode, threshold) result(options)
    ! !! Creates kernel options struct for host kernel selection
    !     type(host_kernel_t),        intent(in) :: kernel_type   !! Selected host kernel type
    !     type(dtfft_access_mode_t),  intent(in) :: access_mode   !! Access mode for kernel execution
    !     integer(int32),             intent(in) :: threshold     !! Cache oblivious threshold for kernel selection
    ! end function create_kernel_host_options

    ! module subroutine get_decision(self, kernel_type, access_mode, threshold)
    ! !! Returns everything this kernel decided during tuning, so that it can be
    ! !! written to a plan file and fed back in through [[kernel_host_options]]
    !     class(kernel_host),         intent(in)  :: self         !! Host kernel class
    !     type(host_kernel_t),        intent(out) :: kernel_type  !! Selected host kernel variant
    !     type(dtfft_access_mode_t),  intent(out) :: access_mode  !! Selected access mode
    !     integer(int32),             intent(out) :: threshold    !! Selected cache oblivious threshold
    ! end subroutine get_decision

    subroutine create_host(self, effort, base_storage, options, force_effort)
    !! Creates host kernel
        class(kernel_host),                 intent(inout)   :: self         !! Host kernel class
        type(dtfft_effort_t),               intent(in)      :: effort       !! Effort level for generating transpose kernels
        integer(int64),                     intent(in)      :: base_storage !! Number of bytes needed to store single element
        type(kernel_options),   optional,   intent(in)      :: options      !! Kernel options struct
        logical,                optional,   intent(in)      :: force_effort !! Should effort be forced or not
        logical                         :: force_effort_    !! Local copy of force_effort
        integer(int32)                  :: n_iters          !! Number of iterations to perform when testing kernel
        integer(int32)                  :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
        real(real32)                    :: best_time        !! Best execution time
        real(real64)                    :: execution_time   !! Execution time
        integer(int8)                   :: test_id          !! Current test configuration id
        type(host_kernel_t)             :: current_kernel   !! Current test configuration
        type(host_kernel_t)             :: best_kernel      !! Best kernel type
        character(len=:),   allocatable :: global_phase     !! Global phase name for profiling
        character(len=:),   allocatable :: local_phase      !! Local phase name for profiling
        integer(int32)                  :: ndims            !! Number of dimensions
        integer(int32),     allocatable :: fixed_dims(:)    !! Fixed dimensions for bandwidth calculation
        type(c_ptr) :: in, out                              !! Host buffers for benchmarking
        type(kernel_type_t)             :: temp_kernel_type !! Temporary storage for kernel type
        logical :: has_options
        type(host_kernel_t) :: kernel_type
        integer(int32), allocatable :: co_thrshlds(:)
        type(dtfft_access_mode_t) :: temp_access_mode
        integer(int32) :: best_co_threshold
        integer(int32) :: block_size

        self%access_mode = get_conf_access_mode()
        block_size = get_conf_block_size()
        select case ( block_size )
        case ( 0 )
            kernel_type = HOST_KERNEL_BASE
        case ( 4 )
            kernel_type = HOST_KERNEL_BLOCK_4
        case ( 8 )
            kernel_type = HOST_KERNEL_BLOCK_8
        case ( 16 )
            kernel_type = HOST_KERNEL_BLOCK_16
        case ( 32 )
            kernel_type = HOST_KERNEL_BLOCK_32
        case ( 64 )
            kernel_type = HOST_KERNEL_BLOCK_64
        endselect
        self%co_threshold = get_conf_co_threshold()

        ! if ( self%access_mode == DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS ) then
        !     kernel_type = HOST_KERNEL_BASE
        ! endif

        OPTVAL(force_effort_, .false., force_effort)

        has_options = present(options)
        if ( has_options ) then
            ! block
            !     type(kernel_host_options), pointer :: host_options
            !     call c_f_pointer(options, host_options)
#ifdef DTFFT_DEBUG
            if ( options%slot_kind /= KERNEL_SLOT_HOST ) then
                INTERNAL_ERROR("options%slot_kind /= KERNEL_SLOT_HOST")
            endif
#endif
            self%access_mode = options%access_mode
            self%co_threshold = options%co_threshold
            kernel_type = host_kernel_t(options%host_kernel)
            ! endblock
        endif
        if ((effort == DTFFT_ESTIMATE .and. force_effort_) &
            .or. .not. ( effort == DTFFT_EXHAUSTIVE .or. get_conf_kernel_autotune_enabled())    &
            .or. any(self%kernel_type == [KERNEL_COPY, KERNEL_COPY_PIPELINED])                  &
            .or. has_options) then
            self%selected_kernel = kernel_type
            call select_kernel(kernel_type, base_storage, self%execute_impl)
            return
        end if

        n_warmup_iters = get_conf_measure_warmup_iters()
        n_iters = get_conf_measure_iters()
        best_time = MAX_REAL32

        ndims = size(self%dims)
        allocate (fixed_dims(ndims))
        fixed_dims(1:ndims) = self%dims(1:ndims)
        if (is_unpack_kernel(self%kernel_type) .or. is_pack_kernel(self%kernel_type)) fixed_dims(1:ndims) = self%neighbor_data(1:ndims, 1)

        in = mem_alloc_host(base_storage * product(self%dims))
        out = mem_alloc_host(base_storage * product(self%dims))

        temp_kernel_type = self%kernel_type
        select case ( self%kernel_type%val )
        case ( KERNEL_PERMUTE_BACKWARD_END%val )
            self%kernel_type = KERNEL_PERMUTE_BACKWARD_END_PIPELINED
        case ( KERNEL_UNPACK%val )
            self%kernel_type = KERNEL_UNPACK_PIPELINED
        case ( KERNEL_PACK%val )
            self%kernel_type = KERNEL_PACK_PIPELINED
        case ( KERNEL_UNPACK_FORWARD%val )
            self%kernel_type = KERNEL_UNPACK_FORWARD_PIPELINED
        case ( KERNEL_UNPACK_BACKWARD%val )
            self%kernel_type = KERNEL_UNPACK_BACKWARD_PIPELINED
        endselect

        allocate( global_phase, source="Benchmarking kernel: '"//self%kernel_string%raw//"'" )
        PHASE_BEGIN(global_phase, COLOR_STEEL_BLUE)
        WRITE_INFO(global_phase)

        do test_id = HOST_KERNEL_BASE%val, HOST_KERNEL_BLOCK_64%val
            current_kernel = host_kernel_t(test_id)

            call select_kernel(current_kernel, base_storage, self%execute_impl)

            if (current_kernel == HOST_KERNEL_BASE .and. .not. (any(self%kernel_type == [KERNEL_PACK_PIPELINED, KERNEL_UNPACK_PIPELINED]))) then
                allocate( local_phase, source="Selecting access mode" )
                REGION_BEGIN(local_phase, COLOR_ORCHID)
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
                allocate( local_phase, source="Testing kernel "//get_host_kernel_string(current_kernel) )
                REGION_BEGIN(local_phase, COLOR_AUTOTUNE2)
                WRITE_INFO("    "//local_phase)

                call self%execute_benchmark(in, out, n_warmup_iters, n_iters, execution_time)
            end if

            call report_timings(execution_time, fixed_dims, base_storage)

            if (execution_time < best_time) then
                best_time = real(execution_time, real32)
                best_kernel = current_kernel
            end if

            REGION_END(local_phase)
            deallocate( local_phase )
        end do

        best_co_threshold = -1
        if ( is_transpose_kernel(self%kernel_type) .or. self%kernel_type == KERNEL_PERMUTE_BACKWARD_END_PIPELINED ) then
            temp_access_mode = self%access_mode
            if( effort%val < DTFFT_EXHAUSTIVE%val ) then
                allocate(co_thrshlds, source=[self%co_threshold])
            else
                allocate(co_thrshlds, source=CO_THRESHOLDS)
            endif

            self%access_mode = DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS
            call select_kernel(HOST_KERNEL_CACHE_OBLIVIOUS, base_storage, self%execute_impl)
            do test_id = 1_int8, size(co_thrshlds, kind=int8)
                self%co_threshold = co_thrshlds(test_id)

                allocate( local_phase, source="Testing kernel "//get_host_kernel_string(HOST_KERNEL_CACHE_OBLIVIOUS)//", threshold = "//to_str(self%co_threshold) )
                REGION_BEGIN(local_phase, COLOR_ORCHID)
                WRITE_INFO("    "//local_phase)

                call self%execute_benchmark(in, out, n_warmup_iters, n_iters, execution_time)

                call report_timings(execution_time, fixed_dims, base_storage)

                if (execution_time < best_time) then
                    best_time = real(execution_time, real32)
                    best_co_threshold = self%co_threshold
                    best_kernel = HOST_KERNEL_CACHE_OBLIVIOUS
                end if

                REGION_END(local_phase)
                deallocate( local_phase )
            enddo

            if ( best_co_threshold == -1 ) then
                self%access_mode = temp_access_mode
            else
                self%co_threshold = best_co_threshold
            endif

            deallocate(co_thrshlds)
        endif

        WRITE_INFO("  Selected kernel: "//get_host_kernel_string(best_kernel))
        if ( best_kernel == HOST_KERNEL_CACHE_OBLIVIOUS ) then
            WRITE_INFO("  Threshold = "//to_str(self%co_threshold))
            ! best_kernel = HOST_KERNEL_BASE
        endif

        self%kernel_type = temp_kernel_type
        self%selected_kernel = best_kernel
        call select_kernel(best_kernel, base_storage, self%execute_impl)

        PHASE_END(global_phase)
        deallocate (fixed_dims)
        call mem_free_host(in)
        call mem_free_host(out)
        deallocate (global_phase)
    end subroutine create_host

    subroutine report_timings(time, dims, base_storage)
        real(real64),   intent(in)  :: time         !! Kernel execution time
        integer(int32), intent(in)  :: dims(:)      !! Local dimensions to process
        integer(int64), intent(in)  :: base_storage !! Number of bytes needed to store single element
        real(real32) :: bandwidth   !! Bandwidth for kernel execution

        WRITE_INFO("        Average execution time = "//to_str(time)//" [ms]")
        if (time > 0._real64) then
            bandwidth = 2000._real32 * real(base_storage * product(dims), real32) / real(1024 * 1024 * 1024, real32) / real(time, real32)
            WRITE_INFO("        Bandwidth = "//to_str(bandwidth)//" [GB/s]")
        end if
    end subroutine report_timings

    subroutine execute_benchmark(self, in, out, n_warmup_iters, n_iters, execution_time)
    !! Executes benchmark for the given kernel
        class(kernel_host), intent(inout)   :: self             !! Host kernel class
        type(c_ptr),        intent(in)      :: in               !! Source host-allocated buffer
        type(c_ptr),        intent(in)      :: out              !! Target host-allocated buffer
        integer(int32),     intent(in)      :: n_warmup_iters   !! Number of warmup iterations to perform before testing kernel
        integer(int32),     intent(in)      :: n_iters          !! Number of iterations to perform when testing kernel
        real(real64),       intent(out)     :: execution_time   !! Execution time of the selected access
        integer(int32) :: iter
        real(real64) :: start_time, end_time

#ifdef DTFFT_DEBUG
        if (.not. associated(self%execute_impl)) then
            INTERNAL_ERROR("kernel_host%execute_benchmark: Kernel execute implementation is not associated!")
        end if
#endif

        REGION_BEGIN("Warmup", COLOR_VIOLET)
        do iter = 1, n_warmup_iters
            call self%execute_impl(in, out, 1)
        end do
        REGION_END("Warmup")

        REGION_BEGIN("Measure", COLOR_DODGER_BLUE)
        start_time = MPI_Wtime()
        do iter = 1, n_iters
            call self%execute_impl(in, out, 1)
        end do
        end_time = MPI_Wtime()
        execution_time = 1000._real64 * (end_time - start_time) / real(n_iters, real64)
        REGION_END("Measure")
    
    end subroutine execute_benchmark

    subroutine execute_host(self, in, out, stream, sync, neighbor)
    !! Executes host kernel
        class(kernel_host),         intent(inout)   :: self         !! Host kernel class
        type(c_ptr),                intent(in)      :: in           !! Source host-allocated buffer
        type(c_ptr),                intent(in)      :: out          !! Target host-allocated buffer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream to execute on, unused here
        logical,                    intent(in)      :: sync         !! Sync stream after kernel execution, unused here
        integer(int32), optional,   intent(in)      :: neighbor     !! Source rank for pipelined unpacking
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

    type(kernel_options) function get_kernel_options(self)
        class(kernel_host), intent(in) :: self !! Host kernel class

        get_kernel_options%slot_kind = KERNEL_SLOT_HOST
        get_kernel_options%access_mode = self%access_mode
        get_kernel_options%co_threshold = self%co_threshold
        get_kernel_options%host_kernel = self%selected_kernel%val
    end function get_kernel_options

MAKE_EQ_FUN(host_kernel_t, host_kernel_eq)
end module dtfft_kernel_host
