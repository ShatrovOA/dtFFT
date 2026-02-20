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
module dtfft_nvrtc_module
!! Module for managing nvRTC compiled CUDA kernels
!! Each module has only one templated kernel that can be instantiated with different parameters
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_kernel
use dtfft_errors
use dtfft_config
use dtfft_nvrtc_block_optimizer, only: kernel_config
use dtfft_interface_cuda
use dtfft_interface_cuda_runtime
use dtfft_interface_nvrtc
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_MOCK_ENABLED
use dtfft_kernel_host
#endif
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: nvrtc_module

    type, extends(string) :: codegen_t
    !! Class for generating CUDA code
    contains
        procedure, pass(self) :: add => add_line  !! Adds new line to CUDA code
    end type codegen_t

    type :: nvrtc_module
    !! Class for managing nvRTC compiled CUDA kernels
    private
        logical                       :: is_created = .false.   !! Is module created
        type(string)                  :: basic_name             !! Basic kernel name
        integer(int32)                :: ndims                  !! Number of dimensions, used only for forward permutation
        type(CUmodule)                :: cumod                  !! CUDA module
        type(nvrtcProgram)            :: prog                   !! nvRTC program
        type(kernel_type_t)           :: kernel_type            !! Type of kernel
        integer(int64)                :: base_storage           !! Number of bytes needed to store single element
        type(kernel_config),  allocatable :: configs(:)         !! Kernel configurations that this module was compiled for
    contains
    private
        procedure, pass(self), public   :: create     !! Creates module with given parameters
        procedure, pass(self), public   :: destroy    !! Destroys module and frees resources
        procedure, pass(self), public   :: get        !! Returns kernel ready to be executed
        generic,               public   :: check =>         &
                                           check_instance,  &
                                           check_module
        !! Checks if kernel is with given parameters is available in this module
        procedure, pass(self)           :: check_instance !! Checks if kernel with given parameters is available in this module
        procedure, pass(self)           :: check_module   !! Basic check that this module provides kernels of given type
    end type nvrtc_module

contains

    subroutine add_line(self, line)
    !! Adds new line to CUDA code
        class(codegen_t), intent(inout) :: self     !! Kernel code
        character(len=*), intent(in)    :: line     !! Line to add

        if (.not. allocated(self%raw)) allocate (self%raw, source="")
        self%raw = self%raw//line//c_new_line
    end subroutine add_line

    function get(self, ndims, kernel_type, base_storage, tile_size, block_rows, fun) result(ierr)
    !! Returns kernel ready to be executed
        class(nvrtc_module),  intent(in)    :: self               !! This module
        integer(int32),       intent(in)    :: ndims              !! Number of dimensions, used only for forward permutation
        type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
        integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element
        integer(int32),       intent(in)    :: tile_size          !! Size of shared memory tile, template parameter
        integer(int32),       intent(in)    :: block_rows         !! Number of rows processed by single thread, template parameter
        type(CUfunction),     intent(out)   :: fun                !! Resulting kernel
        integer(int32)                      :: ierr
        integer(int32)  :: i          !! Loop variable
        integer(int32)  :: config_id  !! Configuration ID
#ifndef DTFFT_WITH_MOCK_ENABLED
        type(c_ptr)     :: mangled    !! Mangled kernel name
        type(kernel_config) :: config !! Found configuration

        fun = CUfunction(c_null_ptr)
#endif
        ierr = -1
        if ( .not. self%check(ndims, kernel_type, base_storage, tile_size, block_rows) ) return
        config_id = -1
        do i = 1, size(self%configs, dim=1)
            if (tile_size == self%configs(i)%tile_size .and. block_rows == self%configs(i)%block_rows) then
                config_id = i
                exit
            end if
        end do
#ifdef DTFFT_DEBUG
        if (config_id < 0) then
            INTERNAL_ERROR("nvrtc_module.get: config_id < 0")
        endif
#endif
#ifndef DTFFT_WITH_MOCK_ENABLED
        config = self%configs(config_id)
        mangled = get_mangled_name(self%basic_name%raw, self%prog, config%tile_size, config%block_rows, config%padding)
        CUDA_CALL( cuModuleGetFunction(fun, self%cumod, mangled) )
#else
        select case ( kernel_type%val )
        case ( KERNEL_PERMUTE_BACKWARD%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%sfun_r32 => permute_backward_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%sfun_r64 => permute_backward_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%sfun_r128 => permute_backward_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_PERMUTE_FORWARD%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%sfun_r32 => permute_forward_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%sfun_r64 => permute_forward_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%sfun_r128 => permute_forward_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_PERMUTE_BACKWARD_START%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%sfun_r32 => permute_backward_start_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%sfun_r64 => permute_backward_start_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%sfun_r128 => permute_backward_start_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        ! case ( KERNEL_COPY%val )
        !     if ( base_storage == FLOAT_STORAGE_SIZE ) then
        !         fun%sfun_r32 => copy_f32
        !     else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
        !         fun%sfun_r64 => copy_f64
        !     else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
        !         fun%sfun_r128 => copy_f128
        !     else
        !         INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
        !     end if
        case ( KERNEL_PACK_PIPELINED%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%pfun_r32 => pack_pipelined_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%pfun_r64 => pack_pipelined_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%pfun_r128 => pack_pipelined_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_PACK_FORWARD%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%pfun_r32 => pack_forward_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%pfun_r64 => pack_forward_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%pfun_r128 => pack_forward_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_PACK_BACKWARD%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%pfun_r32 => pack_backward_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%pfun_r64 => pack_backward_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%pfun_r128 => pack_backward_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_UNPACK_PIPELINED%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%pfun_r32 => unpack_pipelined_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%pfun_r64 => unpack_pipelined_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%pfun_r128 => unpack_pipelined_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            if ( base_storage == FLOAT_STORAGE_SIZE ) then
                fun%pfun_r32 => permute_backward_end_pipelined_write_f32
            else if ( base_storage == DOUBLE_STORAGE_SIZE ) then
                fun%pfun_r64 => permute_backward_end_pipelined_write_f64
            else if ( base_storage == DOUBLE_COMPLEX_STORAGE_SIZE ) then
                fun%pfun_r128 => permute_backward_end_pipelined_write_f128
            else
                INTERNAL_ERROR("nvrtc_module.get: unknown `base_storage`")
            end if
        case default
            block
                type(string) :: str
                str = get_kernel_string(kernel_type)
                INTERNAL_ERROR("nvrtc_module.get: unknown `kernel_type`"//str%raw)
            endblock
        endselect
#endif
        ierr = DTFFT_SUCCESS
    end function get

    logical function check_instance(self, ndims, kernel_type, base_storage, tile_size, block_rows)
    !! Checks if kernel with given parameters is available in this module
        class(nvrtc_module),  intent(in)    :: self               !! This module
        integer(int32),       intent(in)    :: ndims              !! Number of dimensions
        type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
        integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element
        integer(int32),       intent(in)    :: tile_size          !! Size of shared memory tile, template parameter
        integer(int32),       intent(in)    :: block_rows         !! Number of rows processed by single thread, template parameter
        integer(int32)       :: i !! Loop variable

        check_instance = .false.
        if ( .not. self%check(ndims, kernel_type, base_storage) ) return
        do i = 1, size(self%configs, dim=1)
            if (tile_size == self%configs(i)%tile_size .and. block_rows == self%configs(i)%block_rows) then
                check_instance = .true.
                return
            end if
        end do
    end function check_instance

    logical function check_module(self, ndims, kernel_type, base_storage)
    !! Basic check that this module provides kernels of given type
        class(nvrtc_module),  intent(in)    :: self               !! This module
        integer(int32),       intent(in)    :: ndims              !! Number of dimensions
        type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
        integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element

        check_module = .false.
        if (.not. self%is_created) return
        if (kernel_type /= self%kernel_type) return
        if (base_storage /= self%base_storage) return
        if (ndims /= self%ndims .and. kernel_type == KERNEL_PERMUTE_FORWARD) return
        check_module = .true.
    end function check_module

    subroutine create(self, ndims, kernel_type, base_storage, configs, props)
    !! Creates module with given parameters, compiles nvRTC program and loads it as CUDA module
        class(nvrtc_module),  intent(inout) :: self               !! This module
        integer(int32),       intent(in)    :: ndims              !! Number of dimensions, used only for forward permutation
        type(kernel_type_t),  intent(in)    :: kernel_type        !! Type of kernel to build
        integer(int64),       intent(in)    :: base_storage       !! Number of bytes needed to store single element
        type(kernel_config),  intent(in)    :: configs(:)         !! Kernel configurations that this module should be compiled for
        type(device_props),   intent(in)    :: props              !! GPU architecture properties
        type(codegen_t)                   :: code             !! Generated code
        character(len=:),     allocatable :: region_name      !! Phase name for profiling
        integer(int32)                    :: i                !! Loop variable
        integer(c_size_t)                 :: cubinSizeRet     !! Size of cubin
        type(c_ptr)                       :: cubin            !! Cubin data

        call self%destroy()
        self%base_storage = base_storage
        self%kernel_type = kernel_type
        self%ndims = ndims
        self%cumod = CUmodule(c_null_ptr)
        allocate (self%configs( size(configs) ))
        do i = 1, size(configs)
            self%configs(i)%tile_size = configs(i)%tile_size
            self%configs(i)%block_rows = configs(i)%block_rows
            self%configs(i)%padding = configs(i)%padding
        enddo

        self%basic_name = get_kernel_string(kernel_type)

        region_name = "Compiling "//self%basic_name%raw
        REGION_BEGIN(region_name, COLOR_FFT)
        WRITE_DEBUG(region_name)

        code = get_code(self%basic_name%raw, ndims, base_storage, kernel_type)
        WRITE_DEBUG(code%raw)
        self%prog = compile_program(code, self%basic_name%raw, configs, props)
        call code%destroy()

        NVRTC_CALL( nvrtcGetCUBINSize(self%prog, cubinSizeRet) )
        cubin = mem_alloc_host(cubinSizeRet)
        NVRTC_CALL( nvrtcGetCUBIN(self%prog, cubin) )
        CUDA_CALL( cuModuleLoadData(self%cumod, cubin) )
        call mem_free_host(cubin)
        REGION_END(region_name)

        self%is_created = .true.
        deallocate( region_name )
    end subroutine create

    subroutine destroy(self)
    !! Destroys module and frees resources
        class(nvrtc_module), intent(inout) :: self

        if ( .not. self%is_created ) return
        if ( is_null_ptr(self%cumod%ptr) .or. is_null_ptr(self%prog%cptr) ) then
            INTERNAL_ERROR("nvrtc_module.destroy: is_null_ptr(self%cumod%ptr)")
        endif

        NVRTC_CALL( nvrtcDestroyProgram(self%prog) )
        CUDA_CALL( cuModuleUnload(self%cumod) )
        self%cumod = CUmodule(c_null_ptr)
        self%prog = nvrtcProgram(c_null_ptr)
        if( allocated( self%configs ) ) deallocate(self%configs)
        call self%basic_name%destroy()
    end subroutine destroy

    function compile_program(code, prog_name, configs, props) result(prog)
    !! Compiles nvRTC program with given configurations
        type(codegen_t),      intent(in)  :: code               !! CUDA code to compile
        character(len=*),     intent(in)  :: prog_name          !! Basic kernel name
        type(kernel_config),  intent(in)  :: configs(:)         !! Kernel configurations that this module should be compiled for
        type(device_props),   intent(in)  :: props              !! GPU architecture properties
        type(nvrtcProgram)                :: prog               !! Resulting nvRTC program
        integer(int32)                    :: num_options        !! Number of compilation options
        type(string), target, allocatable :: options(:)         !! Compilation options
        type(c_ptr),          allocatable :: c_options(:)       !! C style, null-string terminated options
        character(c_char),    allocatable :: c_code(:)          !! CUDA C Code to compile
        integer(int32)  :: i              !! Loop variable
        integer(int32)  :: compile_result !! Result of compilation
        character(len=:),     allocatable :: prog_name_

#ifdef DTFFT_DEBUG
        num_options = 3
#else
        num_options = 1
#endif

        allocate( c_options(num_options), options(num_options) )
        options(1) = string("--gpu-architecture=sm_"//to_str(props%compute_capability_major)//to_str(props%compute_capability_minor) // c_null_char)
#ifdef DTFFT_DEBUG
        options(2) = string("--device-debug" // c_null_char)
        options(3) = string("--generate-line-info" // c_null_char)
#endif
        do i = 1, num_options
        c_options(i) = c_loc(options(i)%raw)
        enddo

        call astring_f2c(code%raw, c_code)
        prog_name_ = prog_name//".cu"//c_null_char
        NVRTC_CALL( nvrtcCreateProgram(prog, c_code, prog_name_, 0, c_null_ptr, c_null_ptr) )
        deallocate( c_code, prog_name_ )

        do i = 1, size(configs)
            call set_name_expression(prog, prog_name, configs(i)%tile_size, configs(i)%block_rows, configs(i)%padding)
        enddo

        compile_result = nvrtcCompileProgram(prog, num_options, c_options)
        if ( compile_result /= 0 ) then
            block
                type(c_ptr) :: c_log
                integer(c_size_t) :: log_size
                character(len=:), allocatable :: f_log

                NVRTC_CALL( nvrtcGetProgramLogSize(prog, log_size) )
                c_log = mem_alloc_host(log_size)
                NVRTC_CALL( nvrtcGetProgramLog(prog, c_log) )
                call string_c2f(c_log, f_log)

                write(error_unit, "(a)") "dtFFT Internal Error: failed to compile kernel"
                write(error_unit, "(a)") "CUDA Code:"
                write(error_unit, "(a)") code%raw
                write(error_unit, "(a)") "Compilation log:"
                write(error_unit, "(a)") f_log

                INTERNAL_ERROR("compile_program: Compilation failed")
            endblock
        endif

        call destroy_strings(options)
        deallocate( c_options )
    end function compile_program

    function get_name_expression(basic_name, tile_dim, block_rows, padding) result(expression)
    !! Generates name expression for given template parameters
        character(len=*), intent(in)  :: basic_name       !! Basic kernel name
        integer(int32),   intent(in)  :: tile_dim         !! Size of shared memory tile, template parameter
        integer(int32),   intent(in)  :: block_rows       !! Number of rows processed by single thread, template parameter
        integer(int32),   intent(in)  :: padding          !! Padding to avoid shared memory bank conflicts, template parameter
        character(c_char),  allocatable :: expression(:)  !! Resulting name expression
        character(len=:),   allocatable :: str  !! Temporary string

        allocate (str, source=basic_name//"<"//to_str(tile_dim)//","//to_str(block_rows)//","//to_str(padding)//">"//c_null_char)
        call astring_f2c(str, expression)
        deallocate (str)
    end function get_name_expression

    subroutine set_name_expression(prog, basic_name, tile_dim, block_rows, padding)
    !! Sets name expression for given template parameters to nvRTC program
        type(nvrtcProgram), intent(in)  :: prog         !! nvRTC program
        character(len=*),   intent(in)  :: basic_name   !! Basic kernel name
        integer(int32),     intent(in)  :: tile_dim     !! Size of shared memory tile, template parameter
        integer(int32),     intent(in)  :: block_rows   !! Number of rows processed by single thread, template parameter
        integer(int32),     intent(in)  :: padding      !! Padding to avoid shared memory bank conflicts, template parameter
        character(c_char), allocatable  :: expression(:)!! Name expression

        expression = get_name_expression(basic_name, tile_dim, block_rows, padding)
        NVRTC_CALL( nvrtcAddNameExpression(prog, expression) )
        deallocate (expression)
    end subroutine set_name_expression

    function get_mangled_name(basic_name, prog, tile_dim, block_rows, padding) result(mangled)
    !! Gets mangled name for given template parameters from nvRTC program
        character(len=*),   intent(in)  :: basic_name   !! Basic kernel name
        type(nvrtcProgram), intent(in)  :: prog         !! nvRTC program
        integer(int32),     intent(in)  :: tile_dim     !! Size of shared memory tile, template parameter
        integer(int32),     intent(in)  :: block_rows   !! Number of rows processed by single thread, template parameter
        integer(int32),     intent(in)  :: padding      !! Padding to avoid shared memory bank conflicts, template parameter
        type(c_ptr)                     :: mangled      !! Mangled kernel name
        character(c_char), allocatable  :: expression(:)!! Name expression

        expression = get_name_expression(basic_name, tile_dim, block_rows, padding)
        NVRTC_CALL( nvrtcGetLoweredName(prog, expression, mangled) )
        deallocate (expression)
    end function get_mangled_name

    function get_code(kernel_name, ndims, base_storage, kernel_type) result(code)
    !! Generates code that will be used to locally tranpose data and prepares to send it to other processes
        character(len=*),     intent(in)  :: kernel_name  !! Name of CUDA kernel
        integer(int32),       intent(in)  :: ndims        !! Number of dimensions
        integer(int64),       intent(in)  :: base_storage !! Number of bytes needed to store single element
        type(kernel_type_t),  intent(in)  :: kernel_type  !! Type of kernel to generate code for
        type(codegen_t)                   :: code          !! Resulting code
        character(len=:), allocatable     :: buffer_type      !! Type of buffer that should be used
        character(len=2)  :: temp       !! Temporary string
        logical           :: is_packer  !! Is this pack/unpack kernel

        select case (base_storage)
        case (FLOAT_STORAGE_SIZE)
            allocate (buffer_type, source="float")
        case (DOUBLE_STORAGE_SIZE)
            allocate (buffer_type, source="double")
        case (DOUBLE_COMPLEX_STORAGE_SIZE)
            allocate (buffer_type, source="double2")
        case default
            INTERNAL_ERROR("get_code: unknown `base_storage`")
        end select

        is_packer = is_unpack_kernel(kernel_type) .or. is_pack_kernel(kernel_type)

        if (kernel_type == KERNEL_PERMUTE_FORWARD .or. kernel_type == KERNEL_PACK_FORWARD) then
            temp = "ny"
        else
            temp = "nz"
        end if

        call code%add("template <int TILE_DIM, int BLOCK_ROWS, int PADDING>")
        call code%add('__global__ void')
        call code%add(kernel_name)
        call code%add("(")
        call code%add("   "//buffer_type//" * __restrict__ out")
        call code%add("    ,const "//buffer_type//" * __restrict__ in")
        call code%add("    ,const int nx")
        call code%add("    ,const int ny")
        call code%add("    ,const int nz")
        if (is_packer) then
            call code%add("   ,const int nxx")
            call code%add("   ,const int nyy")
            call code%add("   ,const int nzz")
            call code%add("   ,const int din")
            call code%add("   ,const int dout")
        end if
        call code%add(")")
        call code%add("{")
        call code%add("    __shared__ "//buffer_type//" tile[TILE_DIM][TILE_DIM + PADDING];")
        call code%add("    const int x_in = threadIdx.x + TILE_DIM * blockIdx.x;")
        call code%add("    const int y_in = threadIdx.y + TILE_DIM * blockIdx.y;")
        call code%add("    const int z = blockIdx.z;")
        if (is_transpose_kernel(kernel_type)) then
            call code%add("    const int x_out = threadIdx.y + TILE_DIM * blockIdx.x;")
            call code%add("    const int y_out = threadIdx.x + TILE_DIM * blockIdx.y;")
        end if
        if (ndims == 2 .and. .not. is_packer) then
            call code%add("    const int ibase = x_in;")
            call code%add("    const int obase = y_out;")
        else
            select case (kernel_type%val)
            case (KERNEL_PERMUTE_FORWARD%val)
                call code%add("    const int ibase = x_in + z * ny * nx;")
                call code%add("    const int obase = y_out + z * ny;")
            case (KERNEL_PACK_FORWARD%val)
                call code%add("    const int ibase = din + x_in + z * ny * nx;")
                call code%add("    const int obase = dout + y_out + z * nyy;")
            case (KERNEL_PERMUTE_BACKWARD%val)
                call code%add("    const int ibase = x_in + z * nx;")
                call code%add("    const int obase = y_out + z * nx * nz;")
            case (KERNEL_PACK_BACKWARD%val)
                call code%add("    const int ibase = din + x_in + z * nx;")
                call code%add("    const int obase = dout + y_out + z * nxx * nzz;")
            case (KERNEL_PERMUTE_BACKWARD_START%val)
                call code%add("    const int ibase = x_in + z * nx;")
                call code%add("    const int obase = y_out + z * nz;")
            case (KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val)
                call code%add("    const int ibase = din + x_in + z * nxx;")
                call code%add("    const int obase = dout + x_in + z * nx * ny;")
            case (KERNEL_UNPACK_PIPELINED%val)
                call code%add("    const int ibase = din + x_in + z * nxx * nyy;")
                call code%add("    const int obase = dout + x_in + z * nx * ny;")
            case (KERNEL_PACK_PIPELINED%val)
                call code%add("    const int ibase = din + x_in + z * nx * ny;")
                call code%add("    const int obase = dout + x_in + z * nxx * nyy;")
            end select
        end if
        call code%add("    #pragma unroll")
        call code%add("    for(int offset = 0; offset < TILE_DIM; offset+=BLOCK_ROWS) {")
        call code%add("        int y = y_in + offset;")
        if ( is_packer .and.  is_transpose_kernel(kernel_type)) then
            call code%add("        if( x_in < nxx && y < "//temp//temp(2:2)//" ) {")
        else if (is_packer) then
            call code%add("        if( x_in < nxx && y < nyy) {")
        else
            call code%add("        if( x_in < nx && y < "//temp//") {")
        end if
        select case (kernel_type%val)
        case (KERNEL_PERMUTE_FORWARD%val, KERNEL_PACK_PIPELINED%val, KERNEL_PACK_FORWARD%val)
            call code%add("            int iidx = ibase + y * nx;")
        case (KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val)
            call code%add("            int iidx = ibase + y * nxx * nzz;")
        case (KERNEL_UNPACK_PIPELINED%val)
            call code%add("            int iidx = ibase + y * nxx;")
        case default
            call code%add("            int iidx = ibase + y * nx * ny;")
        end select
        call code%add("            tile[threadIdx.x][threadIdx.y + offset] = in[iidx];")
        call code%add("        }")
        call code%add("    }")
        call code%add("    __syncthreads();")
        call code%add("    #pragma unroll")
        call code%add("    for(int offset = 0; offset < TILE_DIM; offset+=BLOCK_ROWS) {")
        if ( is_packer .and. is_transpose_kernel(kernel_type) ) then
            call code%add("      int x = x_out + offset;")
            call code%add("      if( x < nxx && y_out < "//temp//temp(2:2)//" ) {")
        else if (is_packer) then
            call code%add("      int y = y_in + offset;")
            call code%add("      if( x_in < nxx && y < nyy ) {")
        else
            call code%add("      int x = x_out + offset;")
            call code%add("      if( x < nx && y_out < "//temp//" ) {")
        end if
        if (ndims == 2 .and. .not. is_packer) then
            call code%add("        int oidx = obase + x * ny;")
        else
            if (any(kernel_type == [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD_START])) then
                call code%add("        int oidx = obase + x * ny * nz;")
            else if ( kernel_type == KERNEL_PACK_FORWARD ) then
                call code%add("        int oidx = obase + x * nyy * nzz;")
            else if (is_unpack_kernel(kernel_type)) then
                call code%add("        int oidx = obase + y * nx;")
            else if (kernel_type == KERNEL_PACK_PIPELINED) then
                call code%add("        int oidx = obase + y * nxx;")
            else if ( kernel_type == KERNEL_PACK_BACKWARD ) then
                call code%add("        int oidx = obase + x * nzz;")
            else
                call code%add("        int oidx = obase + x * nz;")
            end if
        end if
        if (is_packer .and. .not.is_transpose_kernel(kernel_type)) then
            call code%add("        out[oidx] = tile[threadIdx.x][threadIdx.y + offset];")
        else
            call code%add("        out[oidx] = tile[threadIdx.y + offset][threadIdx.x];")
        end if
        call code%add("      }")
        call code%add("    }")
        call code%add("}")
        deallocate (buffer_type)
    end function get_code
end module dtfft_nvrtc_module
