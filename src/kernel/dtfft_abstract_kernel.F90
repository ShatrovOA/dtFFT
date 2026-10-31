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
module dtfft_abstract_kernel
!! This module defines [[abstract_kernel]] type and its type bound procedures.
!!
!! The abstract kernel is used in `reshape_handle_generic` type and
!! is resposible for packing/unpacking/permute operations.
!! The actual implementation of the kernel is deferred to the
!! `create_private`, `execute_private` and `destroy_private` procedures.
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_compressor, only: abstract_compressor
use dtfft_config
use dtfft_parameters
use dtfft_serializer
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: abstract_kernel, kernel_options
public :: kernel_type_t, get_kernel_string, get_fused
public :: is_unpack_kernel, is_pack_kernel, is_transpose_kernel
public :: bytes_to_floats

public :: operator(==)
    interface operator(==)
        module procedure kernel_type_eq     !! Check if two `kernel_type_t` are equal
    end interface

public :: operator(/=)
    interface operator(/=)
        module procedure kernel_type_ne     !! Check if two `kernel_type_t` are not equal
    end interface

    type :: kernel_type_t
    !! Type of kernel
        integer(int32) :: val
    end type kernel_type_t

    type(kernel_type_t), parameter, public  :: KERNEL_COMPRESSION         = kernel_type_t(-2)
    !! Only compression/decompression kernel
    type(kernel_type_t), parameter, public  :: KERNEL_DUMMY               = kernel_type_t(-1)
    !! Dummy kernel, does nothing
    type(kernel_type_t), parameter, public  :: KERNEL_PACK                = kernel_type_t(1)
    !! Packs local buffer into contiguous memory
    type(kernel_type_t), parameter, public  :: KERNEL_COPY_PIPELINED      = kernel_type_t(2)
    !! Performs partial buffer copy
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK              = kernel_type_t(3)
    !! Unpacks contiguous buffer into strided layout
    type(kernel_type_t), parameter, public  :: KERNEL_COPY                = kernel_type_t(4)
    !! Performs buffer copy
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_PIPELINED    = kernel_type_t(5)
    !! Unpacks only single part of contiguous buffer into strided layout
    type(kernel_type_t), parameter, public  :: KERNEL_PACK_PIPELINED      = kernel_type_t(6)
    !! Packs only single part of local buffer into contiguous memory
    type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_FORWARD     = kernel_type_t(7)
    !! Local permutation forward: XYZ -> YZX
    type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD    = kernel_type_t(8)
    !! Local permutation backward: ZXY -> YZX
    type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_START    = kernel_type_t(9)
    !! Starts Local permutation backward: ZXY -> YXZ
    type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_END    = kernel_type_t(10)
    !! Ends Local permutation backward: YXZ -> YZX
    type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_END_PIPELINED   = kernel_type_t(11)
    !! Ends Local permutation backward: YXZ -> YZX. only partial permutation is performed
    type(kernel_type_t), parameter, public  :: KERNEL_PACK_FORWARD     = kernel_type_t(12)
    !! Partial forward permutation
    type(kernel_type_t), parameter, public  :: KERNEL_PACK_BACKWARD     = kernel_type_t(13)
    !! Partial backward permutation
    ! type(kernel_type_t), parameter, public  :: KERNEL_PACK_BACKWARD_START     = kernel_type_t(14)
    ! !! Starts of Partial backward permutation
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_FORWARD = kernel_type_t(15)
    !! Unpacks contiguous buffer into strided layout and perform forward permutation
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_FORWARD_PIPELINED = kernel_type_t(16)
    !! Unpacks only single part of contiguous buffer into strided layout and perform forward permutation
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_BACKWARD = kernel_type_t(17)
    !! Unpacks contiguous buffer into strided layout and perform backward permutation
    type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_BACKWARD_PIPELINED = kernel_type_t(18)
    !! Unpacks only single part of contiguous buffer into strided layout and perform backward permutation

    type(kernel_type_t), parameter          :: TRANSPOSE_KERNELS(*) = [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD, KERNEL_PERMUTE_BACKWARD_START, KERNEL_PACK_FORWARD, KERNEL_PACK_BACKWARD, KERNEL_UNPACK_FORWARD_PIPELINED, KERNEL_UNPACK_BACKWARD_PIPELINED]
    type(kernel_type_t), parameter          :: UNPACK_KERNELS(*) = [KERNEL_PERMUTE_BACKWARD_END, KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED, KERNEL_UNPACK_FORWARD, KERNEL_UNPACK_FORWARD_PIPELINED, KERNEL_UNPACK_BACKWARD, KERNEL_UNPACK_BACKWARD_PIPELINED]
    !! List of all unpack kernel types
    type(kernel_type_t), parameter          :: PACK_KERNELS(*) = [KERNEL_PACK, KERNEL_COPY_PIPELINED, KERNEL_PACK_PIPELINED, KERNEL_PACK_FORWARD, KERNEL_PACK_BACKWARD]
    !! List of all pack kernel types

    type, abstract :: abstract_kernel
    !! Abstract kernel type
    !!
    !! This kernel type is used in `reshape_handle_generic` type and
    !! is resposible for packing/unpacking/permute operations.
        logical                                 :: is_created = .false.     !! Kernel is created flag.
        logical                                 :: is_dummy = .false.       !! If kernel should do anything or not.
        logical                                 :: is_dummy_kernel = .false.    !! If kernel is of type KERNEL_DUMMY
        logical                                 :: is_dummy_compressed = .false.
        type(kernel_type_t)                     :: kernel_type              !! Type of the kernel
        type(string)                            :: kernel_string
        integer(int32),             allocatable :: neighbor_data(:,:)       !! Neighbor data for pipelined unpacking
        integer(int32),             allocatable :: dims(:)                  !! Local dimensions to process
        class(abstract_compressor), pointer     :: compressor               !! Compressor pointer. Compressor itself is created by generic handle and passed here
        logical                                 :: is_compress              !! Enable compression
        logical                                 :: is_decompress            !! Enable decompression
        integer(int64)                          :: base_storage             !! 
    contains
        procedure,                    pass(self)  :: create           !! Creates kernel
        procedure,                    pass(self)  :: execute          !! Executes kernel
        procedure,                    pass(self)  :: destroy          !! Destroys kernel
        procedure,                    pass(self)  :: set_compressor
        procedure(create_interface),                deferred    :: create_private   !! Creates underlying kernel
        procedure(execute_interface),               deferred    :: execute_private  !! Executes underlying kernel
        procedure(destroy_interface),               deferred    :: destroy_private  !! Destroys underlying kernel
        procedure(get_kernel_options_interface),    deferred    :: get_kernel_options
    end type abstract_kernel

    integer(int8), parameter, public :: KERNEL_SLOT_ABSENT = 0_int8
      !! Kernel slot is not used by this reshape handle
    integer(int8), parameter, public :: KERNEL_SLOT_HOST   = 1_int8
      !! Slot holds a host kernel decision
    integer(int8), parameter, public :: KERNEL_SLOT_DEVICE = 2_int8
      !! Slot holds a device kernel decision

    type :: kernel_options
    !! Result of local kernel autotuning for a single pack or unpack slot.
    !!
    !! Host and device kernels are tuned on different parameters, so both variants share one
    !! fixed-size record and `slot_kind` says which fields are meaningful.
        integer(int8)  :: slot_kind = KERNEL_SLOT_ABSENT
        !! One of `KERNEL_SLOT_*`
        integer(int8)  :: host_kernel = 0
        !! Host: selected host kernel variant
        type(dtfft_access_mode_t) :: access_mode = dtfft_access_mode_t(0)
        !! Host: selected memory access pattern
        ! integer(int8)  :: internal_kernel_type 
        !! Device: kernel type actually used, which may differ from the requested one
        ! integer(int8)  :: padding
        ! !! Device: shared memory padding
        integer(int32) :: co_threshold = 0
        !! Host: cache oblivious threshold
        integer(int16) :: tile_size = 0
        !! Device: tile size
        integer(int16) :: block_rows = 0
        !! Device: number of rows processed per block
    contains
        procedure, pass(self) :: save
        procedure, pass(self) :: load
    end type kernel_options

    abstract interface
        subroutine create_interface(self, effort, base_storage, options, force_effort)
        import
        !! Creates underlying kernel
            class(abstract_kernel),         intent(inout) :: self             !! Abstract kernel
            type(dtfft_effort_t),           intent(in)    :: effort           !! Effort level for generating transpose kernels
            integer(int64),                 intent(in)    :: base_storage     !! Number of bytes needed to store single element
            type(kernel_options), optional, intent(in)    :: options          !! Pointer to kernel options struct
            logical,              optional, intent(in)    :: force_effort     !! Should effort be forced or not
        end subroutine create_interface

        subroutine execute_interface(self, in, out, stream, sync, neighbor)
        import
        !! Executes underlying kernel
            class(abstract_kernel),     intent(inout)   :: self         !! Abstract kernel
            type(c_ptr),                intent(in)      :: in           !! Source buffer, can be device or host pointer
            type(c_ptr),                intent(in)      :: out          !! Target buffer, can be device or host pointer
            type(dtfft_stream_t),       intent(in)      :: stream       !! Stream to execute on, used only for device pointers
            logical,                    intent(in)      :: sync         !! Sync stream after kernel execution
            integer(int32),   optional, intent(in)      :: neighbor     !! Source rank for pipelined unpacking
        end subroutine execute_interface

        subroutine destroy_interface(self)
        import
        !! Destroys underlying kernel
            class(abstract_kernel), intent(inout) :: self               !! Abstract kernel
        end subroutine destroy_interface

        type(kernel_options) function get_kernel_options_interface(self)
        import
            !! Destroys underlying kernel
            class(abstract_kernel), intent(in)  :: self               !! Abstract kernel
        end function get_kernel_options_interface
    end interface

contains

    function get_kernel_string(kernel) result(str)
    !! Gets the string description of a kernel
        type(kernel_type_t), intent(in) :: kernel !! kernel type
        type(string)                    :: str !! kernel string

        select case ( kernel%val )
        case ( KERNEL_PACK%val, KERNEL_PACK_PIPELINED%val )
            str = string("pack")
        case ( KERNEL_COPY%val, KERNEL_COPY_PIPELINED%val )
            str = string("copy")
        case ( KERNEL_UNPACK%val, KERNEL_UNPACK_PIPELINED%val )
            str = string("unpack")
        case ( KERNEL_PERMUTE_FORWARD%val )
            str = string("forward")
        case ( KERNEL_PERMUTE_BACKWARD%val )
            str = string("backward")
        case ( KERNEL_PERMUTE_BACKWARD_START%val )
            str = string("backward_start")
        case ( KERNEL_PERMUTE_BACKWARD_END%val, KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            str = string("backward_end")
        case ( KERNEL_UNPACK_FORWARD%val, KERNEL_UNPACK_FORWARD_PIPELINED%val )
            str = string("unpack_forward")
        case ( KERNEL_UNPACK_BACKWARD%val, KERNEL_UNPACK_BACKWARD_PIPELINED%val )
            str = string("unpack_backward")
        case ( KERNEL_PACK_FORWARD%val )
            str = string("pack_forward")
        case ( KERNEL_PACK_BACKWARD%val )
            str = string("pack_backward")
        case ( KERNEL_COMPRESSION%val )
            str = string("compression")
        case default
            str = string("unknown")
        endselect
    end function get_kernel_string

    function get_fused(kernel) result(fused)
    !! Converts kernel to its fused version if applicable
        type(kernel_type_t),  intent(in)  :: kernel !! kernel type
        type(kernel_type_t)               :: fused  !! fused kernel type

        select case ( kernel%val )
        case ( KERNEL_PACK%val )
            fused = KERNEL_PACK_PIPELINED
        case ( KERNEL_PERMUTE_FORWARD%val )
            fused = KERNEL_PACK_FORWARD
        case ( KERNEL_PERMUTE_BACKWARD%val )
            fused = KERNEL_PACK_BACKWARD
        case default
            fused = kernel
        endselect
    end function get_fused

#if defined(DTFFT_WITH_CUDA) && defined(DTFFT_WITH_MOCK_ENABLED)
    recursive   &
#endif
    subroutine create(self, dims, effort, base_storage, kernel_type, neighbor_data, force_effort, &
        with_compression, with_decompression, options)
    !! Creates kernel
        class(abstract_kernel),         intent(inout) :: self               !! Abstract kernel
        integer(int32),                 intent(in)    :: dims(:)            !! Local dimensions to process
        type(dtfft_effort_t),           intent(in)    :: effort             !! Effort level for generating transpose kernels
        integer(int64),                 intent(in)    :: base_storage       !! Number of bytes needed to store single element
        type(kernel_type_t),            intent(in)    :: kernel_type        !! Type of kernel to build
        integer(int32),       optional, intent(in)    :: neighbor_data(:,:) !! Optional pointers for unpack kernels
        logical,              optional, intent(in)    :: force_effort       !! Should effort be forced or not
        logical,              optional, intent(in)    :: with_compression   !! Enable compression after executing kernel
        logical,              optional, intent(in)    :: with_decompression !! Enable decompression before executing kernel
        type(kernel_options), optional, intent(in)    :: options            !! Pointer to kernel options struct

        call self%destroy()
        OPTVAL( self%is_compress, .false., with_compression )
        OPTVAL( self%is_decompress, .false., with_decompression )
        self%is_dummy = .false.
        self%is_dummy_kernel = .false.
#ifdef DTFFT_WITH_COMPRESSION
        self%compressor => null()
#endif
        if ( any(dims == 0) ) then
            self%is_created = .true.
            self%is_dummy = .true.
            return
        endif
        if ( kernel_type == KERNEL_DUMMY .and. .not.(self%is_compress .or. self%is_decompress) ) then
            self%is_created = .true.
            self%is_dummy_kernel = .true.
            return
        endif

        self%is_dummy_compressed = .false.
        if ( kernel_type == KERNEL_DUMMY .and. (self%is_compress .or. self%is_decompress) ) then
            self%is_dummy_compressed = .true.
            self%kernel_type = KERNEL_COMPRESSION
        else
            self%kernel_type = kernel_type
        endif
        self%base_storage = base_storage

#ifdef DTFFT_DEBUG
        if ( any(kernel_type == [KERNEL_PERMUTE_BACKWARD_START, KERNEL_PERMUTE_BACKWARD_END, KERNEL_PERMUTE_BACKWARD_END_PIPELINED]) ) then
            if ( size(dims) /= 3 ) then
                INTERNAL_ERROR("2-step permutation is only valid for 3d grid")
            endif
        endif
#endif
        if ( is_unpack_kernel(kernel_type) .or. is_pack_kernel(kernel_type) .or. self%is_compress .or. self%is_decompress ) then
#ifdef DTFFT_DEBUG
            if ( .not. present(neighbor_data) ) then
                INTERNAL_ERROR("Neighbor data required")
            endif
#endif
            allocate( self%neighbor_data, source=neighbor_data )
        endif
        allocate( self%dims, source=dims )
        if ( size(dims) == 2 ) then
            select case ( kernel_type%val )
            case ( KERNEL_PACK_BACKWARD%val )
                self%kernel_type = KERNEL_PACK_FORWARD
            case (KERNEL_PERMUTE_BACKWARD%val)
                self%kernel_type = KERNEL_PERMUTE_FORWARD
            case (KERNEL_UNPACK_BACKWARD%val)
                self%kernel_type = KERNEL_UNPACK_FORWARD
            case (KERNEL_UNPACK_BACKWARD_PIPELINED%val)
                self%kernel_type = KERNEL_UNPACK_FORWARD_PIPELINED
            endselect
        endif
        self%kernel_string = get_kernel_string(self%kernel_type)

        if ( .not. self%is_dummy_compressed ) call self%create_private(effort, base_storage, options, force_effort)
        self%is_created = .true.
    end subroutine create

#if defined(DTFFT_WITH_CUDA) && defined(DTFFT_WITH_MOCK_ENABLED)
    recursive   &
#endif
    subroutine execute(self, in, out, stream, neighbor, aux, csize, csizes, skip_compression, skip_rank, sync)
    !! Executes kernel
        class(abstract_kernel),     intent(inout) :: self             !! Abstract kernel
        type(c_ptr),                intent(in)    :: in               !! Source buffer, can be device or host pointer
        type(c_ptr),                intent(in)    :: out              !! Target buffer, can be device or host pointer
        type(dtfft_stream_t),       intent(in)    :: stream           !! Stream to execute on, used only for device pointers
        integer(int32),   optional, intent(in)    :: neighbor         !! Source rank for pipelined unpacking
        type(c_ptr),      optional, intent(in)    :: aux              !! Target buffer, can be device or host pointer
        integer(int64),   optional, intent(inout) :: csize            !! Compressed buffer size
        integer(int64),   optional, intent(inout) :: csizes(:)        !! Multiple compression sizes. This should only be used with CUDA backends
        logical,          optional, intent(in)    :: skip_compression !! Skip compression/decompression stage. Should be used when packing/unpacking from itself.
        integer(int32),   optional, intent(in)    :: skip_rank        !! Skip compression/decompression for specific rank when neighbor is not specified.
        logical,          optional, intent(in)    :: sync             !! Sync stream after packing/compression. Should be used only for fused backends
        logical :: skip_compression_, with_compression, sync_
        integer :: batch_id
        integer(int32) :: skip_rank_
#ifdef DTFFT_WITH_COMPRESSION
        ! integer(int64)  :: compressed_bytes
#endif

        if ( self%is_dummy .or. self%is_dummy_kernel ) return
        OPTVAL(skip_compression_, .false., skip_compression)
        with_compression = self%is_compress .or. self%is_decompress
        if ( self%is_dummy_compressed .and. skip_compression_ ) return
        OPTVAL(skip_rank_, -1, skip_rank)
        OPTVAL(sync_, .false., sync)

#ifdef DTFFT_DEBUG
        if ( .not. self%is_created ) then
            INTERNAL_ERROR("abstract_kernel.execute: kernel not created")
        endif
        ! WRITE_DEBUG("executing "//self%kernel_string%raw)
        if ( any(self%kernel_type == [KERNEL_UNPACK_PIPELINED, KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_COPY_PIPELINED, KERNEL_PACK_BACKWARD, KERNEL_PACK_FORWARD, KERNEL_PACK_PIPELINED]) ) then
            if ( .not.present(neighbor) ) then
                INTERNAL_ERROR("abstract_kernel.execute: Neighbor is not passed")
            endif
            if ( neighbor < 1 .or. neighbor > size(self%neighbor_data, dim=2) ) then
                INTERNAL_ERROR("abstract_kernel.execute: Neighbor index out of bounds")
            endif
        endif
# ifdef DTFFT_WITH_COMPRESSION
        if ( with_compression .and. .not.associated(self%compressor) ) then
            INTERNAL_ERROR("abstract_kernel.execute: .not.associated(self%compressor)")
        endif
        if ( with_compression .and. .not.skip_compression_ .and..not.present(aux) ) then
            INTERNAL_ERROR("abstract_kernel.execute: compression missing workspace")
        endif
        if ( self%is_compress .and. .not.skip_compression_ .and. present(csize) .and. present(csizes) ) then
            INTERNAL_ERROR("abstract_kernel.execute.compress: present(csize) .and. present(csizes)")
        endif
        if ( self%is_compress .and. .not.skip_compression_ .and. .not.present(csize) .and. .not.present(csizes) ) then
            INTERNAL_ERROR("abstract_kernel.execute.compress: .not.present(csize) .and. .not.present(csizes)")
        endif
        if ( self%is_compress .and. .not.skip_compression_ .and. present(csize) .and. .not. present(neighbor) ) then
            INTERNAL_ERROR("abstract_kernel.execute.compress: present(csize) .and. .not. present(neighbor)")
        endif

        if ( self%is_decompress .and. .not.skip_compression_ .and. present(csize) .and. present(csizes) ) then
            INTERNAL_ERROR("abstract_kernel.execute.decompress: present(csize) .and. present(csizes)")
        endif
        if ( self%is_decompress .and. .not.skip_compression_ .and. .not.present(csize) .and. .not.present(csizes) ) then
            INTERNAL_ERROR("abstract_kernel.execute.decompress: .not.present(csize) .and. .not.present(csizes)")
        endif
        if ( self%is_decompress .and. .not.skip_compression_ .and. present(csize) .and. .not. present(neighbor) ) then
            INTERNAL_ERROR("abstract_kernel.execute.decompress: present(csize) .and. .not. present(neighbor)")
        endif
# endif
#endif

        REGION_BEGIN(self%kernel_string%raw, COLOR_CHARTREUSE)
        block
        type(c_ptr) :: work
#ifdef DTFFT_WITH_COMPRESSION
            if ( self%is_decompress .and. .not.skip_compression_ ) then
                work = out; if ( .not. self%is_dummy_compressed ) work = aux
                call self%compressor%pre_sync(stream)
                if ( present(neighbor) .and. present(csize) ) then
                    call self%compressor%decompress(self%neighbor_data(:, neighbor), self%neighbor_data(4, neighbor), in, work, stream, csize)
                else if ( present(neighbor) .and. present(csizes) ) then
                    call self%compressor%decompress(self%neighbor_data(:, neighbor), self%neighbor_data(4, neighbor), in, work, stream, csizes(neighbor))
                else
                    do batch_id = 1, size(self%neighbor_data, dim=2)
                        if ( skip_rank_ == batch_id ) cycle
                        call self%compressor%decompress(self%neighbor_data(:, batch_id), self%neighbor_data(4, batch_id), in, work, stream, csizes(batch_id))
                    enddo
                endif
                call self%compressor%post_sync(stream)
                if ( .not. self%is_dummy_compressed ) call self%execute_private(work, out, stream, .false., neighbor)
            else if ( self%is_compress .and. .not.skip_compression_ ) then
                if ( self%is_dummy_compressed ) then
                    work = in
                else
                    work = aux
                    call self%execute_private(in, work, stream, .false., neighbor)
                endif
                call self%compressor%pre_sync(stream)
                if ( present(neighbor) ) then
                    csize = self%compressor%compress(self%neighbor_data(:, neighbor), self%neighbor_data(5, neighbor), work, out, stream)
                else
                    do batch_id = 1, size(self%neighbor_data, dim=2)
                        if ( skip_rank_ == batch_id ) cycle
                        csizes(batch_id) = self%compressor%compress(self%neighbor_data(:, batch_id), self%neighbor_data(5, batch_id), work, out, stream)
                    enddo
                endif
                call self%compressor%post_sync(stream)
                if ( sync_ ) call self%compressor%sync(stream)
            else
#endif
                call self%execute_private(in, out, stream, sync_, neighbor)
                if ( allocated( self%neighbor_data ) ) then
                    if ( present(csize) ) then
                        csize = self%base_storage * product( self%neighbor_data(1:3, neighbor) )
                    endif
                    if ( present(csizes) ) then
                        do batch_id = 1, size(self%neighbor_data, dim=2)
                            csizes(batch_id) = self%base_storage * product( self%neighbor_data(1:3, batch_id) )
                        enddo
                    endif
                endif
#ifdef DTFFT_WITH_COMPRESSION
            endif
#endif
        endblock
        REGION_END(self%kernel_string%raw)
    end subroutine execute

    subroutine destroy(self)
    !! Destroys kernel
        class(abstract_kernel), intent(inout) :: self                 !! Abstract kernel

        if ( .not. self%is_created ) return
        if ( self%is_dummy ) then
            self%is_created = .false.
            return
        endif
        call self%destroy_private()
#ifdef DTFFT_WITH_COMPRESSION
        self%compressor => null()
#endif
        if ( allocated(self%dims) ) deallocate(self%dims)
        if ( allocated(self%neighbor_data) ) deallocate(self%neighbor_data)
        call self%kernel_string%destroy()
        self%is_dummy_compressed = .false.
        self%is_created = .false.
    end subroutine destroy

    subroutine set_compressor(self, compressor)
    !! Sets created compressor for the kernel
        class(abstract_kernel),             intent(inout) :: self       !! Abstract kernel
        class(abstract_compressor), target, intent(in)    :: compressor !! Compressor to set
        self%compressor => compressor
    end subroutine set_compressor

    subroutine save(self, fh)
        class(kernel_options),  intent(in)    :: self
        type(serializer),       intent(inout) :: fh

        call fh%save(self%slot_kind, .true.)
        call fh%save(self%host_kernel, .true.)
        call fh%save(self%access_mode%val, .true.)
        call fh%save(self%co_threshold, .true.)
        call fh%save(self%tile_size, .true.)
        call fh%save(self%block_rows, .true.)
    end subroutine save

    subroutine load(self, fh)
        class(kernel_options),  intent(inout) :: self
        type(serializer),       intent(inout) :: fh

        call fh%load(self%slot_kind, .true.)
        call fh%load(self%host_kernel, .true.)
        call fh%load(self%access_mode%val, .true.)
        call fh%load(self%co_threshold, .true.)
        call fh%load(self%tile_size, .true.)
        call fh%load(self%block_rows, .true.)
    end subroutine load

    elemental function bytes_to_floats(bytes) result(floats)
    !! Converts number of bytes to number of floats needed to store them
        integer(int64), intent(in)  :: bytes    !! Number of bytes
        integer(int32)              :: floats   !! Number of floats
        integer(int64) :: padding, n_floats

        if (bytes <= 0_int64) then
            floats = 0_int32
            return
        end if

        padding = mod(bytes, FLOAT_STORAGE_SIZE)
        if (padding == 0_int64) then
            n_floats = bytes / FLOAT_STORAGE_SIZE
        else
            n_floats = ( bytes + (FLOAT_STORAGE_SIZE - padding) ) / FLOAT_STORAGE_SIZE
        end if
        floats = int(n_floats, int32)
    end function bytes_to_floats

    MAKE_EQ_FUN(kernel_type_t, kernel_type_eq)
    MAKE_NE_FUN(kernel_type_t, kernel_type_ne)
    MAKE_VALID_FUN_DTYPE(kernel_type_t, is_transpose_kernel, TRANSPOSE_KERNELS)
    MAKE_VALID_FUN_DTYPE(kernel_type_t, is_unpack_kernel, UNPACK_KERNELS)
    MAKE_VALID_FUN_DTYPE(kernel_type_t, is_pack_kernel, PACK_KERNELS)
end module dtfft_abstract_kernel
