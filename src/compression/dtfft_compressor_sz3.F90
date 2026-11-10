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
module dtfft_compressor_sz3
!! SZ3-based compressor implementation
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_abstract_compressor
! #ifdef DTFFT_WITH_CUDA
! use dtfft_interface_cuda_runtime
! #endif
use dtfft_parameters
use dtfft_errors
use dtfft_interface_sz3
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: compressor_sz3

    type, extends(abstract_compressor) :: compressor_sz3
    !! SZ3-based compressor implementation
    private
    contains
        procedure   :: create_private => create         !! Initializes the SZ3 compressor with given configuration
        procedure   :: compress_private => compress     !! Compresses the input data using SZ3
        procedure   :: decompress_private => decompress !! Decompresses the input data using SZ3
        procedure   :: destroy                          !! Cleans up the compressor resources
        procedure   :: sync
        procedure   :: pre_sync
        procedure   :: post_sync
        procedure,  pass(self) :: get_config
        procedure, nopass   :: get_supported_modes
    end type compressor_sz3

contains

    integer(int32) function create(self, platform)
    !! Initializes the SZ3 compressor with given configuration
        class(compressor_sz3),              intent(inout)   :: self         !! Compressor instance
        type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform (CPU or CUDA)

        create = DTFFT_SUCCESS
        ! self%mode = config%compression_mode
        ! self%tolerance = config%tolerance
    end function create

    function get_config(self, dims) result(config)
        class(compressor_sz3),  intent(in)  :: self         !! Compressor instance
        integer(int32),         intent(in)  :: dims(:)  !! Array dimensions
        type(c_ptr)                         :: config
        integer(c_int32_t) :: x, y, z
        integer(c_int32_t) :: ierr

        config = c_null_ptr
        x = dims(1); y = dims(2); z = 0
        if ( size(dims) == 3 ) then
            z = dims(3)
        endif
        if ( self%is_complex ) x = x * 2
        ierr = sz3_create_config(x, y, z, self%compression_mode%val, self%rate_or_tolerance, config)
#ifdef DTFFT_DEBUG
        if ( ierr /= 0 ) then
            INTERNAL_ERROR("Failed to create sz3 config")
        endif
#endif
    end function get_config

    integer(int64) function compress(self, dims, in, out, stream)
    !! Compresses the input data using SZ3
        class(compressor_sz3),      intent(inout)   :: self         !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)      !! Array dimensions
        type(c_ptr),                intent(in)      :: in           !! Pointer to input data
        type(c_ptr),                intent(in)      :: out          !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream handle
        type(c_ptr) :: config
        integer(c_size_t) :: cmpCap, compressed_size

        config = self%get_config(dims)
        cmpCap = self%get_n_elements(dims) * self%true_storage_size * 2

        compressed_size = sz3_compress(config, in, out, cmpCap, self%storage_size)
        call sz3_destroy_config(config)
        compress = int(compressed_size, int64)
    end function compress

    subroutine decompress(self, dims, in, out, stream, compressed_size)
    !! Decompresses the input data using SZ3
        class(compressor_sz3),      intent(inout)   :: self             !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out              !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        integer(int64),             intent(in)      :: compressed_size
        type(c_ptr) :: config
        integer(c_size_t) :: decompSize

        config = self%get_config(dims)
        decompSize = sz3_decompress(config, in, compressed_size, out, self%storage_size)
#ifdef DTFFT_DEBUG
        if ( decompSize == 0 ) then
            INTERNAL_ERROR("Failed to decompress data with SZ3")
        endif
#endif
        call sz3_destroy_config(config)
    end subroutine decompress

    subroutine destroy(self)
    !! Cleans up the compressor resources
        class(compressor_sz3),      intent(inout)   :: self             !! Compressor instance
    end subroutine destroy

    subroutine pre_sync(self, stream)
        class(compressor_sz3),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)     :: stream
    end subroutine pre_sync

    subroutine post_sync(self, stream)
        class(compressor_sz3),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream
    end subroutine post_sync

    subroutine sync(self, stream)
        class(compressor_sz3),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream
    end subroutine sync

    pure subroutine get_supported_modes(host_modes, cuda_modes, use_tolerance)
    !! Returns supported compression modes
    !! Arrays are of size 4, corresponding to the order of `VALID_COMPRESSION_MODES`:
    !! 1. DTFFT_COMPRESSION_MODE_LOSSLESS
    !! 2. DTFFT_COMPRESSION_MODE_FIXED_RATE
    !! 3. DTFFT_COMPRESSION_MODE_FIXED_PRECISION
    !! 4. DTFFT_COMPRESSION_MODE_FIXED_ACCURACY
        logical,    intent(out) :: host_modes(4)        !! Supported host compression modes
        logical,    intent(out) :: cuda_modes(4)        !! Supported cuda compression modes
        logical,    intent(out) :: use_tolerance        !! .true.

        host_modes = [.true., .false., .true., .true.]
        cuda_modes = [.false., .false., .false., .false.]
        use_tolerance = .true.
    end subroutine get_supported_modes
end module dtfft_compressor_sz3