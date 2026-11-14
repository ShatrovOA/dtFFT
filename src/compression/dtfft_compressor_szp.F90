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
module dtfft_compressor_szp
!! SZp-based compressor implementation
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_abstract_compressor
use dtfft_parameters
use dtfft_errors
use dtfft_interface_szp
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_mpi.h"
implicit none
private
public :: compressor_szp

    type, extends(abstract_compressor) :: compressor_szp
    !! SZp-based compressor implementation
    private
        integer(int32)                  :: block_size
        ! type(dtfft_compression_mode_t)  :: mode         !! Compression mode
        ! real(real64)                    :: tolerance    !! Absolute or relative error bound, based on `mode`
    contains
        procedure   :: create_private => create         !! Initializes the SZp compressor with given configuration
        procedure   :: compress_private => compress     !! Compresses the input data using SZp
        procedure   :: decompress_private => decompress !! Decompresses the input data using SZp
        procedure   :: destroy                          !! Cleans up the compressor resources
        procedure   :: sync
        procedure   :: pre_sync
        procedure   :: post_sync
        procedure, nopass   :: get_supported_modes
    end type compressor_szp

contains

    integer(int32) function create(self, platform)
    !! Initializes the SZp compressor with given configuration
        class(compressor_szp),              intent(inout)   :: self         !! Compressor instance
        type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform (CPU or CUDA)

        create = DTFFT_SUCCESS
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            create = DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED
            return
        endif
        ! self%mode = config%compression_mode
        ! self%tolerance = config%tolerance
        self%block_size = 64
        if ( self%storage_size == FLOAT_STORAGE_SIZE ) self%block_size = 128
    end function create

    integer(int64) function compress(self, dims, in, out, stream)
    !! Compresses the input data using SZp
        class(compressor_szp),      intent(inout)   :: self         !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)      !! Array dimensions
        type(c_ptr),                intent(in)      :: in           !! Pointer to input data
        type(c_ptr),                intent(in)      :: out          !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream handle
        integer(c_size_t) :: compressed_size

        compressed_size = szp_compress(self%get_n_elements(dims), in, out, self%storage_size, self%compression_mode%val, self%rate_or_tolerance, self%block_size)
        compress = int(compressed_size, int64)
    end function compress

    subroutine decompress(self, dims, in, out, stream, compressed_size)
    !! Decompresses the input data using SZp
        class(compressor_szp),      intent(inout)   :: self             !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out              !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        integer(int64),             intent(in)      :: compressed_size
        integer(c_size_t) :: decompSize

        decompSize = szp_decompress(self%get_n_elements(dims), in, out, self%storage_size, self%block_size)
#ifdef DTFFT_DEBUG
        if ( decompSize == 0 ) then
            INTERNAL_ERROR("Failed to decompress data with SZp")
        endif
#endif
    end subroutine decompress

    subroutine destroy(self)
    !! Cleans up the compressor resources
        class(compressor_szp),      intent(inout)   :: self             !! Compressor instance
    end subroutine destroy

    subroutine pre_sync(self, stream)
        class(compressor_szp),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)     :: stream
    end subroutine pre_sync

    subroutine post_sync(self, stream)
        class(compressor_szp),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream
    end subroutine post_sync

    subroutine sync(self, stream)
        class(compressor_szp),      intent(inout)  :: self                 !! Compressor instance
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
        logical,    intent(out) :: cuda_modes(4)        !! Supported cuda compression modes: .false. for all modes
        logical,    intent(out) :: use_tolerance        !! .true.

        host_modes = [.false., .false., .true., .true.]
        cuda_modes = [.false., .false., .false., .false.]
        use_tolerance = .true.
    end subroutine get_supported_modes
end module dtfft_compressor_szp
