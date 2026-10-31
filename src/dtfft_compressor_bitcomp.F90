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
module dtfft_compressor_bitcomp
!! bitcomp-based compressor implementation
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_compressor
use dtfft_parameters
use dtfft_errors
use dtfft_interface_bitcomp
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: compressor_bitcomp

    type, extends(abstract_compressor) :: compressor_bitcomp
    !! bitcomp-based compressor implementation
    private
        type(dtfft_compression_mode_t) :: mode
        real(real64) :: tolerance
    contains
        procedure   :: create_private => create         !! Initializes the SZ3 compressor with given configuration
        procedure   :: compress_private => compress     !! Compresses the input data using SZ3
        procedure   :: decompress_private => decompress !! Decompresses the input data using SZ3
        procedure   :: destroy                          !! Cleans up the compressor resources
        procedure   :: sync
        procedure   :: pre_sync
        procedure   :: post_sync
        procedure, nopass   :: get_supported_modes
    end type compressor_bitcomp

contains

    integer(int32) function create(self, config, platform)
    !! Initializes the bitcomp compressor with given configuration
        class(compressor_bitcomp),          intent(inout)   :: self         !! Compressor instance
        type(dtfft_compression_config_t),   intent(in)      :: config       !! Compression settings
        type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform (CPU or CUDA)

        create = DTFFT_SUCCESS
        self%mode = config%compression_mode
        self%tolerance = config%tolerance
    end function create

    integer(int64) function compress(self, dims, in, out, stream)
    !! Compresses the input data using bitcomp
        class(compressor_bitcomp),  intent(inout)   :: self         !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)      !! Array dimensions
        type(c_ptr),                intent(in)      :: in           !! Pointer to input data
        type(c_ptr),                intent(in)      :: out          !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream handle
        integer(c_size_t) :: n, compressed_size

        n = self%get_n_elements(dims)
        compressed_size = bitcompCompress(in, out, n, self%mode, self%storage_size, self%tolerance, stream)
        compress = int(compressed_size, int64)
    end function compress

    subroutine decompress(self, dims, in, out, stream, compressed_size)
    !! Decompresses the input data using bitcomp
        class(compressor_bitcomp),  intent(inout)   :: self         !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions. Unused. bitcomp handle is created with compressed data itself
        type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out              !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        integer(int64),             intent(in)      :: compressed_size  !! Compressed buffer size. Unused
        integer(c_size_t) :: decomp_size

        decomp_size = bitcompDecompress(in, out, stream)
#ifdef DTFFT_DEBUG
        if ( decomp_size == 0 ) then
            INTERNAL_ERROR("Failed to decompress data with bitcomp")
        endif
#endif
    end subroutine decompress

    subroutine destroy(self)
    !! Cleans up the compressor resources
        class(compressor_bitcomp),  intent(inout)   :: self             !! Compressor instance
    end subroutine destroy

    subroutine pre_sync(self, stream)
        class(compressor_bitcomp),  intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)     :: stream
    end subroutine pre_sync

    subroutine post_sync(self, stream)
        class(compressor_bitcomp),  intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream
    end subroutine post_sync

    subroutine sync(self, stream)
        class(compressor_bitcomp),  intent(inout)  :: self                 !! Compressor instance
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
        cuda_modes = [.true., .false., .false., .true.]
        use_tolerance = .true.
    end subroutine get_supported_modes
end module dtfft_compressor_bitcomp