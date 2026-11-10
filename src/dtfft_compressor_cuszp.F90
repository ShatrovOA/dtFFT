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
module dtfft_compressor_cuszp
!! cuSZp-based CUDA compressor implementation
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_abstract_compressor
use dtfft_parameters
use dtfft_errors
use dtfft_interface_cuszp
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: compressor_cuszp

    type, extends(abstract_compressor) :: compressor_cuszp
    !! cuSZp-based CUDA compressor implementation
    private
        ! real(real32)      :: tolerance  !! Compression tolerance for cuSZp
    contains
        procedure   :: create_private => create
        procedure   :: compress_private => compress
        procedure   :: decompress_private => decompress
        procedure   :: destroy
        procedure   :: sync
        procedure   :: pre_sync
        procedure   :: post_sync
        procedure, nopass   :: get_supported_modes
    end type compressor_cuszp

contains

    integer(int32) function create(self, platform)
    !! Create a cuSZp compressor instance with the specified configuration and platform
        class(compressor_cuszp),            intent(inout) :: self       !! cuSZp compressor instance
        type(dtfft_platform_t),             intent(in)    :: platform   !! Platform type (should be CUDA for cuSZp)

        create = DTFFT_SUCCESS
        ! self%tolerance = real(config%tolerance, real32)
    end function create

    integer(int64) function compress(self, dims, in, out, stream)
    !! Compress data using cuSZp
        class(compressor_cuszp), intent(inout) :: self      !! cuSZp compressor instance
        integer(int32),          intent(in)    :: dims(:)   !! Dimensions of the data to compress
        type(c_ptr),             intent(in)    :: in        !! Pointer to the input data to compress
        type(c_ptr),             intent(in)    :: out       !! Pointer to the output buffer for compressed data
        type(dtfft_stream_t),    intent(in)    :: stream    !! CUDA stream for asynchronous operations
        integer(c_size_t)   :: compressed_size  !! Size of the compressed data
        integer(c_int32_t)  :: dims_(3)

        dims_(1:size(dims)) = dims
        if ( self%is_complex ) dims_(1) = dims_(1) * 2
        compressed_size = cuSZp_compress(in, out, self%ndims, dims_, self%storage_size, real(self%rate_or_tolerance, c_float), stream)
        compress = int(compressed_size, int64)
    end function compress

    subroutine decompress(self, dims, in, out, stream, compressed_size)
    !! Decompress data using cuSZp
        class(compressor_cuszp), intent(inout) :: self              !! cuSZp compressor instance
        integer(int32),          intent(in)    :: dims(:)           !! Dimensions of the data to decompress
        type(c_ptr),             intent(in)    :: in                !! Pointer to the input data to decompress
        type(c_ptr),             intent(in)    :: out               !! Pointer to the output buffer for decompressed data
        type(dtfft_stream_t),    intent(in)    :: stream            !! CUDA stream for asynchronous operations
        integer(int64),          intent(in)    :: compressed_size   !! Size of the compressed data
        integer(c_int32_t)  :: dims_(3)

        dims_(1:size(dims)) = dims
        if ( self%is_complex ) dims_(1) = dims_(1) * 2
        call cuSZp_decompress(in, out, self%ndims, dims_, compressed_size, self%storage_size, real(self%rate_or_tolerance, c_float), stream)
    end subroutine decompress

    subroutine destroy(self)
    !! No-op for cuSZp compressor, but required for interface compliance
        class(compressor_cuszp), intent(inout) :: self      !! cuSZp compressor instance
    end subroutine destroy

    subroutine pre_sync(self, stream)
    !! No-op for cuSZp compressor, but required for interface compliance
        class(compressor_cuszp), intent(inout) :: self      !! cuSZp compressor instance
        type(dtfft_stream_t),    intent(in)    :: stream    !! CUDA stream for synchronization
    end subroutine pre_sync

    subroutine post_sync(self, stream)
    !! No-op for cuSZp compressor, but required for interface compliance
        class(compressor_cuszp), intent(inout) :: self      !! cuSZp compressor instance
        type(dtfft_stream_t),    intent(in)    :: stream    !! CUDA stream for synchronization
    end subroutine post_sync

    subroutine sync(self, stream)
    !! Synchronize the CUDA stream for cuSZp compressor, but required for interface compliance
        class(compressor_cuszp), intent(inout) :: self      !! cuSZp compressor instance
        type(dtfft_stream_t),    intent(in)    :: stream    !! CUDA stream for synchronization
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

        host_modes = [.false., .false., .false., .false.]
        cuda_modes = [.false., .false., .false., .true.]
        use_tolerance = .true.
    end subroutine get_supported_modes
end module dtfft_compressor_cuszp