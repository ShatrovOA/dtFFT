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
module dtfft_interface_bitcomp
!! Fortran interface to bitcomp compression library
use, intrinsic :: iso_c_binding
use dtfft_abstract_compressor
use dtfft_parameters
implicit none
private
public :: bitcompCompress, bitcompDecompress

interface
    function bitcompCompress(in, out, n, mode, storage_size, delta, stream) result(compressed_size) bind(C, name="_dtfft_compress_bitcomp")
    !! Compresses the input data using bitcomp compression library
    import
        type(c_ptr),    value :: in     !! Pointer to the original data
        type(c_ptr),    value :: out    !! Pointer to the compressed data
        integer(c_size_t), value :: n   !! Number of scalars in the input data
        type(dtfft_compression_mode_t), value :: mode   !! Compression mode
        integer(c_size_t),  value :: storage_size   !! Size of single input scalar in bytes
        real(c_double), value :: delta  !! Error bound for compression
        type(dtfft_stream_t), value :: stream   !! Stream handle
        integer(c_size_t) :: compressed_size  !! Size of the compressed data in bytes
    end function bitcompCompress

    function bitcompDecompress(in, out, stream) result(decompressed_size) bind(C, name="_dtfft_decompress_bitcomp")
    !! Decompresses the input data using bitcomp compression library
    import
        type(c_ptr),    value :: in !! Pointer to the compressed data
        type(c_ptr),    value :: out    !! Pointer to the decompressed data
        type(dtfft_stream_t), value :: stream   !! Stream handle
        integer(c_size_t) :: decompressed_size  !! Size of the decompressed data in bytes
    end function bitcompDecompress
end interface

end module dtfft_interface_bitcomp