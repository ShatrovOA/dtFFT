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
module dtfft_interface_cuszp
!! Fortran interface to cuSZp compression library
use, intrinsic :: iso_c_binding
use dtfft_interface_cuda
use dtfft_parameters
implicit none
private
public :: cuSZp_compress, cuSZp_decompress

interface

    function cuSZp_compress(in, out, ndims, dims, storage_size, errorBound, stream) result(compressed_size) bind(C, name="_dtfft_compress_cuszp")
    !! Compresses the input data using cuSZp compression library
    import
        type(c_ptr),            value :: in  !! Pointer to the original data on device
        type(c_ptr),            value :: out !! Pointer to the compressed data on device
        integer(c_int8_t),      value :: ndims  !! Number of dimensions in the input data
        integer(c_int32_t)            :: dims(*)    !! Array containing the size of each dimension
        integer(c_size_t),      value :: storage_size   !! Size of single input scalar in bytes
        real(c_float),          value :: errorBound !! Error bound for compression
        type(dtfft_stream_t),   value :: stream !! Stream handle
        integer(c_size_t)             :: compressed_size    !! Size of the compressed data in bytes
    end function cuSZp_compress

    subroutine cuSZp_decompress(in, out, ndims, dims, cmpSize, storage_size, errorBound, stream) bind(C, name="_dtfft_decompress_cuszp")
    !! Decompresses the input data using cuSZp compression library
    import
        type(c_ptr),            value :: in  !! Pointer to the original data on device
        type(c_ptr),            value :: out !! Pointer to the compressed data on device
        integer(c_int8_t),      value :: ndims  !! Number of dimensions in the input data
        integer(c_int32_t)            :: dims(*)    !! Array containing the size of each dimension
        integer(c_size_t),      value :: cmpSize    !! Size of the compressed data in bytes
        integer(c_size_t),      value :: storage_size   !! Size of single input scalar in bytes
        real(c_float),          value :: errorBound !! Error bound for compression
        type(dtfft_stream_t),   value :: stream !! Stream handle
    end subroutine cuSZp_decompress
end interface
end module dtfft_interface_cuszp