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
module dtfft_interface_szp
!! Fortran interface to SZp compression library
use, intrinsic :: iso_c_binding
implicit none
private

public :: szp_compress
    interface
        integer(c_size_t) function szp_compress(n_elements, data, cmpData, storage_size, mode, tolerance, blockSize) bind(C, name="_dtfft_szp_compress")
        !! Compresses `n_elements` scalars into `cmpData`.
        !! Returns total number of bytes written to `cmpData` or 0 in case of failure.
        import
            integer(c_size_t),  value   :: n_elements   !! Number of scalars to compress
            type(c_ptr),        value   :: data         !! Pointer to uncompressed data
            type(c_ptr),        value   :: cmpData      !! Pointer to compressed buffer
            integer(c_size_t),  value   :: storage_size !! Size of single scalar in bytes
            integer(c_int32_t), value   :: mode         !! Compression mode
            real(c_double),     value   :: tolerance    !! Absolute or relative error bound, based on `mode`
            integer(c_int32_t), value   :: blockSize    !! Number of scalars in a single SZp block
        end function szp_compress
    end interface

public :: szp_decompress
    interface
        integer(c_size_t) function szp_decompress(n_elements, cmpData, data, storage_size, blockSize) bind(C, name="_dtfft_szp_decompress")
        !! Decompresses `n_elements` scalars from `cmpData` into `data`.
        !! Returns number of decompressed bytes or 0 in case of failure.
        import
            integer(c_size_t),  value   :: n_elements   !! Number of scalars to decompress
            type(c_ptr),        value   :: cmpData      !! Pointer to compressed data
            type(c_ptr),        value   :: data         !! Pointer to uncompressed buffer
            integer(c_size_t),  value   :: storage_size !! Size of single scalar in bytes
            integer(c_int32_t), value   :: blockSize    !! Number of scalars in a single SZp block
        end function szp_decompress
    end interface
end module dtfft_interface_szp
