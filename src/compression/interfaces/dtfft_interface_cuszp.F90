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
use, intrinsic :: iso_c_binding
use dtfft_interface_cuda
use dtfft_parameters
implicit none
private
public :: cuSZp_compress, cuSZp_decompress

! public :: cuszp_dim_t
!     type, bind(C) :: cuszp_dim_t
!     !! The dimension of cuszp.
!         integer(c_int) :: val   !! Internal value
!     end type cuszp_dim_t

!     type(cuszp_dim_t), parameter, public :: CUSZP_DIM_1D = cuszp_dim_t(1)
!     !! 1D compression
!     type(cuszp_dim_t), parameter, public :: CUSZP_DIM_2D = cuszp_dim_t(2)
!     !! 2D compression
!     type(cuszp_dim_t), parameter, public :: CUSZP_DIM_3D = cuszp_dim_t(3)
!     !! 3D compression

! public :: cuszp_mode_t
!     type, bind(C) :: cuszp_mode_t
!     !! The compression mode of cuszp.
!         integer(c_int) :: val   !! Internal value
!     end type cuszp_mode_t

!     type(cuszp_mode_t), parameter, public :: CUSZP_MODE_FIXED = cuszp_mode_t(0)
!     !! Only fixed-length encoding is used. 
!     !! This mode is designed for compressing unsmooth and non-structured scientific data or machine learning weights/tokens.
!     type(cuszp_mode_t), parameter, public :: CUSZP_MODE_PLAIN = cuszp_mode_t(1)
!     !! Dimension-aware delta encoding + fixed-length encoding. 
!     !! This mode is designed for scientfic datasets with only local smoothness or just sparse (containing lots of 0s).
!     type(cuszp_mode_t), parameter, public :: CUSZP_MODE_OUTLIER = cuszp_mode_t(2)
!     !! Dimension-aware delta encoding + fixed-length encoding + outlier preservation. 
!     !! This mode is designed for non-sparse and globally smooth scientific datasets. Always highest compression ratios.

! public :: cuszp_type_t
!     type, bind(C) :: cuszp_type_t
!     !! The data type of cuszp.
!         integer(c_int) :: val   !! Internal value
!     end type cuszp_type_t

!     type(cuszp_type_t), parameter, public :: CUSZP_TYPE_FLOAT = cuszp_type_t(0)
!     !! Single precision floating point
!     type(cuszp_type_t), parameter, public :: CUSZP_TYPE_DOUBLE = cuszp_type_t(1)
!     !! Double precision floating point

interface

    function cuSZp_compress(in, out, ndims, dims, storage_size, errorBound, stream) result(compressed_size) bind(C, name="_dtfft_compress_cuszp")
    import
        type(c_ptr),            value :: in  !! Pointer to the original data on device
        type(c_ptr),            value :: out !! Pointer to the compressed data on device
        integer(c_int8_t),      value :: ndims
        integer(c_int32_t)            :: dims(*)
        integer(c_size_t),      value :: storage_size
        real(c_float),          value :: errorBound !! Error bound for compression
        type(dtfft_stream_t),   value :: stream
        integer(c_size_t)             :: compressed_size
    end function cuSZp_compress

    subroutine cuSZp_decompress(in, out, ndims, dims, cmpSize, storage_size, errorBound, stream) bind(C, name="_dtfft_decompress_cuszp")
    import
        type(c_ptr),            value :: in  !! Pointer to the original data on device
        type(c_ptr),            value :: out !! Pointer to the compressed data on device
        integer(c_int8_t),      value :: ndims
        integer(c_int32_t)            :: dims(*)
        integer(c_size_t),      value :: cmpSize
        integer(c_size_t),      value :: storage_size
        real(c_float),          value :: errorBound !! Error bound for compression
        type(dtfft_stream_t),   value :: stream
    end subroutine cuSZp_decompress

    ! subroutine cuSZp_compress(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, dim, dims, type, mode, stream) bind(C, name="cuSZp_compress")
    !     import
    !     type(c_ptr),            value :: d_oriData  !! Pointer to the original data on device
    !     type(c_ptr),            value :: d_cmpBytes !! Pointer to the compressed data on device
    !     integer(c_size_t),      value :: nbEle      !! Number of elements in the original data
    !     integer(c_size_t)             :: cmpSize    !! Size of the compressed data (output)
    !     real(c_float),          value :: errorBound !! Error bound for compression
    !     type(cuszp_dim_t),      value :: dim        !! Dimension of the data (1D, 2D, or 3D)
    !     type(dim3),             value :: dims       !! Dimensions of the data (x, y, z)
    !     type(cuszp_type_t),     value :: type       !! Data type of the original data (float or double)
    !     type(cuszp_mode_t),     value :: mode       !! Compression mode (fixed, plain, or outlier)
    !     type(dtfft_stream_t),   value :: stream     !! CUDA stream for asynchronous execution
    ! end subroutine cuSZp_compress

    ! subroutine cuSZp_decompress(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, dim, dims, type, mode, stream) bind(C, name="cuSZp_decompress")
    !     import
    !     type(c_ptr),            value :: d_decData  !! Pointer to the decompressed data on device
    !     type(c_ptr),            value :: d_cmpBytes !! Pointer to the compressed data on device
    !     integer(c_size_t),      value :: nbEle      !! Number of elements in the decompressed data
    !     integer(c_size_t),      value :: cmpSize    !! Size of the compressed data
    !     real(c_float),          value :: errorBound !! Error bound for decompression
    !     type(cuszp_dim_t),      value :: dim        !! Dimension of the data (1D, 2D, or 3D)
    !     type(dim3),             value :: dims       !! Dimensions of the data (x, y, z)
    !     type(cuszp_type_t),     value :: type       !! Data type of the decompressed data (float or double)
    !     type(cuszp_mode_t),     value :: mode       !! Compression mode (fixed, plain, or outlier)
    !     type(dtfft_stream_t),   value :: stream     !! CUDA stream for asynchronous execution
    ! end subroutine cuSZp_decompress
end interface
end module dtfft_interface_cuszp