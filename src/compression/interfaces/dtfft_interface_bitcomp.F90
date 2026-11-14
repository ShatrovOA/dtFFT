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
use, intrinsic :: iso_c_binding
use dtfft_abstract_compressor
use dtfft_parameters
implicit none
private
public :: bitcompCompress, bitcompDecompress

interface
    function bitcompCompress(in, out, n, mode, storage_size, delta, stream) result(s) bind(C, name="_dtfft_compress_bitcomp")
    import
        type(c_ptr),    value :: in
        type(c_ptr),    value :: out
        integer(c_size_t), value :: n
        type(dtfft_compression_mode_t), value :: mode
        integer(c_size_t),  value :: storage_size
        real(c_double), value :: delta
        type(dtfft_stream_t), value :: stream
        integer(c_size_t) :: s
    end function bitcompCompress

    function bitcompDecompress(in, out, stream) result(u) bind(C, name="_dtfft_decompress_bitcomp")
    import
        type(c_ptr),    value :: in
        type(c_ptr),    value :: out
        type(dtfft_stream_t), value :: stream
        integer(c_size_t) :: u
    end function bitcompDecompress
end interface

end module dtfft_interface_bitcomp