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
module dtfft_interface_sz3
!! Fortran interface to SZ3 compression library
use iso_c_binding
implicit none
private

public :: sz3_create_config
    interface
        integer(c_int32_t) function sz3_create_config(nx, ny, nz, mode, tolerance, conf) bind(C, name="_dtfft_sz3_create_config")
        !! Allocates config object
        import
            integer(c_int32_t),     value   :: nx               !! number of scalars in x dimension
            integer(c_int32_t),     value   :: ny               !! number of scalars in y dimension
            integer(c_int32_t),     value   :: nz               !! number of scalars in z dimension
            integer(c_int32_t),     value   :: mode
            real(c_double),         value   :: tolerance
            type(c_ptr)                     :: conf
        end function sz3_create_config
    end interface

public :: sz3_destroy_config
    interface
        subroutine sz3_destroy_config(conf) bind(c, name="_dtfft_sz3_destroy_config")
        !! Destroys config object
        import
            type(c_ptr),        value   :: conf !! Allocated config object
        end subroutine sz3_destroy_config
    end interface

public :: sz3_compress
    interface
        integer(c_size_t) function sz3_compress(conf, data, cmpData, cmpCap, storage_size) bind(c, name="_dtfft_sz3_compress")
        !! Set 2D field strides in number of scalars
        import
            type(c_ptr),        value   :: conf !! Allocated config object
            type(c_ptr),        value   :: data
            type(c_ptr),        value   :: cmpData
            integer(c_size_t),  value   :: cmpCap
            integer(c_size_t),  value   :: storage_size
        end function sz3_compress
    end interface

public :: sz3_decompress
    interface
        integer(c_size_t) function sz3_decompress(conf, cmpData, cmpSize, decData, storage_size) bind(c, name="_dtfft_sz3_decompress")
        !! Set 2D field strides in number of scalars
        import
            type(c_ptr),        value   :: conf !! Allocated config object
            type(c_ptr),        value   :: cmpData
            integer(c_size_t),  value   :: cmpSize
            type(c_ptr),        value   :: decData
            integer(c_size_t),  value   :: storage_size
        end function sz3_decompress
    end interface
end module dtfft_interface_sz3