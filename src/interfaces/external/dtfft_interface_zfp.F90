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
module dtfft_interface_zfp
!! Fortran interface to ZFP compression library
use iso_c_binding
use iso_fortran_env
implicit none
public :: zfp_create_field, zfp_field_set_stride, zfp_stream_set_rate


public :: zfp_field
    type, bind(C) :: zfp_field
    !! ZFP field type, representing an uncompressed array
        type(c_ptr) :: val
    end type zfp_field

public :: zfp_stream
    type, bind(C) :: zfp_stream
    !! ZFP stream type, representing a compressed stream
        type(c_ptr) :: val
    end type zfp_stream

public :: bitstream
    type, bind(C) :: bitstream
    !! Bitstream type for compressed data
        type(c_ptr) :: val
    end type bitstream

public :: zfp_type
    type, bind(C) :: zfp_type
    !! Scalar type enumeration
        integer(c_int)  :: val
    end type zfp_type

    type(zfp_type), parameter, public :: zfp_type_float = zfp_type(3)
    !! Single precision floating point type

    type(zfp_type), parameter, public :: zfp_type_double = zfp_type(4)
    !! Double precision floating point type

public :: zfp_exec_policy
    type, bind(C) :: zfp_exec_policy
    !! Execution policy enumeration
        integer(c_int)  :: val
    end type zfp_exec_policy

    type(zfp_exec_policy), parameter, public :: zfp_exec_serial = zfp_exec_policy(0)
    !! Serial execution policy

    type(zfp_exec_policy), parameter, public :: zfp_exec_cuda = zfp_exec_policy(2)
    !! CUDA parallel execution policy

    interface
        function zfp_field_2d(uncompressed_ptr, scalar_type, nx, ny) result(field) bind(c, name="zfp_field_2d")
        !! Allocate metadata for 2D field f[ny][nx]
        import
            type(c_ptr),        value   :: uncompressed_ptr !! pointer to uncompressed scalars (may be NULL)
            type(zfp_type),     value   :: scalar_type      !! scalar type
            integer(c_size_t),  value   :: nx               !! number of scalars in x dimension
            integer(c_size_t),  value   :: ny               !! number of scalars in y dimension
            type(zfp_field)             :: field            !! allocated field metadata
        end function zfp_field_2d
    end interface

    interface
        function zfp_field_3d(uncompressed_ptr, scalar_type, nx, ny, nz) result(field) bind(c, name="zfp_field_3d")
        !! Allocate metadata for 3D field f[nz][ny][nx]
        import
            type(c_ptr),        value   :: uncompressed_ptr !! pointer to uncompressed scalars (may be NULL)
            type(zfp_type),     value   :: scalar_type      !! scalar type
            integer(c_size_t),  value   :: nx               !! number of scalars in x dimension
            integer(c_size_t),  value   :: ny               !! number of scalars in y dimension
            integer(c_size_t),  value   :: nz               !! number of scalars in z dimension
            type(zfp_field)             :: field            !! allocated field metadata
        end function zfp_field_3d
    end interface

    interface
        subroutine zfp_field_set_stride_2d(field, sx, sy) bind(c, name="zfp_field_set_stride_2d")
        !! Set 2D field strides in number of scalars
        import
            type(zfp_field),        value :: field !! field metadata
            integer(c_ptrdiff_t),   value :: sx    !! stride in x dimension: &f[0][1] - &f[0][0]
            integer(c_ptrdiff_t),   value :: sy    !! stride in y dimension: &f[1][0] - &f[0][0]
        end subroutine zfp_field_set_stride_2d
    end interface

    interface
        subroutine zfp_field_set_stride_3d(field, sx, sy, sz) bind(c, name="zfp_field_set_stride_3d")
        !! Set 3D field strides in number of scalars
        import
            type(zfp_field),        value :: field !! field metadata
            integer(c_ptrdiff_t),   value :: sx    !! stride in x dimension: &f[0][0][1] - &f[0][0][0]
            integer(c_ptrdiff_t),   value :: sy    !! stride in y dimension: &f[0][1][0] - &f[0][0][0]
            integer(c_ptrdiff_t),   value :: sz    !! stride in z dimension: &f[1][0][0] - &f[0][0][0]
        end subroutine zfp_field_set_stride_3d
    end interface

public :: zfp_stream_open
    interface
        function zfp_stream_open(bs) result(stream) bind(c, name="zfp_stream_open")
        !! Open compressed stream and associate with bit stream
        import
            type(bitstream),    value   :: bs     !! bit stream to read from and write to (may be NULL)
            type(zfp_stream)            :: stream !! allocated compressed stream
        end function zfp_stream_open
    end interface

public :: zfp_stream_close
    interface
        subroutine zfp_stream_close(stream) bind(c, name="zfp_stream_close")
        !! Close and deallocate compressed stream (does not affect bit stream)
        import
            type(zfp_stream),   value   :: stream !! compressed stream
        end subroutine zfp_stream_close
    end interface

public :: zfp_stream_maximum_size
    interface
        function zfp_stream_maximum_size(stream, field) result(max_size) bind(c, name="zfp_stream_maximum_size")
        !! Conservative estimate of compressed size in bytes
        import
            type(zfp_stream), value     :: stream   !! compressed stream
            type(zfp_field),  value     :: field    !! array to compress
            integer(c_size_t)           :: max_size !! maximum number of bytes of compressed storage
        end function zfp_stream_maximum_size
    end interface

public :: zfp_field_set_pointer
    interface
        subroutine zfp_field_set_pointer(field, arr_ptr) bind(c, name="zfp_field_set_pointer")
        !! Set pointer to first scalar in field
        import
            type(zfp_field),    value   :: field   !! field metadata
            type(c_ptr),        value   :: arr_ptr !! pointer to first scalar
        end subroutine zfp_field_set_pointer
    end interface

! public :: zfp_field_dimensionality
    interface
        function zfp_field_dimensionality(field) result(dims) bind(c, name="zfp_field_dimensionality")
        !! Field dimensionality (1, 2, 3, or 4)
        import
            type(zfp_field),    value   :: field !! field metadata
            integer(c_int)              :: dims  !! number of dimensions
        end function zfp_field_dimensionality
    end interface

    interface
        function zfp_field_type(field) result(scalar_type) bind(c, name="zfp_field_type")
        !! Field scalar type
        import
            type(zfp_field),    value   :: field       !! field metadata
            type(zfp_type)              :: scalar_type !! scalar type
        end function zfp_field_type
    end interface

public :: zfp_field_free
    interface
        subroutine zfp_field_free(field) bind(c, name="zfp_field_free")
        !! Deallocate field metadata
        import
            type(zfp_field),    value   :: field !! field metadata
        end subroutine zfp_field_free
    end interface

public :: zfp_stream_rewind
    interface
        subroutine zfp_stream_rewind(stream) bind(c, name="zfp_stream_rewind")
        !! Rewind bit stream to beginning for compression or decompression
        import
            type(zfp_stream), value     :: stream !! compressed bit stream
        end subroutine zfp_stream_rewind
    end interface

public :: stream_open
    interface
        function stream_open(buffer, bytes) result(bs) bind(c, name="stream_open")
        !! Open bit stream for reading and writing
        import
            type(c_ptr),        value   :: buffer !! pointer to buffer
            integer(c_size_t),  value   :: bytes  !! buffer size in bytes
            type(bitstream)             :: bs     !! bit stream
        end function stream_open
    end interface

public :: stream_close
    interface
        subroutine stream_close(bs) bind(c, name="stream_close")
        !! Close bit stream
        import
            type(bitstream),    value   :: bs !! bit stream
        end subroutine stream_close
    end interface

public :: zfp_stream_set_bit_stream
    interface
        subroutine zfp_stream_set_bit_stream(stream, bs) bind(c, name="zfp_stream_set_bit_stream")
        !! Associate bit stream with compressed stream
        import
            type(zfp_stream),   value :: stream !! compressed stream
            type(bitstream),    value :: bs     !! bit stream to read from and write to
        end subroutine zfp_stream_set_bit_stream
    end interface

public :: zfp_stream_set_reversible
    interface
        subroutine zfp_stream_set_reversible(stream) bind(c, name="zfp_stream_set_reversible")
        !! Enable reversible (lossless) compression
        import
            type(zfp_stream),   value :: stream !! compressed stream
        end subroutine zfp_stream_set_reversible
    end interface

    interface
        function zfp_stream_set_rate_interface(stream, rate, scalar_type, dims, align) result(rate_result) bind(c, name="zfp_stream_set_rate")
        !! Set size in compressed bits/scalar (fixed-rate mode)
        import
            type(zfp_stream),   value   :: stream      !! compressed stream
            real(c_double),     value   :: rate        !! desired rate in compressed bits/scalar
            type(zfp_type),     value   :: scalar_type !! scalar type to compress
            integer(c_int),     value   :: dims        !! array dimensionality (1, 2, 3, or 4)
            integer(c_int),     value   :: align       !! word-aligned blocks, e.g., for write random access
            real(c_double)              :: rate_result !! actual rate in compressed bits/scalar
        end function zfp_stream_set_rate_interface
    end interface

    interface
        function zfp_stream_set_precision_interface(stream, prec) result(prec_result) bind(c, name="zfp_stream_set_precision")
        !! Set precision in uncompressed bits/scalar (fixed-precision mode)
        import
            type(zfp_stream),   value   :: stream       !! compressed stream
            integer(c_int),     value   :: prec         !! desired precision in uncompressed bits/scalar
            integer(c_int)              :: prec_result  !! actual precision
        end function zfp_stream_set_precision_interface
    end interface

    interface
        function zfp_stream_set_accuracy_interface(stream, acc) result(acc_result) bind(c, name="zfp_stream_set_accuracy")
        !! Set accuracy as absolute error tolerance (fixed-accuracy mode)
        import
            type(zfp_stream),   value   :: stream     !! compressed stream
            real(c_double),     value   :: acc        !! desired error tolerance
            real(c_double)              :: acc_result !! actual error tolerance
        end function zfp_stream_set_accuracy_interface
    end interface

public :: zfp_stream_set_execution
    interface
        function zfp_stream_set_execution(stream, execution_policy) result(is_success) bind(c, name="zfp_stream_set_execution")
        !! Set execution policy
        import
            type(zfp_stream),       value   :: stream           !! compressed stream
            type(zfp_exec_policy),  value   :: execution_policy !! execution policy
            integer(c_int)                  :: is_success       !! true upon success
        end function zfp_stream_set_execution
    end interface

public :: zfp_decompress
    interface
        function zfp_decompress(stream, field) result(bitstream_offset_bytes) bind(c, name="zfp_decompress")
        !! Decompress entire field (nonzero return value upon success)
        import
            type(zfp_stream),   value   :: stream                  !! compressed stream
            type(zfp_field),    value   :: field                   !! field metadata
            integer(c_size_t)           :: bitstream_offset_bytes  !! cumulative number of bytes of compressed storage
        end function zfp_decompress
    end interface

public :: zfp_compress
    interface
        function zfp_compress(stream, field) result(bitstream_offset_bytes) bind(c, name="zfp_compress")
        !! Compress entire field (nonzero return value upon success)
        import
            type(zfp_stream),   value   :: stream                  !! compressed stream
            type(zfp_field),    value   :: field                   !! field metadata
            integer(c_size_t)           :: bitstream_offset_bytes  !! cumulative number of bytes of compressed storage
        end function zfp_compress
    end interface

contains

    function zfp_create_field(uncompressed_ptr, scalar_type, dims) result(field)
    !! Create field for 2D or 3D arrays
        type(c_ptr),    intent(in)  :: uncompressed_ptr !! pointer to uncompressed scalars
        type(zfp_type), intent(in)  :: scalar_type      !! scalar type
        integer(int32), intent(in)  :: dims(:)          !! array dimensions
        type(zfp_field)             :: field            !! field metadata

        if ( size(dims) == 2 ) then
            field = zfp_field_2d(uncompressed_ptr, scalar_type, int(dims(1), c_size_t), int(dims(2), c_size_t))
        else
            field = zfp_field_3d(uncompressed_ptr, scalar_type, int(dims(1), c_size_t), int(dims(2), c_size_t), int(dims(3), c_size_t))
        endif
    end function zfp_create_field

    subroutine zfp_field_set_stride(field, strides)
    !! Set field strides for 2D or 3D arrays
        type(zfp_field),    intent(in)  :: field   !! field metadata
        integer(int32),     intent(in)  :: strides(:) !! strides per dimension
        integer(c_int) :: ndims

        ndims = zfp_field_dimensionality(field)
#ifdef DTFFT_DEBUG
        if ( ndims /= size(strides) ) then
            INTERNAL_ERROR("zfp_field_set_stride: ndims /= size(strides)")
        endif
#endif

        if ( ndims == 2 ) then
            call zfp_field_set_stride_2d(field, int(strides(1), c_ptrdiff_t), int(strides(2), c_ptrdiff_t))
        else
            call zfp_field_set_stride_3d(field, int(strides(1), c_ptrdiff_t), int(strides(2), c_ptrdiff_t), int(strides(3), c_ptrdiff_t))
        endif
    end subroutine zfp_field_set_stride

    subroutine zfp_stream_set_rate(stream, field, rate, is_complex)
    !! Set compression rate with alignment for complex data
        type(zfp_stream),   intent(in)  :: stream    !! compressed stream
        type(zfp_field),    intent(in)  :: field     !! field metadata
        real(c_double),     intent(in)  :: rate      !! desired rate in compressed bits/scalar
        logical,            intent(in)  :: is_complex !! whether data is complex
        real(c_double) :: dummy
        integer(c_int) :: align

        align = 1; if( is_complex ) align = 0
        dummy = zfp_stream_set_rate_interface(stream, rate, zfp_field_type(field), zfp_field_dimensionality(field), align)
    end subroutine zfp_stream_set_rate

    subroutine zfp_stream_set_precision(stream, prec)
    !! Set precision in uncompressed bits/scalar
        type(zfp_stream),   intent(in)  :: stream !! compressed stream
        integer(c_int),     intent(in)  :: prec   !! desired precision
        integer(c_int) :: dummy

        dummy = zfp_stream_set_precision_interface(stream, prec)
    end subroutine zfp_stream_set_precision

    subroutine zfp_stream_set_accuracy(stream, acc)
    !! Set accuracy as absolute error tolerance
        type(zfp_stream),   intent(in)  :: stream !! compressed stream
        real(c_double),     intent(in)  :: acc    !! desired error tolerance
        real(c_double) :: dummy

        dummy = zfp_stream_set_accuracy_interface(stream, acc)
    end subroutine zfp_stream_set_accuracy

end module dtfft_interface_zfp