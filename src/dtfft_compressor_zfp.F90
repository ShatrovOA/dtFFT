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
module dtfft_compressor_zfp
!! ZFP-based compressor implementation
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_compressor
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
use dtfft_interface_zfp
use dtfft_parameters
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: compressor_zfp

    type, extends(abstract_compressor) :: compressor_zfp
    !! ZFP-based compressor implementation
    private
        logical                         :: is_complex   !! Indicates if the data type is complex
        integer(int64)                  :: imag_offset  !! Byte offset for imaginary part in complex data
        type(dtfft_compression_config_t):: config       !! Compression configuration parameters
        type(zfp_exec_policy)           :: policy       !! ZFP execution policy (serial or CUDA)
        type(zfp_type)                  :: scalar_type  !! ZFP scalar type (float or double)
#ifdef DTFFT_WITH_CUDA
        type(cudaEvent)                 :: event
#endif
    contains
        procedure   :: create_private => create         !! Initializes the ZFP compressor with given configuration
        procedure   :: compress_private => compress     !! Compresses the input data using ZFP
        procedure   :: decompress_private => decompress !! Decompresses the input data using ZFP
        procedure   :: destroy                          !! Cleans up the compressor resources
        procedure   :: sync
        procedure   :: pre_sync
        procedure   :: post_sync
        procedure   :: init                             !! Initializes ZFP stream and field for compression/decompression
    end type compressor_zfp

contains

    integer(int32) function create(self, config, platform, base_type)
    !! Initializes the ZFP compressor with given configuration
        class(compressor_zfp),              intent(inout)   :: self         !! Compressor instance
        type(dtfft_compression_config_t),   intent(in)      :: config       !! Compression settings
        type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform (CPU or CUDA)
        TYPE_MPI_DATATYPE,                  intent(in)      :: base_type    !! MPI data type

        create = DTFFT_SUCCESS
        self%imag_offset = 0
        self%is_complex = GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_COMPLEX) .or. GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_DOUBLE_COMPLEX)
        self%config = config
        self%policy = zfp_exec_serial
        if ( platform == DTFFT_PLATFORM_CUDA ) then
#if !defined(ZFP_WITH_CUDA) && !defined(DTFFT_WITH_MOCK_ENABLED)
            create = DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED
            return
#endif
            self%policy = zfp_exec_cuda
        endif
        select case ( GET_MPI_VALUE(base_type) )
        case ( GET_MPI_VALUE(MPI_COMPLEX), GET_MPI_VALUE(MPI_REAL) )
            self%scalar_type = zfp_type_float
            if ( self%is_complex ) self%imag_offset = 4_int64
        case default
            self%scalar_type = zfp_type_double
            if ( self%is_complex ) self%imag_offset = 8_int64
        endselect

        if ( self%scalar_type%val == zfp_type_float%val ) then
            if ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. config%rate >= 32.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
                return
            elseif ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. config%precision >= 32 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
                return
            elseif ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_ACCURACY .and. config%tolerance < nearest(1._real32, 1._real32) - nearest(1._real32,-1._real32) ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
                return
            endif
        else
            if ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. config%rate >= 64.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
                return
            elseif ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. config%precision >= 64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
                return
            elseif ( config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_ACCURACY .and. config%tolerance < nearest(1._real64, 1._real64) - nearest(1._real64,-1._real64) ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
                return
            endif
        endif

        select case ( config%compression_mode%val )
        case ( DTFFT_COMPRESSION_MODE_FIXED_RATE%val )
            if ( config%rate <= 1.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
                return
            endif
        case ( DTFFT_COMPRESSION_MODE_FIXED_PRECISION%val )
            if ( config%precision <= 1 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
                return
            endif
        case ( DTFFT_COMPRESSION_MODE_FIXED_ACCURACY%val )
            if ( config%tolerance <= 0.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
                return
            endif
        endselect
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL( cudaEventCreateWithFlags(self%event, cudaEventDisableTiming) )
        endif
#endif
    end function create

    subroutine init(self, uncompressed_ptr, dims, zfp, field)
    !! Initializes ZFP stream and field for compression/decompression
        class(compressor_zfp),      intent(inout)   :: self             !! Compressor instance
        type(c_ptr),                intent(in)      :: uncompressed_ptr !! Pointer to uncompressed data
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        type(zfp_stream),           intent(out)     :: zfp              !! ZFP stream
        type(zfp_field),            intent(out)     :: field            !! ZFP field
        integer(c_int) :: exec
        integer(int32), allocatable :: strides(:)

        field = zfp_create_field(uncompressed_ptr, self%scalar_type, dims)
        if ( self%is_complex ) then
            allocate( strides(self%ndims) )

            strides(1) = 2
            strides(2) = strides(1) * dims(1)
            if ( self%ndims == 3 ) then
                strides(3) = strides(2) * dims(2)
            endif
            call zfp_field_set_stride(field, strides)

            deallocate( strides )
        endif

        zfp = zfp_stream_open(bitstream(c_null_ptr))
        exec = zfp_stream_set_execution(zfp, self%policy)
        select case ( self%config%compression_mode%val )
        case ( DTFFT_COMPRESSION_MODE_LOSSLESS%val )
            call zfp_stream_set_reversible(zfp)
        case ( DTFFT_COMPRESSION_MODE_FIXED_RATE%val )
            call zfp_stream_set_rate(zfp, field, self%config%rate, self%is_complex)
        case ( DTFFT_COMPRESSION_MODE_FIXED_PRECISION%val )
            call zfp_stream_set_precision(zfp, self%config%precision)
        case ( DTFFT_COMPRESSION_MODE_FIXED_ACCURACY%val )
            call zfp_stream_set_accuracy(zfp, self%config%tolerance)
        endselect
    end subroutine init

    integer(int64) function compress(self, dims, in, out, stream)
    !! Compresses the input data using ZFP
        class(compressor_zfp),      intent(inout)   :: self         !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)      !! Array dimensions
        type(c_ptr),                intent(in)      :: in           !! Pointer to input data
        type(c_ptr),                intent(in)      :: out          !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream handle
        type(zfp_field) :: field
        type(zfp_stream) :: zfp
        type(bitstream) :: bs
        integer(c_size_t) :: max_size

        compress = 0_int64
        call self%init(in, dims, zfp, field)
        max_size = zfp_stream_maximum_size(zfp, field)
        if ( self%is_complex ) max_size = max_size * 2
        bs = stream_open(out, max_size)
        call zfp_stream_set_bit_stream(zfp, bs)
        call zfp_stream_rewind(zfp)

        compress = zfp_compress(zfp, field)

        if ( self%is_complex ) then
            call zfp_field_set_pointer(field, ptr_offset(in, self%imag_offset))
            compress = zfp_compress(zfp, field)
        endif

        call zfp_field_free(field)
        call zfp_stream_close(zfp)
        call stream_close(bs)
    end function compress

    subroutine decompress(self, dims, in, out, stream)
    !! Decompresses the input data using ZFP
        class(compressor_zfp),      intent(inout)   :: self             !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out              !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        type(zfp_field) :: field
        type(zfp_stream) :: zfp
        type(bitstream) :: bs
        integer(int64) :: decompressed_size

        call self%init(out, dims, zfp, field)
        bs = stream_open(in, product(dims) * self%storage_size)
        call zfp_stream_set_bit_stream(zfp, bs)
        call zfp_stream_rewind(zfp)

        decompressed_size = zfp_decompress(zfp, field)
        if ( self%is_complex ) then
            call zfp_field_set_pointer(field, ptr_offset(out, self%imag_offset))
            decompressed_size = zfp_decompress(zfp, field)
        endif
        ! if ( decompressed_size /= compressed_size ) then
        !     INTERNAL_ERROR("compressor_zfp.decompress: decompressed_size /= compressed_size")
        ! endif
        call zfp_field_free(field)
        call zfp_stream_close(zfp)
        call stream_close(bs)
    end subroutine decompress

    subroutine destroy(self)
    !! Cleans up the compressor resources
        class(compressor_zfp),      intent(inout)   :: self !! Compressor instance

        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaEventDestroy(self%event) )
#endif
    end subroutine destroy

    subroutine pre_sync(self, stream)
        class(compressor_zfp),      intent(inout)  :: self                 !! Abstract kernel
        type(dtfft_stream_t),       intent(in)     :: stream

        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaEventRecord(self%event, stream) )
        ! Waiting for transpose kernel to finish execution on stream `stream`
        CUDA_CALL( cudaStreamWaitEvent(NULL_STREAM, self%event, 0) )
#endif
    end subroutine pre_sync

    subroutine post_sync(self, stream)
        class(compressor_zfp),      intent(inout)  :: self                 !! Abstract kernel
        type(dtfft_stream_t),       intent(in)  :: stream

        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaEventRecord(self%event, NULL_STREAM) )
        ! Waiting for compression kernel to finish execution on stream `stream`
        CUDA_CALL( cudaStreamWaitEvent(stream, self%event, 0) )
#endif
    end subroutine post_sync

    subroutine sync(self, stream)
        class(compressor_zfp),      intent(inout)  :: self                 !! Abstract kernel
        type(dtfft_stream_t),       intent(in)  :: stream

        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaStreamSynchronize(NULL_STREAM) )
#endif
    end subroutine sync
end module dtfft_compressor_zfp