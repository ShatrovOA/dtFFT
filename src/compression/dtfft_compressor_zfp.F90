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
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
use dtfft_interface_zfp
use dtfft_parameters
use dtfft_errors
use dtfft_config
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_mpi.h"
implicit none
private
public :: compressor_zfp

    type, extends(abstract_compressor) :: compressor_zfp
    !! ZFP-based compressor implementation
    private
        logical                         :: is_fixed_rate
        integer(int64)                  :: imag_offset  !! Byte offset for imaginary part in complex data
        ! type(dtfft_compression_config_t):: config       !! Compression configuration parameters
        type(zfp_exec_policy)           :: policy       !! ZFP execution policy (serial or CUDA)
        type(zfp_type)                  :: scalar_type  !! ZFP scalar type (float or double)
#ifdef DTFFT_WITH_CUDA
        type(cudaEvent)                 :: event
#endif
    contains
        procedure           :: create_private => create         !! Initializes the ZFP compressor with given configuration
        procedure           :: compress_private => compress     !! Compresses the input data using ZFP
        procedure           :: decompress_private => decompress !! Decompresses the input data using ZFP
        procedure           :: destroy                          !! Cleans up the compressor resources
        procedure           :: sync
        procedure           :: pre_sync
        procedure           :: post_sync
        procedure           :: init                             !! Initializes ZFP stream and field for compression/decompression
        procedure, nopass   :: get_supported_modes
    end type compressor_zfp

#ifdef DTFFT_WITH_OPENMP
    logical, save :: OMP_WARNING_ISSUED = .false.
#endif

contains

    integer(int32) function create(self, platform)
    !! Initializes the ZFP compressor with given configuration
        class(compressor_zfp),              intent(inout)   :: self         !! Compressor instance
        type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform (CPU or CUDA)

        create = DTFFT_SUCCESS
        ! self%config = config
#ifdef DTFFT_WITH_OPENMP
# ifdef ZFP_WITH_OPENMP
        self%policy = zfp_exec_omp
        ! self%policy = zfp_exec_serial
        if ( .not. OMP_WARNING_ISSUED ) then
            WRITE_WARN("ZFP Does not support OpenMP decompression. All decompressions will be performed in a serial manner.")
            OMP_WARNING_ISSUED = .true.
        endif
# else
        if ( .not. OMP_WARNING_ISSUED ) then
            WRITE_WARN("ZFP Does not support OpenMP execution policy. All compressions and decompressions will be performed in a serial manner.")
            OMP_WARNING_ISSUED = .true.
        endif
        self%policy = zfp_exec_serial
# endif
#else
        self%policy = zfp_exec_serial
#endif
        if ( platform == DTFFT_PLATFORM_CUDA ) then
#if !defined(ZFP_WITH_CUDA) && !defined(DTFFT_WITH_MOCK_ENABLED)
            create = DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED
            return
#endif
            ! ZFP CUDA execution policy supports fixed rate mode only, both for compression
            ! and decompression. `zfp_compress` and `zfp_decompress` simply return zero for
            ! any other mode, so such configuration is rejected here instead of failing later
            ! if ( config%compression_mode /= DTFFT_COMPRESSION_MODE_FIXED_RATE ) then
            !     create = DTFFT_ERROR_COMPRESSION_INVALID_MODE
            !     return
            ! endif
            self%policy = zfp_exec_cuda
        endif

        self%imag_offset = 0
        if ( self%is_complex ) self%imag_offset = self%storage_size
        if ( self%storage_size == FLOAT_STORAGE_SIZE ) then
            self%scalar_type = zfp_type_float
        else
            self%scalar_type = zfp_type_double
        endif

        self%is_fixed_rate = .false.
        if ( self%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE ) self%is_fixed_rate = .true.
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL( cudaEventCreateWithFlags(self%event, cudaEventDisableTiming) )
        endif
#endif
    end function create

    subroutine init(self, uncompressed_ptr, dims, is_decompression, zfp, field)
    !! Initializes ZFP stream and field for compression/decompression
        class(compressor_zfp),      intent(inout)   :: self             !! Compressor instance
        type(c_ptr),                intent(in)      :: uncompressed_ptr !! Pointer to uncompressed data
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        logical,                    intent(in)      :: is_decompression !! Decompression flag. Used only in omp build
        type(zfp_stream),           intent(out)     :: zfp              !! ZFP stream
        type(zfp_field),            intent(out)     :: field            !! ZFP field
        integer(c_int) :: exec
        integer(int32), allocatable :: strides(:), work_dims(:)

        allocate(work_dims, source=dims)
        if ( self%is_complex .and. self%is_fixed_rate ) then
            work_dims(1) = 2 * work_dims(1)
        endif

        field = zfp_create_field(uncompressed_ptr, self%scalar_type, work_dims)

        deallocate(work_dims)
        if ( self%is_complex .and. .not. self%is_fixed_rate) then
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
        if ( is_decompression .and. self%policy%val == zfp_exec_omp%val) then
            exec = zfp_stream_set_execution(zfp, zfp_exec_serial)
        else
            exec = zfp_stream_set_execution(zfp, self%policy)
        endif
        select case ( self%compression_mode%val )
        case ( DTFFT_COMPRESSION_MODE_LOSSLESS%val )
            call zfp_stream_set_reversible(zfp)
        case ( DTFFT_COMPRESSION_MODE_FIXED_RATE%val )
            call zfp_stream_set_rate(zfp, field, self%rate_or_tolerance)
        case ( DTFFT_COMPRESSION_MODE_FIXED_PRECISION%val )
            call zfp_stream_set_precision(zfp, self%precision)
        case ( DTFFT_COMPRESSION_MODE_FIXED_ACCURACY%val )
            call zfp_stream_set_accuracy(zfp, self%rate_or_tolerance)
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
        call self%init(in, dims, .false., zfp, field)
        max_size = zfp_stream_maximum_size(zfp, field)
        if ( self%is_complex .and. .not. self%is_fixed_rate ) max_size = max_size * 2
        bs = stream_open(out, max_size)
        call zfp_stream_set_bit_stream(zfp, bs)
        call zfp_stream_rewind(zfp)

        compress = zfp_compress(zfp, field)

        if ( self%is_complex .and. .not. self%is_fixed_rate) then
            call zfp_field_set_pointer(field, ptr_offset(in, self%imag_offset))
            compress = zfp_compress(zfp, field)
        endif
        call zfp_field_free(field)
        call zfp_stream_close(zfp)
        call stream_close(bs)
    end function compress

    subroutine decompress(self, dims, in, out, stream, compressed_size)
    !! Decompresses the input data using ZFP
        class(compressor_zfp),      intent(inout)   :: self             !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
        type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out              !! Pointer to output buffer
        type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        integer(int64),             intent(in)      :: compressed_size
        type(zfp_field) :: field
        type(zfp_stream) :: zfp
        type(bitstream) :: bs
        integer(int64) :: decompressed_size

        call self%init(out, dims, .true., zfp, field)
        bs = stream_open(in, compressed_size)
        call zfp_stream_set_bit_stream(zfp, bs)
        call zfp_stream_rewind(zfp)

        decompressed_size = zfp_decompress(zfp, field)
        if ( self%is_complex .and. .not. self%is_fixed_rate) then
            call zfp_field_set_pointer(field, ptr_offset(out, self%imag_offset))
            decompressed_size = zfp_decompress(zfp, field)
        endif
#ifdef DTFFT_DEBUG
        if ( decompressed_size == 0 ) then
            INTERNAL_ERROR("Decompression failed")
        endif
#endif
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
        class(compressor_zfp),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)     :: stream
        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaEventRecord(self%event, stream) )
        ! Waiting for transpose kernel to finish execution on stream `stream`
        CUDA_CALL( cudaStreamWaitEvent(NULL_STREAM, self%event, 0) )
#endif
    end subroutine pre_sync

    subroutine post_sync(self, stream)
        class(compressor_zfp),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream

        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaEventRecord(self%event, NULL_STREAM) )
        ! Waiting for compression kernel to finish execution on stream `stream`
        CUDA_CALL( cudaStreamWaitEvent(stream, self%event, 0) )
#endif
    end subroutine post_sync

    subroutine sync(self, stream)
        class(compressor_zfp),      intent(inout)  :: self                 !! Compressor instance
        type(dtfft_stream_t),       intent(in)  :: stream
        if ( self%policy%val /= zfp_exec_cuda%val ) return
#ifdef DTFFT_WITH_CUDA
        CUDA_CALL( cudaStreamSynchronize(NULL_STREAM) )
#endif
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
        logical,    intent(out) :: use_tolerance        !! .false.

        host_modes = [.true., .true., .true., .true.]
        cuda_modes = [.false., .true., .false., .false.]
        use_tolerance = .false.
    end subroutine get_supported_modes
end module dtfft_compressor_zfp