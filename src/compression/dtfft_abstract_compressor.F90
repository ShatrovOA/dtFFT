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
module dtfft_abstract_compressor
!! Abstract base class for compression implementations [[abstract_compressor]]
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_errors
use dtfft_parameters
use dtfft_serializer
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: abstract_compressor
public :: is_valid_compression_mode


public :: dtfft_compression_mode_t
    type, bind(C) :: dtfft_compression_mode_t
    !! Type that specifies compression mode
        integer(c_int32_t) :: val !! Internal value
    end type dtfft_compression_mode_t

    type(dtfft_compression_mode_t), parameter,  public :: DTFFT_COMPRESSION_MODE_LOSSLESS = dtfft_compression_mode_t(CONF_DTFFT_COMPRESSION_MODE_LOSSLESS)
    !! Lossless compression mode

    type(dtfft_compression_mode_t), parameter,  public :: DTFFT_COMPRESSION_MODE_FIXED_RATE = dtfft_compression_mode_t(CONF_DTFFT_COMPRESSION_MODE_FIXED_RATE)
    !! Fixed rate compression mode

    type(dtfft_compression_mode_t), parameter,  public :: DTFFT_COMPRESSION_MODE_FIXED_PRECISION = dtfft_compression_mode_t(CONF_DTFFT_COMPRESSION_MODE_FIXED_PRECISION)
    !! Fixed precision compression mode

    type(dtfft_compression_mode_t), parameter,  public :: DTFFT_COMPRESSION_MODE_FIXED_ACCURACY = dtfft_compression_mode_t(CONF_DTFFT_COMPRESSION_MODE_FIXED_ACCURACY)
    !! Fixed accuracy compression mode

public :: dtfft_compression_lib_t
    type, bind(C) :: dtfft_compression_lib_t
    !! Type that specifies compression library
        integer(c_int32_t) :: val !! Internal value
    end type dtfft_compression_lib_t

    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_NONE = dtfft_compression_lib_t(VARIABLE_NOT_SET)

    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_ZFP = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_ZFP)
    !! ZFP compression library
    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_SZ3 = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_SZ3)
    !! SZ3 compression library
    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_SZP = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_SZP)
    !! SZp compression library
    ! type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_CUSZ = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_CUSZ)
    ! !! cuSZ GPU compression library
    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_CUSZP = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_CUSZP)
    !! cuSZp GPU compression library
    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_BITCOMP = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_BITCOMP)

    type(dtfft_compression_lib_t),  parameter,  public :: VALID_COMPRESSION_LIBS(*) = [ DTFFT_COMPRESSION_LIB_NONE      &
#ifdef DTFFT_WITH_ZFP
                                                                                        , DTFFT_COMPRESSION_LIB_ZFP     &
#endif
#ifdef DTFFT_WITH_SZ3
                                                                                        , DTFFT_COMPRESSION_LIB_SZ3     &
#endif
#ifdef DTFFT_WITH_SZP
                                                                                        , DTFFT_COMPRESSION_LIB_SZP     &
#endif
! #ifdef DTFFT_WITH_CUSZ
!                                                                                         , DTFFT_COMPRESSION_LIB_CUSZ    &
! #endif
#ifdef DTFFT_WITH_CUSZP
                                                                                        , DTFFT_COMPRESSION_LIB_CUSZP   &
#endif
#ifdef DTFFT_WITH_BITCOMP
                                                                                        , DTFFT_COMPRESSION_LIB_BITCOMP &
#endif
                                                                                        ]
    !! List of valid compression libraries

    type(dtfft_compression_mode_t), parameter,  public :: VALID_COMPRESSION_MODES(*) = [    DTFFT_COMPRESSION_MODE_LOSSLESS,        &
                                                                                            DTFFT_COMPRESSION_MODE_FIXED_RATE,      &
                                                                                            DTFFT_COMPRESSION_MODE_FIXED_PRECISION, &
                                                                                            DTFFT_COMPRESSION_MODE_FIXED_ACCURACY ]
    !! List of valid compression modes


    type, abstract :: abstract_compressor
    !! Abstract base class for compression implementations
        logical         :: is_complex        !! Indicates if the data type is complex
        integer(int8)   :: ndims             !! Number of dimensions
        integer(int64)  :: storage_size      !! Size of storage per element
        integer(int64)  :: true_storage_size !! Size of storage per element
        real(real64)    :: compressed_rate   !! Accumulated compression rate
        integer(int64)  :: execution_count   !! Number of compression executions
        integer(int32)  :: dims_permutation  !! Dimension permutation flag
        type(dtfft_compression_mode_t) :: compression_mode !! Compression mode
        real(real64) :: rate_or_tolerance !! Compression rate (for fixed rate mode) or tolerance (for fixed accuracy mode or fixed precision mode)
        integer(int32) :: precision !! Precision (for fixed precision mode)
    contains
        procedure,  non_overridable,    public,                 pass(self)  :: create             !! Initializes the compressor with given parameters
        procedure,  non_overridable,    public,                 pass(self)  :: compress           !! Compresses the input data and returns the compressed size
        procedure,  non_overridable,    public,                 pass(self)  :: decompress         !! Decompresses the input data into the output buffer
        procedure,  non_overridable,    public,                 pass(self)  :: get_average_rate   !! Returns the average compression rate over all executions
        procedure,  non_overridable,    public,                 pass(self)  :: check_tolerance
        procedure,  non_overridable,    public,                 pass(self)  :: get_n_elements
        procedure(sync_interface),                  deferred,   pass(self)  :: pre_sync           !!
        procedure(sync_interface),                  deferred,   pass(self)  :: post_sync
        procedure(sync_interface),                  deferred,   pass(self)  :: sync
        procedure(create_private_interface),        deferred,   pass(self)  :: create_private    !! Creates the compressor implementation
        procedure(compress_private_interface),      deferred,   pass(self)  :: compress_private  !! Performs compression using the implementation
        procedure(decompress_private_interface),    deferred,   pass(self)  :: decompress_private !! Performs decompression using the implementation
        procedure(destroy_interface),               deferred,   pass(self)  :: destroy           !! Destroys the compressor implementation
        procedure(get_supported_modes_interface),   deferred,   nopass      :: get_supported_modes
    end type abstract_compressor

    abstract interface
        integer(int32) function create_private_interface(self, platform)
        !! Creates the compressor implementation
        import
            class(abstract_compressor),         intent(inout)   :: self         !! Compressor instance
            type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform
        end function create_private_interface

        integer(int64) function compress_private_interface(self, dims, in, out, stream)
        !! Performs compression using the implementation
        import
            class(abstract_compressor), intent(inout)   :: self     !! Compressor instance
            integer(int32),             intent(in)      :: dims(:)  !! Array dimensions
            type(c_ptr),                intent(in)      :: in       !! Pointer to uncompressed data
            type(c_ptr),                intent(in)      :: out      !! Pointer to compressed buffer
            type(dtfft_stream_t),       intent(in)      :: stream   !! Stream handle
        end function compress_private_interface

        subroutine decompress_private_interface(self, dims, in, out, stream, compressed_size)
        !! Performs decompression using the implementation
        import
            class(abstract_compressor), intent(inout)   :: self             !! Compressor instance
            integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
            type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
            type(c_ptr),                intent(in)      :: out              !! Pointer to uncompressed buffer
            type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
            integer(int64),             intent(in)      :: compressed_size
        end subroutine decompress_private_interface

        subroutine destroy_interface(self)
        !! Destroys the compressor implementation
        import
            class(abstract_compressor), intent(inout)   :: self !! Compressor instance
        end subroutine destroy_interface

        subroutine sync_interface(self, stream)
        import
            class(abstract_compressor), intent(inout)   :: self                 !! Compressor instance
            type(dtfft_stream_t),       intent(in)      :: stream
        end subroutine sync_interface

        pure subroutine get_supported_modes_interface(host_modes, cuda_modes, use_tolerance)
        !! Returns supported compression modes
        !! Arrays are of size 4, corresponding to the order of `VALID_COMPRESSION_MODES`:
        !! 1. DTFFT_COMPRESSION_MODE_LOSSLESS
        !! 2. DTFFT_COMPRESSION_MODE_FIXED_RATE
        !! 3. DTFFT_COMPRESSION_MODE_FIXED_PRECISION
        !! 4. DTFFT_COMPRESSION_MODE_FIXED_ACCURACY
        import
            logical,    intent(out) :: host_modes(4)        !! Supported host compression modes
            logical,    intent(out) :: cuda_modes(4)        !! Supported cuda compression modes
            logical,    intent(out) :: use_tolerance        !! Should `tolerance` parameter be used in `DTFFT_COMPRESSION_MODE_FIXED_PRECISION`
                                                            !! mode instead of `precision`
        end subroutine get_supported_modes_interface
    end interface

public :: operator(==)
    interface operator(==)
        module procedure compression_mode_eq    !! Check if two `dtfft_compression_mode_t` are equal
    end interface

public :: operator(/=)
    interface operator(/=)
        module procedure compression_mode_ne    !! Check if two `dtfft_compression_mode_t` are not equal
    end interface

contains

    integer(int32) function create(self, ndims, compression_mode, rate_or_tolerance, precision,  platform, base_type, dims_permutation)
    !! Initializes the compressor with given parameters
        class(abstract_compressor),         intent(inout)   :: self             !! Compressor instance
        integer(int8),                      intent(in)      :: ndims            !! Number of dimensions
        type(dtfft_compression_mode_t),     intent(in)      :: compression_mode !! Compression mode
        real(real64),                       intent(in)      :: rate_or_tolerance!! Compression rate (for fixed rate mode) or tolerance (for fixed accuracy mode or fixed precision mode)
        integer(int32),                     intent(in)      :: precision        !! Precision (for fixed precision
        type(dtfft_platform_t),             intent(in)      :: platform         !! Target platform
        TYPE_MPI_DATATYPE,                  intent(in)      :: base_type        !! MPI data type
        integer(int32),                     intent(in)      :: dims_permutation !! Dimension permutation flag
        logical :: host_modes(4), cuda_modes(4), use_tolerance

        create = DTFFT_SUCCESS

        self%ndims = ndims
        self%compressed_rate = 0._real64
        self%execution_count = 0
        self%compression_mode = compression_mode
        self%rate_or_tolerance = rate_or_tolerance
        self%precision = precision
        self%dims_permutation = dims_permutation
#ifdef DTFFT_DEBUG
        if ( .not. any(GET_MPI_VALUE(base_type) == [GET_MPI_VALUE(MPI_COMPLEX), GET_MPI_VALUE(MPI_DOUBLE_COMPLEX), GET_MPI_VALUE(MPI_REAL), GET_MPI_VALUE(MPI_REAL8)]) ) then
            INTERNAL_ERROR("abstract_compressor.create: invalid base_type")
        endif
#endif
        self%is_complex = any(GET_MPI_VALUE(base_type) == [GET_MPI_VALUE(MPI_COMPLEX), GET_MPI_VALUE(MPI_DOUBLE_COMPLEX)])
        ! self%is_complex = GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_COMPLEX) .or. GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_DOUBLE_COMPLEX)
        select case ( GET_MPI_VALUE(base_type) )
        case ( GET_MPI_VALUE(MPI_COMPLEX), GET_MPI_VALUE(MPI_REAL) )
            self%storage_size = FLOAT_STORAGE_SIZE
        case default
            self%storage_size = DOUBLE_STORAGE_SIZE
        endselect
        self%true_storage_size = self%storage_size
        if ( self%is_complex ) self%true_storage_size = self%true_storage_size * 2

        call self%get_supported_modes(host_modes, cuda_modes, use_tolerance)

        if ( platform == DTFFT_PLATFORM_CUDA .and. .not.any(cuda_modes) ) then
            create = DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED
        endif
        if ( platform == DTFFT_PLATFORM_HOST .and. .not.any(host_modes) ) then
            create = DTFFT_ERROR_COMPRESSION_HOST_NOT_SUPPORTED
        endif
        if ( create /= DTFFT_SUCCESS ) return

        if ( platform == DTFFT_PLATFORM_HOST .and. .not. host_modes(compression_mode%val)        &
        .or. platform == DTFFT_PLATFORM_CUDA .and. .not. cuda_modes(compression_mode%val) ) then
            create = DTFFT_ERROR_COMPRESSION_INVALID_MODE
            return
        endif
        if ( (compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. use_tolerance)    &
         .or. compression_mode == DTFFT_COMPRESSION_MODE_FIXED_ACCURACY ) then
            CHECK_CALL( self%check_tolerance(rate_or_tolerance), create )
        endif
        if ( self%storage_size == FLOAT_STORAGE_SIZE ) then
            if ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. rate_or_tolerance >= 32.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
            elseif ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. .not. use_tolerance .and. precision >= 32 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
            endif
        else
            if ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. rate_or_tolerance >= 64.0_real64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
            elseif ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. .not. use_tolerance .and. precision >= 64 ) then
                create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
            endif
        endif
        if ( create /= DTFFT_SUCCESS ) return

        if ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. rate_or_tolerance <= 1.0_real64 ) then
            create = DTFFT_ERROR_COMPRESSION_INVALID_RATE
        else if ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. .not. use_tolerance .and. precision <= 1 ) then
            create = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
        else if ( compression_mode == DTFFT_COMPRESSION_MODE_FIXED_ACCURACY .and. rate_or_tolerance >= 1.0_real64 ) then
            create = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
        endif
        if ( create /= DTFFT_SUCCESS ) return

        CHECK_CALL( self%create_private(platform), create )
    end function create

    integer(int64) function compress(self, dims, displ, in, out, stream)
    !! Compresses the input data and returns the compressed size
        class(abstract_compressor), intent(inout)   :: self     !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)  !! Array dimensions
        integer(int32),             intent(in)      :: displ    !! Displacement in arrays
        type(c_ptr),                intent(in)      :: in       !! Pointer to uncompressed data
        type(c_ptr),                intent(in)      :: out      !! Pointer to compressed buffer
        type(dtfft_stream_t),       intent(in)      :: stream   !! Stream handle
        type(c_ptr) :: inptr, outptr
        real(real64) :: compressed_rate
        integer(int64)  :: uncompressed_size

        if ( product(dims(1:self%ndims)) == 0 ) then
            compress = 0
            return
        endif

        REGION_BEGIN("compress", COLOR_SALMON)

        inptr = ptr_offset(in, self%true_storage_size * displ)
        outptr = ptr_offset(out, self%true_storage_size * displ)
#ifdef DTFFT_DEBUG
        if ( is_same_ptr(inptr, outptr) ) then
            INTERNAL_ERROR("compress: is_same_ptr(inptr, outptr)")
        endif
#endif

        compress = self%compress_private(dims(1:self%ndims), inptr, outptr, stream)

        uncompressed_size = self%true_storage_size * product(dims(1:self%ndims))
        if ( compress >= uncompressed_size ) then
            INTERNAL_ERROR("abstract_compressor.compress: compress >= uncompressed_size: "//to_str(compress)//" >= "//to_str(uncompressed_size))
        endif
        if ( compress == 0 ) then
            INTERNAL_ERROR("Compression failed")
        endif

        ! Aggregate compression stats
        compressed_rate = real(uncompressed_size, real64) / real(compress, real64)
        self%compressed_rate = self%compressed_rate + compressed_rate
        self%execution_count = self%execution_count + 1
        REGION_END("compress")
    end function compress

    subroutine decompress(self, dims, displ, in, out, stream, compressed_size)
    !! Decompresses the input data into the output buffer
        class(abstract_compressor), intent(inout)   :: self     !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)  !! Array dimensions
        integer(int32),             intent(in)      :: displ    !! Displacement in arrays
        type(c_ptr),                intent(in)      :: in       !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out      !! Pointer to uncompressed buffer
        type(dtfft_stream_t),       intent(in)      :: stream   !! Stream handle
        integer(int64),             intent(in)      :: compressed_size
        type(c_ptr) :: inptr, outptr
        integer(int32) :: permuted_dims(3)

        if ( product(dims(1:self%ndims)) == 0 ) then
            return
        endif
        REGION_BEGIN("decompress", COLOR_CORAL)

        inptr = ptr_offset(in, self%true_storage_size * displ)
        outptr = ptr_offset(out, self%true_storage_size * displ)
#ifdef DTFFT_DEBUG
        if ( is_same_ptr(inptr, outptr) ) then
            INTERNAL_ERROR("decompress: is_same_ptr(inptr, outptr)")
        endif
#endif
        permuted_dims(1:self%ndims) = dims(1:self%ndims)
        if ( self%dims_permutation == DIMS_PERMUTE_FORWARD ) then
            permuted_dims(1) = dims(2)
            if ( self%ndims == 2 ) then
                permuted_dims(2) = dims(1)
            else
                permuted_dims(2) = dims(3)
                permuted_dims(3) = dims(1)
            endif
        else if ( self%dims_permutation == DIMS_PERMUTE_BACKWARD) then
            if ( self%ndims == 2 ) then
                permuted_dims(1) = dims(2)
                permuted_dims(2) = dims(1)
            else
                permuted_dims(1) = dims(3)
                permuted_dims(2) = dims(1)
                permuted_dims(3) = dims(2)
            endif
        endif
        call self%decompress_private(permuted_dims(1:self%ndims), inptr, outptr, stream, compressed_size)
    REGION_END("decompress")
    end subroutine decompress

    function check_tolerance(self, tolerance) result(error_code)
    !! Returns the average compression rate over all executions
        class(abstract_compressor), intent(in)      :: self !! Compressor instance
        real(c_double),             intent(in)      :: tolerance
        integer(int32)                              :: error_code

        error_code = DTFFT_SUCCESS
        if ( (self%storage_size == FLOAT_STORAGE_SIZE .and. real(tolerance, real32) < epsilon(1.0_real32)) .or. &
            (self%storage_size == DOUBLE_STORAGE_SIZE .and. real(tolerance, real64) < epsilon(1.0_real64)) ) then
            error_code = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
        endif
    end function check_tolerance

    pure real(real64) function get_average_rate(self)
    !! Returns the average compression rate over all executions
        class(abstract_compressor), intent(in)      :: self !! Compressor instance
        get_average_rate = self%compressed_rate / real(self%execution_count, real64)
    end function get_average_rate

    pure function get_n_elements(self, dims) result(n_elements)
    !! Returns number of `real` elements
        class(abstract_compressor), intent(in)  :: self !! Compressor instance
        integer(int32),             intent(in)  :: dims(:)
        integer(int64)                          :: n_elements

        n_elements = 1_int64 * product(dims)
        if ( self%is_complex ) n_elements = n_elements * 2
    end function get_n_elements

MAKE_EQ_FUN(dtfft_compression_mode_t, compression_mode_eq)
MAKE_NE_FUN(dtfft_compression_mode_t, compression_mode_ne)

MAKE_VALID_FUN_DTYPE(dtfft_compression_mode_t, is_valid_compression_mode, VALID_COMPRESSION_MODES)
end module dtfft_abstract_compressor