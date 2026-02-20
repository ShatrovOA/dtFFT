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
use iso_c_binding
use iso_fortran_env
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: abstract_compressor
public :: check_compression_config


    type, abstract :: abstract_compressor
    !! Abstract base class for compression implementations
        integer(int8)   :: ndims             !! Number of dimensions
        integer(int64)  :: storage_size      !! Size of storage per element
        real(real64)    :: compressed_rate   !! Accumulated compression rate
        integer(int64)  :: execution_count   !! Number of compression executions
        integer(int32)  :: dims_permutation  !! Dimension permutation flag
    contains
        procedure,  non_overridable,    public,                 pass(self)  :: create             !! Initializes the compressor with given parameters
        procedure,  non_overridable,    public,                 pass(self)  :: compress           !! Compresses the input data and returns the compressed size
        procedure,  non_overridable,    public,                 pass(self)  :: decompress         !! Decompresses the input data into the output buffer
        procedure,  non_overridable,    public,                 pass(self)  :: get_average_rate   !! Returns the average compression rate over all executions
        procedure(sync_interface),                  deferred,   pass(self)  :: pre_sync           !!
        procedure(sync_interface),                  deferred,   pass(self)  :: post_sync
        procedure(sync_interface),                  deferred,   pass(self)  :: sync
        procedure(create_private_interface),        deferred,   pass(self)  :: create_private    !! Creates the compressor implementation
        procedure(compress_private_interface),      deferred,   pass(self)  :: compress_private  !! Performs compression using the implementation
        procedure(decompress_private_interface),    deferred,   pass(self)  :: decompress_private !! Performs decompression using the implementation
        procedure(destroy_interface),               deferred,   pass(self)  :: destroy           !! Destroys the compressor implementation
    end type abstract_compressor

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

    type(dtfft_compression_mode_t), parameter  :: VALID_COMPRESSION_MODES(*) = [    DTFFT_COMPRESSION_MODE_LOSSLESS,        &
                                                                                    DTFFT_COMPRESSION_MODE_FIXED_RATE,      &
                                                                                    DTFFT_COMPRESSION_MODE_FIXED_PRECISION, &
                                                                                    DTFFT_COMPRESSION_MODE_FIXED_ACCURACY ]

public :: dtfft_compression_lib_t
    type, bind(C) :: dtfft_compression_lib_t
    !! Type that specifies compression library
        integer(c_int32_t) :: val !! Internal value
    end type dtfft_compression_lib_t

    type(dtfft_compression_lib_t),  parameter,  public :: DTFFT_COMPRESSION_LIB_ZFP = dtfft_compression_lib_t(CONF_DTFFT_COMPRESSION_LIB_ZFP)
    !! ZFP compression library

    type(dtfft_compression_lib_t),  parameter  :: VALID_COMPRESSION_LIBS(*) = [ DTFFT_COMPRESSION_LIB_ZFP]
    !! List of valid compression libraries


public :: dtfft_compression_config_t
    type, bind(C) :: dtfft_compression_config_t
        type(dtfft_compression_lib_t)   :: compression_lib
        !! Library to use. Currently only support ZFP

        type(dtfft_compression_mode_t)  :: compression_mode
        !! Compression mode to use

        real(c_double)                  :: rate
        !! Rate for `DTFFT_COMPRESSION_MODE_FIXED_RATE`

        integer(c_int32_t)              :: precision
        !! Precision for `DTFFT_COMPRESSION_MODE_FIXED_PRECISION`

        real(c_double)                  :: tolerance
        !! Tolerance for `DTFFT_COMPRESSION_MODE_FIXED_ACCURACY`
    end type dtfft_compression_config_t

    interface dtfft_compression_config_t
    !! Type bound constuctor for dtfft_compression_config_t
        module procedure create_compression_config_t    !! Creates config object
    end interface dtfft_compression_config_t

    type(dtfft_compression_config_t), parameter, public :: DEFAULT_COMPRESSION_CONFIG = dtfft_compression_config_t(DTFFT_COMPRESSION_LIB_ZFP, DTFFT_COMPRESSION_MODE_LOSSLESS, -1.0, -1, -1.0)
    !! Default compression configuration

    abstract interface
        integer(int32) function create_private_interface(self, config, platform, base_type)
        !! Creates the compressor implementation
        import
            class(abstract_compressor),         intent(inout)   :: self         !! Compressor instance
            type(dtfft_compression_config_t),   intent(in)      :: config       !! Compression configuration
            type(dtfft_platform_t),             intent(in)      :: platform     !! Target platform
            TYPE_MPI_DATATYPE,                  intent(in)      :: base_type    !! MPI data type
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

        subroutine decompress_private_interface(self, dims, in, out, stream)
        !! Performs decompression using the implementation
        import
            class(abstract_compressor), intent(inout)   :: self             !! Compressor instance
            integer(int32),             intent(in)      :: dims(:)          !! Array dimensions
            type(c_ptr),                intent(in)      :: in               !! Pointer to compressed data
            type(c_ptr),                intent(in)      :: out              !! Pointer to uncompressed buffer
            type(dtfft_stream_t),       intent(in)      :: stream           !! Stream handle
        end subroutine decompress_private_interface

        subroutine destroy_interface(self)
        !! Destroys the compressor implementation
        import
            class(abstract_compressor), intent(inout)   :: self !! Compressor instance
        end subroutine destroy_interface

        subroutine sync_interface(self, stream)
        import
            class(abstract_compressor), intent(inout)   :: self                 !! Abstract kernel
            type(dtfft_stream_t),       intent(in)      :: stream
        end subroutine sync_interface
    end interface

public :: operator(==)
    interface operator(==)
        module procedure compression_mode_eq    !! Check if two `dtfft_compression_mode_t` are equal
        module procedure compression_lib_eq     !! Check if two `dtfft_compression_lib_t` are equal
    end interface

public :: operator(/=)
    interface operator(/=)
        module procedure compression_mode_ne    !! Check if two `dtfft_compression_mode_t` are not equal
        module procedure compression_lib_ne     !! Check if two `dtfft_compression_lib_t` are not equal
    end interface

contains

    pure type(dtfft_compression_config_t) function create_compression_config_t(lib, mode, rate, precision, tolerance) result(config)
    !! Creates a compression configuration object
        type(dtfft_compression_lib_t),  intent(in)  :: lib        !! Compression library to use
        type(dtfft_compression_mode_t), intent(in)  :: mode       !! Compression mode
        real(c_double),     optional,   intent(in)  :: rate       !! Compression rate (for fixed rate mode)
        integer(c_int32_t), optional,   intent(in)  :: precision  !! Precision (for fixed precision mode)
        real(c_double),     optional,   intent(in)  :: tolerance  !! Tolerance (for fixed accuracy mode)

        config%compression_lib = lib
        config%compression_mode = mode
        config%rate = -1._c_double;         if ( present(rate) ) config%rate = rate
        config%precision = -1;              if ( present(precision) ) config%precision = precision
        config%tolerance = -1._c_double;    if ( present(tolerance) ) config%tolerance = tolerance
    end function create_compression_config_t

    integer(int32) function create(self, ndims, config, platform, base_type, storage_size, dims_permutation)
    !! Initializes the compressor with given parameters
        class(abstract_compressor),         intent(inout)   :: self             !! Compressor instance
        integer(int8),                      intent(in)      :: ndims            !! Number of dimensions
        type(dtfft_compression_config_t),   intent(in)      :: config           !! Compression configuration
        type(dtfft_platform_t),             intent(in)      :: platform         !! Target platform
        TYPE_MPI_DATATYPE,                  intent(in)      :: base_type        !! MPI data type
        integer(int64),                     intent(in)      :: storage_size     !! Storage size per element
        integer(int32),                     intent(in)      :: dims_permutation !! Dimension permutation flag

        self%ndims = ndims
        self%storage_size = storage_size
        self%compressed_rate = 0._real64
        self%execution_count = 0
        self%dims_permutation = dims_permutation
        CHECK_CALL( self%create_private(config, platform, base_type), create )
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

        REGION_BEGIN("Compressing", COLOR_SALMON)

        inptr = ptr_offset(in, self%storage_size * displ)
        outptr = ptr_offset(out, self%storage_size * displ)
#ifdef DTFFT_DEBUG
        if ( is_same_ptr(inptr, outptr) ) then
            INTERNAL_ERROR("compress: is_same_ptr(inptr, outptr)")
        endif
#endif

        compress = self%compress_private(dims(1:self%ndims), inptr, outptr, stream)

        uncompressed_size = self%storage_size * product(dims(1:self%ndims))
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
        REGION_END("Compressing")
    end function compress

    subroutine decompress(self, dims, displ, in, out, stream)
    !! Decompresses the input data into the output buffer
        class(abstract_compressor), intent(inout)   :: self     !! Compressor instance
        integer(int32),             intent(in)      :: dims(:)  !! Array dimensions
        integer(int32),             intent(in)      :: displ    !! Displacement in arrays
        type(c_ptr),                intent(in)      :: in       !! Pointer to compressed data
        type(c_ptr),                intent(in)      :: out      !! Pointer to uncompressed buffer
        type(dtfft_stream_t),       intent(in)      :: stream   !! Stream handle
        type(c_ptr) :: inptr, outptr
        integer(int32) :: permuted_dims(3)

        if ( product(dims(1:self%ndims)) == 0 ) then
            return
        endif
        REGION_BEGIN("Decompressing", COLOR_CORAL)

        inptr = ptr_offset(in, self%storage_size * displ)
        outptr = ptr_offset(out, self%storage_size * displ)
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
        call self%decompress_private(permuted_dims(1:self%ndims), inptr, outptr, stream)
    REGION_END("Decompressing")
    end subroutine decompress

    pure real(real64) function get_average_rate(self)
    !! Returns the average compression rate over all executions
        class(abstract_compressor), intent(in)      :: self !! Compressor instance
        get_average_rate = self%compressed_rate / real(self%execution_count, real64)
    end function get_average_rate

MAKE_EQ_FUN(dtfft_compression_mode_t, compression_mode_eq)
MAKE_NE_FUN(dtfft_compression_mode_t, compression_mode_ne)

MAKE_EQ_FUN(dtfft_compression_lib_t, compression_lib_eq)
MAKE_NE_FUN(dtfft_compression_lib_t, compression_lib_ne)

MAKE_VALID_FUN_DTYPE(dtfft_compression_mode_t, is_valid_compression_mode, VALID_COMPRESSION_MODES)
MAKE_VALID_FUN_DTYPE(dtfft_compression_lib_t, is_valid_compression_lib, VALID_COMPRESSION_LIBS)

    pure function check_compression_config(config) result(error_code)
    !! Checks if the given compression configuration is valid
        type(dtfft_compression_config_t), intent(in)    :: config !! Compression configuration to check
        integer(int32)                                  :: error_code

        error_code = DTFFT_SUCCESS
        if ( .not. is_valid_compression_lib(config%compression_lib) ) then
            error_code = DTFFT_ERROR_COMPRESSION_INVALID_LIBRARY
            return
        endif
        if ( .not. is_valid_compression_mode(config%compression_mode) ) then
            error_code = DTFFT_ERROR_COMPRESSION_INVALID_MODE
            return
        endif
    end function check_compression_config

end module dtfft_abstract_compressor