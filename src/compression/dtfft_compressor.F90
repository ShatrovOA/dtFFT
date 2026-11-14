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
module dtfft_compressor
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
use dtfft_errors
use dtfft_parameters
use dtfft_abstract_compressor
#ifdef DTFFT_WITH_ZFP
use dtfft_compressor_zfp,       only: compressor_zfp
#endif
#ifdef DTFFT_WITH_CUSZP
use dtfft_compressor_cuszp,     only: compressor_cuszp
#endif
#ifdef DTFFT_WITH_SZ3
use dtfft_compressor_sz3,       only: compressor_sz3
#endif
#ifdef DTFFT_WITH_SZP
use dtfft_compressor_szp,       only: compressor_szp
#endif
! #ifdef DTFFT_WITH_CUSZ
! use dtfft_compressor_cusz,      only: compressor_cusz
! #endif
#ifdef DTFFT_WITH_BITCOMP
use dtfft_compressor_bitcomp,   only: compressor_bitcomp
#endif
use dtfft_serializer
use dtfft_utils
#include "_dtfft_private.h"
#include "_dtfft_mpi.h"
implicit none
private
public :: create_compressor
public :: get_compressor_string
public :: save_compression_config, load_compression_config
public :: check_compression_config

public :: dtfft_compression_config_t
    type, bind(C) :: dtfft_compression_config_t
    !! Type that specifies compression configuration
        type(dtfft_compression_lib_t)   :: compression_lib
        !! Compression library to use

        type(dtfft_compression_mode_t)  :: compression_mode
        !! Compression mode to use

        real(c_double)                  :: rate
        !! Rate for `DTFFT_COMPRESSION_MODE_FIXED_RATE`

        integer(c_int32_t)              :: precision
        !! Precision for `DTFFT_COMPRESSION_MODE_FIXED_PRECISION`

        real(c_double)                  :: tolerance
        !! Tolerance for `DTFFT_COMPRESSION_MODE_FIXED_ACCURACY`
    end type dtfft_compression_config_t

    type(dtfft_compression_config_t), parameter, public :: DEFAULT_COMPRESSION_CONFIG = dtfft_compression_config_t(DTFFT_COMPRESSION_LIB_NONE, DTFFT_COMPRESSION_MODE_LOSSLESS, -1.0, -1, -1.0)
    !! Default compression configuration

    interface dtfft_compression_config_t
    !! Type bound constuctor for dtfft_compression_config_t
        module procedure create_compression_config_t    !! Creates config object
    end interface dtfft_compression_config_t

public :: operator(==)
    interface operator(==)
        module procedure compression_lib_eq     !! Check if two `dtfft_compression_lib_t` are equal
    end interface

public :: operator(/=)
    interface operator(/=)
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
        OPTVAL(config%rate, -1._c_double, rate)
        OPTVAL(config%precision, -1, precision)
        OPTVAL(config%tolerance, -1._c_double, tolerance)
    end function create_compression_config_t

    type(string) function get_compressor_string(compressor)
        type(dtfft_compression_lib_t), intent(in) :: compressor

        select case (compressor%val)
        case (DTFFT_COMPRESSION_LIB_SZ3%val)
            get_compressor_string = string("SZ3")
        case (DTFFT_COMPRESSION_LIB_SZP%val)
            get_compressor_string = string("SZp")
        ! case (DTFFT_COMPRESSION_LIB_CUSZ%val)
        !     get_compressor_string = string("cuSZ")
        case (DTFFT_COMPRESSION_LIB_CUSZP%val)
            get_compressor_string = string("cuSZp")
        case (DTFFT_COMPRESSION_LIB_ZFP%val)
            get_compressor_string = string("ZFP")
        case (DTFFT_COMPRESSION_LIB_BITCOMP%val)
            get_compressor_string = string("Bitcomp")
        case default
            get_compressor_string = string("Unknown")
        end select
    end function get_compressor_string

    subroutine allocate_compressor(lib, compressor)
    !! Allocates compressor of a dynamic type that corresponds to `lib`
        type(dtfft_compression_lib_t),               intent(in)  :: lib          !! Compression library
        class(abstract_compressor),     allocatable, intent(out) :: compressor   !! Compressor to allocate

        select case ( lib%val )
#ifdef DTFFT_WITH_ZFP
        case ( DTFFT_COMPRESSION_LIB_ZFP%val )
            allocate( compressor_zfp :: compressor )
#endif
#ifdef DTFFT_WITH_SZ3
        case ( DTFFT_COMPRESSION_LIB_SZ3%val )
            allocate( compressor_sz3 :: compressor )
#endif
#ifdef DTFFT_WITH_SZP
        case ( DTFFT_COMPRESSION_LIB_SZP%val )
            allocate( compressor_szp :: compressor )
#endif
! #ifdef DTFFT_WITH_CUSZ
!         case ( DTFFT_COMPRESSION_LIB_CUSZ%val )
!             allocate( compressor_cusz :: compressor )
! #endif
#ifdef DTFFT_WITH_CUSZP
        case ( DTFFT_COMPRESSION_LIB_CUSZP%val )
            allocate( compressor_cuszp :: compressor )
#endif
#ifdef DTFFT_WITH_BITCOMP
        case ( DTFFT_COMPRESSION_LIB_BITCOMP%val )
            allocate( compressor_bitcomp :: compressor )
#endif
        endselect
    end subroutine allocate_compressor

    function create_compressor(config, ndims, platform, base_type, dims_permutation, compressor) result(error_code)
    !! Allocates and initializes compressor of a dynamic type that corresponds to `config%compression_lib`
        type(dtfft_compression_config_t),           intent(in)  :: config       !! Compression configuration
        integer(int8),                              intent(in)  :: ndims            !! Number of dimensions
        type(dtfft_platform_t),                     intent(in)  :: platform         !! Target platform
        TYPE_MPI_DATATYPE,                          intent(in)  :: base_type        !! MPI data type
        integer(int32),                             intent(in)  :: dims_permutation !! Dimension permutation flag
        class(abstract_compressor), allocatable,    intent(out) :: compressor   !! Compressor to allocate
        integer(int32)                                          :: error_code   !! Error code
        logical :: m1(4), m2(4)
        logical :: use_tolerance
        real(real64) :: rate_or_tolerance

        error_code = DTFFT_SUCCESS
        call allocate_compressor(config%compression_lib, compressor)
        if (.not. allocated(compressor)) then
            error_code = DTFFT_ERROR_COMPRESSION_INVALID_LIBRARY
            return
        end if
        call compressor%get_supported_modes(m1, m2, use_tolerance)

        rate_or_tolerance = config%rate
        if ( (config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_PRECISION .and. use_tolerance)   &
            .or. (config%compression_mode == DTFFT_COMPRESSION_MODE_FIXED_ACCURACY)) then
            rate_or_tolerance = config%tolerance
        end if

        error_code = compressor%create(ndims, config%compression_mode, rate_or_tolerance, config%precision, platform, base_type, dims_permutation)
        if ( error_code /= DTFFT_SUCCESS ) then
            deallocate(compressor)
        endif
    end function create_compressor

    subroutine save_compression_config(config, fh)
        type(dtfft_compression_config_t),   intent(in)    :: config !! Compression configuration to save
        type(serializer),                   intent(inout) :: fh

        call fh%save(int(config%compression_lib%val, int16))
        call fh%save(int(config%compression_mode%val, int8))
        call fh%save(config%rate)
        call fh%save(config%precision)
        call fh%save(config%tolerance)
    end subroutine save_compression_config

    type(dtfft_compression_config_t) function load_compression_config(fh) result(config)
        type(serializer),                   intent(inout) :: fh
        integer(int16) :: t1
        integer(int8) :: t2

        call fh%load(t1);   config%compression_lib%val = int(t1, int32)
        call fh%load(t2);   config%compression_mode%val = int(t2, int32)
        call fh%load(config%rate)
        call fh%load(config%precision)
        call fh%load(config%tolerance)
    end function load_compression_config

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

MAKE_EQ_FUN(dtfft_compression_lib_t, compression_lib_eq)
MAKE_NE_FUN(dtfft_compression_lib_t, compression_lib_ne)

MAKE_VALID_FUN_DTYPE(dtfft_compression_lib_t, is_valid_compression_lib, VALID_COMPRESSION_LIBS)
end module dtfft_compressor