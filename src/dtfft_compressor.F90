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
use iso_c_binding
use iso_fortran_env
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
use dtfft_utils
#include "_dtfft_private.h"
#include "_dtfft_mpi.h"
implicit none
private
public :: create_compressor
public :: get_compressor_string

contains

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
end module dtfft_compressor