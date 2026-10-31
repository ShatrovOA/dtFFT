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
program test_compressors_host
!! Unit tests for all compressors available in current build:
!! creation and validation of compression configs, round-trip of compress + decompress
!! for every supported compression mode, data type, number of dimensions and dimension permutation
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_compressor
use dtfft_compressor
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none

    integer(int32),  parameter :: N_MODES = size(VALID_COMPRESSION_MODES)
    !! Number of compression modes defined by dtFFT

    real(real64),    parameter :: TEST_RATE = 32._real64
    !! Rate used by `DTFFT_COMPRESSION_MODE_FIXED_RATE`. Halved for single precision
    integer(int32),  parameter :: TEST_PRECISION = 32
    !! Precision used by `DTFFT_COMPRESSION_MODE_FIXED_PRECISION`. Halved for single precision
    real(real64),    parameter :: TEST_TOLERANCE = 1.e-5_real64
    !! Tolerance used by both `DTFFT_COMPRESSION_MODE_FIXED_PRECISION` and
    !! `DTFFT_COMPRESSION_MODE_FIXED_ACCURACY`

    type :: compressor_info
    !! Description of a single compressor available in current build
        type(dtfft_compression_lib_t)   :: lib                  !! Compression library
        logical                         :: supported(N_MODES)   !! Modes supported by library
        logical                         :: host_only            !! Library has no device implementation
    end type compressor_info

    integer(int32)          :: ierror
    integer(int32)          :: n_compressors, i
    type(compressor_info)   :: compressors(size(VALID_COMPRESSION_LIBS))
    type(dtfft_compression_lib_t) :: current_lib
    !! Library that is currently under test

    call MPI_Init(ierror)

    n_compressors = 0
#ifdef DTFFT_WITH_ZFP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_ZFP, [.true., .true., .true., .true.], .false.)
#endif
#ifdef DTFFT_WITH_SZ3
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_SZ3, [.true., .false., .true., .true.], .true.)
#endif
#ifdef DTFFT_WITH_SZP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_SZP, [.false., .false., .true., .true.], .true.)
#endif
#ifdef DTFFT_WITH_BITCOMP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_BITCOMP, [.true., .false., .true., .true.], .false.)
#endif

#if !defined(DTFFT_WITH_CUSZ) && !defined(DTFFT_WITH_CUSZP)
    if ( n_compressors == 0 ) then
        INTERNAL_ERROR("No compressors available while DTFFT_WITH_COMPRESSION is set")
    endif
#endif

    call test_compressor_strings()
    call test_config_checks()

    do i = 1, n_compressors
        call report_compressor(compressors(i))
        call test_invalid_create(compressors(i))
        call test_round_trips(compressors(i))
    enddo

    print*,'test_compressors: ALL TESTS PASSED'

    call MPI_Finalize(ierror)

contains

    subroutine report_compressor(info)
    !! Prints name of compressor that is about to be tested
        type(compressor_info),  intent(in)  :: info !! Compressor description
        type(string) :: name

        name = get_compressor_string(info%lib)
        print*,''
        print*,'--- Testing compressor '//name%raw//' ---'
        call name%destroy()
    end subroutine report_compressor

    subroutine test_compressor_strings()
    !! Checks that every valid compression library has its own name
    !! and that unknown library is reported as such
        type(string) :: name
        integer(int32) :: lib

        print*,'Testing get_compressor_string'
        do lib = 2, size(VALID_COMPRESSION_LIBS)
            name = get_compressor_string(VALID_COMPRESSION_LIBS(lib))
            print*,VALID_COMPRESSION_LIBS(lib)
            if ( name%raw == "Unknown" ) then
                INTERNAL_ERROR("get_compressor_string returned `Unknown` for valid library")
            endif
            print*,'    '//name%raw
            call name%destroy()
        enddo

        name = get_compressor_string(dtfft_compression_lib_t(1))
        if ( name%raw /= "Unknown" ) then
            INTERNAL_ERROR("get_compressor_string returned `"//name%raw//"` for invalid library")
        endif
        call name%destroy()
        print*,'SUCCESS'
    end subroutine test_compressor_strings

    subroutine test_config_checks()
    !! Checks validation of `dtfft_compression_config_t`
        type(dtfft_compression_config_t) :: config
        integer(int32) :: lib, mode

        print*,'Testing check_compression_config'

        ! Any combination of valid library and valid mode must pass basic validation
        do lib = 2, size(VALID_COMPRESSION_LIBS)
            do mode = 1, N_MODES
                config = dtfft_compression_config_t(VALID_COMPRESSION_LIBS(lib), VALID_COMPRESSION_MODES(mode))
                call check_error(check_compression_config(config), DTFFT_SUCCESS, "check_compression_config(valid)")
            enddo
        enddo

        config = dtfft_compression_config_t(dtfft_compression_lib_t(-1), DTFFT_COMPRESSION_MODE_LOSSLESS)
        call check_error(check_compression_config(config), DTFFT_ERROR_COMPRESSION_INVALID_LIBRARY, "invalid library")

        config = dtfft_compression_config_t(VALID_COMPRESSION_LIBS(1), dtfft_compression_mode_t(-1))
        call check_error(check_compression_config(config), DTFFT_ERROR_COMPRESSION_INVALID_MODE, "invalid mode")

        ! Default config must be a valid one
        call check_error(check_compression_config(DEFAULT_COMPRESSION_CONFIG), DTFFT_SUCCESS, "default config")
        print*,'SUCCESS'
    end subroutine test_config_checks

    subroutine test_invalid_create(info)
    !! Checks that compressor rejects unsupported modes and invalid mode parameters
        type(compressor_info),  intent(in)  :: info !! Compressor description
        type(dtfft_compression_mode_t) :: mode
        integer(int32) :: imode

        current_lib = info%lib
        print*,'  Testing invalid compressor configurations'

        do imode = 1, N_MODES
            mode = VALID_COMPRESSION_MODES(imode)
            if ( .not. info%supported(imode) ) then
                ! All parameters are set, mode itself is not supported by library
                call check_create(info, config_for_mode(mode, .false.), DTFFT_PLATFORM_HOST, MPI_REAL8,   &
                                  DTFFT_ERROR_COMPRESSION_INVALID_MODE, "unsupported mode")
                cycle
            endif
            if ( mode == DTFFT_COMPRESSION_MODE_LOSSLESS ) cycle

            ! Supported mode, but its parameter is left unset
            call check_create(info, dtfft_compression_config_t(info%lib, mode), DTFFT_PLATFORM_HOST, MPI_REAL8, &
                              expected_parameter_error(mode), "unset mode parameter")

            ! Single precision data cannot be compressed with double precision parameters
            call check_create(info, config_for_mode(mode, .false.), DTFFT_PLATFORM_HOST, MPI_REAL,   &
                              expected_precision_error(mode), "double precision parameter for real32")
        enddo

        if ( info%host_only ) then
            call check_create(info, config_for_mode(DTFFT_COMPRESSION_MODE_FIXED_ACCURACY, .false.),         &
                              DTFFT_PLATFORM_CUDA, MPI_REAL8, DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED,   &
                              "CUDA platform")
        else if ( is_zfp() ) then
            ! ZFP CUDA execution policy is limited to fixed rate mode. Depending on whether
            ! ZFP itself was built with CUDA, any other mode is rejected either as an
            ! unsupported mode or as a missing device implementation
            do imode = 1, N_MODES
                mode = VALID_COMPRESSION_MODES(imode)
                if ( mode == DTFFT_COMPRESSION_MODE_FIXED_RATE ) cycle
                call check_create_fails(info, config_for_mode(mode, .false.), DTFFT_PLATFORM_CUDA, MPI_REAL8,        &
                                        [DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED, DTFFT_ERROR_COMPRESSION_INVALID_MODE], &
                                        "CUDA platform, mode other then fixed rate")
            enddo
        endif
        print*,'  SUCCESS'
    end subroutine test_invalid_create

    subroutine check_create_fails(info, config, platform, base_type, expected, message)
    !! Creates compressor and checks that it failed with one of `expected` error codes
        type(compressor_info),              intent(in)  :: info         !! Compressor description
        type(dtfft_compression_config_t),   intent(in)  :: config       !! Compression config
        type(dtfft_platform_t),             intent(in)  :: platform     !! Target platform
        TYPE_MPI_DATATYPE,                  intent(in)  :: base_type    !! MPI data type
        integer(int32),                     intent(in)  :: expected(:)  !! Accepted error codes
        character(len=*),                   intent(in)  :: message      !! Test description
        class(abstract_compressor), allocatable :: compressor
        integer(int32) :: error_code, i

        call allocate_compressor(info%lib, compressor)
        if ( .not. allocated(compressor) ) then
            INTERNAL_ERROR("check_create_fails: Unknown compression lib")
        endif
        error_code = compressor%create(3_int8, config, platform, base_type, DIMS_PERMUTE_NONE)
        if ( error_code == DTFFT_SUCCESS ) then
            call compressor%destroy()
            INTERNAL_ERROR(message//": compressor accepted unsupported configuration, mode = "//to_str(config%compression_mode%val))
        endif
        if ( .not. any(error_code == expected) ) then
            print*,'Obtained: '//dtfft_get_error_string(error_code)
            do i = 1, size(expected)
                print*,'Expected: '//dtfft_get_error_string(expected(i))
            enddo
            INTERNAL_ERROR(message//", mode = "//to_str(config%compression_mode%val))
        endif
        deallocate( compressor )
    end subroutine check_create_fails

    integer(int32) function expected_parameter_error(mode)
    !! Error that is expected when parameter of `mode` is left unset
        type(dtfft_compression_mode_t), intent(in)  :: mode !! Compression mode

        select case ( mode%val )
        case ( DTFFT_COMPRESSION_MODE_FIXED_RATE%val )
            expected_parameter_error = DTFFT_ERROR_COMPRESSION_INVALID_RATE
        case ( DTFFT_COMPRESSION_MODE_FIXED_PRECISION%val )
            ! Only ZFP treats fixed precision mode as a number of bitplanes.
            ! Both SZ3 and SZp use it as a relative error bound and rely on tolerance
            if ( is_zfp() ) then
                expected_parameter_error = DTFFT_ERROR_COMPRESSION_INVALID_PRECISION
            else
                expected_parameter_error = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
            endif
        case default
            expected_parameter_error = DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE
        endselect
    end function expected_parameter_error

    integer(int32) function expected_precision_error(mode)
    !! Error that is expected when double precision parameter of `mode`
    !! is used to compress single precision data
        type(dtfft_compression_mode_t), intent(in)  :: mode !! Compression mode

        expected_precision_error = expected_parameter_error(mode)
        ! `TEST_TOLERANCE` is a valid value for both single and double precision,
        ! while rate and precision are limited by number of bits in a scalar
        if ( expected_precision_error == DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE ) expected_precision_error = DTFFT_SUCCESS
    end function expected_precision_error

    logical function is_zfp()
    !! Whether compressor under test is ZFP or not
        is_zfp = .false.
#ifdef DTFFT_WITH_ZFP
        is_zfp = current_lib == DTFFT_COMPRESSION_LIB_ZFP
#endif
    end function is_zfp

    type(dtfft_compression_config_t) function config_for_mode(mode, is_float) result(config)
    !! Creates a valid config for given `mode`
        type(dtfft_compression_mode_t), intent(in)  :: mode     !! Compression mode
        logical,                        intent(in)  :: is_float !! Single precision data flag
        real(real64)    :: rate
        integer(int32)  :: precision_

        rate = TEST_RATE
        precision_ = TEST_PRECISION
        ! ZFP requires rate and precision to be less than number of bits in a scalar
        if ( is_float ) then
            rate = 0.5_real64 * TEST_RATE
            precision_ = TEST_PRECISION / 2
        endif
        config = dtfft_compression_config_t(current_lib, mode, rate, precision_, TEST_TOLERANCE)
    end function config_for_mode

    subroutine check_create(info, config, platform, base_type, expected, message)
    !! Creates compressor and checks returned error code
        type(compressor_info),              intent(in)  :: info         !! Compressor description
        type(dtfft_compression_config_t),   intent(in)  :: config       !! Compression config
        type(dtfft_platform_t),             intent(in)  :: platform     !! Target platform
        TYPE_MPI_DATATYPE,                  intent(in)  :: base_type    !! MPI data type
        integer(int32),                     intent(in)  :: expected     !! Expected error code
        character(len=*),                   intent(in)  :: message      !! Test description
        class(abstract_compressor), allocatable :: compressor
        integer(int32) :: error_code

        call allocate_compressor(info%lib, compressor)
        if ( .not. allocated(compressor) ) then
            INTERNAL_ERROR("check_create: Unknown compression lib")
        endif
        error_code = compressor%create(3_int8, config, platform, base_type, DIMS_PERMUTE_NONE)
        call check_error(error_code, expected, message//", mode = "//to_str(config%compression_mode%val))
        if ( error_code == DTFFT_SUCCESS ) call compressor%destroy()
        deallocate( compressor )
    end subroutine check_create

    subroutine test_round_trips(info)
    !! Runs compress + decompress for all supported modes, data types,
    !! number of dimensions and dimension permutations
        type(compressor_info),  intent(in)  :: info !! Compressor description
        integer(int32) :: imode

        current_lib = info%lib
        do imode = 1, N_MODES
            if ( .not. info%supported(imode) ) cycle

            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_REAL,           DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_REAL8,          DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_COMPLEX,        DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_DOUBLE_COMPLEX, DIMS_PERMUTE_NONE)

            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_REAL,           DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_REAL8,          DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_COMPLEX,        DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_DOUBLE_COMPLEX, DIMS_PERMUTE_NONE)

            ! Compressor is created by pack kernel and reused by unpack kernel on the
            ! other side of communication, where data layout is permuted
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_REAL8,          DIMS_PERMUTE_FORWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 128],       MPI_REAL8,          DIMS_PERMUTE_BACKWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_REAL8,          DIMS_PERMUTE_FORWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [16, 32, 64],    MPI_REAL8,          DIMS_PERMUTE_BACKWARD)

            ! Empty block must be a no-op
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [0, 128],        MPI_REAL8,          DIMS_PERMUTE_NONE)
        enddo
    end subroutine test_round_trips

    subroutine test_round_trip(info, mode, unpack_dims, base_type, permutation)
    !! Compresses reference data and decompresses it back, checking obtained accuracy
        type(compressor_info),          intent(in)  :: info         !! Compressor description
        type(dtfft_compression_mode_t), intent(in)  :: mode         !! Compression mode
        integer(int32),                 intent(in)  :: unpack_dims(:) !! Dimensions known to decompressor
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type    !! MPI data type
        integer(int32),                 intent(in)  :: permutation  !! Dimension permutation flag
        class(abstract_compressor), allocatable :: compressor
        integer(int8)   :: ndims
        integer(int32)  :: pack_dims(size(unpack_dims))
        integer(int64)  :: storage_size, compressed_bytes, uncompressed_bytes
        integer(int64)  :: n_scalars
        logical         :: is_float, is_complex
        type(c_ptr)     :: in, out
        real(real64)    :: error, threshold, rate
        type(string)    :: name

        ndims = size(unpack_dims, kind=int8)
        pack_dims = permute_dims(unpack_dims, permutation)

        is_float = GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_REAL) .or. GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_COMPLEX)
        is_complex = GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_COMPLEX) .or. GET_MPI_VALUE(base_type) == GET_MPI_VALUE(MPI_DOUBLE_COMPLEX)

        if ( is_float ) then
            storage_size = FLOAT_STORAGE_SIZE
        else
            storage_size = DOUBLE_STORAGE_SIZE
        endif
        if ( is_complex ) storage_size = 2 * storage_size

        ! Complex values are compressed as a flat array of interleaved real values
        n_scalars = product(int(pack_dims, int64))
        if ( is_complex ) n_scalars = 2 * n_scalars

        name = get_compressor_string(info%lib)
        print '(a)', '    '//name%raw//': mode = '//to_str(mode%val)//', storage = '//to_str(storage_size)// &
                     ', permutation = '//to_str(permutation)//', dims = '//dims_to_str(pack_dims)
        call name%destroy()

        call allocate_compressor(info%lib, compressor)
        call check_error( compressor%create(ndims, config_for_mode(mode, is_float), DTFFT_PLATFORM_HOST,   &
                                            base_type, permutation),                        &
                          DTFFT_SUCCESS, "create for round trip" )

        uncompressed_bytes = storage_size * product(int(pack_dims, int64))
        ! Buffers are always able to hold uncompressed data
        in = mem_alloc_host(max(uncompressed_bytes, 1_int64))
        out = mem_alloc_host(max(uncompressed_bytes, 1_int64))

        call fill_reference(in, n_scalars, is_float)

        compressed_bytes = compressor%compress(pack_dims, 0, in, out, NULL_STREAM)
        if ( n_scalars == 0 ) then
            if ( compressed_bytes /= 0 ) then
                INTERNAL_ERROR("Empty block reported non zero compressed size")
            endif
        else
            if ( compressed_bytes <= 0 .or. compressed_bytes >= uncompressed_bytes ) then
                INTERNAL_ERROR("Unexpected compressed size: "//to_str(compressed_bytes)//" of "//to_str(uncompressed_bytes))
            endif
        endif

        ! Wipe destination buffer to make sure data is actually restored by decompressor
        call wipe(in, n_scalars, is_float)
        call compressor%decompress(unpack_dims, 0, out, in, NULL_STREAM, compressed_bytes)

        error = max_error(in, n_scalars, is_float)
        threshold = error_threshold(mode, is_float)
        if ( n_scalars > 0 ) then
            print '(a,ES16.8,a,ES16.8)', '        error = ', error, ', threshold = ', threshold
            if ( .not. (error <= threshold) ) then
                INTERNAL_ERROR("Round trip error is too big")
            endif

            rate = compressor%get_average_rate()
            print '(a,F12.4)', '        rate  = ', rate
            if ( .not. (rate > 1._real64) ) then
                INTERNAL_ERROR("Average compression rate is not greater then 1")
            endif
        endif

        call compressor%destroy()
        deallocate( compressor )
        call mem_free_host(in)
        call mem_free_host(out)
    end subroutine test_round_trip

    function dims_to_str(dims) result(str)
    !! Returns dimensions as a `d1 x d2 [x d3]` string
        integer(int32),     intent(in)  :: dims(:)  !! Dimensions to convert
        character(len=:),   allocatable :: str      !! Resulting string
        integer(int32) :: i

        str = to_str(dims(1))
        do i = 2, size(dims)
            str = str//'x'//to_str(dims(i))
        enddo
    end function dims_to_str

    function permute_dims(dims, permutation) result(permuted)
    !! Recovers dimensions that were used during compression.
    !! Mirrors permutation performed by `abstract_compressor%decompress`
        integer(int32), intent(in)  :: dims(:)      !! Dimensions known to decompressor
        integer(int32), intent(in)  :: permutation  !! Dimension permutation flag
        integer(int32)              :: permuted(size(dims))

        permuted(:) = dims(:)
        if ( permutation == DIMS_PERMUTE_FORWARD ) then
            permuted(1) = dims(2)
            if ( size(dims) == 2 ) then
                permuted(2) = dims(1)
            else
                permuted(2) = dims(3)
                permuted(3) = dims(1)
            endif
        else if ( permutation == DIMS_PERMUTE_BACKWARD ) then
            if ( size(dims) == 2 ) then
                permuted(1) = dims(2)
                permuted(2) = dims(1)
            else
                permuted(1) = dims(3)
                permuted(2) = dims(1)
                permuted(3) = dims(2)
            endif
        endif
    end function permute_dims

    pure real(real64) function reference(i)
    !! Smooth reference data. Any lossy compressor should be able to
    !! reach requested accuracy on such data
        integer(int64), intent(in)  :: i  !! Scalar index
        reference = 1._real64 + sin(0.01_real64 * real(i, real64)) * cos(0.003_real64 * real(i, real64))
        ! reference = 1._real64 + real(i, real64) * (-0.5_real64 ** (i / 1000000._real64))
    end function reference

    subroutine fill_reference(ptr, n_scalars, is_float)
    !! Fills buffer with reference data
        type(c_ptr),    intent(in)  :: ptr          !! Buffer to fill
        integer(int64), intent(in)  :: n_scalars    !! Number of scalars in buffer
        logical,        intent(in)  :: is_float     !! Single precision data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)
        integer(int64) :: i

        if ( n_scalars == 0 ) return
        if ( is_float ) then
            call c_f_pointer(ptr, pf, [n_scalars])
            do i = 1, n_scalars
                pf(i) = real(reference(i), real32)
            enddo
        else
            call c_f_pointer(ptr, pd, [n_scalars])
            do i = 1, n_scalars
                pd(i) = reference(i)
            enddo
        endif
    end subroutine fill_reference

    subroutine wipe(ptr, n_scalars, is_float)
    !! Overwrites buffer with zeros
        type(c_ptr),    intent(in)  :: ptr          !! Buffer to wipe
        integer(int64), intent(in)  :: n_scalars    !! Number of scalars in buffer
        logical,        intent(in)  :: is_float     !! Single precision data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)

        if ( n_scalars == 0 ) return
        if ( is_float ) then
            call c_f_pointer(ptr, pf, [n_scalars])
            pf(:) = 0._real32
        else
            call c_f_pointer(ptr, pd, [n_scalars])
            pd(:) = 0._real64
        endif
    end subroutine wipe

    real(real64) function max_error(ptr, n_scalars, is_float)
    !! Returns maximum absolute difference between buffer and reference data
        type(c_ptr),    intent(in)  :: ptr          !! Buffer to check
        integer(int64), intent(in)  :: n_scalars    !! Number of scalars in buffer
        logical,        intent(in)  :: is_float     !! Single precision data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)
        integer(int64) :: i

        max_error = 0._real64
        if ( n_scalars == 0 ) return
        if ( is_float ) then
            call c_f_pointer(ptr, pf, [n_scalars])
            do i = 1, n_scalars
                max_error = max(max_error, abs(real(pf(i), real64) - real(real(reference(i), real32), real64)))
            enddo
        else
            call c_f_pointer(ptr, pd, [n_scalars])
            do i = 1, n_scalars
                max_error = max(max_error, abs(pd(i) - reference(i)))
            enddo
        endif
    end function max_error

    real(real64) function error_threshold(mode, is_float)
    !! Maximum error allowed for given compression mode
        type(dtfft_compression_mode_t), intent(in)  :: mode     !! Compression mode
        logical,                        intent(in)  :: is_float !! Single precision data flag

        select case ( mode%val )
        case ( DTFFT_COMPRESSION_MODE_LOSSLESS%val )
            ! Data must be restored exactly
            error_threshold = 0._real64
        case ( DTFFT_COMPRESSION_MODE_FIXED_RATE%val )
            ! Amount of bits per scalar is fixed, resulting accuracy is data dependent
            error_threshold = 1.e-3_real64
        case ( DTFFT_COMPRESSION_MODE_FIXED_PRECISION%val )
            ! ZFP treats precision as a number of bitplanes to keep, so accuracy
            ! is only indirectly related to it. SZ3 and SZp use relative error bound
            error_threshold = 1.e-2_real64
        case default
            ! Absolute error bound is guaranteed by all libraries
            error_threshold = 10._real64 * TEST_TOLERANCE
        endselect
        if ( is_float ) error_threshold = max(error_threshold, real(epsilon(1._real32), real64))
    end function error_threshold

!     subroutine allocate_compressor(lib, compressor)
!     !! Allocates compressor of a dynamic type that corresponds to `lib`
!         type(dtfft_compression_lib_t),              intent(in)  :: lib          !! Compression library
!         class(abstract_compressor), allocatable,    intent(out) :: compressor   !! Compressor to allocate

!         select case ( lib%val )
! #ifdef DTFFT_WITH_ZFP
!         case ( DTFFT_COMPRESSION_LIB_ZFP%val )
!             allocate( compressor_zfp :: compressor )
! #endif
! #ifdef DTFFT_WITH_SZ3
!         case ( DTFFT_COMPRESSION_LIB_SZ3%val )
!             allocate( compressor_sz3 :: compressor )
! #endif
! #ifdef DTFFT_WITH_SZP
!         case ( DTFFT_COMPRESSION_LIB_SZP%val )
!             allocate( compressor_szp :: compressor )
! #endif
! #ifdef DTFFT_WITH_BITCOMP
!         case ( DTFFT_COMPRESSION_LIB_BITCOMP%val )
!             allocate( compressor_bitcomp :: compressor )
! #endif
!         case default
!             INTERNAL_ERROR("Unknown compression lib")
!         endselect
!     end subroutine allocate_compressor

    subroutine check_error(actual, expected, message)
    !! Aborts execution in case `actual` error code differs from `expected` one
        integer(int32),     intent(in)  :: actual   !! Obtained error code
        integer(int32),     intent(in)  :: expected !! Expected error code
        character(len=*),   intent(in)  :: message  !! Test description

        if ( actual == expected ) return
        print*,'Expected: '//dtfft_get_error_string(expected)
        print*,'Obtained: '//dtfft_get_error_string(actual)
        print*,'actual = ',actual,'expected = ',expected
        INTERNAL_ERROR(message)
    end subroutine check_error
end program test_compressors_host
