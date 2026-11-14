#include "dtfft_config.h"
program test_compressors_device
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_compressor
use dtfft_compressor
use dtfft_errors
use dtfft_interface_cuda_runtime
use dtfft_parameters
use dtfft_utils
use test_suite
#include "_dtfft_cuda.h"
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
    end type compressor_info

    ! integer(int32)          :: ierror
    integer(int32)          :: n_compressors, i
    type(compressor_info)   :: compressors(size(VALID_COMPRESSION_LIBS))
    type(dtfft_compression_lib_t) :: current_lib

    integer(int32), parameter :: N = 64
    integer(int32), parameter :: DIMS(3) = [N, N, N]
    integer(int32) :: ierr
    integer(c_int) :: device_count
    type(test_suite_t) :: test

    call test%create("Compressors device")
    ierr = cudaGetDeviceCount(device_count)
#ifdef DTFFT_WITH_MOCK_ENABLED
    device_count = 0_c_int
#endif
    if ( ierr /= cudaSuccess .or. device_count == 0 ) then
        call MPI_Finalize(ierr)
        stop 77
    endif

    n_compressors = 0
#ifdef DTFFT_WITH_ZFP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_ZFP, [.false., .true., .false., .false.])
#endif
#ifdef DTFFT_WITH_CUSZP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_CUSZP, [.false., .false., .false., .true.])
#endif
#ifdef DTFFT_WITH_BITCOMP
    n_compressors = n_compressors + 1
    compressors(n_compressors) = compressor_info(DTFFT_COMPRESSION_LIB_BITCOMP, [.true., .false., .false., .true.])
#endif

    do i = 1, n_compressors
        call report_compressor(compressors(i))
        call test_round_trips(compressors(i))
    enddo

    call test%finalize()

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

    subroutine test_round_trips(info)
    !! Runs compress + decompress for all supported modes, data types,
    !! number of dimensions and dimension permutations
        type(compressor_info),  intent(in)  :: info !! Compressor description
        integer(int32) :: imode

        current_lib = info%lib
        do imode = 1, N_MODES
            if ( .not. info%supported(imode) ) cycle

            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_REAL,           DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_REAL8,          DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_COMPLEX,        DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_DOUBLE_COMPLEX, DIMS_PERMUTE_NONE)

            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128, 64],    MPI_REAL,           DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128, 64],    MPI_REAL8,          DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128, 64],    MPI_COMPLEX,        DIMS_PERMUTE_NONE)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128, 64],    MPI_DOUBLE_COMPLEX, DIMS_PERMUTE_NONE)

            ! Compressor is created by pack kernel and reused by unpack kernel on the
            ! other side of communication, where data layout is permuted
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_REAL8,          DIMS_PERMUTE_FORWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [256, 128],       MPI_REAL8,          DIMS_PERMUTE_BACKWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 32, 256],    MPI_REAL8,          DIMS_PERMUTE_FORWARD)
            call test_round_trip(info, VALID_COMPRESSION_MODES(imode), [64, 32, 256],    MPI_REAL8,          DIMS_PERMUTE_BACKWARD)

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
        type(c_ptr)     :: in, out, d_in, d_out
        real(real64)    :: error, threshold, rate
        type(string)    :: name
        type(dtfft_stream_t) :: stream


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

        call test%check("create compressor", &
            create_compressor(config_for_mode(mode, is_float), ndims, DTFFT_PLATFORM_CUDA, base_type, permutation, compressor), &
            DTFFT_SUCCESS &
        )

        uncompressed_bytes = storage_size * product(int(pack_dims, int64))
        ! Buffers are always able to hold uncompressed data
        in = mem_alloc_host(max(uncompressed_bytes, 1_int64))
        out = mem_alloc_host(max(uncompressed_bytes, 1_int64))

        call fill_reference(in, n_scalars, is_float)

        CUDA_CALL(cudaMalloc(d_in, uncompressed_bytes))
        CUDA_CALL(cudaMalloc(d_out, uncompressed_bytes))
        CUDA_CALL(cudaMemcpy(d_in, in, uncompressed_bytes, cudaMemcpyHostToDevice))
        CUDA_CALL(cudaStreamCreate(stream))

        compressed_bytes = compressor%compress(pack_dims, 0, d_in, d_out, stream)
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

        call compressor%decompress(unpack_dims, 0, d_out, d_in, stream, compressed_bytes)

        CUDA_CALL(cudaMemcpy(in, d_in, uncompressed_bytes, cudaMemcpyDeviceToHost))

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

        CUDA_CALL(cudaFree(d_in))
        CUDA_CALL(cudaFree(d_out))
        CUDA_CALL(cudaStreamDestroy(stream))

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

    pure real(real64) function reference(i)
    !! Smooth reference data. Any lossy compressor should be able to
    !! reach requested accuracy on such data
        integer(int64), intent(in)  :: i  !! Scalar index
        ! reference = 1._real64 + sin(0.01_real64 * real(i, real64)) * cos(0.003_real64 * real(i, real64))
        reference = 1._real64 + (-0.5_real64 ** i)
    end function reference

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

end program test_compressors_device