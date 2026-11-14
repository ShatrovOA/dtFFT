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
program test_backend_compressed
!! Tests compressed backends with real and complex data on all platforms available in current build.
!!
!! Every rank owns a `NX x ny_local` slab and sends `NX x cy` blocks to all other ranks.
!! Pack kernel compresses each block before it is sent, unpack kernel decompresses received
!! blocks, so the whole `pack -> compress -> exchange -> decompress -> unpack` chain is covered
!! for every backend, compressor and compression mode available in current build.
!!
!! On `DTFFT_PLATFORM_CUDA` buffers live in device memory: reference data is staged through
!! a host buffer and copied to and from the device, exactly as a user of the library would do.
!! Compressors without a device implementation report
!! `DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED` and such cases are reported as skipped.
!!
!! Note that self communication is never compressed, so nothing is actually compressed on a
!! single rank. Such a run is pointless and is not registered by CTest at all.
!!
!! Blocks are kept reasonably large: compressed backends abort when compressed data does not fit
!! into the original buffer, and lossless compressors expand blocks of just a few kilobytes.
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_backend
use dtfft_abstract_compressor
use dtfft_compressor
use dtfft_abstract_kernel
use dtfft_backend_mpi
#ifdef DTFFT_WITH_NCCL
use dtfft_backend_nccl_m,   only: backend_nccl
#endif
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
use dtfft_kernel_device,    only: kernel_device
#endif
use test_suite
use dtfft_kernel_host,      only: kernel_host
use dtfft_parameters
use dtfft_pencil,           only: pencil
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none

    integer(int32),  parameter :: NX = 128
    !! Number of scalars in contiguous dimension of a slab

    integer(int32),  parameter :: MIN_COLUMNS = 32
    !! Minimum number of columns in a single exchanged block.
    !! Blocks must be large enough for a lossless compressor to win over its own
    !! overhead: general purpose entropy coders expand a block of a few kilobytes

    integer(int32),  parameter :: N_MODES = size(VALID_COMPRESSION_MODES)
    !! Number of compression modes defined by dtFFT

    real(real64),    parameter :: TEST_RATE = 32._real64
    !! Rate used by `DTFFT_COMPRESSION_MODE_FIXED_RATE`. Halved for single precision
    integer(int32),  parameter :: TEST_PRECISION = 32
    !! Precision used by `DTFFT_COMPRESSION_MODE_FIXED_PRECISION`. Halved for single precision
    real(real64),    parameter :: TEST_TOLERANCE = 1.e-5_real64
    !! Tolerance used by error bounded compression modes

    type :: compressor_info
    !! Description of a single compressor available in current build
        type(dtfft_compression_lib_t)   :: lib                  !! Compression library
        logical                         :: host_modes(N_MODES)  !! Modes supported on host
        logical                         :: cuda_modes(N_MODES)  !! Modes supported on device
    end type compressor_info

    integer(int32)          :: ierr, comm_rank, comm_size
    integer(int32)          :: n_backends, n_compressors, n_platforms
    type(dtfft_backend_t)   :: backends(4)
    type(compressor_info)   :: compressors(size(VALID_COMPRESSION_LIBS))
    type(dtfft_platform_t)  :: platforms(2)
    integer(int32)          :: ny_globals(2)
    integer(int64)          :: storages(4)
    TYPE_MPI_DATATYPE       :: data_types(4)
    type(pencil), target    :: pencils(1)
    TYPE_MPI_COMM           :: comms(1)
    integer(int32)          :: n_passed, n_skipped
    type(test_suite_t)      :: test

    call test%create("Compressed backends")

    ! call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

    if ( comm_size == 1 ) then
        call MPI_Finalize(ierr)
        stop 77
    endif

    platforms(1) = DTFFT_PLATFORM_HOST
    n_platforms = 1
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
    n_platforms = 2
    platforms(2) = DTFFT_PLATFORM_CUDA
    ierr = cudaSetDevice(0)
#endif

    n_compressors = 0
#ifdef DTFFT_WITH_ZFP
    ! ZFP supports every mode dtFFT defines on host, but its CUDA execution policy
    ! is limited to fixed rate mode, both for compression and decompression
    call add_compressor(DTFFT_COMPRESSION_LIB_ZFP, [.true.,  .true., .true., .true.],   &
                                                   [.false., .true., .false., .false.])
#endif
#ifdef DTFFT_WITH_SZ3
    ! SZ3 has no fixed rate mode and no device implementation
    call add_compressor(DTFFT_COMPRESSION_LIB_SZ3, [.true., .false., .true., .true.],   &
                                                   [.false., .false., .false., .false.])
#endif
#ifdef DTFFT_WITH_SZP
    ! SZp is error bounded only compressor and has no device implementation
    call add_compressor(DTFFT_COMPRESSION_LIB_SZP, [.false., .false., .true., .true.],  &
                                                   [.false., .false., .false., .false.])
#endif
#ifdef DTFFT_WITH_CUSZP
    ! cuSZp supports fixed-precision and fixed-accuracy modes on CUDA only
    call add_compressor(DTFFT_COMPRESSION_LIB_CUSZP, [.false., .false., .false., .false.], &
                                                      [.false., .false., .false., .true.])
#endif
#ifdef DTFFT_WITH_BITCOMP
    call add_compressor(DTFFT_COMPRESSION_LIB_BITCOMP, [.true., .false., .true., .true.],  &
                                                       [.true., .false., .false., .true.])
#endif

    if ( n_compressors == 0 ) then
        INTERNAL_ERROR("No compressors available while DTFFT_WITH_COMPRESSION is set")
    endif

    ! Sizes are derived from number of ranks to keep every block at least
    ! `MIN_COLUMNS` wide. First one splits evenly, second one does not
    ny_globals(1) = MIN_COLUMNS * comm_size * comm_size
    ny_globals(2) = MIN_COLUMNS * comm_size * comm_size + comm_size + 1

    storages(1) = FLOAT_STORAGE_SIZE
    storages(2) = DOUBLE_STORAGE_SIZE
    storages(3) = 2 * FLOAT_STORAGE_SIZE
    storages(4) = 2 * DOUBLE_STORAGE_SIZE
    data_types = [MPI_REAL, MPI_REAL8, MPI_COMPLEX, MPI_DOUBLE_COMPLEX]

    n_passed = 0
    n_skipped = 0
    block
        integer(int32) :: ip, ib, ic, imode, isize, itype

        do ip = 1, n_platforms
            ! NCCL backends are only available on device, so backend list is per platform
            n_backends = 0
            call add_backend(DTFFT_BACKEND_MPI_P2P_COMPRESSED)
#ifdef DTFFT_WITH_RMA
            call add_backend(DTFFT_BACKEND_MPI_RMA_COMPRESSED)
#endif
! #ifdef DTFFT_WITH_NCCL
!             if ( platforms(ip) == DTFFT_PLATFORM_CUDA ) call add_backend(DTFFT_BACKEND_NCCL_COMPRESSED)
! #endif
            if ( comm_rank == 0 ) then
                write(output_unit, '(a,i0,a)') "Testing platform "//get_platform_string(platforms(ip))// &
                                              " with ", n_backends, " backend(s)"
            endif

            do ib = 1, n_backends
                do ic = 1, n_compressors
                    do imode = 1, N_MODES
                        do isize = 1, size(ny_globals)
                            do itype = 1, size(storages)
                                call run_case(backends(ib), platforms(ip), compressors(ic),                      &
                                              VALID_COMPRESSION_MODES(imode), ny_globals(isize),                &
                                              data_types(itype), storages(itype),                                &
                                              is_mode_supported(compressors(ic), imode, platforms(ip)))
                            enddo
                        enddo
                    enddo
                enddo
            enddo
        enddo
    endblock

    call test%finalize()

contains

    subroutine add_backend(b)
    !! Registers backend to test
        type(dtfft_backend_t),  intent(in)  :: b !! Backend to register
        n_backends = n_backends + 1
        backends(n_backends) = b
    end subroutine add_backend

    subroutine add_compressor(lib, host_modes, cuda_modes)
    !! Registers compressor to test
        type(dtfft_compression_lib_t),  intent(in)  :: lib                  !! Compression library
        logical,                        intent(in)  :: host_modes(N_MODES)  !! Modes supported on host
        logical,                        intent(in)  :: cuda_modes(N_MODES)  !! Modes supported on device
        n_compressors = n_compressors + 1
        compressors(n_compressors) = compressor_info(lib, host_modes, cuda_modes)
    end subroutine add_compressor

    logical function is_mode_supported(info, imode, platform)
    !! Whether compressor supports given mode on given platform
        type(compressor_info),  intent(in)  :: info     !! Compressor description
        integer(int32),         intent(in)  :: imode    !! Index of a mode in `VALID_COMPRESSION_MODES`
        type(dtfft_platform_t), intent(in)  :: platform !! Target platform

        if ( platform == DTFFT_PLATFORM_HOST ) then
            is_mode_supported = info%host_modes(imode)
        else
            is_mode_supported = info%cuda_modes(imode)
        endif
    end function is_mode_supported

    pure subroutine local_size(n_global, nprocs, rank, lstart, lcount)
    !! Computes local slice of `n_global` for a given rank.
    !! Mirrors dtfft_pencil get_local_size: the last `res` ranks get one extra
        integer(int32), intent(in)  :: n_global !! Global number of elements
        integer(int32), intent(in)  :: nprocs   !! Number of processes
        integer(int32), intent(in)  :: rank     !! Rank to compute slice for
        integer(int32), intent(out) :: lstart   !! Start index of a slice, 0-based
        integer(int32), intent(out) :: lcount   !! Number of elements in a slice
        integer(int32) :: res, base, i

        base   = n_global / nprocs
        res    = mod(n_global, nprocs)
        lcount = base + merge(1, 0, rank >= nprocs - res)
        lstart = 0
        do i = 0, rank - 1
            lstart = lstart + base + merge(1, 0, i >= nprocs - res)
        end do
    end subroutine local_size

    subroutine run_case(b, platform, info, mode, ny_global, base_type, base_storage, is_supported)
    !! Runs single exchange and checks obtained data
        type(dtfft_backend_t),          intent(in)  :: b            !! Backend to test
        type(dtfft_platform_t),         intent(in)  :: platform     !! Platform to run on
        type(compressor_info),          intent(in)  :: info         !! Compressor to test
        type(dtfft_compression_mode_t), intent(in)  :: mode         !! Compression mode
        integer(int32),                 intent(in)  :: ny_global    !! Global number of columns
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type    !! MPI data type
        integer(int64),                 intent(in)  :: base_storage !! Size of a single element in bytes
        logical,                        intent(in)  :: is_supported !! Whether combination is expected to work

        integer(int32), allocatable :: send_counts(:), recv_counts(:)
        integer(int32), allocatable :: send_displs(:), recv_displs(:)
        integer(int32), allocatable :: pack_locals(:,:), unpack_locals(:,:)
        class(abstract_backend),    allocatable :: backend
        class(abstract_kernel),     allocatable :: pack, unpack
        class(abstract_compressor), allocatable :: compressor
        type(backend_helper)    :: helper
        type(kernel_type_t)     :: pack_kernel_type
        type(dtfft_stream_t)    :: stream
        type(c_ptr)     :: in, out, aux, host
        integer(int32)  :: r, mpi_ierr, err_local
        integer(int32)  :: ny_local, ny_start, cy_start, cy_count
        integer(int32)  :: src_ny_local, total_send, total_recv, buffer_elems
        integer(int64)  :: aux_bytes, buffer_bytes
        logical         :: is_float, is_complex, supported_type
        real(real64)    :: error, max_error, max_error_global, threshold, rate

        is_float = any(GET_MPI_VALUE(base_type) == [GET_MPI_VALUE(MPI_REAL), GET_MPI_VALUE(MPI_COMPLEX)])
        is_complex = any(GET_MPI_VALUE(base_type) == [GET_MPI_VALUE(MPI_COMPLEX), GET_MPI_VALUE(MPI_DOUBLE_COMPLEX)])
        ! ---- compressor is shared by pack and unpack kernels, exactly as reshape handle does ----
        err_local = create_compressor(config_for_mode(info%lib, mode, is_float), 2_int8, platform, &
                                      base_type, DIMS_PERMUTE_NONE, compressor)
        if ( .not. is_supported ) then
            ! Unsupported combination must be rejected right here instead of failing
            ! somewhere inside compression of the very first block
            supported_type = err_local == DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED .or. &
                             err_local == DTFFT_ERROR_COMPRESSION_HOST_NOT_SUPPORTED .or. &
                             err_local == DTFFT_ERROR_COMPRESSION_INVALID_MODE
            call test%check("Unsupported compressor combination was accepted", supported_type)

            if ( allocated(compressor) ) deallocate( compressor )
            n_skipped = n_skipped + 1
            return
        endif
        call test%check("Failed to create compressor: "//dtfft_get_error_string(err_local), &
                        err_local, DTFFT_SUCCESS)

        ! ---- data distribution: `ny_local` columns of `NX` scalars each ----
        call local_size(ny_global, comm_size, comm_rank, ny_start, ny_local)

        allocate( send_counts(comm_size), recv_counts(comm_size) )
        allocate( send_displs(comm_size), recv_displs(comm_size) )
        allocate( pack_locals(5, comm_size), unpack_locals(5, comm_size) )

        send_displs(1) = 0
        do r = 0, comm_size - 1
            call local_size(ny_local, comm_size, r, cy_start, cy_count)
            send_counts(r + 1) = NX * cy_count
            if ( r > 0 ) send_displs(r + 1) = send_displs(r) + send_counts(r)

            call check_block(cy_count)
            pack_locals(1, r + 1) = NX
            pack_locals(2, r + 1) = cy_count
            pack_locals(3, r + 1) = 1
            pack_locals(4:5, r + 1) = send_displs(r + 1)
        enddo

        call MPI_Alltoall(send_counts, 1, MPI_INTEGER, recv_counts, 1, MPI_INTEGER, MPI_COMM_WORLD, mpi_ierr)

        recv_displs(1) = 0
        do r = 1, comm_size
            if ( r > 1 ) recv_displs(r) = recv_displs(r - 1) + recv_counts(r - 1)
            unpack_locals(1, r) = NX
            unpack_locals(2, r) = recv_counts(r) / NX
            unpack_locals(3, r) = 1
            unpack_locals(4:5, r) = recv_displs(r)
        enddo

        total_send = sum(send_counts)
        total_recv = sum(recv_counts)
        buffer_elems = max(total_send, total_recv, 1)
        buffer_bytes = base_storage * buffer_elems

        comms(1) = MPI_COMM_WORLD
        call helper%create(platform, MPI_COMM_WORLD, comms, is_backend_nccl(b), pencils)

        call allocate_backend(b, backend)
        call backend%create(b, helper, platform, 1_int8, &
                            send_displs, send_counts, recv_displs, recv_counts, base_storage)

        ! NCCL backend compresses all blocks in a single call, so its pack kernel
        ! has to process the whole local slab at once
        pack_kernel_type = KERNEL_COPY_PIPELINED
        if ( is_backend_nccl(b) ) pack_kernel_type = KERNEL_COPY

        call allocate_kernel(platform, pack)
        call allocate_kernel(platform, unpack)
        call pack%create([NX, max(ny_local, 1)], DTFFT_ESTIMATE, base_storage, pack_kernel_type,  &
                         pack_locals, with_compression=.true.)
        call unpack%create([NX, max(total_recv / NX, 1)], DTFFT_ESTIMATE, base_storage, KERNEL_COPY_PIPELINED, &
                           unpack_locals, with_decompression=.true.)
        call pack%set_compressor(compressor)
        call unpack%set_compressor(compressor)
        call backend%set_pack_kernel(pack)
        call backend%set_unpack_kernel(unpack)

        ! ---- buffers. `in` and `out` are both used as a workspace by compressed backend ----
        in  = alloc_buffer(platform, buffer_bytes)
        out = alloc_buffer(platform, buffer_bytes)
        aux_bytes = backend%get_aux_bytes()
        if ( aux_bytes <= 0 ) then
            INTERNAL_ERROR("Compressed backend reported no aux buffer")
        endif
        aux = alloc_buffer(platform, aux_bytes)
        ! Reference data is always built on host and staged to the platform buffers
        host = mem_alloc_host(buffer_bytes)

        call fill(host, buffer_elems, comm_rank, is_float, is_complex)
        call copy_to_platform(platform, in, host, buffer_bytes)
        call wipe(host, buffer_elems, is_float, is_complex)
        call copy_to_platform(platform, out, host, buffer_bytes)

        stream = NULL_STREAM
#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL( cudaStreamCreate(stream) )
        endif
#endif

        call backend%execute(in, out, stream, aux, EXEC_BLOCKING, err_local)
        call test%check("backend execute", err_local, DTFFT_SUCCESS)

#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL( cudaStreamSynchronize(stream) )
            CUDA_CALL( cudaDeviceSynchronize() )
        endif
#endif

        call copy_from_platform(platform, host, out, buffer_bytes)

        ! ---- every received block must match data that sender owned ----
        max_error = 0._real64
        do r = 0, comm_size - 1
            if ( recv_counts(r + 1) == 0 ) cycle
            call local_size(ny_global, comm_size, r, ny_start, src_ny_local)
            call local_size(src_ny_local, comm_size, comm_rank, cy_start, cy_count)
            call test%check("Expected block size differs from received one", &
                            NX * cy_count == recv_counts(r + 1))
            error = block_error(host, recv_displs(r + 1), recv_counts(r + 1), r, &
                                NX * cy_start, is_float, is_complex)
            max_error = max(max_error, error)
        enddo
        call MPI_Allreduce(max_error, max_error_global, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, mpi_ierr)
        threshold = error_threshold(mode)
        block
            type(string) :: rep_string
            rep_string = get_report_string(b, platform, info, mode, ny_global, base_type, &
                                           "FAILED", max_error_global, 0.0_real64)
            call test%check(rep_string%raw, max_error_global <= threshold)
            call rep_string%destroy()
        end block
        rate = compressor%get_average_rate()

        if ( max_error_global <= threshold ) then
            n_passed = n_passed + 1
            if ( comm_rank == 0 ) call report(b, platform, info, mode, ny_global, base_type, &
                                              "PASSED", max_error, rate)
        endif

#ifdef DTFFT_WITH_CUDA
        if ( platform == DTFFT_PLATFORM_CUDA ) then
            CUDA_CALL( cudaStreamDestroy(stream) )
        endif
#endif
        call pack%destroy()
        call unpack%destroy()
        deallocate( pack, unpack )
        call compressor%destroy()
        deallocate( compressor )
        call backend%destroy()
        deallocate( backend )
        call helper%destroy()
        call free_buffer(platform, in)
        call free_buffer(platform, out)
        call free_buffer(platform, aux)
        call mem_free_host(host)
        deallocate( send_counts, recv_counts, send_displs, recv_displs )
        deallocate( pack_locals, unpack_locals )
    end subroutine run_case

    function get_report_string(b, platform, info, mode, ny_global, base_type, status, error, rate) &
        result(report_string)
        type(dtfft_backend_t),          intent(in)  :: b            !! Tested backend
        type(dtfft_platform_t),         intent(in)  :: platform     !! Tested platform
        type(compressor_info),          intent(in)  :: info         !! Tested compressor
        type(dtfft_compression_mode_t), intent(in)  :: mode         !! Tested compression mode
        integer(int32),                 intent(in)  :: ny_global    !! Global number of columns
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type    !! MPI data type
        character(len=*),               intent(in)  :: status       !! Case status
        real(real64),                   intent(in)  :: error        !! Obtained error
        real(real64),                   intent(in)  :: rate         !! Obtained compression rate
        type(string)                                :: report_string
        type(string) :: name

        name = get_compressor_string(info%lib)
        report_string = string("  "//status//" ["//get_platform_string(platform)//"/"//dtfft_get_backend_string(b)//"] "// &
            name%raw//": type = "//trim(data_type_name(base_type))//", mode = "//to_str(mode%val)//                 &
            ", n = "//to_str(ny_global)//", error = "//to_str(error, .true.)//", rate = "//to_str(rate, .true.))
        call name%destroy()
    end function get_report_string

    subroutine report(b, platform, info, mode, ny_global, base_type, status, error, rate)
    !! Reports result of a single case
        type(dtfft_backend_t),          intent(in)  :: b            !! Tested backend
        type(dtfft_platform_t),         intent(in)  :: platform     !! Tested platform
        type(compressor_info),          intent(in)  :: info         !! Tested compressor
        type(dtfft_compression_mode_t), intent(in)  :: mode         !! Tested compression mode
        integer(int32),                 intent(in)  :: ny_global    !! Global number of columns
        TYPE_MPI_DATATYPE,              intent(in)  :: base_type    !! MPI data type
        character(len=*),               intent(in)  :: status       !! Case status
        real(real64),                   intent(in)  :: error        !! Obtained error
        real(real64),                   intent(in)  :: rate         !! Obtained compression rate
        type(string) :: rep_string

        if ( comm_rank /= 0 ) return
        rep_string = get_report_string(b, platform, info, mode, ny_global, base_type, status, error, rate)
        write(output_unit, "(a)") rep_string%raw
        call rep_string%destroy()
    end subroutine report

    function data_type_name(base_type) result(name)
        TYPE_MPI_DATATYPE, intent(in) :: base_type
        character(len=12) :: name

        select case ( GET_MPI_VALUE(base_type) )
        case ( GET_MPI_VALUE(MPI_REAL) )
            name = "real32"
        case ( GET_MPI_VALUE(MPI_REAL8) )
            name = "real64"
        case ( GET_MPI_VALUE(MPI_COMPLEX) )
            name = "complex32"
        case ( GET_MPI_VALUE(MPI_DOUBLE_COMPLEX) )
            name = "complex64"
        case default
            name = "unknown"
        endselect
    end function data_type_name

    subroutine check_block(cy_count)
    !! Makes sure block is either empty or big enough to be compressed
        integer(int32), intent(in)  :: cy_count !! Number of columns in a block

        if ( cy_count == 0 .or. cy_count >= MIN_COLUMNS ) return
        INTERNAL_ERROR("Test is broken: block of "//to_str(cy_count)//" columns is too small to be compressed")
    end subroutine check_block

    type(dtfft_compression_config_t) function config_for_mode(lib, mode, is_float) result(config)
    !! Creates a valid config for given `mode`
        type(dtfft_compression_lib_t),  intent(in)  :: lib      !! Compression library
        type(dtfft_compression_mode_t), intent(in)  :: mode     !! Compression mode
        logical,                        intent(in)  :: is_float !! Single precision data flag
        real(real64)    :: rate!, tolerance
        integer(int32)  :: precision

        rate = TEST_RATE
        precision = TEST_PRECISION
        ! tolerance = TEST_TOLERANCE
        ! ZFP requires rate and precision to be less than number of bits in a scalar
        if ( is_float ) then
            rate = 0.5_real64 * TEST_RATE
            precision = TEST_PRECISION / 2
            ! tolerance = tolerance * 100
        endif
        config = dtfft_compression_config_t(lib, mode, rate, precision, TEST_TOLERANCE)
    end function config_for_mode

    real(real64) function error_threshold(mode)
    !! Maximum error allowed for given compression mode
        type(dtfft_compression_mode_t), intent(in)  :: mode     !! Compression mode

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
    end function error_threshold

    pure complex(real64) function reference(rank, i)
    !! Reproducible reference values with distinct, exactly representable components
        integer(int32), intent(in)  :: rank !! Rank that owns the data
        integer(int32), intent(in)  :: i    !! Scalar index inside rank local slab, 1-based
        reference = cmplx( &
            1._real64 + real(modulo(i, 128), real64) / 128._real64 + real(rank, real64) / 16._real64, &
            -0.5_real64 + real(modulo(3 * i + rank, 128), real64) / 256._real64 + real(rank, real64) / 32._real64, &
            kind=real64)
    end function reference

    subroutine fill(ptr, n_elements, rank, is_float, is_complex)
    !! Fills host buffer with real or complex reference data
        type(c_ptr),    intent(in)  :: ptr          !! Buffer to fill
        integer(int32), intent(in)  :: n_elements   !! Number of elements in buffer
        integer(int32), intent(in)  :: rank         !! Rank that owns the data
        logical,        intent(in)  :: is_float     !! Single precision data flag
        logical,        intent(in)  :: is_complex   !! Complex data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)
        complex(real32), pointer :: pcf(:)
        complex(real64), pointer :: pcd(:)
        integer(int32) :: i

        if ( is_complex .and. is_float ) then
            call c_f_pointer(ptr, pcf, [n_elements])
            do i = 1, n_elements
                pcf(i) = cmplx(real(reference(rank, i), real32), &
                                real(aimag(reference(rank, i)), real32), kind=real32)
            enddo
        else if ( is_complex ) then
            call c_f_pointer(ptr, pcd, [n_elements])
            do i = 1, n_elements
                pcd(i) = reference(rank, i)
            enddo
        else if ( is_float ) then
            call c_f_pointer(ptr, pf, [n_elements])
            do i = 1, n_elements
                pf(i) = real(reference(rank, i), real32)
            enddo
        else
            call c_f_pointer(ptr, pd, [n_elements])
            do i = 1, n_elements
                pd(i) = real(reference(rank, i), real64)
            enddo
        endif
    end subroutine fill

    subroutine wipe(ptr, n_elements, is_float, is_complex)
    !! Overwrites host buffer with a value that cannot be produced by `reference`
        type(c_ptr),    intent(in)  :: ptr          !! Buffer to wipe
        integer(int32), intent(in)  :: n_elements   !! Number of elements in buffer
        logical,        intent(in)  :: is_float     !! Single precision data flag
        logical,        intent(in)  :: is_complex   !! Complex data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)
        complex(real32), pointer :: pcf(:)
        complex(real64), pointer :: pcd(:)

        if ( is_complex .and. is_float ) then
            call c_f_pointer(ptr, pcf, [n_elements])
            pcf(:) = cmplx(-3._real32, 7._real32, kind=real32)
        else if ( is_complex ) then
            call c_f_pointer(ptr, pcd, [n_elements])
            pcd(:) = cmplx(-3._real64, 7._real64, kind=real64)
        else if ( is_float ) then
            call c_f_pointer(ptr, pf, [n_elements])
            pf(:) = -3._real32
        else
            call c_f_pointer(ptr, pd, [n_elements])
            pd(:) = -3._real64
        endif
    end subroutine wipe

    real(real64) function block_error(ptr, displ, n_elements, src_rank, src_displ, is_float, is_complex)
    !! Returns maximum absolute difference between received block and data owned by sender
        type(c_ptr),    intent(in)  :: ptr          !! Host buffer with received data
        integer(int32), intent(in)  :: displ        !! Displacement of a block, 0-based
        integer(int32), intent(in)  :: n_elements   !! Number of elements in a block
        integer(int32), intent(in)  :: src_rank     !! Rank that sent the block
        integer(int32), intent(in)  :: src_displ    !! Displacement of a block inside sender slab
        logical,        intent(in)  :: is_float     !! Single precision data flag
        logical,        intent(in)  :: is_complex   !! Complex data flag
        real(real32),   pointer :: pf(:)
        real(real64),   pointer :: pd(:)
        complex(real32), pointer :: pcf(:)
        complex(real64), pointer :: pcd(:)
        complex(real64) :: expected
        complex(real32) :: expected_c32
        real(real32) :: block_error_r32
        integer(int32) :: i

        block_error = 0._real64
        if ( is_complex .and. is_float ) then
            call c_f_pointer(ptr, pcf, [displ + n_elements])
            do i = 1, n_elements
                expected_c32 = cmplx(real(reference(src_rank, src_displ + i), real32), &
                                     real(aimag(reference(src_rank, src_displ + i)), real32), kind=real32)
                block_error = max(block_error, real(abs(pcf(displ + i) - expected_c32), real64))
            enddo
        else if ( is_complex ) then
            call c_f_pointer(ptr, pcd, [displ + n_elements])
            do i = 1, n_elements
                expected = reference(src_rank, src_displ + i)
                block_error = max(block_error, abs(pcd(displ + i) - expected))
            enddo
        else if ( is_float ) then
            block_error_r32 = 0._real32
            call c_f_pointer(ptr, pf, [displ + n_elements])
            do i = 1, n_elements
                block_error_r32 = max(block_error_r32, &
                                       abs(pf(displ + i) - real(reference(src_rank, src_displ + i), real32)))
            enddo
            block_error = real(block_error_r32, real64)
        else
            call c_f_pointer(ptr, pd, [displ + n_elements])
            do i = 1, n_elements
                expected = reference(src_rank, src_displ + i)
                block_error = max(block_error, abs(pd(displ + i) - real(expected, real64)))
            enddo
        endif
    end function block_error

    function alloc_buffer(platform, bytes) result(ptr)
    !! Allocates buffer on a given platform
        type(dtfft_platform_t), intent(in)  :: platform !! Target platform
        integer(int64),         intent(in)  :: bytes    !! Number of bytes to allocate
        type(c_ptr)                         :: ptr      !! Allocated buffer

        if ( platform == DTFFT_PLATFORM_HOST ) then
            ptr = mem_alloc_host(bytes)
        else
#ifdef DTFFT_WITH_CUDA
            CUDA_CALL( cudaMalloc(ptr, int(bytes, c_size_t)) )
#else
            INTERNAL_ERROR("CUDA platform requested in a build without CUDA")
#endif
        endif
        if ( is_null_ptr(ptr) ) then
            INTERNAL_ERROR("Failed to allocate "//to_str(bytes)//" bytes")
        endif
    end function alloc_buffer

    subroutine free_buffer(platform, ptr)
    !! Frees buffer allocated by `alloc_buffer`
        type(dtfft_platform_t), intent(in)  :: platform !! Target platform
        type(c_ptr),            intent(in)  :: ptr      !! Buffer to free

        if ( platform == DTFFT_PLATFORM_HOST ) then
            call mem_free_host(ptr)
        else
#ifdef DTFFT_WITH_CUDA
            CUDA_CALL( cudaFree(ptr) )
#endif
        endif
    end subroutine free_buffer

    subroutine copy_to_platform(platform, dst, src, bytes)
    !! Copies data from host buffer to a platform one
        type(dtfft_platform_t), intent(in)  :: platform !! Target platform
        type(c_ptr),            intent(in)  :: dst      !! Platform buffer
        type(c_ptr),            intent(in)  :: src      !! Host buffer
        integer(int64),         intent(in)  :: bytes    !! Number of bytes to copy

        if ( platform == DTFFT_PLATFORM_HOST ) then
            call copy_host(dst, src, bytes)
        else
#ifdef DTFFT_WITH_CUDA
            CUDA_CALL( cudaMemcpy(dst, src, int(bytes, c_size_t), cudaMemcpyHostToDevice) )
#endif
        endif
    end subroutine copy_to_platform

    subroutine copy_from_platform(platform, dst, src, bytes)
    !! Copies data from platform buffer to a host one
        type(dtfft_platform_t), intent(in)  :: platform !! Target platform
        type(c_ptr),            intent(in)  :: dst      !! Host buffer
        type(c_ptr),            intent(in)  :: src      !! Platform buffer
        integer(int64),         intent(in)  :: bytes    !! Number of bytes to copy

        if ( platform == DTFFT_PLATFORM_HOST ) then
            call copy_host(dst, src, bytes)
        else
#ifdef DTFFT_WITH_CUDA
            CUDA_CALL( cudaMemcpy(dst, src, int(bytes, c_size_t), cudaMemcpyDeviceToHost) )
#endif
        endif
    end subroutine copy_from_platform

    subroutine copy_host(dst, src, bytes)
    !! Copies data between two host buffers
        type(c_ptr),    intent(in)  :: dst    !! Destination buffer
        type(c_ptr),    intent(in)  :: src    !! Source buffer
        integer(int64), intent(in)  :: bytes  !! Number of bytes to copy
        real(real32),   pointer :: pd(:), ps(:)

        call c_f_pointer(dst, pd, [bytes / FLOAT_STORAGE_SIZE])
        call c_f_pointer(src, ps, [bytes / FLOAT_STORAGE_SIZE])
        pd(:) = ps(:)
    end subroutine copy_host

    subroutine allocate_kernel(platform, kernel)
    !! Allocates kernel that corresponds to `platform`
        type(dtfft_platform_t),                 intent(in)  :: platform !! Target platform
        class(abstract_kernel), allocatable,    intent(out) :: kernel   !! Kernel to allocate

        if ( platform == DTFFT_PLATFORM_HOST ) then
            allocate( kernel_host :: kernel )
        else
#ifdef DTFFT_WITH_CUDA
            allocate( kernel_device :: kernel )
#else
            INTERNAL_ERROR("CUDA platform requested in a build without CUDA")
#endif
        endif
    end subroutine allocate_kernel

    subroutine allocate_backend(b, backend)
    !! Allocates backend of a dynamic type that corresponds to `b`
        type(dtfft_backend_t),                  intent(in)  :: b        !! Backend to allocate
        class(abstract_backend), allocatable,   intent(out) :: backend  !! Allocated backend

        if ( is_backend_nccl(b) ) then
#ifdef DTFFT_WITH_NCCL
            allocate( backend_nccl :: backend )
#else
            INTERNAL_ERROR("NCCL backend requested in a build without NCCL")
#endif
        else
            allocate( backend_mpi :: backend )
        endif
    end subroutine allocate_backend
end program test_backend_compressed
