#include "dtfft_config.h"
program test_compression
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_abstract_compressor
use dtfft_compressor
! #ifdef DTFFT_WITH_ZFP
! use dtfft_compressor_zfp,                 only: compressor_zfp
! #endif
! #ifdef DTFFT_WITH_SZ3
! use dtfft_compressor_sz3,                 only: compressor_sz3
! #endif
! #ifdef DTFFT_WITH_SZP
! use dtfft_compressor_szp,                 only: compressor_szp
! #endif
! #ifdef DTFFT_WITH_BITCOMP
! use dtfft_compressor_bitcomp,               only: compressor_bitcomp
! #endif
use dtfft_config
use dtfft_kernel_host
use dtfft_parameters
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
    integer(int32) :: ierror
    type :: compressor_t
        type(dtfft_compression_lib_t) :: lib
        logical :: has_fixed_rate
        logical :: has_lossless
    end type compressor_t
    integer(int32) :: n_compressors, iter
    type(compressor_t) :: compressors(10)


    call MPI_Init(ierror)

    n_compressors = 0
#ifdef DTFFT_WITH_ZFP
    n_compressors = n_compressors + 1; compressors(n_compressors) = compressor_t(DTFFT_COMPRESSION_LIB_ZFP, .true.,  .true.)
#endif
#ifdef DTFFT_WITH_SZ3
    n_compressors = n_compressors + 1; compressors(n_compressors) = compressor_t(DTFFT_COMPRESSION_LIB_SZ3, .false., .true.)
#endif
#ifdef DTFFT_WITH_SZP
    n_compressors = n_compressors + 1; compressors(n_compressors) = compressor_t(DTFFT_COMPRESSION_LIB_SZP, .false., .false.)
#endif

    do iter = 1, n_compressors
        call test_pack_unpack([16, 16, 16], compressors(iter))
        call test_pack_unpack([16, 16], compressors(iter))

        call test_transpose([32, 64, 128], KERNEL_PERMUTE_FORWARD, KERNEL_UNPACK, KERNEL_PERMUTE_BACKWARD, KERNEL_UNPACK, compressors(iter))
        call test_transpose([32, 64, 128], KERNEL_PACK, KERNEL_UNPACK_FORWARD, KERNEL_PACK, KERNEL_UNPACK_BACKWARD, compressors(iter))
        call test_transpose([32, 64, 128], KERNEL_PACK_FORWARD, KERNEL_UNPACK, KERNEL_PACK_BACKWARD, KERNEL_UNPACK, compressors(iter))
        call test_transpose([64, 128], KERNEL_PERMUTE_FORWARD, KERNEL_UNPACK, KERNEL_PERMUTE_BACKWARD, KERNEL_UNPACK, compressors(iter))
        call test_transpose([64, 128], KERNEL_PACK, KERNEL_UNPACK_FORWARD, KERNEL_PACK, KERNEL_UNPACK_BACKWARD, compressors(iter))
        call test_transpose([64, 128], KERNEL_PACK_FORWARD, KERNEL_UNPACK, KERNEL_PACK_BACKWARD, KERNEL_UNPACK, compressors(iter))
    enddo

    call MPI_Finalize(ierror)
contains

    subroutine test_pack_unpack(dims, comp)
        integer(int32), intent(in) :: dims(:)
        type(compressor_t), intent(in) :: comp
        type(dtfft_compression_lib_t) :: lib
        type(kernel_host) :: packer, unpacker
        class(abstract_compressor), allocatable :: compressor
        integer(int32) :: locals(5, 1)
        type(c_ptr) :: in, out, aux
        complex(real32), pointer :: in_ptr(:)
        complex(real32), allocatable :: test(:)
        integer(int64) :: compressed_sizes(1)
        integer(int32) :: i
        type(dtfft_compression_config_t) :: config
        real(real32) :: threshold

        lib = comp%lib
        locals(:, 1) = 0
        locals(1:size(dims), 1) = dims

        print*,'Testing Pack/Unpack kernels with compression, ndims = ', size(dims)

        if ( comp%has_lossless ) then
            config = dtfft_compression_config_t(lib, DTFFT_COMPRESSION_MODE_LOSSLESS)
            threshold = 1.e-5_real32
        else
            ! Error bounded compressors cannot restore data exactly.
            ! Test data grows up to `product(dims)`, so tolerance must be a loose one
            config = dtfft_compression_config_t(lib, DTFFT_COMPRESSION_MODE_FIXED_ACCURACY, tolerance=1.e-2_real64)
            threshold = 1.e-1_real32
        endif
        ierror = check_compression_config(config)

        call allocate_compressor(lib, compressor)

        if ( .not.allocated(compressor) ) then
            INTERNAL_ERROR("test_pack_unpack: Unknown compression lib")
        endif

!         if ( lib == DTFFT_COMPRESSION_LIB_ZFP ) then
! #ifdef DTFFT_WITH_ZFP
!             allocate( compressor_zfp :: compressor )
! #endif
!         else if ( lib == DTFFT_COMPRESSION_LIB_SZ3 ) then
! #ifdef DTFFT_WITH_SZ3
!             allocate( compressor_sz3 :: compressor )
! #endif
!         else if ( lib == DTFFT_COMPRESSION_LIB_SZP ) then
! #ifdef DTFFT_WITH_SZP
!             allocate( compressor_szp :: compressor )
! #endif
!         else if ( lib == DTFFT_COMPRESSION_LIB_BITCOMP ) then
! #ifdef DTFFT_WITH_BITCOMP
!             allocate( compressor_bitcomp :: compressor )
! #endif
!         endif

        ierror = compressor%create(size(dims, kind=int8), config, DTFFT_PLATFORM_HOST, MPI_COMPLEX, DIMS_PERMUTE_NONE)

        call packer%create(dims, DTFFT_ESTIMATE, COMPLEX_STORAGE_SIZE, KERNEL_PACK, locals, with_compression=.true.)
        call packer%set_compressor(compressor)

        call unpacker%create(dims, DTFFT_ESTIMATE, COMPLEX_STORAGE_SIZE, KERNEL_UNPACK, locals, with_decompression=.true.)
        call unpacker%set_compressor(compressor)

        in = mem_alloc_host(COMPLEX_STORAGE_SIZE * product(dims))
        out = mem_alloc_host(COMPLEX_STORAGE_SIZE * product(dims))
        aux = mem_alloc_host(COMPLEX_STORAGE_SIZE * product(dims))

        call c_f_pointer(in, in_ptr, [product(dims)])
        allocate(test(product(dims)))
        do i = 1, product(dims)
            in_ptr(i) = cmplx(real(i, real32), 0.0_real32)
            test(i) = in_ptr(i)
        end do

        call packer%execute(in, out, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
        call unpacker%execute(out, in, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)

        if ( any(abs(in_ptr - test) > threshold) ) then
            INTERNAL_ERROR("Test FAILED")
        end if

        call compressor%destroy()
        deallocate( compressor )
        call packer%destroy()
        call unpacker%destroy()

        call mem_free_host(in)
        call mem_free_host(out)
        call mem_free_host(aux)

        print*,'SUCCESS'
    end subroutine test_pack_unpack

    subroutine test_transpose(dims, f, fu, b, bu, lib)
        integer(int32),      intent(in) :: dims(:)
        type(kernel_type_t), intent(in) :: f, fu, b, bu
        type(compressor_t), intent(in) :: lib
        type(kernel_host) :: forward, backward, forward_unpacker, backward_unpacker
        class(abstract_compressor), allocatable :: compressor_forward, compressor_backward
        integer(int32) :: locals(5, 1)
        integer(int32), allocatable :: temp_dims(:)
        type(c_ptr) :: in, out, aux
        real(real64), pointer :: in_ptr(:), out_ptr(:)
        real(real64), allocatable :: test(:)
        integer(int64) :: compressed_sizes(1)
        integer(int32) :: i
        type(string) :: sf, sfu, sb, sbu, comp
        type(dtfft_compression_config_t) :: config
        integer(int32) :: mode
        type(dtfft_compression_mode_t) :: current_mode
        real(real64) :: frate, brate, error, rnd

        sf = get_kernel_string(f); sfu = get_kernel_string(fu)
        sb = get_kernel_string(b); sbu = get_kernel_string(bu)
        comp = get_compressor_string(lib%lib)
        print*,'Testing transpose kernels with compressor = '//comp%raw//', ndims = '//to_str(size(dims))
        print*,'    Forward method = '//sf%raw//" + "//sfu%raw
        print*,'    Backward method = '//sb%raw//" + "//sbu%raw

        call sf%destroy(); call sfu%destroy()
        call sb%destroy(); call sbu%destroy()
        call comp%destroy()

        allocate( temp_dims(size(dims)) )
        if ( size(dims) == 2 ) then
            temp_dims(:) = [dims(2), dims(1)]
        else
            temp_dims(:) = [dims(2), dims(3), dims(1)]
        end if
        locals(:, 1) = 0

        call allocate_compressor(lib%lib, compressor_forward)
        call allocate_compressor(lib%lib, compressor_backward)

        if ( .not.allocated(compressor_forward) .or. .not.allocated(compressor_backward) ) then
            INTERNAL_ERROR("test_transpose: Unknown compression lib")
        endif

!         if ( lib%lib == DTFFT_COMPRESSION_LIB_ZFP ) then
! #ifdef DTFFT_WITH_ZFP
!             allocate( compressor_zfp :: compressor_forward )
!             allocate( compressor_zfp :: compressor_backward )
! #endif
!         else if ( lib%lib == DTFFT_COMPRESSION_LIB_SZ3 ) then
! #ifdef DTFFT_WITH_SZ3
!             allocate( compressor_sz3 :: compressor_forward )
!             allocate( compressor_sz3 :: compressor_backward )
! #endif
!         else if ( lib%lib == DTFFT_COMPRESSION_LIB_SZP ) then
! #ifdef DTFFT_WITH_SZP
!             allocate( compressor_szp :: compressor_forward )
!             allocate( compressor_szp :: compressor_backward )
! #endif
!         endif

        in = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        out = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        aux = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

        call c_f_pointer(in, in_ptr, [product(dims)])
        call c_f_pointer(out, out_ptr, [product(dims)])
        allocate(test(product(dims)))

        do mode = 1, size(VALID_COMPRESSION_MODES)
            current_mode = VALID_COMPRESSION_MODES(mode)

            if ( current_mode == DTFFT_COMPRESSION_MODE_FIXED_RATE .and. .not. lib%has_fixed_rate ) cycle
            if ( current_mode == DTFFT_COMPRESSION_MODE_LOSSLESS .and. .not. lib%has_lossless ) cycle

            do i = 1, product(dims)
                call random_number(rnd)
                in_ptr(i) = rnd
                test(i) = in_ptr(i)
            end do
            print*,'        Compression mode = ',current_mode%val

            config = dtfft_compression_config_t(lib%lib, current_mode, 32.0_real64, 32, 1e-7)

            ierror = compressor_forward%create(size(dims, kind=int8), config, DTFFT_PLATFORM_HOST, MPI_REAL8, DIMS_PERMUTE_BACKWARD)
            ierror = compressor_backward%create(size(dims, kind=int8), config, DTFFT_PLATFORM_HOST, MPI_REAL8, DIMS_PERMUTE_FORWARD)

            locals(1:size(dims), 1) = dims
            call forward%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, f, locals, with_compression=.true.)
            call forward%set_compressor(compressor_forward)

            locals(1:size(dims), 1) = temp_dims
            call forward_unpacker%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, fu, locals, with_decompression=.true.)
            call forward_unpacker%set_compressor(compressor_forward)

            locals(1:size(dims), 1) = temp_dims
            call backward%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, b, locals, with_compression=.true.)
            call backward%set_compressor(compressor_backward)

            locals(1:size(dims), 1) = dims
            call backward_unpacker%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, bu, locals, with_decompression=.true.)
            call backward_unpacker%set_compressor(compressor_backward)

            if ( f == KERNEL_PERMUTE_FORWARD .or. f == KERNEL_PACK ) then
                call forward%execute(in, out, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
                call forward_unpacker%execute(out, in, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
            else
                call forward%execute(in, out, NULL_STREAM, neighbor=1, aux=aux, csize=compressed_sizes(1), sync=.true.)
                call forward_unpacker%execute(out, in, NULL_STREAM, neighbor=1, aux=aux, csize=compressed_sizes(1), sync=.true.)
            endif

            if ( b == KERNEL_PERMUTE_BACKWARD .or. b == KERNEL_PACK ) then
                call backward%execute(in, out, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
                call backward_unpacker%execute(out, in, NULL_STREAM, neighbor=1, aux=aux, csizes=compressed_sizes, sync=.true.)
            else
                call backward%execute(in, out, NULL_STREAM, neighbor=1, aux=aux, csize=compressed_sizes(1), sync=.true.)
                call backward_unpacker%execute(out, in, NULL_STREAM, neighbor=1, csize=compressed_sizes(1), aux=aux, sync=.true.)
            endif

            error = maxval(abs(in_ptr - test))
            print'(a,ES16.8)','         error = ',error
            if ( error > 1.e-6_real64) then
                INTERNAL_ERROR("Test FAILED")
            end if

            frate = compressor_forward%get_average_rate()
            brate = compressor_backward%get_average_rate()
            print'(a,ES16.8,ES16.8)','         rates :',frate,brate
            print*,'      SUCCESS'

            call compressor_forward%destroy()
            call compressor_backward%destroy()
            call forward%destroy()
            call backward%destroy()
            call forward_unpacker%destroy()
            call backward_unpacker%destroy()
        enddo

        deallocate(compressor_forward, compressor_backward)
        call mem_free_host(in)
        call mem_free_host(out)
        call mem_free_host(aux)
        deallocate(temp_dims, test)

        
    end subroutine test_transpose
end program test_compression