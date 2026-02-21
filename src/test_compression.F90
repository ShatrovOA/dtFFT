program test_compression
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_abstract_compressor
use dtfft_compressor_zfp
use dtfft_config
use dtfft_kernel_host
use dtfft_parameters
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
integer(int32) :: ierror
type(dtfft_stream_t) :: stream

    call MPI_Init(ierror)
    ierror = init_internal()

    call test_pack_unpack([16, 16, 16])
    call test_pack_unpack([16, 16])

    call test_transpose([32, 64, 128], KERNEL_PERMUTE_FORWARD, KERNEL_UNPACK, KERNEL_PERMUTE_BACKWARD, KERNEL_UNPACK)
    call test_transpose([32, 64, 128], KERNEL_PACK, KERNEL_UNPACK_FORWARD, KERNEL_PACK, KERNEL_UNPACK_BACKWARD)
    call test_transpose([32, 64, 128], KERNEL_PACK_FORWARD, KERNEL_UNPACK, KERNEL_PACK_BACKWARD, KERNEL_UNPACK)
    call test_transpose([64, 128], KERNEL_PERMUTE_FORWARD, KERNEL_UNPACK, KERNEL_PERMUTE_BACKWARD, KERNEL_UNPACK)
    call test_transpose([64, 128], KERNEL_PACK, KERNEL_UNPACK_FORWARD, KERNEL_PACK, KERNEL_UNPACK_BACKWARD)
    call test_transpose([64, 128], KERNEL_PACK_FORWARD, KERNEL_UNPACK, KERNEL_PACK_BACKWARD, KERNEL_UNPACK)

    call MPI_Finalize(ierror)
contains

    subroutine test_pack_unpack(dims)
        integer(int32), intent(in) :: dims(:)
        type(kernel_host) :: packer, unpacker
        type(compressor_zfp) :: compressor
        integer(int32) :: locals(5, 1)
        type(c_ptr) :: in, out, aux
        complex(real32), pointer :: in_ptr(:)
        complex(real32), allocatable :: test(:)
        integer(int32) :: compressed_sizes(1)
        integer(int32) :: i

        locals(:, 1) = 0
        locals(1:size(dims), 1) = dims

        print*,'Testing Pack/Unpack kernels with compression, ndims = ', size(dims)

        ierror = compressor%create(size(dims, kind=int8), DEFAULT_COMPRESSION_CONFIG, DTFFT_PLATFORM_HOST, MPI_COMPLEX, COMPLEX_STORAGE_SIZE, DIMS_PERMUTE_NONE)

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
        call unpacker%execute(out, in, NULL_STREAM, aux=aux, sync=.true.)

        if ( any(abs(in_ptr - test) > 1.e-5_real32) ) then
            INTERNAL_ERROR("Test FAILED")
        end if

        call compressor%destroy()
        call packer%destroy()
        call unpacker%destroy()

        call mem_free_host(in)
        call mem_free_host(out)
        call mem_free_host(aux)

        print*,'SUCCESS'
    end subroutine test_pack_unpack

    subroutine test_transpose(dims, f, fu, b, bu)
        integer(int32),      intent(in) :: dims(:)
        type(kernel_type_t), intent(in) :: f, fu, b, bu
        type(kernel_host) :: forward, backward, forward_unpacker, backward_unpacker
        type(compressor_zfp) :: compressor_forward, compressor_backward
        integer(int32) :: locals(5, 1)
        integer(int32), allocatable :: temp_dims(:)
        type(c_ptr) :: in, out, aux
        real(real64), pointer :: in_ptr(:), out_ptr(:)
        real(real64), allocatable :: test(:)
        integer(int32) :: compressed_sizes(1)
        integer(int32) :: i
        type(string) :: sf, sfu, sb, sbu

        sf = get_kernel_string(f); sfu = get_kernel_string(fu)
        sb = get_kernel_string(b); sbu = get_kernel_string(bu)
        print*,'Testing transpose kernels with compression, ndims = '//to_str(size(dims))
        print*,'    Forward method = '//sf%raw//" + "//sfu%raw
        print*,'    Backward method = '//sb%raw//" + "//sbu%raw

        call sf%destroy(); call sfu%destroy()
        call sb%destroy(); call sbu%destroy()

        allocate( temp_dims(size(dims)) )
        if ( size(dims) == 2 ) then
            temp_dims(:) = [dims(2), dims(1)]
        else
            temp_dims(:) = [dims(2), dims(3), dims(1)]
        end if
        locals(:, 1) = 0

        ierror = compressor_forward%create(size(dims, kind=int8), DEFAULT_COMPRESSION_CONFIG, DTFFT_PLATFORM_HOST, MPI_REAL8, DOUBLE_STORAGE_SIZE, DIMS_PERMUTE_BACKWARD)
        ierror = compressor_backward%create(size(dims, kind=int8), DEFAULT_COMPRESSION_CONFIG, DTFFT_PLATFORM_HOST, MPI_REAL8, DOUBLE_STORAGE_SIZE, DIMS_PERMUTE_FORWARD)

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

        in = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        out = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        aux = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

        call c_f_pointer(in, in_ptr, [product(dims)])
        call c_f_pointer(out, out_ptr, [product(dims)])
        allocate(test(product(dims)))
        do i = 1, product(dims)
            in_ptr(i) = real(i, real64)
            test(i) = in_ptr(i)
        end do

        if ( f == KERNEL_PERMUTE_FORWARD .or. f == KERNEL_PACK ) then
            call forward%execute(in, out, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
            call forward_unpacker%execute(out, in, NULL_STREAM, aux=aux, sync=.true.)
        else
            call forward%execute(in, out, NULL_STREAM, neighbor=1, aux=aux, csize=compressed_sizes(1), sync=.true.)
            call forward_unpacker%execute(out, in, NULL_STREAM, aux=aux, sync=.true.)
        endif

        if ( b == KERNEL_PERMUTE_BACKWARD .or. b == KERNEL_PACK ) then
            call backward%execute(in, out, NULL_STREAM, aux=aux, csizes=compressed_sizes, sync=.true.)
            call backward_unpacker%execute(out, in, NULL_STREAM, neighbor=1, aux=aux, sync=.true.)
        else
            call backward%execute(in, out, NULL_STREAM, neighbor=1, aux=aux, csize=compressed_sizes(1), sync=.true.)
            call backward_unpacker%execute(out, in, NULL_STREAM, neighbor=1, aux=aux, sync=.true.)
        endif

        if ( any(abs(in_ptr - test) > 1.e-15_real64) ) then
            INTERNAL_ERROR("Test FAILED")
        end if

        call compressor_forward%destroy()
        call compressor_backward%destroy()
        call forward%destroy()
        call backward%destroy()
        call forward_unpacker%destroy()
        call backward_unpacker%destroy()

        call mem_free_host(in)
        call mem_free_host(out)
        call mem_free_host(aux)
        deallocate(temp_dims, test)

        print*,'SUCCESS'
    end subroutine test_transpose
end program test_compression