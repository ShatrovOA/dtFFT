program test_host_kernels
use iso_fortran_env
use iso_c_binding
use dtfft_parameters
use dtfft_abstract_kernel
use dtfft_kernel_host
use dtfft_utils
implicit none
#include "_dtfft_private.h"
    real(real32), pointer :: in(:), out(:), gold(:), temp(:)
    type(c_ptr) :: in_ptr, out_ptr, gold_ptr, temp_ptr
    integer(int32) :: dims3(3, 3), dims2(2, 3), i
    integer(int32) :: x, y, z, d
    integer(int32) :: product_dims


    dims3 = reshape([256, 64, 128, 1, 13, 19, 0, 3, 15], [3, 3])
    dims2 =reshape([903, 574, 1, 77, 0, 13], [2, 3])

    do d = 1, size(dims3, dim=2)
        call run_permute_forward(dims3(:, d))
        call run_permute_backward(dims3(:, d))
        call run_permute_backward_start(dims3(:, d))
        call run_permute_backward_end(dims3(:, d))
        call run_pack_unpack(dims3(:, d))
    enddo

    do d = 1, size(dims2, dim=2)
        call run_permute_forward(dims2(:, d))
        call run_pack_unpack(dims2(:, d))
    enddo

contains

    subroutine execute_test(test_name, kernel_type, dims, block_size, access_mode, threshold, locals, ptr_in)
        character(len=*), intent(in) :: test_name
        type(kernel_type_t), intent(in) :: kernel_type
        integer(int32), intent(in) :: dims(:)
        type(host_kernel_t), intent(in) :: block_size
        type(dtfft_access_mode_t), optional, intent(in) :: access_mode
        integer(int32), optional, intent(in) :: threshold
        integer(int32), intent(in), optional :: locals(:,:)
        type(c_ptr), optional, intent(in) :: ptr_in
        type(kernel_host) :: kernel
        type(kernel_options) :: options
        type(dtfft_access_mode_t) :: access_mode_
        type(c_ptr) :: ptr_in_
        integer(int32) :: threshold_
        real(real32) :: ts, te

        ! allocate(options)
        OPTVAL(access_mode_, DTFFT_ACCESS_MODE_WRITE, access_mode)
        OPTVAL(threshold_, 32, threshold)
        OPTVAL(ptr_in_, in_ptr, ptr_in)

        options%slot_kind = KERNEL_SLOT_HOST
        options%host_kernel = block_size%val
        options%access_mode = access_mode_
        options%co_threshold = threshold_

        call kernel%create(dims, DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, kernel_type, locals, options=options)
        call cpu_time(ts)
        call kernel%execute(ptr_in_, out_ptr, NULL_STREAM, 1)
        call cpu_time(te)
        call compare(test_name)
        print*,"    "//test_name//": SUCCESS", te - ts
        call kernel%destroy()
    end subroutine execute_test

    subroutine execute_tests(test_name, kernel_type, dims, switch_access_modes, locals, ptr_in)
        character(len=*), intent(in) :: test_name
        type(kernel_type_t), intent(in) :: kernel_type
        integer(int32), intent(in) :: dims(:)
        logical, optional, intent(in) :: switch_access_modes
        integer(int32), intent(in), optional :: locals(:,:)
        type(c_ptr), optional, intent(in) :: ptr_in
        logical :: switch_access_modes_

        OPTVAL(switch_access_modes_, .true., switch_access_modes)
        if ( switch_access_modes_ ) then
            call execute_test(test_name//"_write", kernel_type, dims, HOST_KERNEL_BASE, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_write_block_4", kernel_type, dims, HOST_KERNEL_BLOCK_4, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_write_block_8", kernel_type, dims, HOST_KERNEL_BLOCK_8, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_write_block_16", kernel_type, dims, HOST_KERNEL_BLOCK_16, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_write_block_32", kernel_type, dims, HOST_KERNEL_BLOCK_32, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_write_block_64", kernel_type, dims, HOST_KERNEL_BLOCK_64, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)

            call execute_test(test_name//"_read", kernel_type, dims, HOST_KERNEL_BASE, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_read_block_4", kernel_type, dims, HOST_KERNEL_BLOCK_4, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_read_block_8", kernel_type, dims, HOST_KERNEL_BLOCK_8, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_read_block_16", kernel_type, dims, HOST_KERNEL_BLOCK_16, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_read_block_32", kernel_type, dims, HOST_KERNEL_BLOCK_32, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_read_block_64", kernel_type, dims, HOST_KERNEL_BLOCK_64, DTFFT_ACCESS_MODE_READ, locals=locals, ptr_in=ptr_in)

            call execute_test(test_name//"_cache_oblivious_4", kernel_type, dims, HOST_KERNEL_CACHE_OBLIVIOUS, DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS, threshold=4, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_cache_oblivious_16", kernel_type, dims, HOST_KERNEL_CACHE_OBLIVIOUS, DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS, threshold=16, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_cache_oblivious_32", kernel_type, dims, HOST_KERNEL_CACHE_OBLIVIOUS, DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS, threshold=32, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_cache_oblivious_1024", kernel_type, dims, HOST_KERNEL_CACHE_OBLIVIOUS, DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS, threshold=1024, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_cache_oblivious_4096", kernel_type, dims, HOST_KERNEL_CACHE_OBLIVIOUS, DTFFT_ACCESS_MODE_CACHE_OBLIVIOUS, threshold=4096, locals=locals, ptr_in=ptr_in)
        else
            call execute_test(test_name, kernel_type, dims, HOST_KERNEL_BASE, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_block_4", kernel_type, dims, HOST_KERNEL_BLOCK_4, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_block_8", kernel_type, dims, HOST_KERNEL_BLOCK_8, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_block_16", kernel_type, dims, HOST_KERNEL_BLOCK_16, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_block_32", kernel_type, dims, HOST_KERNEL_BLOCK_32, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
            call execute_test(test_name//"_block_64", kernel_type, dims, HOST_KERNEL_BLOCK_64, DTFFT_ACCESS_MODE_WRITE, locals=locals, ptr_in=ptr_in)
        endif
    end subroutine execute_tests

    subroutine run_permute_forward(dims)
        integer(int32), intent(in) :: dims(:)
        integer(int32) :: locals(5, 1)
        integer(int32) :: unpack_dims(3)

        print*,''
        print'(a,*(i4))','Testing permute_forward kernels: dims = ',dims
        print*,''

        product_dims = product(dims)

        in_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        out_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        gold_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        call c_f_pointer(in_ptr, in, [product_dims])
        call c_f_pointer(out_ptr, out, [product_dims])
        call c_f_pointer(gold_ptr, gold, [product_dims])

        do i = 1, size(in)
            in(i) = real(i, real32)
        enddo

        if ( size(dims) == 3 ) then
            do x = 0, dims(1) - 1
                do z = 0, dims(3) - 1
                    do y = 0, dims(2) - 1
                        gold(x * dims(2) * dims(3) + 1 + z * dims(2) + y) = in(z * dims(1) * dims(2) + 1 + x + y * dims(1))
                    enddo
                enddo
            enddo
        else
            do x = 0, dims(1) - 1
                do y = 0, dims(2) - 1
                    gold(x * dims(2) + 1 + y) = in(y * dims(1) + 1 + x)
                enddo
            enddo
        endif

        call execute_tests("permute_forward", KERNEL_PERMUTE_FORWARD, dims)

        locals(1:size(dims), 1) = dims
        locals(4, 1) = 0
        locals(5, 1) = 0

        call execute_tests("pack_forward", KERNEL_PACK_FORWARD, dims, locals=locals)

        if ( size(dims) == 2 ) then
            unpack_dims = [dims(2), dims(1), 1]
        else
            unpack_dims = [dims(2), dims(3), dims(1)]
        endif
        locals(1:size(unpack_dims), 1) = unpack_dims

        call execute_tests("unpack_forward", KERNEL_UNPACK_FORWARD, unpack_dims(1:size(dims)), locals=locals)

        nullify(in, out, gold)
        call mem_free_host(in_ptr)
        call mem_free_host(out_ptr)
        call mem_free_host(gold_ptr)
    end subroutine run_permute_forward

    subroutine run_permute_backward(dims)
        integer(int32), intent(in) :: dims(:)
        integer(int32) :: locals(5, 1)
        integer(int32) :: unpack_dims(3)

        product_dims = product(dims)
        print*,''
        print*,'Testing permute_backward kernels, dims = ',dims
        print*,''
        in_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        out_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        gold_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        call c_f_pointer(in_ptr, in, [product_dims])
        call c_f_pointer(out_ptr, out, [product_dims])
        call c_f_pointer(gold_ptr, gold, [product_dims])

        do i = 1, size(in)
            in(i) = real(i, real32)
        enddo

        if ( size(dims) == 3 ) then
            do z = 0, dims(3) - 1
                do y = 0, dims(2) - 1
                    do x = 0, dims(1) - 1
                        gold(y * dims(3) * dims(1) + z + 1 + x * dims(3)) = in(z * dims(1) * dims(2) + 1 + y * dims(1) + x)
                    enddo
                enddo
            enddo
        else
            do x = 0, dims(1) - 1
                do y = 0, dims(2) - 1
                    gold(x * dims(2) + 1 + y) = in(y * dims(1) + 1 + x)
                enddo
            enddo
        endif

        call execute_tests("permute_backward", KERNEL_PERMUTE_BACKWARD, dims)

        locals(1:size(dims), 1) = dims
        locals(4, 1) = 0
        locals(5, 1) = 0

        call execute_tests("pack_backward", KERNEL_PACK_BACKWARD, dims, locals=locals)

        unpack_dims = [dims(3), dims(1), dims(2)]
        locals(1:size(unpack_dims), 1) = unpack_dims

        call execute_tests("unpack_backward", KERNEL_UNPACK_BACKWARD, unpack_dims(1:size(dims)), locals=locals)

        nullify(in, out, gold)
        call mem_free_host(in_ptr)
        call mem_free_host(out_ptr)
        call mem_free_host(gold_ptr)
    end subroutine run_permute_backward

    subroutine run_permute_backward_start(dims)
        integer(int32), intent(in) :: dims(:)

        print*,''
        print*,'Testing permute_backward_start kernels, dims = ',dims
        print*,''

        product_dims = product(dims)
        in_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        out_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        gold_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        call c_f_pointer(in_ptr, in, [product_dims])
        call c_f_pointer(out_ptr, out, [product_dims])
        call c_f_pointer(gold_ptr, gold, [product_dims])

        do i = 1, size(in)
            in(i) = real(i, real32)
        enddo

        do z = 0, dims(3) - 1
            do y = 0, dims(2) - 1
                do x = 0, dims(1) - 1
                    gold(y * dims(3) + z + 1 + x * dims(3) * dims(2)) = in(z * dims(1) * dims(2) + 1 + y * dims(1) + x)
                enddo
            enddo
        enddo

        call execute_tests("permute_backward_start", KERNEL_PERMUTE_BACKWARD_START, dims)

        nullify(in, out, gold)
        call mem_free_host(in_ptr)
        call mem_free_host(out_ptr)
        call mem_free_host(gold_ptr)
    end subroutine run_permute_backward_start

    subroutine run_permute_backward_end(dims)
        integer(int32), intent(in) :: dims(:)
        integer(int32) :: temp_dims(size(dims))
        integer(int32) :: locals(5, 1)
        type(kernel_host) :: temp_kernel

        print*,''
        print*,'Testing permute_backward_end kernels, dims = ',dims
        print*,''

        product_dims = product(dims)
        in_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        temp_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        out_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        gold_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        call c_f_pointer(in_ptr, in, [product_dims])
        call c_f_pointer(temp_ptr, temp, [product_dims])
        call c_f_pointer(out_ptr, out, [product_dims])
        call c_f_pointer(gold_ptr, gold, [product_dims])

        do i = 1, size(in)
            in(i) = real(i, real32)
        enddo
        call temp_kernel%create(dims, DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)
        call temp_kernel%execute(in_ptr, gold_ptr, NULL_STREAM)
        call temp_kernel%destroy()
        call temp_kernel%create(dims, DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD_START)
        call temp_kernel%execute(in_ptr, temp_ptr, NULL_STREAM)
        call temp_kernel%destroy()

        temp_dims = [dims(3), dims(1), dims(2)]

        locals(1:size(dims), 1) = temp_dims
        locals(4, 1) = 0
        locals(5, 1) = 0

        call execute_tests("permute_backward_end", KERNEL_PERMUTE_BACKWARD_END, temp_dims, locals=locals, ptr_in=temp_ptr)

        nullify(in, out, gold, temp)
        call mem_free_host(in_ptr)
        call mem_free_host(out_ptr)
        call mem_free_host(gold_ptr)
        call mem_free_host(temp_ptr)
    end subroutine run_permute_backward_end

    subroutine run_pack_unpack(dims)
        integer(int32), intent(in) :: dims(:)
        integer(int32) :: locals(5, 1)

        print*,''
        print'(a,*(i4))','Testing pack/unpack kernels, dims = ',dims
        print*,''

        product_dims = product(dims)
        in_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        out_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        gold_ptr = mem_alloc_host(FLOAT_STORAGE_SIZE * product_dims)
        call c_f_pointer(in_ptr, in, [product_dims])
        call c_f_pointer(out_ptr, out, [product_dims])
        call c_f_pointer(gold_ptr, gold, [product_dims])

        do i = 1, size(in)
            in(i) = real(i, real32)
            gold(i) = in(i)
        enddo
        ! gold(:) = -1._real32
        out(:) = -1._real32

        locals(1:size(dims), 1) = dims
        locals(4, 1) = 0
        locals(5, 1) = 0

        call execute_tests("unpack", KERNEL_UNPACK, dims, .false., locals=locals, ptr_in=in_ptr)
        call execute_tests("pack", KERNEL_PACK, dims, .false., locals=locals, ptr_in=in_ptr)

        nullify(in, out, gold)
        call mem_free_host(in_ptr)
        call mem_free_host(out_ptr)
        call mem_free_host(gold_ptr)
    end subroutine run_pack_unpack

    subroutine compare(test_name)
        character(len=*), intent(in) :: test_name
        do i = 1, size(gold)
            if ( abs(out(i) - gold(i)) > 1e-6 ) then
                print*, i, out(i), gold(i)
                error stop "Test failed: "//test_name
            endif
            out(i) = -1._real32
        enddo
    end subroutine compare
end program test_host_kernels
