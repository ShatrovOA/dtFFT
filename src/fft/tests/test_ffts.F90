#include "dtfft_config.h"
program test_ffts
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_executor
use dtfft_abstract_kernel
use dtfft_executor
use dtfft_errors
use dtfft_kernel_host
#ifdef DTFFT_WITH_CUDA
use dtfft_kernel_device
use dtfft_interface_cuda_runtime
#endif
use dtfft_parameters
use dtfft_pencil
use dtfft_utils
use test_suite
#include "_dtfft_mpi.h"
#ifdef DTFFT_WITH_CUDA
#include "_dtfft_cuda.h"
#endif
implicit none
    type :: executor_t
        type(dtfft_executor_t) :: executor
        logical :: is_device
        logical :: is_r2r
    end type executor_t
    type(executor_t) :: executors(4)
    integer(int32) :: n_executors
    integer(int32) :: exec, error_code
    type(test_suite_t) :: test

    call test%create("FFT")

    n_executors = 0
#ifdef DTFFT_WITH_FFTW
    n_executors = n_executors + 1
    executors(n_executors) = executor_t(DTFFT_EXECUTOR_FFTW3, .false., .true.)
#endif
#ifdef DTFFT_WITH_MKL
    n_executors = n_executors + 1
    executors(n_executors) = executor_t(DTFFT_EXECUTOR_MKL, .false., .false.)
#endif
#if defined(DTFFT_WITH_CUFFT) && !defined(DTFFT_RUNNING_CICD)
    n_executors = n_executors + 1
    executors(n_executors) = executor_t(DTFFT_EXECUTOR_CUFFT, .true., .false.)
#endif
#if defined(DTFFT_WITH_VKFFT) && !defined(DTFFT_RUNNING_CICD)
    n_executors = n_executors + 1
    executors(n_executors) = executor_t(DTFFT_EXECUTOR_VKFFT, .true., .true.)
#endif

    if( n_executors == 0 ) then
        call MPI_Finalize(error_code)
        stop
    endif

    do exec = 1, n_executors
        call test_executor(executors(exec), [17, 1])
        call test_executor(executors(exec), [64, 128])
        call test_executor(executors(exec), [13, 27, 95])
        call test_executor(executors(exec), [16, 32, 64])
    end do

    call test%finalize()
contains

#define PI 3.14159265358979323846264_real64
#define HALF 0.5_real64
#define NX real(dims(1), real64)
#define NY real(dims(2), real64)

    subroutine allocate_kernel(platform, kernel)
        type(dtfft_platform_t), intent(in) :: platform
        class(abstract_kernel), intent(out), allocatable :: kernel

        if ( platform == DTFFT_PLATFORM_HOST ) then
            allocate( kernel_host :: kernel )
#ifdef DTFFT_WITH_CUDA
        else
            allocate( kernel_device :: kernel )
#endif
        endif
    end subroutine allocate_kernel

    recursive subroutine compute_dct(source, result, fft_rank, dims, sign, platform)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        type(dtfft_platform_t), intent(in) :: platform
        real(real64), pointer, contiguous :: source_(:)
        real(real64), pointer, contiguous :: result_(:)
        type(c_ptr) :: temp1, temp2
        class(abstract_kernel), allocatable :: f, b
        integer(int32) :: how_many, i, idx, n1, offset
        real(real64) :: coeff, phase
        integer(int32) :: temp_dims(size(dims))

        if(fft_rank == FFT_2D .and. sign == FFT_BACKWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp1 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            call f%execute(source, temp1, NULL_STREAM)
            call compute_dct(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, source, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif

        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(source, source_, [product(dims)])

        how_many = product(dims) / dims(1)
        do idx = 1, how_many
            offset = (idx - 1) * dims(1)
            if ( sign == FFT_FORWARD ) then
                do i = 1, dims(1)
                    coeff = 0.0_real64
                    do n1 = 1, dims(1)
                        phase = PI * (real(n1 - 1, real64) + HALF) * real(i - 1, real64) / NX
                        coeff = coeff + source_(offset + n1) * cos(phase)
                    enddo
                    result_(i + offset) = 2.0_real64 * coeff
                enddo
            else
                do i = 1, dims(1)
                    coeff = 0.0_real64
                    do n1 = 2, dims(1)
                        phase = PI * (real(i - 1, real64) + HALF) * real(n1 - 1, real64) / NX
                        coeff = coeff + source_(offset + n1) * cos(phase)
                    enddo

                    result_(i + offset) = source_(offset + 1) + 2.0_real64 * coeff
                enddo
            endif
        enddo

        if ( fft_rank == FFT_2D .and. sign == FFT_FORWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            temp1 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

            call f%execute(result, temp1, NULL_STREAM)
            call compute_dct(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, result, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif
    end subroutine compute_dct

    subroutine verify_dct(source, result, fft_rank, dims, sign)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        real(real64), pointer, contiguous :: expected_(:)
        type(c_ptr) :: expected
        real(real64), pointer, contiguous :: result_(:)

        expected = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        call compute_dct(source, expected, fft_rank, dims, sign, DTFFT_PLATFORM_HOST)

        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(expected, expected_, [product(dims)])

        call test%assert_false("DCT Failure", any(abs(result_ - expected_) > 1.0e-5_real64))
        call mem_free_host(expected)
    end subroutine verify_dct

    recursive subroutine compute_dst(source, result, fft_rank, dims, sign, platform)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        type(dtfft_platform_t), intent(in) :: platform
        real(real64), pointer, contiguous :: source_(:)
        real(real64), pointer, contiguous :: result_(:)
        type(c_ptr) :: temp1, temp2
        class(abstract_kernel), allocatable :: f, b
        integer(int32) :: how_many, i, idx, n1, offset
        real(real64) :: coeff, phase
        integer(int32) :: temp_dims(size(dims))

        if(fft_rank == FFT_2D .and. sign == FFT_BACKWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp1 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            call f%execute(source, temp1, NULL_STREAM)
            call compute_dst(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, source, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif

        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(source, source_, [product(dims)])

        how_many = product(dims) / dims(1)
        do idx = 1, how_many
            offset = (idx - 1) * dims(1)
            if ( sign == FFT_FORWARD ) then
                do i = 1, dims(1)
                    coeff = 0.0_real64
                    do n1 = 1, dims(1)
                        phase = PI * (real(n1 - 1, real64) + HALF) * real(i, real64) / NX
                        coeff = coeff + source_(offset + n1) * sin(phase)
                    enddo
                    result_(i + offset) = 2.0_real64 * coeff
                enddo
            else
                do i = 1, dims(1)
                    coeff = 0.0_real64
                    do n1 = 1, dims(1) - 1
                        phase = PI * real(n1, real64) * (real(i - 1, real64) + HALF) / NX
                        coeff = coeff + source_(offset + n1) * sin(phase)
                    enddo
                    result_(i + offset) = (-1) ** (i - 1) * source_(dims(1) + offset) + 2.0_real64 * coeff
                enddo
            endif
        enddo

        if ( fft_rank == FFT_2D .and. sign == FFT_FORWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            temp1 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))

            call f%execute(result, temp1, NULL_STREAM)
            call compute_dst(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, result, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif
    end subroutine compute_dst

    subroutine verify_dst(source, result, fft_rank, dims, sign)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        real(real64), pointer, contiguous :: expected_(:)
        type(c_ptr) :: expected
        real(real64), pointer, contiguous :: result_(:)

        expected = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
        call compute_dst(source, expected, fft_rank, dims, sign, DTFFT_PLATFORM_HOST)
        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(expected, expected_, [product(dims)])

        call test%assert_false("DST Failure", any(abs(result_ - expected_) > 1.0e-5_real64))
        call mem_free_host(expected)
    end subroutine verify_dst

    recursive subroutine compute_c2c(source, result, fft_rank, dims, sign, platform)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        type(dtfft_platform_t), intent(in) :: platform
        complex(real64), pointer, contiguous :: source_(:)
        complex(real64), pointer, contiguous :: result_(:)
        type(c_ptr) :: temp1, temp2
        class(abstract_kernel), allocatable :: f, b
        integer(int32) :: how_many, i, idx, n1, offset
        complex(real64) :: coeff, factor
        real(real64) :: phase
        integer(int32) :: temp_dims(size(dims))

        if(fft_rank == FFT_2D .and. sign == FFT_BACKWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp1 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims))

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            call f%execute(source, temp1, NULL_STREAM)
            call compute_c2c(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, source, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif

        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(source, source_, [product(dims)])

        how_many = product(dims) / dims(1)
        do idx = 1, how_many
            offset = (idx - 1) * dims(1)
            do i = 1, dims(1)
                coeff = (0.0_real64, 0.0_real64)
                do n1 = 1, dims(1)
                    phase = 2.0_real64 * PI * real((i - 1) * (n1 - 1), real64) / real(dims(1), real64)
                    factor = cmplx(cos(phase), real(sign, real64) * sin(phase), kind=real64)
                    coeff = coeff + source_(offset + n1) * factor
                enddo
                result_(i + offset) = coeff
            enddo
        enddo

        if ( fft_rank == FFT_2D .and. sign == FFT_FORWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp_dims = [dims(2), dims(3), dims(1)]
            call f%create(dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            temp1 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims))
            temp2 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims))

            call f%execute(result, temp1, NULL_STREAM)
            call compute_c2c(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, result, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif
    end subroutine compute_c2c

    subroutine verify_c2c(source, result, fft_rank, dims, sign)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        complex(real64), pointer, contiguous :: expected_(:)
        type(c_ptr) :: expected
        complex(real64), pointer, contiguous :: result_(:)

        expected = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims))
        call compute_c2c(source, expected, fft_rank, dims, sign, DTFFT_PLATFORM_HOST)
        call c_f_pointer(result, result_, [product(dims)])
        call c_f_pointer(expected, expected_, [product(dims)])
        call test%assert_false("C2C Failure", any(abs(result_ - expected_) > 1.0e-5_real64))
        call mem_free_host(expected)
    end subroutine verify_c2c

    subroutine test_c2c(executor, dims)
        type(executor_t), intent(in) :: executor
        integer(int32), intent(in) :: dims(:)
        complex(real64), allocatable, target :: source(:), result(:)
        integer(int32) :: i, error_code
        type(c_ptr) :: source_ptr, result_ptr
        type(c_ptr) :: source_fft_ptr, result_fft_ptr
        integer(c_size_t) :: buffer_bytes
        class(abstract_executor), allocatable :: fft
        type(pencil) :: p
        TYPE_MPI_COMM :: comms(size(dims))

        print*,'  Testing C2C FFT'

        allocate(source(product(dims)))
        allocate(result(product(dims)))

        do i = 1, product(dims)
            source(i) = cmplx(real(i, real64), real(i + 1, real64), kind=real64)
        end do
        source_ptr = c_loc(source)
        result_ptr = c_loc(result)
        buffer_bytes = int(DOUBLE_COMPLEX_STORAGE_SIZE * product(dims), c_size_t)

        comms(:) = MPI_COMM_WORLD
        p = pencil(size(dims, kind=int8), 1_int8, dims, comms)

        print*,'    Testing C2C FFT 1D'
        call allocate_executor(executor%executor, fft)
        call test%assert_true("Allocated C2C executor", allocated(fft))
        error_code = fft%create(FFT_1D, FFT_C2C, DTFFT_DOUBLE, complex_pencil=p)
        call test%assert_true("Created C2C executor", error_code == DTFFT_SUCCESS)
        call allocate_fft_buffer(executor, buffer_bytes, source_ptr, source_fft_ptr)
        call allocate_fft_buffer(executor, buffer_bytes, result_ptr, result_fft_ptr)
        call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_FORWARD)
        call verify_c2c(source_ptr, result_ptr, FFT_1D, dims, FFT_FORWARD)

        call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_BACKWARD)
        call verify_c2c(result_ptr, source_ptr, FFT_1D, dims, FFT_BACKWARD)
        call fft%destroy()

        if ( size(dims) == 3 ) then
            print*,'    Testing C2C FFT 2D'
            error_code = fft%create(FFT_2D, FFT_C2C, DTFFT_DOUBLE, complex_pencil=p)
            call test%assert_true("Created C2C executor 2", error_code == DTFFT_SUCCESS)
            do i = 1, product(dims)
                source(i) = cmplx(real(i, real64), real(i + 1, real64), kind=real64)
            end do
            call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_FORWARD)
            call verify_c2c(source_ptr, result_ptr, FFT_2D, dims, FFT_FORWARD)

            call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_BACKWARD)
            call verify_c2c(result_ptr, source_ptr, FFT_2D, dims, FFT_BACKWARD)
            call fft%destroy()
        endif

        call free_fft_buffer(executor, source_fft_ptr)
        call free_fft_buffer(executor, result_fft_ptr)
        deallocate(source)
        deallocate(result)
        deallocate(fft)
    end subroutine test_c2c

    subroutine compute_r2c(source, result, fft_rank, dims, sign, platform)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        type(dtfft_platform_t), intent(in) :: platform
        real(real64), pointer, contiguous :: rl(:)
        complex(real64), pointer, contiguous :: cl(:)
        type(c_ptr) :: temp1, temp2
        class(abstract_kernel), allocatable :: f, b
        integer(int32) :: how_many, i, idx, n1, out_dim, shift, offset_in, offset_out
        complex(real64) :: coeff, factor
        real(real64) :: phase, r_coeff
        integer(int32) :: temp_dims(size(dims))
        class(abstract_executor), allocatable :: work

        out_dim = dims(1) / 2 + 1
        if(fft_rank == FFT_2D .and. sign == FFT_BACKWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp_dims = [dims(2), dims(3), out_dim]
            temp1 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(temp_dims))
            temp2 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(temp_dims))

            call f%create([out_dim, dims(2), dims(3)], DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            call f%execute(source, temp1, NULL_STREAM)
            call compute_c2c(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, source, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)
            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif

        if ( sign == FFT_FORWARD ) then
            how_many = product(dims) / dims(1)
            call c_f_pointer(source, rl, [product(dims)])
            call c_f_pointer(result, cl, [out_dim * product(dims(2:size(dims)))])

            do idx = 1, how_many
                offset_in = (idx - 1) * dims(1)
                offset_out = (idx - 1) * out_dim
                do i = 1, out_dim
                    coeff = (0.0_real64, 0.0_real64)
                    do n1 = 1, dims(1)
                        phase = 2.0_real64 * PI * real((i - 1) * (n1 - 1), real64) / NX
                        factor = cmplx(cos(phase), -sin(phase), kind=real64)
                        coeff = coeff + cmplx(rl(offset_in + n1), 0.0_real64, kind=real64) * factor
                    enddo
                    cl(i + offset_out) = coeff
                enddo
            enddo

        else
            how_many = product(dims(2:size(dims)))
            call c_f_pointer(source, cl, [out_dim * product(dims(2:size(dims)))])
            call c_f_pointer(result, rl, [product(dims)])

            shift = 1; if ( mod(dims(1), 2) == 0 ) shift = 0

            do idx = 1, how_many
                offset_in = (idx - 1) * out_dim
                offset_out = (idx - 1) * dims(1)
                do i = 1, dims(1)
                    r_coeff = real(cl(offset_in + 1), real64)
                    do n1 = 2, dims(1) / 2 + shift
                        phase =  2.0_real64 * PI * real((n1 - 1) * (i - 1), real64) / NX
                        r_coeff = r_coeff + 2.0_real64 * (real(cl(offset_in + n1), real64) * cos(phase) - aimag(cl(offset_in + n1)) * sin(phase))
                    enddo

                    if (mod(dims(1), 2) == 0) then
                        phase = PI * real(i - 1, real64)
                        r_coeff = r_coeff + real(cl(offset_in + out_dim)) * cos(phase)
                    endif
                    rl(offset_out + i) = r_coeff
                enddo
            enddo
        endif

        if ( fft_rank == FFT_2D .and. sign == FFT_FORWARD) then
            call allocate_kernel(platform, f)
            call allocate_kernel(platform, b)

            temp_dims = [dims(2), dims(3), out_dim]
            call f%create([out_dim, dims(2), dims(3)], DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_FORWARD)
            call b%create(temp_dims, DTFFT_ESTIMATE, DOUBLE_COMPLEX_STORAGE_SIZE, KERNEL_PERMUTE_BACKWARD)

            temp1 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(temp_dims))
            temp2 = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * product(temp_dims))

            call f%execute(result, temp1, NULL_STREAM)
            call compute_c2c(temp1, temp2, FFT_1D, temp_dims, sign, platform)
            call b%execute(temp2, result, NULL_STREAM)

            call mem_free_host(temp1)
            call mem_free_host(temp2)

            call f%destroy()
            call b%destroy()
            deallocate(f, b)
        endif
    end subroutine compute_r2c

    subroutine verify_r2c(source, result, fft_rank, dims, sign)
        type(c_ptr), intent(in) :: source
        type(c_ptr), intent(in) :: result
        type(fft_rank_t), intent(in) :: fft_rank
        integer(int32), intent(in) :: dims(:)
        integer(int8), intent(in) :: sign
        complex(real64), pointer, contiguous :: result_complex(:)
        real(real64), pointer, contiguous :: result_real(:)
        complex(real64), pointer, contiguous :: expected_complex(:)
        real(real64), pointer, contiguous :: expected_real(:)
        integer(int32) :: i, out_dim
        type(c_ptr) :: expected

        out_dim = dims(1) / 2 + 1
        if ( sign == FFT_FORWARD ) then
            expected = mem_alloc_host(DOUBLE_COMPLEX_STORAGE_SIZE * out_dim * product(dims(2:size(dims))))
            call compute_r2c(source, expected, fft_rank, dims, sign, DTFFT_PLATFORM_HOST)
            call c_f_pointer(expected, expected_complex, [out_dim * product(dims(2:size(dims)))])
            call c_f_pointer(result, result_complex, [out_dim * product(dims(2:size(dims)))] )
            do i = 1, size(expected_complex)
                if ( abs(result_complex(i) - expected_complex(i)) > 1.0e-5_real64 ) then
                    print *, "R2C forward test failed at index ", i, " : expected ", expected_complex(i), " but got ", result_complex(i)
                    call MPI_Abort(MPI_COMM_WORLD, -1, error_code)
                endif
            enddo
        else
            expected = mem_alloc_host(DOUBLE_STORAGE_SIZE * product(dims))
            call compute_r2c(source, expected, fft_rank, dims, sign, DTFFT_PLATFORM_HOST)
            call c_f_pointer(expected, expected_real, [ product(dims)])
            call c_f_pointer(result, result_real, [product(dims)] )
            do i = 1, size(expected_real)
                if ( abs(result_real(i) - expected_real(i)) > 1.0e-5_real64 ) then
                    print *, "R2C backward test failed at index ", i, " : expected ", expected_real(i), " but got ", result_real(i)
                    call MPI_Abort(MPI_COMM_WORLD, -1, error_code)
                endif
            enddo
        endif
        call mem_free_host(expected)
    end subroutine verify_r2c

    subroutine allocate_fft_buffer(executor, bytes, host_ptr, fft_ptr)
        type(executor_t), intent(in) :: executor
        integer(c_size_t), intent(in) :: bytes
        type(c_ptr), intent(in) :: host_ptr
        type(c_ptr), intent(out) :: fft_ptr

#ifdef DTFFT_WITH_CUDA
        if ( executor%is_device ) then
            CUDA_CALL( cudaMalloc(fft_ptr, bytes) )
        else
            fft_ptr = host_ptr
        endif
#else
        fft_ptr = host_ptr
#endif
    end subroutine allocate_fft_buffer

    subroutine free_fft_buffer(executor, fft_ptr)
        type(executor_t), intent(in) :: executor
        type(c_ptr), intent(in) :: fft_ptr

#ifdef DTFFT_WITH_CUDA
        if ( executor%is_device ) then
            CUDA_CALL( cudaFree(fft_ptr) )
        endif
#endif
    end subroutine free_fft_buffer

    subroutine execute_fft(executor, fft, input_host, output_host, input_fft, output_fft, input_bytes, output_bytes, sign)
        type(executor_t), intent(in) :: executor
        class(abstract_executor), intent(in) :: fft
        type(c_ptr), intent(in) :: input_host, output_host
        type(c_ptr), intent(in) :: input_fft, output_fft
        integer(c_size_t), intent(in) :: input_bytes, output_bytes
        integer(int8), intent(in) :: sign

#ifdef DTFFT_WITH_CUDA
        if ( executor%is_device ) then
            CUDA_CALL( cudaMemcpy(input_fft, input_host, input_bytes, cudaMemcpyHostToDevice) )
            call fft%execute(input_fft, output_fft, sign)
            CUDA_CALL( cudaMemcpy(output_host, output_fft, output_bytes, cudaMemcpyDeviceToHost) )
        else
            call fft%execute(input_host, output_host, sign)
        endif
#else
        call fft%execute(input_host, output_host, sign)
#endif
    end subroutine execute_fft

    subroutine test_r2c(executor, dims)
        type(executor_t), intent(in) :: executor
        integer(int32), intent(in) :: dims(:)
        real(real64), allocatable, target :: source(:)
        complex(real64), allocatable, target :: result(:), result2(:)
        integer(int32) :: i, error_code, out_dim
        type(c_ptr) :: source_ptr, result_ptr, result2_ptr
        type(c_ptr) :: source_fft_ptr, result_fft_ptr
        integer(c_size_t) :: source_bytes, result_bytes
        class(abstract_executor), allocatable :: fft
        type(pencil) :: real_p, complex_p
        TYPE_MPI_COMM :: comms(size(dims))

        print*,'  Testing R2C FFT'

        out_dim = dims(1) / 2 + 1
        allocate(source(product(dims)))
        allocate(result(product(dims(2:)) * out_dim))

        do i = 1, product(dims)
            source(i) = real(i, real64)
        end do
        source_ptr = c_loc(source)
        result_ptr = c_loc(result)
        source_bytes = int(DOUBLE_STORAGE_SIZE * product(dims), c_size_t)
        result_bytes = int(DOUBLE_COMPLEX_STORAGE_SIZE * size(result), c_size_t)

        comms(:) = MPI_COMM_WORLD
        real_p = pencil(size(dims, kind=int8), 1_int8, dims, comms)
        complex_p = pencil(size(dims, kind=int8), 1_int8, [out_dim, dims(2:size(dims))], comms)

        print*,'    Testing R2C FFT 1D'
        call allocate_executor(executor%executor, fft)
        call test%assert_true("Allocated R2C executor", allocated(fft))
        error_code = fft%create(FFT_1D, FFT_R2C, DTFFT_DOUBLE, real_pencil=real_p, complex_pencil=complex_p)
        call test%assert_true("Created R2C executor", error_code == DTFFT_SUCCESS)
        call allocate_fft_buffer(executor, source_bytes, source_ptr, source_fft_ptr)
        call allocate_fft_buffer(executor, result_bytes, result_ptr, result_fft_ptr)
        call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                         source_bytes, result_bytes, FFT_FORWARD)
        call verify_r2c(source_ptr, result_ptr, FFT_1D, dims, FFT_FORWARD)

        allocate(result2, source=result)
        result2_ptr = c_loc(result2)
        call execute_fft(executor, fft, result2_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                         result_bytes, source_bytes, FFT_BACKWARD)
        call verify_r2c(result_ptr, source_ptr, FFT_1D, dims, FFT_BACKWARD)
        deallocate(result2)
        call fft%destroy()

        if ( size(dims) == 3 ) then
            print*,'    Testing R2C FFT 2D'
            error_code = fft%create(FFT_2D, FFT_R2C, DTFFT_DOUBLE, real_pencil=real_p, complex_pencil=complex_p)
            call test%assert_true("Created R2C executor 2", error_code == DTFFT_SUCCESS)
            do i = 1, product(dims)
                source(i) = real(i, real64)
            end do
            call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                             source_bytes, result_bytes, FFT_FORWARD)
            call verify_r2c(source_ptr, result_ptr, FFT_2D, dims, FFT_FORWARD)

            allocate(result2, source=result)
            result_ptr = c_loc(result2)
            call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                             result_bytes, source_bytes, FFT_BACKWARD)
            result_ptr = c_loc(result)
            call verify_r2c(result_ptr, source_ptr, FFT_2D, dims, FFT_BACKWARD)
            call fft%destroy()
            deallocate(result2)
        endif

        call free_fft_buffer(executor, source_fft_ptr)
        call free_fft_buffer(executor, result_fft_ptr)
        deallocate(source)
        deallocate(result)
        deallocate(fft)
    end subroutine test_r2c

    subroutine test_r2r(executor, dims)
        type(executor_t), intent(in) :: executor
        integer(int32), intent(in) :: dims(:)
        real(real64), allocatable, target :: source(:), result(:)
        integer(int32) :: i, error_code
        type(c_ptr) :: source_ptr, result_ptr
        type(c_ptr) :: source_fft_ptr, result_fft_ptr
        integer(c_size_t) :: buffer_bytes
        class(abstract_executor), allocatable :: fft
        type(pencil) :: p
        TYPE_MPI_COMM :: comms(size(dims))
        type(dtfft_r2r_kind_t) :: r2rs(size(dims))

        if ( .not. executor%is_r2r ) return

        print*,'  Testing R2R FFT'

        allocate(source(product(dims)))
        allocate(result(product(dims)))

        do i = 1, product(dims)
            source(i) = real(i, real64)
        end do
        source_ptr = c_loc(source)
        result_ptr = c_loc(result)
        buffer_bytes = int(DOUBLE_STORAGE_SIZE * product(dims), c_size_t)

        comms(:) = MPI_COMM_WORLD
        p = pencil(size(dims, kind=int8), 1_int8, dims, comms)

        print*,'    Testing DCT FFT 1D'
        r2rs(:) = DTFFT_DCT_2
        call allocate_executor(executor%executor, fft)
        call test%assert_true("Allocated DCT executor", allocated(fft))
        error_code = fft%create(FFT_1D, FFT_R2R, DTFFT_DOUBLE, real_pencil=p, r2r_kinds=r2rs)
        call test%assert_true("Created DCT executor", error_code == DTFFT_SUCCESS)

        call allocate_fft_buffer(executor, buffer_bytes, source_ptr, source_fft_ptr)
        call allocate_fft_buffer(executor, buffer_bytes, result_ptr, result_fft_ptr)
        print*,'      Forward'
        call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_FORWARD)
        call verify_dct(source_ptr, result_ptr, FFT_1D, dims, FFT_FORWARD)

        print*,'      Backward'
        call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_BACKWARD)
        call verify_dct(result_ptr, source_ptr, FFT_1D, dims, FFT_BACKWARD)

        call fft%destroy()

        if ( size(dims) == 3 ) then
            print*,'    Testing DCT FFT 2D'
            error_code = fft%create(FFT_2D, FFT_R2R, DTFFT_DOUBLE, real_pencil=p, r2r_kinds=r2rs)
            call test%assert_true("Created DCT executor 2", error_code == DTFFT_SUCCESS)
            do i = 1, product(dims)
                source(i) = real(i, real64)
            end do
            print*,'      Forward'
            call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_FORWARD)
            call verify_dct(source_ptr, result_ptr, FFT_2D, dims, FFT_FORWARD)

            print*,'      Backward'
            call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_BACKWARD)
            call verify_dct(result_ptr, source_ptr, FFT_2D, dims, FFT_BACKWARD)
            call fft%destroy()
        endif

        print*,'    Testing DST FFT 1D'
        r2rs(:) = DTFFT_DST_2
        call allocate_executor(executor%executor, fft)
        call test%assert_true("Allocated DST executor", allocated(fft))
        error_code = fft%create(FFT_1D, FFT_R2R, DTFFT_DOUBLE, real_pencil=p, r2r_kinds=r2rs)
        call test%assert_true("Created DST executor", error_code == DTFFT_SUCCESS)
        print*,'      Forward'
        call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_FORWARD)
        call verify_dst(source_ptr, result_ptr, FFT_1D, dims, FFT_FORWARD)

        print*,'      Backward'
        call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                         buffer_bytes, buffer_bytes, FFT_BACKWARD)
        call verify_dst(result_ptr, source_ptr, FFT_1D, dims, FFT_BACKWARD)

        call fft%destroy()

        if ( size(dims) == 3 ) then
            print*,'    Testing DST FFT 2D'
            error_code = fft%create(FFT_2D, FFT_R2R, DTFFT_DOUBLE, real_pencil=p, r2r_kinds=r2rs)
            call test%assert_true("Created DST executor 2", error_code == DTFFT_SUCCESS)
            do i = 1, product(dims)
                source(i) = real(i, real64)
            end do
            print*,'      Forward'
            call execute_fft(executor, fft, source_ptr, result_ptr, source_fft_ptr, result_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_FORWARD)
            call verify_dst(source_ptr, result_ptr, FFT_2D, dims, FFT_FORWARD)

            print*,'      Backward'
            call execute_fft(executor, fft, result_ptr, source_ptr, result_fft_ptr, source_fft_ptr, &
                             buffer_bytes, buffer_bytes, FFT_BACKWARD)
            call verify_dst(result_ptr, source_ptr, FFT_2D, dims, FFT_BACKWARD)
            call fft%destroy()
        endif
        call free_fft_buffer(executor, source_fft_ptr)
        call free_fft_buffer(executor, result_fft_ptr)
        deallocate(source)
        deallocate(result)
        deallocate(fft)
    end subroutine test_r2r

    subroutine test_executor(executor, dims)
        type(executor_t), intent(in) :: executor
        integer(int32), intent(in) :: dims(:)

        print'(a,*(i4))','Testing executor '//dtfft_get_executor_string(executor%executor)//' with dims ', dims

        call test_r2c(executor, dims)
        call test_r2r(executor, dims)
        call test_c2c(executor, dims)
    end subroutine test_executor

end program test_ffts