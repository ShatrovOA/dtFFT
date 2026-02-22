program test_device_kernels
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_interface_cuda_runtime
use dtfft_interface_cuda
use dtfft_interface_nvrtc
use dtfft_config
use dtfft_kernel_device
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
    ierror = load_cuda()
    ierror = load_nvrtc()

    CUDA_CALL( cudaStreamCreate(stream) )

    call execute_test([18, 155], KERNEL_PERMUTE_FORWARD)
    call execute_test([18, 155], KERNEL_PACK)
    call execute_test([18, 155], KERNEL_UNPACK)
    call execute_test([18, 155], KERNEL_PACK_FORWARD)


    call execute_test([18, 33, 155], KERNEL_PERMUTE_FORWARD)
    call execute_test([18, 33, 155], KERNEL_PERMUTE_BACKWARD)
    call execute_test([18, 33, 155], KERNEL_PERMUTE_BACKWARD_START)
    call execute_test([18, 33, 155], KERNEL_PERMUTE_BACKWARD_END)
    call execute_test([18, 33, 155], KERNEL_PACK)
    call execute_test([18, 33, 155], KERNEL_UNPACK)
    call execute_test([18, 33, 155], KERNEL_PACK_FORWARD)
    call execute_test([18, 33, 155], KERNEL_PACK_BACKWARD)

    CUDA_CALL( cudaStreamDestroy(stream) )

    call MPI_Finalize(ierror)
contains

    subroutine execute_test(dims, kernel_type)
        integer(int32),         intent(in) :: dims(:)
        type(kernel_type_t),    intent(in) :: kernel_type
        type(c_ptr) :: p1, p2, p2_host, p3, gold
        integer(int32) :: locals(5, 1)
        type(kernel_device) :: dev
        type(kernel_host) :: host
        real(real32), pointer, contiguous :: p2_host_(:), p3_(:), gold_(:)
        integer(int32) :: i
        real(real32) :: rnd
        type(string) :: kernel_string

        locals(1:size(dims), 1) = dims(:)
        if ( size(dims) == 2 ) locals(3, 1) = 1
        locals(4, 1) = 0
        locals(5, 1) = 0

        kernel_string = get_kernel_string(kernel_type)
        print*,'Testing kernel '//kernel_string%raw//', ndims = '//to_str(size(dims))

        call dev%create(dims, DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, kernel_type, locals)
        call host%create(dims, DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, kernel_type, locals)

        CUDA_CALL( cudaMalloc(p1, FLOAT_STORAGE_SIZE * product(dims)) )
        CUDA_CALL( cudaMalloc(p2, FLOAT_STORAGE_SIZE * product(dims)) )
        p2_host = mem_alloc_host(FLOAT_STORAGE_SIZE * product(dims))
        p3 = mem_alloc_host(FLOAT_STORAGE_SIZE * product(dims))
        gold = mem_alloc_host(FLOAT_STORAGE_SIZE * product(dims))

        call c_f_pointer(p3, p3_, [product(dims)])
        call c_f_pointer(gold, gold_, [product(dims)])
        do i = 1, product(dims)
            call random_number(rnd)
            p3_(i) = rnd * real(i, real32)
        enddo

        CUDA_CALL( cudaMemcpy(p1, p3, FLOAT_STORAGE_SIZE * product(dims), cudaMemcpyHostToDevice) )
        CUDA_CALL( cudaDeviceSynchronize() )

        call host%execute(p3, gold, stream, 1)
        call dev%execute(p1, p2, stream, 1)

        CUDA_CALL( cudaDeviceSynchronize() )
        CUDA_CALL( cudaMemcpy(p2_host, p2, FLOAT_STORAGE_SIZE * product(dims), cudaMemcpyDeviceToHost) )

        call c_f_pointer(p2_host, p2_host_, [product(dims)])
        do i = 1, product(dims)
            if (abs(p2_host_(i) - gold_(i)) > 1e-6) then
                print*,'i = ',i, 'gold = ',gold_(i), 'dev = ',p2_host_(i)
                INTERNAL_ERROR("Mismatch between device and host results")
            end if
        enddo
        call mem_free_host(p2_host)
        call mem_free_host(p3)
        call mem_free_host(gold)
        CUDA_CALL( cudaFree(p1) )
        CUDA_CALL( cudaFree(p2) )
        call dev%destroy()
        call host%destroy()
        print*,'SUCCESS'
    end subroutine execute_test
end program test_device_kernels