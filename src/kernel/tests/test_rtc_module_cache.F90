#include "dtfft_config.h"
program test_rtc_module_cache
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_config
use dtfft_errors
use dtfft_interface_cuda, only: CUfunction, load_cuda
use dtfft_interface_nvrtc, only: load_nvrtc
use dtfft_interface_cuda_runtime, only: device_props
use dtfft_parameters
use dtfft_rtc_block_optimizer, only: kernel_config, get_ampere_architecture
use dtfft_rtc_module
use dtfft_rtc_module_cache
use dtfft_test_utils, only: test_context
#include "_dtfft_mpi.h"
implicit none
    type(device_props) :: props
    type(kernel_config) :: configs(2)
    type(kernel_config) :: extra_configs(2)
    type(nvrtc_module) :: mod
    type(CUfunction) :: fun
    integer(int32) :: ierr
    type(test_context) :: test

    call MPI_Init(ierr)

    ierr = init_internal()
    call test%assert_true(ierr == DTFFT_SUCCESS, "init_internal failed")

    ierr = load_cuda()
    call test%assert_true(ierr == DTFFT_SUCCESS, "load_cuda failed")

    ierr = load_nvrtc()
    call test%assert_true(ierr == DTFFT_SUCCESS, "load_nvrtc failed")

    props = get_ampere_architecture()

    configs(1)%tile_size = 32
    configs(1)%block_rows = 8
    configs(1)%padding = 1
    configs(2)%tile_size = 64
    configs(2)%block_rows = 8
    configs(2)%padding = 1

    call mod%create(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, configs, props)

    call test%assert_true(mod%check(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE), &
                          "check_module failed after create")
    call test%assert_true(mod%check(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, 32, 8), &
                          "check_instance failed for 32x8")

    ierr = mod%get(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, 32, 8, fun)
    call test%assert_true(ierr == DTFFT_SUCCESS, "module get failed for 32x8")
    call verify_function(fun, 3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE)

    ierr = mod%get(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, 64, 8, fun)
    call test%assert_true(ierr == DTFFT_SUCCESS, "module get failed for 64x8")
    call verify_function(fun, 3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE)

    call mod%destroy()

    call create_nvrtc_module(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, configs, props)

    fun = get_kernel_instance(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, 32, 8)
    call verify_function(fun, 3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE)

    extra_configs(1) = configs(1)
    extra_configs(2)%tile_size = 16
    extra_configs(2)%block_rows = 16
    extra_configs(2)%padding = 1

    call create_nvrtc_module(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, extra_configs, props)

    fun = get_kernel_instance(3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE, 16, 16)
    call verify_function(fun, 3, KERNEL_PERMUTE_FORWARD, FLOAT_STORAGE_SIZE)

    call MPI_Finalize(ierr)

contains

    subroutine verify_function(fun, ndims, kernel_type, base_storage)
        type(CUfunction), intent(in) :: fun
        integer(int32), intent(in) :: ndims
        type(kernel_type_t), intent(in) :: kernel_type
        integer(int64), intent(in) :: base_storage
        integer(int32) :: consume_args

        consume_args = ndims + kernel_type%val + int(base_storage, int32) + storage_size(fun)
        if (consume_args == -1) then
            call test%assert_true(.false., "unexpected consumed argument marker")
        end if

#ifdef DTFFT_WITH_MOCK_ENABLED
        call test%assert_true(fun%ndims == ndims, "mock function ndims mismatch")
        call test%assert_true(fun%kernel_type == kernel_type, "mock function kernel_type mismatch")
        call test%assert_true(fun%base_storage == base_storage, "mock function base_storage mismatch")
#else
        call test%assert_true(.true., "kernel handle retrieved")
#endif
    end subroutine verify_function

end program test_rtc_module_cache
