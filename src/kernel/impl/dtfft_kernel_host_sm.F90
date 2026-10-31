#include "dtfft_config.h"
submodule (dtfft_kernel_host) dtfft_kernel_host_sm
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
#include "_dtfft_kernel_defines.inc"
implicit none

    interface
#define PREC _f32
#define MACRO_NAME GET_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_CO_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_BLOCK_KERNEL_NAME
#define BLOCK_SIZE 4
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 8
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 16
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 32
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 64
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE

#undef PREC
#undef MACRO_NAME

#define PREC _f64
#define MACRO_NAME GET_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_CO_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_BLOCK_KERNEL_NAME
#define BLOCK_SIZE 4
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 8
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 16
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 32
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 64
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE

#undef PREC
#undef MACRO_NAME

#define PREC _f128
#define MACRO_NAME GET_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_CO_SUB_NAME
# include "_dtfft_kernel_host_interface.inc"
#undef MACRO_NAME
#define MACRO_NAME GET_BLOCK_KERNEL_NAME
#define BLOCK_SIZE 4
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 8
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 16
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 32
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 64
# include "_dtfft_kernel_host_interface.inc"
#undef BLOCK_SIZE

#undef PREC
#undef MACRO_NAME
    end interface

contains

    module subroutine select_kernel(kernel, base_storage, fun)
    !! Selects the kernel implementation based on the given id and base storage size
        type(host_kernel_t), intent(in) :: kernel           !! Kernel id
        integer(int64),      intent(in) :: base_storage     !! Size of single element in bytes
        procedure(execute_host_interface), pointer, intent(out) :: fun   !! Selected kernel implementation
        fun => null()
        select case (kernel%val)
        case (HOST_KERNEL_BASE%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_BLOCK_4%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_block_4
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_block_4
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_block_4
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_BLOCK_8%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_block_8
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_block_8
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_block_8
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_BLOCK_16%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_block_16
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_block_16
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_block_16
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_BLOCK_32%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_block_32
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_block_32
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_block_32
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_BLOCK_64%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_block_64
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_block_64
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_block_64
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case (HOST_KERNEL_CACHE_OBLIVIOUS%val)
            select case (base_storage)
            case (FLOAT_STORAGE_SIZE)
                fun => execute_f32_co
            case (DOUBLE_STORAGE_SIZE)
                fun => execute_f64_co
            case (DOUBLE_COMPLEX_STORAGE_SIZE)
                fun => execute_f128_co
            case default
                INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
            end select
        case default
            INTERNAL_ERROR("select_kernel: Unknown kernel "//to_str(kernel%val)//" "//to_str(base_storage))
        end select
    end subroutine select_kernel
end submodule dtfft_kernel_host_sm