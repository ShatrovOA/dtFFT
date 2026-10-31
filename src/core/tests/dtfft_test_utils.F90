module dtfft_test_utils
use iso_fortran_env, only: int32, output_unit, error_unit
#include "dtfft_config.h"
#include "_dtfft_mpi.h"
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#include "_dtfft_cuda.h"
#endif
implicit none
private

public :: test_context
public :: attach_gpu_to_process

type :: test_context
    private
    integer(int32) :: failures = 0_int32
contains
    procedure, private :: check_integer
    procedure, private :: check_logical
    generic, public :: check => check_integer, check_logical
    procedure, public :: assert_true
    procedure, public :: failure_count
end type test_context

contains

    subroutine attach_gpu_to_process()
#if defined(DTFFT_WITH_CUDA)
        integer(int32) :: comm_rank, ierr, host_rank, host_size, num_devices
        TYPE_MPI_COMM :: host_comm
        character(len=5) :: platform_env

        call get_environment_variable("DTFFT_PLATFORM", platform_env)
        if (trim(adjustl(platform_env)) == "host") return

        CUDA_CALL(cudaGetDeviceCount(num_devices))
        if (num_devices == 0) return

        call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank, &
                                 MPI_INFO_NULL, host_comm, ierr)
        call MPI_Comm_rank(host_comm, host_rank, ierr)
        call MPI_Comm_size(host_comm, host_size, ierr)
        call MPI_Comm_free(host_comm, ierr)

        if (host_size > num_devices) return
        CUDA_CALL(cudaSetDevice(host_rank))
#endif
    end subroutine attach_gpu_to_process

    subroutine check_integer(self, name, actual, expected)
        class(test_context), intent(inout) :: self
        character(len=*), intent(in) :: name
        integer(int32), intent(in) :: actual, expected
        integer(int32) :: failed, local_failed, reported_actual
        integer(int32) :: rank, comm_size, first_rank, ierr

        local_failed = merge(1_int32, 0_int32, actual /= expected)
        call MPI_Allreduce(local_failed, failed, 1, MPI_INTEGER4, MPI_MAX, MPI_COMM_WORLD, ierr)
        if ( failed == 0 ) return

        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
        call MPI_Allreduce(merge(actual, expected, local_failed /= 0), reported_actual, 1, &
                           MPI_INTEGER4, MPI_MAX, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(merge(rank, comm_size, local_failed /= 0), first_rank, 1, &
                           MPI_INTEGER4, MPI_MIN, MPI_COMM_WORLD, ierr)
        self%failures = self%failures + 1_int32
        if ( rank == 0 ) write(output_unit, "(a, a, a, i0, a, i0, a, i0, a)") &
            "FAIL ", name, ": got ", reported_actual, ", expected ", expected, &
            " (first failing rank ", first_rank, ")"
    end subroutine check_integer

    subroutine check_logical(self, name, condition)
        class(test_context), intent(inout) :: self
        character(len=*), intent(in) :: name
        logical, intent(in) :: condition
        integer(int32) :: failed, rank, ierr

        call MPI_Allreduce(merge(0_int32, 1_int32, condition), failed, 1, &
                           MPI_INTEGER4, MPI_MAX, MPI_COMM_WORLD, ierr)
        if ( failed == 0 ) return

        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        self%failures = self%failures + 1_int32
        if ( rank == 0 ) write(output_unit, "(a, a)") "FAIL ", name
    end subroutine check_logical

    subroutine assert_true(self, condition, message)
        class(test_context), intent(inout) :: self
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer(int32) :: rank, ierr

        if ( condition ) return

        self%failures = self%failures + 1_int32
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        if ( rank == 0 ) write(error_unit, "(a)") "Assertion failed: "//trim(message)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine assert_true

    integer(int32) function failure_count(self)
        class(test_context), intent(in) :: self

        failure_count = self%failures
    end function failure_count

end module dtfft_test_utils
