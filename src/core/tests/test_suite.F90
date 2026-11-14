module test_suite
use iso_fortran_env, only: int32, output_unit, error_unit
#include "dtfft_config.h"
#include "_dtfft_mpi.h"
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#include "_dtfft_cuda.h"
#endif
implicit none
private

public :: test_suite_t

    type :: test_suite_t
    private
        integer(int32) :: comm_rank
        integer(int32) :: failures = 0_int32
    contains
        procedure :: create
        procedure :: finalize
        procedure, private :: check_integer
        procedure, private :: check_logical
        generic, public :: check => check_integer, check_logical
        procedure, public :: assert_true, assert_false
        procedure, public :: failure_count
    end type test_suite_t

contains

    subroutine create(self, name)
        class(test_suite_t), intent(inout)  :: self
        character(len=*),    intent(in)     :: name
        integer(int32)    :: ierr             !! Error code
        logical           :: is_mpi_init      !! Is MPI initialized?
        integer(int32) :: comm_size


        call MPI_Initialized(is_mpi_init, ierr)
        if ( .not. is_mpi_init ) then
            call MPI_Init(ierr)
        endif
        call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
        call MPI_Comm_rank(MPI_COMM_WORLD, self%comm_rank, ierr)
        if ( self%comm_rank == 0 ) then
            write(output_unit, '(a)') repeat("=", 30)
            write(output_unit, '(a)') "  Test "//name
            write(output_unit, '(a,i0)')     "  Ranks: ", comm_size
            write(output_unit, '(a)') repeat("=", 30)
            flush(output_unit)
        endif
    end subroutine create

    subroutine finalize(self)
        class(test_suite_t), intent(inout)  :: self
        integer(int32)    :: ierr             !! Error code

        if ( self%comm_rank == 0 ) then
            if ( self%failures == 0 ) then
                write(output_unit, "(a)") "All tests PASSED"
            else
                write(output_unit, "(i0,a)") self%failures, " test(s) FAILED"
            endif
            flush(output_unit)
        endif
        if ( self%failures == 0 ) then
            call MPI_Finalize(ierr)
        else
            call MPI_Abort(MPI_COMM_WORLD, self%failures, ierr)
        endif
    end subroutine finalize

    subroutine check_integer(self, name, actual, expected)
        class(test_suite_t), intent(inout) :: self
        character(len=*), intent(in) :: name
        integer(int32), intent(in) :: actual, expected
        integer(int32) :: failed, local_failed, reported_actual
        integer(int32) :: comm_size, first_rank, ierr

        local_failed = merge(1_int32, 0_int32, actual /= expected)
        call MPI_Allreduce(local_failed, failed, 1, MPI_INTEGER4, MPI_MAX, MPI_COMM_WORLD, ierr)
        if ( failed == 0 ) return

        call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
        call MPI_Allreduce(merge(actual, expected, local_failed /= 0), reported_actual, 1, &
                           MPI_INTEGER4, MPI_MAX, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(merge(self%comm_rank, comm_size, local_failed /= 0), first_rank, 1, &
                           MPI_INTEGER4, MPI_MIN, MPI_COMM_WORLD, ierr)
        self%failures = self%failures + 1_int32
        if ( self%comm_rank == 0 ) then
            write(output_unit, "(a, a, a, i0, a, i0, a, i0, a)") &
            "FAIL ", name, ": got ", reported_actual, ", expected ", expected, &
            " (first failing rank ", first_rank, ")"
            flush(output_unit)
        endif
    end subroutine check_integer

    subroutine check_logical(self, name, condition)
        class(test_suite_t), intent(inout) :: self
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

    subroutine assert_true(self, message, condition)
        class(test_suite_t), intent(inout) :: self
        character(len=*), intent(in) :: message
        logical, intent(in) :: condition
        integer(int32) :: rank, ierr

        if ( condition ) return

        self%failures = self%failures + 1_int32
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        if ( rank == 0 ) write(error_unit, "(a)") "Assertion failed: "//trim(message)
        ! call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine assert_true

    subroutine assert_false(self, message, condition)
        class(test_suite_t), intent(inout) :: self
        character(len=*), intent(in) :: message
        logical, intent(in) :: condition
        ! integer(int32) :: rank, ierr

        call self%assert_true(message, .not.condition)

        ! if ( .not. condition ) return

        ! self%failures = self%failures + 1_int32
        ! call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        ! if ( rank == 0 ) write(error_unit, "(a)") "Assertion failed: "//trim(message)
        ! ! call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine assert_false

    integer(int32) function failure_count(self)
        class(test_suite_t), intent(in) :: self

        failure_count = self%failures
    end function failure_count

end module test_suite
