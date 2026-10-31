#include "dtfft_config.h"
program test_mem_alloc
use iso_fortran_env, only: int32, int64, real32, real64, output_unit
use dtfft
use dtfft_test_utils, only: test_context
#include "_dtfft_mpi.h"
implicit none

    integer(int32) :: ierr, comm_rank
    type(dtfft_plan_c2c_t) :: plan
    type(test_context) :: test

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

    call plan%create([8_int32, 10_int32, 12_int32], executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
    call test%check("create plan", ierr, DTFFT_SUCCESS)
    call test%assert_true(ierr == DTFFT_SUCCESS, "plan creation must succeed before memory allocation")

    call test_real32_allocations()
    call test_real64_allocations()
    call test_complex32_allocations()
    call test_complex64_allocations()
    call test_default_lower_bounds()

    call plan%destroy(ierr)
    call test%check("destroy plan", ierr, DTFFT_SUCCESS)

    if ( comm_rank == 0 ) then
        if ( test%failure_count() == 0 ) then
            write(output_unit, "(a)") "test_mem_alloc: all tests PASSED"
        else
            write(output_unit, "(a,i0,a)") "test_mem_alloc: ", test%failure_count(), " test(s) FAILED"
        endif
        flush(output_unit)
    endif

    call MPI_Finalize(ierr)
    if ( test%failure_count() /= 0 ) error stop 1

contains

    subroutine test_real32_allocations()
        real(real32), pointer :: v1(:), v2(:, :), v3(:, :, :)

        call plan%mem_alloc(5_int64, v1, lbound=-2_int32, error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v1), "allocate real32 rank 1")
        call test%check("real32 rank 1 bounds", lbound(v1, 1) == -2 .and. ubound(v1, 1) == 2)
        call plan%mem_free(v1, ierr)
        call test%check("free real32 rank 1", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(6_int64, v2, sizes=[2_int32, 3_int32], &
                            lbounds=[-1_int32, 0_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v2), "allocate real32 rank 2")
        call test%check("real32 rank 2 bounds", all(lbound(v2) == [-1, 0]) .and. &
                        all(ubound(v2) == [0, 2]))
        call plan%mem_free(v2, ierr)
        call test%check("free real32 rank 2", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(24_int64, v3, sizes=[2_int32, 3_int32, 4_int32], &
                            lbounds=[-2_int32, 1_int32, 0_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v3), "allocate real32 rank 3")
        call test%check("real32 rank 3 bounds", all(lbound(v3) == [-2, 1, 0]) .and. &
                        all(ubound(v3) == [-1, 3, 3]))
        call plan%mem_free(v3, ierr)
        call test%check("free real32 rank 3", ierr, DTFFT_SUCCESS)
    end subroutine test_real32_allocations

    subroutine test_real64_allocations()
        real(real64), pointer :: v1(:), v2(:, :), v3(:, :, :)

        call plan%mem_alloc(7_int64, v1, lbound=4_int32, error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v1), "allocate real64 rank 1")
        call test%check("real64 rank 1 bounds", lbound(v1, 1) == 4 .and. ubound(v1, 1) == 10)
        call plan%mem_free(v1, ierr)
        call test%check("free real64 rank 1", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(2_int64, v2, sizes=[1_int32, 2_int32], &
                            lbounds=[0_int32, -3_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v2), "allocate real64 rank 2")
        call test%check("real64 rank 2 bounds", all(lbound(v2) == [0, -3]) .and. &
                        all(ubound(v2) == [0, -2]))
        call plan%mem_free(v2, ierr)
        call test%check("free real64 rank 2", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(8_int64, v3, sizes=[2_int32, 2_int32, 2_int32], &
                            lbounds=[3_int32, -2_int32, 0_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v3), "allocate real64 rank 3")
        call test%check("real64 rank 3 bounds", all(lbound(v3) == [3, -2, 0]) .and. &
                        all(ubound(v3) == [4, -1, 1]))
        call plan%mem_free(v3, ierr)
        call test%check("free real64 rank 3", ierr, DTFFT_SUCCESS)
    end subroutine test_real64_allocations

    subroutine test_complex32_allocations()
        complex(real32), pointer :: v1(:), v2(:, :), v3(:, :, :)

        call plan%mem_alloc(9_int64, v1, lbound=-3_int32, error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v1), "allocate complex32 rank 1")
        call test%check("complex32 rank 1 bounds", lbound(v1, 1) == -3 .and. ubound(v1, 1) == 5)
        call plan%mem_free(v1, ierr)
        call test%check("free complex32 rank 1", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(6_int64, v2, sizes=[3_int32, 2_int32], &
                            lbounds=[-2_int32, 5_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v2), "allocate complex32 rank 2")
        call test%check("complex32 rank 2 bounds", all(lbound(v2) == [-2, 5]) .and. &
                        all(ubound(v2) == [0, 6]))
        call plan%mem_free(v2, ierr)
        call test%check("free complex32 rank 2", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(6_int64, v3, sizes=[2_int32, 1_int32, 3_int32], &
                            lbounds=[0_int32, -4_int32, 2_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v3), "allocate complex32 rank 3")
        call test%check("complex32 rank 3 bounds", all(lbound(v3) == [0, -4, 2]) .and. &
                        all(ubound(v3) == [1, -4, 4]))
        call plan%mem_free(v3, ierr)
        call test%check("free complex32 rank 3", ierr, DTFFT_SUCCESS)
    end subroutine test_complex32_allocations

    subroutine test_complex64_allocations()
        complex(real64), pointer :: v1(:), v2(:, :), v3(:, :, :)

        call plan%mem_alloc(4_int64, v1, lbound=2_int32, error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v1), "allocate complex64 rank 1")
        call test%check("complex64 rank 1 bounds", lbound(v1, 1) == 2 .and. ubound(v1, 1) == 5)
        call plan%mem_free(v1, ierr)
        call test%check("free complex64 rank 1", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(8_int64, v2, sizes=[4_int32, 2_int32], &
                            lbounds=[-5_int32, -2_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v2), "allocate complex64 rank 2")
        call test%check("complex64 rank 2 bounds", all(lbound(v2) == [-5, -2]) .and. &
                        all(ubound(v2) == [-2, -1]))
        call plan%mem_free(v2, ierr)
        call test%check("free complex64 rank 2", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(4_int64, v3, sizes=[1_int32, 2_int32, 2_int32], &
                            lbounds=[-1_int32, 0_int32, 6_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v3), "allocate complex64 rank 3")
        call test%check("complex64 rank 3 bounds", all(lbound(v3) == [-1, 0, 6]) .and. &
                        all(ubound(v3) == [-1, 1, 7]))
        call plan%mem_free(v3, ierr)
        call test%check("free complex64 rank 3", ierr, DTFFT_SUCCESS)
    end subroutine test_complex64_allocations

    subroutine test_default_lower_bounds()
        real(real32), pointer :: v1(:), v2(:, :), v3(:, :, :)

        call plan%mem_alloc(3_int64, v1, error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v1), "allocate default-bound rank 1")
        call test%check("default rank 1 bounds", lbound(v1, 1) == 1 .and. ubound(v1, 1) == 3)
        call plan%mem_free(v1, ierr)
        call test%check("free default-bound rank 1", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(6_int64, v2, sizes=[2_int32, 3_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v2), "allocate default-bound rank 2")
        call test%check("default rank 2 bounds", all(lbound(v2) == [1, 1]) .and. &
                        all(ubound(v2) == [2, 3]))
        call plan%mem_free(v2, ierr)
        call test%check("free default-bound rank 2", ierr, DTFFT_SUCCESS)

        call plan%mem_alloc(24_int64, v3, sizes=[2_int32, 3_int32, 4_int32], error_code=ierr)
        call test%assert_true(ierr == DTFFT_SUCCESS .and. associated(v3), "allocate default-bound rank 3")
        call test%check("default rank 3 bounds", all(lbound(v3) == [1, 1, 1]) .and. &
                        all(ubound(v3) == [2, 3, 4]))
        call plan%mem_free(v3, ierr)
        call test%check("free default-bound rank 3", ierr, DTFFT_SUCCESS)
    end subroutine test_default_lower_bounds

end program test_mem_alloc
