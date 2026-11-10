#include "dtfft_config.h"
program test_reshape_plans
use iso_fortran_env
use iso_c_binding
use dtfft_config, only: dtfft_config_t, dtfft_create_config, dtfft_set_config
use dtfft_errors, only: DTFFT_SUCCESS
use dtfft_parameters
use dtfft_pencil, only: dtfft_pencil_t, pencil, pencil_init
use dtfft_reshape_plan, only: reshape_plan
use dtfft_transpose_plan, only: transpose_plan
use test_suite, only: test_suite_t
#include "_dtfft_mpi.h"
implicit none
    integer(int32), parameter :: DIMS(3) = [64, 32, 16]
    type(dtfft_config_t) :: config
    type(dtfft_pencil_t) :: public_pencil
    type(pencil_init) :: ipencil
    type(transpose_plan) :: tplan
    type(reshape_plan) :: rplan
    type(pencil) :: pencils(3), bricks(2)
    TYPE_MPI_COMM :: cart_comm, comms(3)
    integer(int32) :: comm_rank, comm_size, ierr
    type(test_suite_t) :: test
    integer(int32), pointer :: init_grid(:), final_grid(:)
    logical :: final_enabled

    call test%create("Reshape plan")
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    cart_comm = MPI_COMM_NULL
    comms = MPI_COMM_NULL

    call dtfft_create_config(config)
    config%enable_z_slab = .false.
    config%enable_y_slab = .false.
    config%backend = DTFFT_BACKEND_MPI_DATATYPE
    config%reshape_backend = DTFFT_BACKEND_MPI_DATATYPE
    call dtfft_set_config(config, ierr)
    call test%check("set config", ierr, DTFFT_SUCCESS)

    call make_z_pencil(DIMS, public_pencil)
    ierr = ipencil%create(public_pencil, MPI_COMM_WORLD)
    call test%check("create pencil decomposition", ierr, DTFFT_SUCCESS)
    if ( all_ok(ierr) ) then
        ierr = tplan%create(DTFFT_PLATFORM_HOST, DIMS, MPI_COMM_WORLD, DTFFT_ESTIMATE, &
                            MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, &
                            cart_comm, comms, pencils, ipencil)
        call test%check("create transpose_plan", ierr, DTFFT_SUCCESS)
        if ( all_ok(ierr) ) then
            ierr = rplan%create(DTFFT_PLATFORM_HOST, ipencil, pencils, MPI_COMM_WORLD, comms, &
                                MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, DTFFT_ESTIMATE, &
                                DTFFT_BACKEND_MPI_DATATYPE, MPI_DOUBLE_COMPLEX, &
                                DOUBLE_COMPLEX_STORAGE_SIZE, bricks, final_enabled)
            call test%check("create reshape_plan", ierr, DTFFT_SUCCESS)
            if ( all_ok(ierr) ) then
                call rplan%get_grid(1_int8, init_grid)
                call rplan%get_grid(2_int8, final_grid)
                call test%check("initial grid available", associated(init_grid))
                call test%check("final grid available", associated(final_grid))
                if ( associated(init_grid) ) call test%check("initial grid rank", size(init_grid), 3)
                if ( associated(final_grid) ) call test%check("final grid rank", size(final_grid), 3)
                call test_transpose_execute(tplan, pencils)
                call test_reshape_execute(rplan, pencils(3), bricks(2))
            endif
        endif
    endif

    call rplan%destroy()
    call tplan%destroy()
    call ipencil%destroy()
    call public_pencil%destroy()
    call destroy_pencils(pencils)
    call destroy_pencils(bricks)
    call free_communicators()
    call test%finalize()

contains
    subroutine test_transpose_execute(plan, layouts)
        type(transpose_plan), intent(inout) :: plan
        type(pencil), intent(in) :: layouts(:)
        complex(real64), allocatable, target :: input(:), output(:), restored(:)
        integer(int64) :: buffer_size

        buffer_size = max(product(int(layouts(1)%counts, int64)), &
                          product(int(layouts(2)%counts, int64)))
        allocate(input(buffer_size), output(buffer_size), restored(buffer_size))
        call fill_data(input, layouts(1))
        output = cmplx(-1.0_real64, -1.0_real64, real64)
        restored = cmplx(-2.0_real64, -2.0_real64, real64)

        call plan%execute(c_loc(input(1)), c_loc(output(1)), DTFFT_TRANSPOSE_X_TO_Y%val, &
                          EXEC_BLOCKING, c_null_ptr, ierr)
        call test%check("transpose_plan execute X->Y", ierr, DTFFT_SUCCESS)
        call check_data("transpose_plan X->Y data", output, layouts(2))

        call plan%execute(c_loc(output(1)), c_loc(restored(1)), DTFFT_TRANSPOSE_Y_TO_X%val, &
                          EXEC_BLOCKING, c_null_ptr, ierr)
        call test%check("transpose_plan execute Y->X", ierr, DTFFT_SUCCESS)
        call check_data("transpose_plan roundtrip data", restored, layouts(1))
    end subroutine test_transpose_execute

    subroutine test_reshape_execute(plan, pencil_layout, brick_layout)
        type(reshape_plan), intent(inout) :: plan
        type(pencil), intent(in) :: pencil_layout, brick_layout
        complex(real64), allocatable, target :: input(:), output(:), restored(:)
        integer(int64) :: buffer_size

        buffer_size = max(product(int(pencil_layout%counts, int64)), &
                          product(int(brick_layout%counts, int64)))
        allocate(input(buffer_size), output(buffer_size), restored(buffer_size))
        call fill_data(input, brick_layout)
        output = cmplx(-1.0_real64, -1.0_real64, real64)
        restored = cmplx(-2.0_real64, -2.0_real64, real64)

        call plan%execute(c_loc(input(1)), c_loc(output(1)), DTFFT_RESHAPE_X_BRICKS_TO_PENCILS%val, &
                          EXEC_BLOCKING, c_null_ptr, ierr)
        call test%check("reshape_plan execute X-bricks->pencils", ierr, DTFFT_SUCCESS)
        call check_data("reshape_plan X-bricks->pencils data", output, brick_layout)

        call plan%execute(c_loc(output(1)), c_loc(restored(1)), DTFFT_RESHAPE_X_PENCILS_TO_BRICKS%val, &
                          EXEC_BLOCKING, c_null_ptr, ierr)
        call test%check("reshape_plan execute X-pencils->bricks", ierr, DTFFT_SUCCESS)
        call check_data("reshape_plan roundtrip data", restored, brick_layout)
    end subroutine test_reshape_execute

    subroutine fill_data(buffer, layout)
        complex(real64), intent(out) :: buffer(:)
        type(pencil), intent(in) :: layout
        integer(int32) :: local_index(3), global_index(3), global_axis, offset
        integer(int32) :: i, j, k

        offset = 0
        do k = 1, layout%counts(3)
            do j = 1, layout%counts(2)
                do i = 1, layout%counts(1)
                    offset = offset + 1
                    local_index = [i, j, k]
                    global_index = 0
                    do global_axis = 1, 3
                        global_index(layout_axis(layout%aligned_dim, global_axis)) = &
                            layout%starts(global_axis) + local_index(global_axis) - 1
                    enddo
                    buffer(offset) = reference_value(global_index)
                enddo
            enddo
        enddo
    end subroutine fill_data

    subroutine check_data(tag, buffer, layout)
        character(len=*), intent(in) :: tag
        complex(real64), intent(in) :: buffer(:)
        type(pencil), intent(in) :: layout
        integer(int32) :: local_index(3), global_index(3), global_axis, offset, mismatches
        integer(int32) :: i, j, k

        offset = 0
        mismatches = 0
        do k = 1, layout%counts(3)
            do j = 1, layout%counts(2)
                do i = 1, layout%counts(1)
                    offset = offset + 1
                    local_index = [i, j, k]
                    global_index = 0
                    do global_axis = 1, 3
                        global_index(layout_axis(layout%aligned_dim, global_axis)) = &
                            layout%starts(global_axis) + local_index(global_axis) - 1
                    enddo
                    if ( any(transfer(buffer(offset), [0_int64, 0_int64]) /= &
                             transfer(reference_value(global_index), [0_int64, 0_int64])) ) then
                        mismatches = mismatches + 1
                    endif
                enddo
            enddo
        enddo
        call test%check(tag, mismatches, 0)
    end subroutine check_data

    integer(int32) function layout_axis(aligned_dim, slot)
        integer(int8), intent(in) :: aligned_dim
        integer(int32), intent(in) :: slot

        select case ( aligned_dim )
        case ( 1_int8 )
            layout_axis = slot
        case ( 2_int8 )
            layout_axis = mod(slot, 3) + 1
        case default
            layout_axis = mod(slot + 1, 3) + 1
        endselect
    end function layout_axis

    pure complex(real64) function reference_value(global_index)
        integer(int32), intent(in) :: global_index(3)
        real(real64) :: value

        value = real(mod(int(global_index(1), int64) * 1009_int64 + &
                         int(global_index(2), int64) * 101_int64 + &
                         int(global_index(3), int64) * 11_int64, 65521_int64), real64)
        reference_value = cmplx(value, -value * 0.5_real64, real64)
    end function reference_value

    subroutine make_z_pencil(dims_in, pencil_out)
        integer(int32),         intent(in)  :: dims_in(:)
        type(dtfft_pencil_t),   intent(out) :: pencil_out
        integer(int32) :: start, count, base, remainder, first_extra_rank

        base = dims_in(3) / comm_size
        write(output_unit, *)dims_in,size(dims_in)
        write(output_unit, *)lbound(dims_in),ubound(dims_in)
        flush(output_unit)
        write(output_unit, *)dims_in(2), dims_in(1)
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        remainder = mod(dims_in(3), comm_size)
        first_extra_rank = comm_size - remainder
        count = base + merge(1, 0, comm_rank >= first_extra_rank .and. remainder > 0)
        start = comm_rank * base + max(0, comm_rank - first_extra_rank)
        pencil_out = dtfft_pencil_t([0_int32, 0_int32, start], [dims_in(1), dims_in(2), count])
    end subroutine make_z_pencil

    subroutine destroy_pencils(items)
        type(pencil), intent(inout) :: items(:)
        integer(int32) :: item

        do item = 1, size(items)
            call items(item)%destroy()
        enddo
    end subroutine destroy_pencils

    subroutine free_communicators()
        integer(int32) :: comm_id

        do comm_id = 1, size(comms)
            if ( comms(comm_id) /= MPI_COMM_NULL ) call MPI_Comm_free(comms(comm_id), ierr)
        enddo
        if ( cart_comm /= MPI_COMM_NULL ) call MPI_Comm_free(cart_comm, ierr)
    end subroutine free_communicators

    logical function all_ok(local_code)
        integer(int32), intent(in) :: local_code
        integer(int32) :: failed

        call MPI_Allreduce(merge(0, 1, local_code == DTFFT_SUCCESS), failed, 1, MPI_INTEGER4, &
                           MPI_MAX, MPI_COMM_WORLD, ierr)
        all_ok = failed == 0
    end function all_ok

end program test_reshape_plans
