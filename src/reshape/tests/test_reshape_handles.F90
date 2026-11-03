#include "dtfft_config.h"
program test_reshape_handles
use iso_fortran_env
use iso_c_binding
use dtfft_abstract_backend, only: backend_helper
use dtfft_abstract_reshape_handle, only: abstract_reshape_handle, create_args, execute_args
use dtfft_errors, only: DTFFT_SUCCESS
use dtfft_parameters
use dtfft_pencil, only: pencil
use dtfft_reshape_handle_datatype, only: reshape_handle_datatype
use dtfft_reshape_handle_generic, only: reshape_handle_generic
use test_suite, only: test_suite_t
! #ifdef DTFFT_WITH_COMPRESSION
! use dtfft_config, only: get_conf_reshape
! #endif
#include "_dtfft_mpi.h"
implicit none
    integer(int32) :: DIMS(3) = [32, 24, 40]
    integer(int32) :: comm_rank, comm_size, ierr
    type(test_suite_t) :: test
#ifdef DTFFT_WITH_RMA
    type(dtfft_backend_t) :: backends(5)
#else
    type(dtfft_backend_t) :: backends(4)
#endif
    integer(int32) :: divs(2), div_count

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    DIMS = ((DIMS + comm_size - 1) / comm_size) * comm_size

    backends = [DTFFT_BACKEND_MPI_P2P, DTFFT_BACKEND_MPI_P2P_SCHEDULED, DTFFT_BACKEND_MPI_P2P_FUSED, DTFFT_BACKEND_MPI_DATATYPE &
#ifdef DTFFT_WITH_RMA
                , DTFFT_BACKEND_MPI_RMA_FUSED                                                                                   &
#endif
    ]

    call find_first_two_divisors(comm_size, divs, div_count)
    block
        integer(int32) :: backend_id
        character(len=:), allocatable :: backend_name

        do backend_id = 1, size(backends)
            allocate(backend_name, source=dtfft_get_backend_string(backends(backend_id)))
            ! backend_name = 
            call test_transpose_2d(backends(backend_id), backend_name//" 2D X->Y")
            call test_brick_pencil_2d(backends(backend_id), backend_name//" 2D bricks-pencils")
            call test_transpose_3d(backends(backend_id), DTFFT_TRANSPOSE_X_TO_Y)
            call test_transpose_3d(backends(backend_id), DTFFT_TRANSPOSE_Y_TO_Z)
            call test_transpose_3d_internal(backends(backend_id), DTFFT_TRANSPOSE_X_TO_Z, [1, 1, comm_size])

            call test_brick_pencil_scenario(backends(backend_id), 1, backend_name)
            call test_brick_pencil_scenario(backends(backend_id), 2, backend_name)
            call test_brick_pencil_scenario(backends(backend_id), 3, backend_name)
            call test_bricks_x_to_yz(backends(backend_id), backend_name)

            deallocate(backend_name)
        enddo
    end block

    
    if ( test%failure_count() /= 0 ) error stop 1
    if ( comm_rank == 0 ) then
        print*,'All tests passed'
    endif
    call MPI_Finalize(ierr)
contains
    subroutine test_transpose_2d(backend, tag)
        type(dtfft_backend_t), intent(in) :: backend
        character(len=*), intent(in) :: tag
        type(pencil) :: x_pencil, y_pencil
        TYPE_MPI_COMM :: cart_comm, axis_comms(3), helper_comms(3)
        integer(int32) :: grid_dims(3)

        grid_dims = [1, comm_size, 1]
        call create_grid(grid_dims, cart_comm, axis_comms)
        helper_comms = [cart_comm, axis_comms(2), axis_comms(3)]
        x_pencil = pencil(2_int8, 1_int8, DIMS(1:2), axis_comms(1:2))
        y_pencil = pencil(2_int8, 2_int8, [DIMS(2), DIMS(1)], axis_comms(1:2))

        call check_handle_roundtrip(backend, x_pencil, y_pencil, cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_PACK)
        call check_handle_roundtrip(backend, x_pencil, y_pencil, cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_UNPACK)

        call x_pencil%destroy()
        call y_pencil%destroy()
        call free_grid(cart_comm, axis_comms)
    end subroutine test_transpose_2d

    subroutine test_brick_pencil_2d(backend, tag)
        type(dtfft_backend_t), intent(in) :: backend
        character(len=*), intent(in) :: tag
        type(pencil) :: bricks, x_pencil
        TYPE_MPI_COMM :: cart_comm, axis_comms(3), helper_comms(3)
        integer(int32) :: grid_dims(3), brick_starts(2), brick_counts(2)
        integer(int32) :: pencil_starts(2), pencil_counts(2)

        grid_dims = [comm_size, 1, 1]
        call create_grid(grid_dims, cart_comm, axis_comms)
        call local_range(DIMS(1), comm_rank, comm_size, brick_starts(1), brick_counts(1))
        brick_starts(2) = 0
        brick_counts(2) = DIMS(2)
        bricks = pencil(2_int8, 1_int8, DIMS(1:2), [axis_comms(1), MPI_COMM_SELF], &
                        lstarts=brick_starts, lcounts=brick_counts)

        call local_range(DIMS(2), comm_rank, comm_size, pencil_starts(2), pencil_counts(2))
        pencil_starts(1) = 0
        pencil_counts(1) = DIMS(1)
        x_pencil = pencil(2_int8, 1_int8, DIMS(1:2), [MPI_COMM_SELF, axis_comms(1)], &
                          lstarts=pencil_starts, lcounts=pencil_counts)

        call test%check(tag//": bricks distribution", merge(1, 0, bricks%is_distributed), &
                        merge(1, 0, comm_size > 1))
        call test%check(tag//": pencil distribution", merge(1, 0, .not.x_pencil%is_distributed), 1)
        helper_comms = [cart_comm, cart_comm, cart_comm]
        call check_handle_roundtrip(backend, bricks, x_pencil, cart_comm, helper_comms, tag)

        call bricks%destroy()
        call x_pencil%destroy()
        call free_grid(cart_comm, axis_comms)
    end subroutine test_brick_pencil_2d

    subroutine test_transpose_3d(backend, transpose_type)
        type(dtfft_backend_t), intent(in) :: backend
        type(dtfft_transpose_t), intent(in) :: transpose_type

        call test_transpose_3d_internal(backend, transpose_type, [1, comm_size, 1])
        call test_transpose_3d_internal(backend, transpose_type, [1, 1, comm_size])
        if ( div_count == 1 ) then
            call test_transpose_3d_internal(backend, transpose_type, [1, divs(1), divs(1)])
        else if ( div_count == 2 ) then
            call test_transpose_3d_internal(backend, transpose_type, [1, divs(1), divs(2)])
            call test_transpose_3d_internal(backend, transpose_type, [1, divs(2), divs(1)])
        endif
    end subroutine test_transpose_3d

    subroutine test_transpose_3d_internal(backend, transpose_type, grid_dims)
        type(dtfft_backend_t), intent(in) :: backend
        type(dtfft_transpose_t), intent(in) :: transpose_type
        integer(int32), intent(in) :: grid_dims(3)
        character(len=:), allocatable :: back, tag
        type(pencil) :: pencils(3)
        TYPE_MPI_COMM :: cart_comm, axis_comms(3), y_comms(3), helper_comms(3)
        integer(int32) :: y_dims(3), z_dims(3)

        call create_grid(grid_dims, cart_comm, axis_comms)
        helper_comms = [cart_comm, axis_comms(2), axis_comms(3)]
        pencils(1) = pencil(3_int8, 1_int8, DIMS, axis_comms)
        y_dims = [DIMS(2), DIMS(3), DIMS(1)]
        y_comms = [axis_comms(1), axis_comms(3), axis_comms(2)]
        pencils(2) = pencil(3_int8, 2_int8, y_dims, y_comms)
        z_dims = [DIMS(3), DIMS(1), DIMS(2)]
        pencils(3) = pencil(3_int8, 3_int8, z_dims, axis_comms)

        allocate(back, source=dtfft_get_backend_string(backend))
        allocate(tag, source="")

        select case ( transpose_type%val )
        case ( DTFFT_TRANSPOSE_X_TO_Y%val )
            tag = "Transpose 3D; X->Y; DTFFT_TRANSPOSE_MODE_PACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(1), pencils(2), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_PACK)
            tag = "Transpose 3D; X->Y; DTFFT_TRANSPOSE_MODE_UNPACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(1), pencils(2), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_UNPACK)
        case ( DTFFT_TRANSPOSE_Y_TO_Z%val )
            tag = "Transpose 3D; Y->Z; DTFFT_TRANSPOSE_MODE_PACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(2), pencils(3), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_PACK)
            tag = "Transpose 3D; Y->Z; DTFFT_TRANSPOSE_MODE_UNPACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(2), pencils(3), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_UNPACK)
        case ( DTFFT_TRANSPOSE_X_TO_Z%val )
            tag = "Transpose 3D; X->Z; DTFFT_TRANSPOSE_MODE_PACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(1), pencils(3), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_PACK)
            tag = "Transpose 3D; X->Z; DTFFT_TRANSPOSE_MODE_UNPACK; Backend = " // back
            call check_handle_roundtrip(backend, pencils(1), pencils(3), cart_comm, helper_comms, tag, transpose_mode=DTFFT_TRANSPOSE_MODE_UNPACK)
        end select

        deallocate(tag)
        call destroy_pencils(pencils)
        call free_grid(cart_comm, axis_comms)
    end subroutine test_transpose_3d_internal

    subroutine test_brick_pencil_scenario(backend, distributed_axes, backend_name)
        type(dtfft_backend_t), intent(in) :: backend
        integer(int32), intent(in) :: distributed_axes
        character(len=*), intent(in) :: backend_name
        integer(int32) :: grid_dims(3), dims_2d(2), dims_3d(3)
        character(len=64) :: tag

        select case ( distributed_axes )
        case ( 1 )
            grid_dims = [comm_size, 1, 1]
        case ( 2 )
            dims_2d = 0
            call MPI_Dims_create(comm_size, 2, dims_2d, ierr)
            if ( any(dims_2d < 2) ) return
            grid_dims = [dims_2d(1), dims_2d(2), 1]
        case ( 3 )
            dims_3d = 0
            call MPI_Dims_create(comm_size, 3, dims_3d, ierr)
            if ( any(dims_3d < 2) ) return
            grid_dims = dims_3d
        end select

        write(tag, "(a, i0, a, a)") "bricks-pencils ", distributed_axes, "D ", backend_name
        call run_brick_pencil_case(backend, grid_dims, trim(tag))
    end subroutine test_brick_pencil_scenario

    subroutine test_bricks_x_to_yz(backend, backend_name)
        type(dtfft_backend_t), intent(in) :: backend
        character(len=*), intent(in) :: backend_name
        integer(int32) :: grid_dims(3), process_dims(2)
        character(len=64) :: tag

        if ( comm_size < 4 ) return
        process_dims = 0
        call MPI_Dims_create(comm_size, 2, process_dims, ierr)
        if ( any(process_dims < 2) ) return
        grid_dims = [1, process_dims(1), process_dims(2)]
        if ( product(grid_dims(2:3)) /= comm_size ) return

        write(tag, "(a, i0, a, i0, a, i0, a, a)") "bricks [", comm_size, ",1,1] -> [1,", &
            grid_dims(2), ",", grid_dims(3), "] ", backend_name
        call run_brick_pencil_case(backend, grid_dims, trim(tag), source_x_slab=.true.)
    end subroutine test_bricks_x_to_yz

    subroutine run_brick_pencil_case(backend, grid_dims, tag, source_x_slab)
        type(dtfft_backend_t), intent(in) :: backend
        integer(int32), intent(in) :: grid_dims(3)
        character(len=*), intent(in) :: tag
        logical, optional, intent(in) :: source_x_slab
        type(pencil) :: bricks, x_pencil
        TYPE_MPI_COMM :: cart_comm, axis_comms(3), helper_comms(3)
        integer(int32) :: coords(3), brick_starts(3), brick_counts(3)
        integer(int32) :: pencil_starts(3), pencil_counts(3), xy_rank, xy_size, d
        logical :: source_x_slab_

        source_x_slab_ = .false.
        if ( present(source_x_slab) ) source_x_slab_ = source_x_slab
        call create_grid(grid_dims, cart_comm, axis_comms)
        call MPI_Cart_coords(cart_comm, comm_rank, 3, coords, ierr)
        if ( source_x_slab_ ) then
            call local_range(DIMS(1), comm_rank, comm_size, brick_starts(1), brick_counts(1))
            brick_starts(2:3) = 0
            brick_counts(2:3) = DIMS(2:3)
        else
            do d = 1, 3
                call local_range(DIMS(d), coords(d), grid_dims(d), brick_starts(d), brick_counts(d))
            enddo
        endif
        bricks = pencil(3_int8, 1_int8, DIMS, [cart_comm, MPI_COMM_SELF, MPI_COMM_SELF], &
                        lstarts=brick_starts, lcounts=brick_counts)

        pencil_starts(1) = 0
        pencil_counts(1) = DIMS(1)
        if ( source_x_slab_ ) then
            call local_range(DIMS(2), coords(2), grid_dims(2), pencil_starts(2), pencil_counts(2))
        else
            xy_rank = coords(2) * grid_dims(1) + coords(1)
            xy_size = grid_dims(1) * grid_dims(2)
            call local_range(DIMS(2), xy_rank, xy_size, pencil_starts(2), pencil_counts(2))
        endif
        call local_range(DIMS(3), coords(3), grid_dims(3), pencil_starts(3), pencil_counts(3))
        x_pencil = pencil(3_int8, 1_int8, DIMS, [cart_comm, MPI_COMM_SELF, MPI_COMM_SELF], &
                          lstarts=pencil_starts, lcounts=pencil_counts)

        if ( source_x_slab_ ) then
            call test%check(tag//": bricks [x,1,1]", merge(1, 0, bricks%is_distributed), 1)
            helper_comms = [cart_comm, cart_comm, axis_comms(3)]
        else
            call test%check(tag//": bricks distribution", merge(1, 0, bricks%is_distributed), &
                            merge(1, 0, comm_size > 1))
            helper_comms = [cart_comm, axis_comms(1), axis_comms(3)]
        endif
        call test%check(tag//": pencil distribution", merge(1, 0, .not.x_pencil%is_distributed), 1)
        call check_handle_roundtrip(backend, bricks, x_pencil, cart_comm, helper_comms, tag)

        call bricks%destroy()
        call x_pencil%destroy()
        call free_grid(cart_comm, axis_comms)
    end subroutine run_brick_pencil_case

    subroutine check_handle_roundtrip(backend, send_layout, recv_layout, base_comm, helper_comms, tag, transpose_mode)
        type(dtfft_backend_t), intent(in) :: backend
        type(pencil), intent(in) :: send_layout, recv_layout
        TYPE_MPI_COMM, intent(in) :: base_comm, helper_comms(3)
        character(len=*), intent(in) :: tag
        type(dtfft_transpose_mode_t), optional, intent(in) :: transpose_mode
        integer(int64) :: buffer_size
        real(real32), allocatable, target :: input(:), transposed(:), restored(:)

        buffer_size = max(product(int(send_layout%counts, int64)), &
                          product(int(recv_layout%counts, int64)))
        buffer_size = max(buffer_size, 1_int64)
        allocate(input(buffer_size), transposed(buffer_size), restored(buffer_size))
        call fill_layout(input, send_layout)
        transposed = -1.0_real32
        restored = -2.0_real32

        call execute_handle(backend, send_layout, recv_layout, input, transposed, &
                    base_comm, helper_comms, tag//": forward", transpose_mode=transpose_mode)
        call check_layout(transposed, recv_layout, tag//": forward data")
        call execute_handle(backend, recv_layout, send_layout, transposed, restored, &
                    base_comm, helper_comms, tag//": backward", transpose_mode=transpose_mode)
        call check_layout(restored, send_layout, tag//": roundtrip data")
    end subroutine check_handle_roundtrip

    subroutine execute_handle(backend, send_layout, recv_layout, input, output, base_comm, helper_comms, tag, transpose_mode)
        type(dtfft_backend_t), intent(in) :: backend
        type(pencil), intent(in) :: send_layout, recv_layout
        real(real32), target, intent(inout) :: input(:), output(:)
        TYPE_MPI_COMM, intent(in) :: base_comm, helper_comms(3)
        character(len=*), intent(in) :: tag
        type(dtfft_transpose_mode_t), optional, intent(in) :: transpose_mode
        class(abstract_reshape_handle), allocatable :: handle
        type(backend_helper) :: helper
        type(pencil), target :: helper_pencils(2)
        type(create_args) :: create_kwargs
        type(execute_args) :: execute_kwargs
        real(real32), allocatable, target :: aux(:)
        type(c_ptr) :: aux_ptr
        integer(int64) :: aux_bytes

        helper_pencils = [send_layout, recv_layout]
        call helper%create(DTFFT_PLATFORM_HOST, base_comm, helper_comms, .false., helper_pencils)

        create_kwargs%platform = DTFFT_PLATFORM_HOST
        create_kwargs%helper = helper
        create_kwargs%effort = DTFFT_ESTIMATE
        create_kwargs%backend = backend
        create_kwargs%force_effort = .false.
        create_kwargs%base_type = MPI_REAL
        create_kwargs%base_storage = FLOAT_STORAGE_SIZE
        if ( present(transpose_mode) ) then
            create_kwargs%transpose_mode = transpose_mode
        else
            create_kwargs%transpose_mode = DTFFT_TRANSPOSE_MODE_PACK
        endif
! #ifdef DTFFT_WITH_COMPRESSION
!         create_kwargs%compression_config = get_conf_reshape()
! #endif

        if ( backend == DTFFT_BACKEND_MPI_DATATYPE ) then
            allocate(reshape_handle_datatype :: handle)
        else
            allocate(reshape_handle_generic :: handle)
        endif
        call handle%create(send_layout, recv_layout, create_kwargs)

        aux_bytes = handle%get_aux_bytes()
        aux_ptr = c_null_ptr
        if ( aux_bytes > 0_int64 ) then
            allocate(aux(aux_bytes / FLOAT_STORAGE_SIZE))
            aux_ptr = c_loc(aux(1))
        endif
        execute_kwargs%stream = NULL_STREAM
        execute_kwargs%exec_type = EXEC_BLOCKING
        execute_kwargs%p1 = aux_ptr
        execute_kwargs%p2 = c_null_ptr
        execute_kwargs%p3 = c_null_ptr
        call handle%execute(c_loc(input(1)), c_loc(output(1)), execute_kwargs, ierr)
        call test%check(tag, ierr, DTFFT_SUCCESS)

        call handle%destroy()
        deallocate(handle)
        if ( allocated(aux) ) deallocate(aux)
        call helper%destroy()
    end subroutine execute_handle

    subroutine create_grid(grid_dims, cart_comm, axis_comms)
        integer(int32), intent(in) :: grid_dims(3)
        TYPE_MPI_COMM, intent(out) :: cart_comm, axis_comms(3)
        logical :: periods(3), remain_dims(3)
        integer(int32) :: axis

        periods = .false.
        call MPI_Cart_create(MPI_COMM_WORLD, 3, grid_dims, periods, .false., cart_comm, ierr)
        do axis = 1, 3
            remain_dims = .false.
            remain_dims(axis) = .true.
            call MPI_Cart_sub(cart_comm, remain_dims, axis_comms(axis), ierr)
        enddo
    end subroutine create_grid

    subroutine free_grid(cart_comm, axis_comms)
        TYPE_MPI_COMM, intent(inout) :: cart_comm, axis_comms(3)
        integer(int32) :: axis

        do axis = 1, 3
            call MPI_Comm_free(axis_comms(axis), ierr)
        enddo
        call MPI_Comm_free(cart_comm, ierr)
    end subroutine free_grid

    subroutine destroy_pencils(pencils)
        type(pencil), intent(inout) :: pencils(:)
        integer(int32) :: pencil_id

        do pencil_id = 1, size(pencils)
            call pencils(pencil_id)%destroy()
        enddo
    end subroutine destroy_pencils

    subroutine local_range(global_size, coordinate, process_count, start, count)
        integer(int32), intent(in) :: global_size, coordinate, process_count
        integer(int32), intent(out) :: start, count
        integer(int32) :: base, remainder, first_extra_rank

        base = global_size / process_count
        remainder = mod(global_size, process_count)
        first_extra_rank = process_count - remainder
        count = base + merge(1, 0, coordinate >= first_extra_rank .and. remainder > 0)
        start = coordinate * base + max(0, coordinate - first_extra_rank)
    end subroutine local_range

    subroutine fill_layout(buffer, layout)
        real(real32), intent(out) :: buffer(:)
        type(pencil), intent(in) :: layout
        integer(int32) :: index_x, index_y, index_z, offset, global_coords(3), axis
        integer(int32) :: z_count

        z_count = 1
        if ( layout%rank == 3 ) z_count = layout%counts(3)
        offset = 0
        do index_z = 1, z_count
            do index_y = 1, layout%counts(2)
                do index_x = 1, layout%counts(1)
                    offset = offset + 1
                    global_coords = 0
                    do axis = 1, layout%rank
                        global_coords(layout_axis(layout%rank, layout%aligned_dim, axis)) = &
                            layout%starts(axis) + merge(index_x, merge(index_y, index_z, axis == 2), axis == 1) - 1
                    enddo
                    buffer(offset) = reference_value(global_coords(1), global_coords(2), global_coords(3))
                enddo
            enddo
        enddo
    end subroutine fill_layout

    subroutine check_layout(buffer, layout, tag)
        real(real32), intent(in) :: buffer(:)
        type(pencil), intent(in) :: layout
        character(len=*), intent(in) :: tag
        integer(int32) :: index_x, index_y, index_z, offset, global_coords(3), axis
        integer(int32) :: z_count, mismatches, local_indices(3)

        z_count = 1
        if ( layout%rank == 3 ) z_count = layout%counts(3)
        offset = 0
        mismatches = 0
        if ( comm_rank == 0 ) then
            write(output_unit, '(A)') 'Testing '//tag
        endif
        do index_z = 1, z_count
            do index_y = 1, layout%counts(2)
                do index_x = 1, layout%counts(1)
                    offset = offset + 1
                    local_indices = [index_x, index_y, index_z]
                    global_coords = 0
                    do axis = 1, layout%rank
                        global_coords(layout_axis(layout%rank, layout%aligned_dim, axis)) = &
                            layout%starts(axis) + local_indices(axis) - 1
                    enddo
                    if ( transfer(buffer(offset), 0_int32) /= &
                         transfer(reference_value(global_coords(1), global_coords(2), global_coords(3)), 0_int32) ) then
                        mismatches = mismatches + 1
                    endif
                enddo
            enddo
        enddo
        call test%check(tag, mismatches, 0)
    end subroutine check_layout

    integer(int32) function layout_axis(rank, aligned_dim, slot)
        integer(int8), intent(in) :: rank, aligned_dim
        integer(int32), intent(in) :: slot

        if ( rank == 2 ) then
            if ( aligned_dim == 1_int8 ) then
                layout_axis = slot
            else
                layout_axis = 3 - slot
            endif
        else
            select case ( aligned_dim )
            case ( 1_int8 )
                layout_axis = slot
            case ( 2_int8 )
                layout_axis = mod(slot, 3) + 1
            case default
                layout_axis = mod(slot + 1, 3) + 1
            end select
        endif
    end function layout_axis

    real(real32) function reference_value(global_x, global_y, global_z)
        integer(int32), intent(in) :: global_x, global_y, global_z

        reference_value = real(mod(int(global_x, int64) * 1009_int64 + int(global_y, int64) * 101_int64 &
                                   + int(global_z, int64) * 11_int64, 65521_int64), real32)
    end function reference_value

    subroutine find_first_two_divisors(n, divisors, count)
        integer(int32), intent(in) :: n
        integer(int32), intent(out) :: divisors(2)
        integer(int32), intent(out) :: count
        integer :: i

        divisors = 0
        count = 0

        do i = 2, n/2
            if (mod(n, i) == 0) then
                count = count + 1
                divisors(count) = i
                if (count == 2) exit
            end if
        end do
    end subroutine find_first_two_divisors
end program test_reshape_handles
