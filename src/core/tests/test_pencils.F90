!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
#include "dtfft_config.h"
program test_r2r_2d
use iso_fortran_env
use iso_c_binding
use dtfft
use test_suite
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "dtfft.f03"
implicit none
    integer(int32), parameter :: AXIS_X = 1, AXIS_Y = 2, AXIS_Z = 4
    integer(int32), parameter :: PARTITION_CASES(8) = &
        [0, AXIS_X, AXIS_Y, AXIS_Z, AXIS_X + AXIS_Y, AXIS_X + AXIS_Z, &
         AXIS_Y + AXIS_Z, AXIS_X + AXIS_Y + AXIS_Z]
    character(len=4), parameter :: PARTITION_NAMES(8) = &
        [character(len=4) :: "none", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]
    integer(int32) :: comm_size, comm_rank, ierr, case_index
    type(dtfft_plan_r2r_t) :: plan
    integer(int32) :: lbounds(3), sizes(3), global_dims(3), process_dims(3)
    integer(int32) :: starts_4d(4), counts_4d(4)
    type(dtfft_pencil_t) :: pencil
    integer(int32), pointer :: dims(:)
    type(test_suite_t) :: test
    type(dtfft_config_t) :: config

    call test%create("Pencil plan creation")

    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

    config = dtfft_config_t(enable_fourier_reshape=.true.)
    ! config%enable_fourier_reshape = .true.
    call dtfft_set_config(config)

    call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
    call test%check("Pencil not created", ierr, DTFFT_ERROR_PENCIL_NOT_INITIALIZED)

    call test_invalid_array_shapes()
    do case_index = 1, size(PARTITION_CASES)
        call test_partition_case(PARTITION_CASES(case_index), trim(PARTITION_NAMES(case_index)))
    enddo
    call test_zero_sized_slabs()

    call test%finalize()
contains
    subroutine test_invalid_array_shapes()
        call set_partition([1, 1, 1])
        pencil = dtfft_pencil_t(lbounds(1:2), sizes)
        call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
        call test%check("Pencil arrays size mismatch", ierr, DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH)

        pencil = dtfft_pencil_t(lbounds(1:1), sizes(1:1))
        call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
        call test%check("Pencil too few dimensions", ierr, DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES)

        starts_4d = 0
        counts_4d = 4
        pencil = dtfft_pencil_t(starts_4d, counts_4d)
        call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
        call test%check("Pencil too many dimensions", ierr, DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES)
    end subroutine test_invalid_array_shapes

    subroutine test_partition_case(axis_mask, case_name)
        integer(int32), intent(in) :: axis_mask
        character(len=*), intent(in) :: case_name
        integer(int32) :: d, changed_axis, count_axis
        integer(int32) :: case_global_dims(3), case_process_dims(3)
        logical :: possible

        call make_process_grid(axis_mask, case_process_dims, possible)
        if ( .not. possible ) return
        if ( axis_mask == 0 .and. comm_size /= 1 ) return
        call set_partition(case_process_dims)
        case_global_dims = global_dims

        do d = 1, 3
            call set_partition(case_process_dims)
            if ( comm_rank == 0 ) lbounds(d) = -1
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": negative start", ierr, DTFFT_ERROR_PENCIL_INVALID_STARTS)

            call set_partition(case_process_dims)
            if ( comm_rank == 0 ) sizes(d) = -1
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": negative count", ierr, DTFFT_ERROR_PENCIL_INVALID_COUNTS)
        enddo

        if ( comm_size > 1 ) then
            call set_partition(case_process_dims)
            if ( comm_rank == 1 ) lbounds = 0
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": full overlap", ierr, DTFFT_ERROR_PENCIL_OVERLAP)

            call set_partition(case_process_dims)
            changed_axis = first_nonzero_coordinate(case_process_dims, 1)
            if ( comm_rank == 1 ) lbounds(changed_axis) = lbounds(changed_axis) - 1
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": partial overlap", ierr, DTFFT_ERROR_PENCIL_OVERLAP)

            call set_partition(case_process_dims)
            changed_axis = first_active_axis(case_process_dims)
            count_axis = 1
            if ( count_axis == changed_axis ) count_axis = 2
            if ( comm_rank == 0 ) sizes(count_axis) = sizes(count_axis) - 1
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": shape mismatch", ierr, DTFFT_ERROR_PENCIL_SHAPE_MISMATCH)

            call set_partition(case_process_dims)
            changed_axis = first_active_axis(case_process_dims)
            if ( coordinate(changed_axis, case_process_dims) >= 1 ) &
                lbounds(changed_axis) = lbounds(changed_axis) + 1
            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check(trim(case_name)//": discontinuous", ierr, DTFFT_ERROR_PENCIL_NOT_CONTINUOUS)
        endif

        call set_partition(case_process_dims)
        pencil = dtfft_pencil_t(lbounds, sizes)
        call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
        call test%check(trim(case_name)//": valid partition", ierr, DTFFT_SUCCESS)
        call plan%get_dims(dims, error_code=ierr)
        call test%check(trim(case_name)//": get_dims", ierr, DTFFT_SUCCESS)
        call test%assert_false(trim(case_name)//": global dimensions", any(dims /= case_global_dims))
        call check_roundtrip(trim(case_name))
        call plan%destroy()
    end subroutine test_partition_case

    subroutine test_zero_sized_slabs()
        integer(int32) :: d, global_size, base, remainder, rank
        integer(int32) :: expected_dims(3)

        if ( comm_size <= 1 ) return
        do d = 1, 3
            global_size = max(1, comm_size / 2)
            lbounds = 0
            sizes = [4, 4, 4]
            sizes(d) = 0
            base = global_size / comm_size
            remainder = mod(global_size, comm_size)
            do rank = 0, comm_rank - 1
                lbounds(d) = lbounds(d) + base + merge(1, 0, rank >= comm_size - remainder)
            enddo
            sizes(d) = base + merge(1, 0, comm_rank >= comm_size - remainder)
            expected_dims = [4, 4, 4]
            expected_dims(d) = global_size

            pencil = dtfft_pencil_t(lbounds, sizes)
            call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
            call test%check("zero-sized slabs: plan", ierr, DTFFT_SUCCESS)
            call plan%get_dims(dims, error_code=ierr)
            call test%check("zero-sized slabs: get_dims", ierr, DTFFT_SUCCESS)
            call test%assert_false("zero-sized slabs: global dimensions", any(dims /= expected_dims))
            call check_roundtrip("zero_sized_slabs")
            call plan%destroy()
        enddo
    end subroutine test_zero_sized_slabs

    subroutine check_roundtrip(case_name)
        character(len=*), intent(in) :: case_name
        integer(int8), pointer :: input_buffer(:), output_buffer(:)
        integer(int32) :: local_counts(3), n_mismatches
        integer(int64) :: buffer_bytes, aux_bytes, data_bytes, element_size, i
        type(c_ptr) :: input_ptr, output_ptr, aux_ptr

        call plan%report()
        buffer_bytes = plan%get_alloc_bytes(ierr)
        call test%check(case_name//": get allocation size", ierr, DTFFT_SUCCESS)
        call plan%get_local_sizes(in_counts=local_counts, error_code=ierr)
        call test%check(case_name//": get local sizes", ierr, DTFFT_SUCCESS)
        element_size = plan%get_element_size(ierr)
        call test%check(case_name//": get element size", ierr, DTFFT_SUCCESS)
        aux_bytes = plan%get_aux_bytes(ierr)
        call test%check(case_name//": get auxiliary size", ierr, DTFFT_SUCCESS)
        data_bytes = product(int(local_counts, int64)) * element_size
        buffer_bytes = max(buffer_bytes, 8_int64)

        input_ptr = plan%mem_alloc_ptr(buffer_bytes, ierr)
        call test%check(case_name//": allocate input", ierr, DTFFT_SUCCESS)
        output_ptr = plan%mem_alloc_ptr(buffer_bytes, ierr)
        call test%check(case_name//": allocate output", ierr, DTFFT_SUCCESS)
        aux_ptr = plan%mem_alloc_ptr(max(aux_bytes, 8_int64), ierr)
        call test%check(case_name//": allocate auxiliary buffer", ierr, DTFFT_SUCCESS)
        call c_f_pointer(input_ptr, input_buffer, [int(buffer_bytes)])
        call c_f_pointer(output_ptr, output_buffer, [int(buffer_bytes)])

        do i = 1, data_bytes
            input_buffer(i) = int(mod(i - 1_int64, 127_int64), int8)
            output_buffer(i) = 0_int8
        enddo

        call plan%execute_ptr(input_ptr, output_ptr, DTFFT_EXECUTE_FORWARD, aux_ptr, ierr)
        call test%check(case_name//": forward execute", ierr, DTFFT_SUCCESS)
        call plan%execute_ptr(output_ptr, input_ptr, DTFFT_EXECUTE_BACKWARD, aux_ptr, ierr)
        call test%check(case_name//": backward execute", ierr, DTFFT_SUCCESS)

        n_mismatches = 0
        do i = 1, data_bytes
            ! if(comm_rank == 2) print*,input_buffer(i), int(mod(i - 1_int64, 127_int64), int8)
            if ( input_buffer(i) /= int(mod(i - 1_int64, 127_int64), int8) ) then
                n_mismatches = n_mismatches + 1
            endif
        enddo
        call test%check(case_name//": forward/backward exact roundtrip", n_mismatches, 0)

        call plan%mem_free_ptr(input_ptr, ierr)
        call plan%mem_free_ptr(output_ptr, ierr)
        call plan%mem_free_ptr(aux_ptr, ierr)
    end subroutine check_roundtrip

    subroutine make_process_grid(axis_mask, grid, possible)
        integer(int32), intent(in) :: axis_mask
        integer(int32), intent(out) :: grid(3)
        logical, intent(out) :: possible
        integer(int32) :: remaining, factor, axis, active_count
        integer(int32) :: axes(3), n_axes

        grid = 1
        n_axes = 0
        do axis = 1, 3
            if ( btest(axis_mask, axis - 1) ) then
                n_axes = n_axes + 1
                axes(n_axes) = axis
            endif
        enddo

        if ( n_axes == 0 ) then
            possible = comm_size == 1
            return
        endif

        remaining = comm_size
        active_count = 0
        do factor = 2, remaining
            do while ( mod(remaining, factor) == 0 )
                axis = axes(mod(active_count, n_axes) + 1)
                grid(axis) = grid(axis) * factor
                active_count = active_count + 1
                remaining = remaining / factor
            enddo
        enddo
        possible = active_count >= n_axes
    end subroutine make_process_grid

    subroutine set_partition(grid)
        integer(int32), intent(in) :: grid(3)
        integer(int32) :: x, y, z

        process_dims = grid
        global_dims = 4 * grid
        x = comm_rank / (grid(2) * grid(3))
        y = mod(comm_rank / grid(3), grid(2))
        z = mod(comm_rank, grid(3))
        lbounds = [4 * x, 4 * y, 4 * z]
        sizes = 4
    end subroutine set_partition

    integer(int32) function coordinate(axis, grid, rank) result(value)
        integer(int32), intent(in) :: axis, grid(3)
        integer(int32), intent(in), optional :: rank
        integer(int32) :: rank_value

        rank_value = comm_rank
        if ( present(rank) ) rank_value = rank
        select case (axis)
        case (1)
            value = rank_value / (grid(2) * grid(3))
        case (2)
            value = mod(rank_value / grid(3), grid(2))
        case default
            value = mod(rank_value, grid(3))
        end select
    end function coordinate

    integer(int32) function first_active_axis(grid) result(axis)
        integer(int32), intent(in) :: grid(3)
        do axis = 1, 3
            if ( grid(axis) > 1 ) return
        enddo
    end function first_active_axis

    integer(int32) function first_nonzero_coordinate(grid, rank) result(axis)
        integer(int32), intent(in) :: grid(3)
        integer(int32), intent(in) :: rank
        do axis = 1, 3
            if ( coordinate(axis, grid, rank) > 0 ) return
        enddo
    end function first_nonzero_coordinate

end program test_r2r_2d