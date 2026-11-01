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
program test_plan_save_load
!! Checks that a plan written with `plan%save` can be rebuilt with `plan%load` and that the
!! rebuilt plan is indistinguishable from the tuned one -- without running any autotuning.
use iso_fortran_env
use iso_c_binding
use dtfft
use dtfft_pencil,      only: dtfft_pencil_t
! Reading the plan kind out of a file is deliberately not part of the public API: nothing
! outside the library needs it, since `load` works the kind out on its own. Reaching into the
! internal modules keeps it under test without putting it on the public surface.
use dtfft_parameters,  only: plan_kind_t, PLAN_C2C, PLAN_R2C, PLAN_R2R, DTFFT_PLATFORM_CUDA
use dtfft_plan,        only: get_plan_kind_from_file
use dtfft_config,      only: get_conf_platform
use dtfft_test_utils,  only: test_context
#include "_dtfft_mpi.h"
implicit none
    integer(int32), parameter :: DIMS_3D(3) = [32, 24, 16]
    integer(int32), parameter :: DIMS_2D(2) = [32, 24]

    integer(int32) :: comm_rank, comm_size, ierr
    type(test_context) :: test

    call MPI_Init(ierr)

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

    call report("test_plan_save_load: "//to_string(comm_size)//" ranks")

    call test_c2c_roundtrip()
    call test_patient_roundtrip()
    call test_exhaustive_roundtrip()
    call test_adaptive_roundtrip(.true.)
    call test_adaptive_roundtrip(.false.)
    call test_r2c_roundtrip()
    call test_pencil_roundtrip()
    call test_pencil_reshape_roundtrip()
    call test_config_ignored_on_load()
#if defined(DTFFT_WITH_COMPRESSION) && defined(DTFFT_WITH_ZFP)
    call test_compressed_roundtrip()
#endif
    call test_r2r_roundtrip()
    call test_2d_roundtrip()
    call test_plan_kind_probe()
    call test_missing_file()
    call test_corrupted_file()
    call test_format_version_gate()
    ! call test_self_describing_sizes()
    ! call test_unavailable_backend()
    call test_kind_mismatch()
    call test_comm_size_mismatch()
    call test_save_before_create()

    if ( test%failure_count() == 0 ) then
        call report("All tests PASSED")
    else
        call report(to_string(test%failure_count())//" test(s) FAILED")
    endif

    call MPI_Finalize(ierr)
    if ( test%failure_count() /= 0 ) error stop 1

contains

!------------------------------------------------------------------------------------------------
! Round-trip tests
!------------------------------------------------------------------------------------------------
    subroutine test_c2c_roundtrip()
    !! Tunes a 3D C2C plan, saves it, loads it back and compares every observable property
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_c2c.dtfft"

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_MEASURE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("c2c: create tuned plan", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("c2c: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)

        call check("c2c: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("c2c", size(DIMS_3D), tuned, loaded)
        call check_transpose_roundtrip("c2c-tuned", size(DIMS_3D), tuned)
        call check_transpose_roundtrip("c2c", size(DIMS_3D), loaded)
        call check_resave_identical("c2c", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        ! call delete_file(filename)
    end subroutine test_c2c_roundtrip

    subroutine test_patient_roundtrip()
    !! `DTFFT_PATIENT` additionally benchmarks communication backends and, for the
    !! MPI-datatype backend, per-direction transpose modes. All of that has to be replayed.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_patient.dtfft"

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_PATIENT, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("patient: create tuned plan", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("patient: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("patient: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("patient", size(DIMS_3D), tuned, loaded)
        call check_transpose_roundtrip("patient-tuned", size(DIMS_3D), tuned)
        call check_transpose_roundtrip("patient", size(DIMS_3D), loaded)
        call check_resave_identical("patient", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        call delete_file(filename)
    end subroutine test_patient_roundtrip

    subroutine test_exhaustive_roundtrip()
    !! `DTFFT_EXHAUSTIVE` also benchmarks the local pack/unpack kernels. Those results are
    !! per rank, so they live in the per-rank part of the file.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_exhaustive.dtfft"

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_EXHAUSTIVE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("exhaustive: create tuned plan", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("exhaustive: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("exhaustive: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("exhaustive", size(DIMS_3D), tuned, loaded)
        call check_transpose_roundtrip("exhaustive-tuned", size(DIMS_3D), tuned)
        call check_transpose_roundtrip("exhaustive", size(DIMS_3D), loaded)
        call check_resave_identical("exhaustive", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        call delete_file(filename)
    end subroutine test_exhaustive_roundtrip

    subroutine test_adaptive_roundtrip(use_z_slab)
    !! `DTFFT_BACKEND_ADAPTIVE` picks a separate backend for every transposition, so the
    !! plan keeps a whole table of them rather than one value.
    !!
    !! It is also the case where replaying used to be refused outright: loading forces effort
    !! down to `DTFFT_ESTIMATE`, and an adaptive backend is not something that effort level is
    !! ever allowed to *choose*. Choosing and replaying are different things.
    !!
    !! Run both with and without Z-slab: the number of transpositions differs between the two,
    !! and with it how much of the per-direction backend table is actually filled in. Only the
    !! shorter case can expose a tail that was written to file without being set.
        logical, intent(in)     :: use_z_slab   !! Allow the Z-slab optimization
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        type(dtfft_config_t)    :: conf, default_conf
        character(len=:), allocatable :: tag
        character(len=*), parameter :: filename = "test_save_load_adaptive.dtfft"

        ! Strange gfortran bug workaround
        if(use_z_slab) then
            allocate( tag, source="adaptive(z_slab=ON)" )
        else
            allocate( tag, source="adaptive(z_slab=OFF)" )
        endif

        call dtfft_create_config(default_conf)
        conf = default_conf
        conf%backend = DTFFT_BACKEND_ADAPTIVE
        conf%enable_mpi_backends = .true.
        conf%enable_z_slab = use_z_slab
        ! The adaptive backend is rejected outright on the CUDA platform, so there is nothing
        ! to replay there. Bail out before touching the configuration, since restoring it is
        ! what the rest of this routine would otherwise have to guarantee on every path.
        if ( get_conf_platform() == DTFFT_PLATFORM_CUDA ) then
            call report("  skip "//tag//": adaptive backend is not supported on CUDA platform")
            deallocate( tag )
            return
        endif

        call dtfft_set_config(conf, ierr)
        call check(tag//": set config", ierr, DTFFT_SUCCESS)

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_EXHAUSTIVE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check(tag//": create tuned plan", ierr, DTFFT_SUCCESS)

        ! No early returns past this point: the configuration is global state and has to be
        ! handed back whatever happens.
        if ( all_ok(ierr) ) then
            call tuned%save(filename, ierr)
            call check(tag//": save", ierr, DTFFT_SUCCESS)

            call loaded%load(filename, MPI_COMM_WORLD, ierr)
            call check(tag//": load", ierr, DTFFT_SUCCESS)

            if ( all_ok(ierr) ) then
                call compare_plans(tag, size(DIMS_3D), tuned, loaded)
                call check_transpose_roundtrip(tag//"-tuned", size(DIMS_3D), tuned)
                call check_transpose_roundtrip(tag, size(DIMS_3D), loaded)
                call check_resave_identical(tag, filename, loaded)
                call loaded%destroy(ierr)
            endif

            call tuned%destroy(ierr)
            call delete_file(filename)
        endif

        call dtfft_set_config(default_conf, ierr)
        call check(tag//": restore config", ierr, DTFFT_SUCCESS)
        deallocate( tag )
    end subroutine test_adaptive_roundtrip

    subroutine test_r2r_roundtrip()
    !! R2R plans carry `kinds`, which have to survive the round trip as well
        type(dtfft_plan_r2r_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_r2r.dtfft"
        type(dtfft_r2r_kind_t)  :: kinds(3)

        kinds = [DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_2]

        call tuned%create(DIMS_3D, kinds=kinds, precision=DTFFT_SINGLE, effort=DTFFT_MEASURE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("r2r: create tuned plan", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("r2r: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("r2r: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("r2r", size(DIMS_3D), tuned, loaded)
        call check_transpose_roundtrip("r2r-tuned", size(DIMS_3D), tuned)
        call check_transpose_roundtrip("r2r", size(DIMS_3D), loaded)
        call check_resave_identical("r2r", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        call delete_file(filename)
    end subroutine test_r2r_roundtrip

    subroutine test_2d_roundtrip()
    !! 2D plans take a different path through plan creation, so they get their own check
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_2d.dtfft"

        call tuned%create(DIMS_2D, precision=DTFFT_DOUBLE, effort=DTFFT_ESTIMATE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("2d: create tuned plan", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("2d: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("2d: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("2d", size(DIMS_2D), tuned, loaded)
        call check_transpose_roundtrip("2d-tuned", size(DIMS_2D), tuned)
        call check_transpose_roundtrip("2d", size(DIMS_2D), loaded)
        call check_resave_identical("2d", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        call delete_file(filename)
    end subroutine test_2d_roundtrip

    subroutine test_r2c_roundtrip()
    !! R2C has a separate real-space pencil and a different FFT reconstruction path.
        type(dtfft_plan_r2c_t)  :: tuned, loaded
        character(len=*), parameter :: filename = "test_save_load_r2c.dtfft"

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_ESTIMATE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("r2c: create", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call tuned%save(filename, ierr)
        call check("r2c: save", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("r2c: load", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) return

        call compare_plans("r2c", size(DIMS_3D), tuned, loaded)
        call check_resave_identical("r2c", filename, loaded)

        call tuned%destroy(ierr)
        call loaded%destroy(ierr)
        call delete_file(filename)
    end subroutine test_r2c_roundtrip

    subroutine test_pencil_roundtrip()
    !! A plan created from an explicit pencil must save and restore that user decomposition.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        type(dtfft_pencil_t)    :: user_pencil
        character(len=*), parameter :: filename = "test_save_load_pencil.dtfft"

        call make_z_pencil(user_pencil)
        if ( .not. allocated(user_pencil%starts) ) return

        call tuned%create(user_pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, &
                          DTFFT_EXECUTOR_NONE, ierr)
        call check("pencil: create", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) then
            call user_pencil%destroy()
            return
        endif

        call tuned%save(filename, ierr)
        call check("pencil: save", ierr, DTFFT_SUCCESS)
        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("pencil: load", ierr, DTFFT_SUCCESS)
        if ( all_ok(ierr) ) then
            call compare_plans("pencil", size(DIMS_3D), tuned, loaded)
            call check_transpose_roundtrip("pencil-tuned", size(DIMS_3D), tuned)
            call check_transpose_roundtrip("pencil", size(DIMS_3D), loaded)
            call check_resave_identical("pencil", filename, loaded)
            call loaded%destroy(ierr)
        endif

        call tuned%destroy(ierr)
        call user_pencil%destroy()
        call delete_file(filename)
    end subroutine test_pencil_roundtrip

    subroutine test_pencil_reshape_roundtrip()
    !! Distributing the explicit pencil in one direction must preserve the reshape plan too.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        type(dtfft_pencil_t)    :: user_pencil
        character(len=*), parameter :: filename = "test_save_load_pencil_reshape.dtfft"

        if ( comm_size == 1 .or. comm_size > DIMS_3D(1) ) return
        call make_x_pencil(user_pencil)
        if ( .not. allocated(user_pencil%starts) ) return

        call tuned%create(user_pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, &
                          DTFFT_EXECUTOR_NONE, ierr)
        call check("pencil reshape: create", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) then
            call user_pencil%destroy()
            return
        endif
        block
        type(dtfft_backend_t) :: tuned_reshape_backend

        tuned_reshape_backend = tuned%get_reshape_backend(ierr)
        call check("pencil reshape: backend available", ierr, DTFFT_SUCCESS)
        end block

        call tuned%save(filename, ierr)
        call check("pencil reshape: save", ierr, DTFFT_SUCCESS)
        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("pencil reshape: load", ierr, DTFFT_SUCCESS)
        if ( all_ok(ierr) ) then
            block
            type(dtfft_backend_t) :: tuned_reshape_backend, loaded_reshape_backend

            tuned_reshape_backend = tuned%get_reshape_backend(ierr)
            loaded_reshape_backend = loaded%get_reshape_backend(ierr)
            call compare_plans("pencil reshape", size(DIMS_3D), tuned, loaded)
            call check("pencil reshape: backend", loaded_reshape_backend%val, tuned_reshape_backend%val)
            call check_transpose_roundtrip("pencil reshape-tuned", size(DIMS_3D), tuned)
            call check_transpose_roundtrip("pencil reshape", size(DIMS_3D), loaded)
            call check_resave_identical("pencil reshape", filename, loaded)
            call loaded%destroy(ierr)
            end block
        endif

        call tuned%destroy(ierr)
        call user_pencil%destroy()
        call delete_file(filename)
    end subroutine test_pencil_reshape_roundtrip

    subroutine test_config_ignored_on_load()
    !! Loading must use decisions stored in the file, not the current global configuration.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        type(dtfft_config_t)    :: saved_conf, file_conf, other_conf
        character(len=*), parameter :: filename = "test_save_load_config.dtfft"

        if ( comm_size == 1 ) return
        call dtfft_create_config(saved_conf)
        file_conf = saved_conf
        file_conf%enable_z_slab = .false.
        file_conf%enable_y_slab = .true.
        file_conf%backend = DTFFT_BACKEND_MPI_DATATYPE
        file_conf%reshape_backend = DTFFT_BACKEND_MPI_DATATYPE
        file_conf%enable_mpi_backends = .true.
        call dtfft_set_config(file_conf, ierr)
        call check("config ignored: set file config", ierr, DTFFT_SUCCESS)

        call tuned%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("config ignored: create", ierr, DTFFT_SUCCESS)
        if ( .not. all_ok(ierr) ) then
            call dtfft_set_config(saved_conf, ierr)
            return
        endif
        call tuned%save(filename, ierr)
        call check("config ignored: save", ierr, DTFFT_SUCCESS)

        other_conf = file_conf
        other_conf%enable_z_slab = .true.
        other_conf%enable_y_slab = .false.
        other_conf%backend = DTFFT_BACKEND_MPI_P2P
        other_conf%reshape_backend = DTFFT_BACKEND_MPI_P2P
        call dtfft_set_config(other_conf, ierr)
        call check("config ignored: set other config", ierr, DTFFT_SUCCESS)

        call loaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("config ignored: load", ierr, DTFFT_SUCCESS)
        if ( all_ok(ierr) ) then
            call compare_plans("config ignored", size(DIMS_3D), tuned, loaded)
            call check("config ignored: z-slab", merge(1, 0, loaded%get_z_slab_enabled(ierr)), &
                       merge(1, 0, tuned%get_z_slab_enabled(ierr)))
            call check("config ignored: y-slab", merge(1, 0, loaded%get_y_slab_enabled(ierr)), &
                       merge(1, 0, tuned%get_y_slab_enabled(ierr)))
            block
            type(dtfft_backend_t) :: tuned_backend, loaded_backend

            tuned_backend = tuned%get_backend(ierr)
            loaded_backend = loaded%get_backend(ierr)
            call check("config ignored: backend", loaded_backend%val, tuned_backend%val)
            call check_resave_identical("config ignored", filename, loaded)
            call loaded%destroy(ierr)
            end block
        endif

        call tuned%destroy(ierr)
        call dtfft_set_config(saved_conf, ierr)
        call check("config ignored: restore config", ierr, DTFFT_SUCCESS)
        call delete_file(filename)
    end subroutine test_config_ignored_on_load

#if defined(DTFFT_WITH_COMPRESSION) && defined(DTFFT_WITH_ZFP)
    subroutine test_compressed_roundtrip()
    !! A compressed backend exercises serialization of compression settings and backend choice.
        type(dtfft_plan_c2c_t)  :: tuned, loaded
        type(dtfft_config_t)    :: saved_conf, conf
        character(len=*), parameter :: filename = "test_save_load_compressed.dtfft"

        if ( comm_size == 1 ) return
        call dtfft_create_config(saved_conf)
        conf = saved_conf
        ! conf%enable_mpi_backends = .true.
        ! conf%enable_pipelined_backends = .false.
        ! conf%enable_datatype_backend = .false.
        ! conf%enable_compressed_backends = .true.
        conf%backend = DTFFT_BACKEND_MPI_P2P_COMPRESSED
        conf%compression_config_transpose = dtfft_compression_config_t( &
            DTFFT_COMPRESSION_LIB_ZFP, DTFFT_COMPRESSION_MODE_FIXED_RATE, rate=32.0_real64)
        call dtfft_set_config(conf, ierr)
        call check("compressed: set config", ierr, DTFFT_SUCCESS)

        call tuned%create(DIMS_3D, precision=DTFFT_DOUBLE, effort=DTFFT_ESTIMATE, &
                          executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call check("compressed: create", ierr, DTFFT_SUCCESS)
        if ( all_ok(ierr) ) then
            call tuned%save(filename, ierr)
            call check("compressed: save", ierr, DTFFT_SUCCESS)

            conf%compression_config_transpose = dtfft_compression_config_t( &
                DTFFT_COMPRESSION_LIB_ZFP, DTFFT_COMPRESSION_MODE_FIXED_RATE, rate=16.0_real64)
            call dtfft_set_config(conf, ierr)

            call loaded%load(filename, MPI_COMM_WORLD, ierr)
            call check("compressed: load", ierr, DTFFT_SUCCESS)
            if ( all_ok(ierr) ) then
                call compare_plans("compressed", size(DIMS_3D), tuned, loaded)
                block
                type(dtfft_backend_t) :: tuned_backend, loaded_backend

                tuned_backend = tuned%get_backend(ierr)
                loaded_backend = loaded%get_backend(ierr)
                call check("compressed: backend", loaded_backend%val, tuned_backend%val)
                call check_resave_identical("compressed", filename, loaded)
                call loaded%destroy(ierr)
                end block
            endif
            call tuned%destroy(ierr)
        endif

        call dtfft_set_config(saved_conf, ierr)
        call check("compressed: restore config", ierr, DTFFT_SUCCESS)
        call delete_file(filename)
    end subroutine test_compressed_roundtrip
#endif

    subroutine make_x_pencil(user_pencil)
    !! Creates a contiguous pencil decomposition distributed over the X direction.
        type(dtfft_pencil_t), intent(out) :: user_pencil
        integer(int32) :: base, remainder, x_start, x_count

        if ( comm_size > DIMS_3D(1) ) return
        base = DIMS_3D(1) / comm_size
        remainder = mod(DIMS_3D(1), comm_size)
        x_count = base + merge(1, 0, comm_rank < remainder)
        x_start = comm_rank * base + min(comm_rank, remainder)
        user_pencil = dtfft_pencil_t([x_start, 0_int32, 0_int32], [x_count, DIMS_3D(2), DIMS_3D(3)])
    end subroutine make_x_pencil

    subroutine make_z_pencil(user_pencil)
    !! Creates a contiguous pencil decomposition distributed over the Z direction.
        type(dtfft_pencil_t), intent(out) :: user_pencil
        integer(int32) :: base, remainder, z_start, z_count

        if ( comm_size > DIMS_3D(3) ) return
        base = DIMS_3D(3) / comm_size
        remainder = mod(DIMS_3D(3), comm_size)
        z_count = base + merge(1, 0, comm_rank < remainder)
        z_start = comm_rank * base + min(comm_rank, remainder)
        user_pencil = dtfft_pencil_t([0_int32, 0_int32, z_start], [DIMS_3D(1), DIMS_3D(2), z_count])
    end subroutine make_z_pencil

!------------------------------------------------------------------------------------------------
! `read_plan_kind`
!------------------------------------------------------------------------------------------------
    subroutine test_plan_kind_probe()
    !! The plan kind must be readable before any plan object exists. That is what lets the C
    !! wrapper allocate the right concrete class before loading into it.
        type(dtfft_plan_c2c_t)  :: c2c
        type(dtfft_plan_r2r_t)  :: r2r
        type(plan_kind_t) :: plan_kind
        type(dtfft_r2r_kind_t)  :: kinds(3)
        class(dtfft_plan_t), allocatable :: plan

        kinds = [DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_2]

        call c2c%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call c2c%save("test_save_load_kind_c2c.dtfft", ierr)
        call c2c%destroy(ierr)

        call r2r%create(DIMS_3D, kinds=kinds, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call r2r%save("test_save_load_kind_r2r.dtfft", ierr)
        call r2r%destroy(ierr)

        ierr = get_plan_kind_from_file("test_save_load_kind_c2c.dtfft", MPI_COMM_WORLD, plan_kind)
        call check("probe: c2c error code", ierr, DTFFT_SUCCESS)
        ! call check("probe: c2c kind", plan_kind%val, PLAN_C2C%val)

        ierr = get_plan_kind_from_file("test_save_load_kind_r2r.dtfft", MPI_COMM_WORLD, plan_kind)
        call check("probe: r2r error code", ierr, DTFFT_SUCCESS)
        ! call check("probe: r2r kind", plan_kind%val, PLAN_R2R%val)

        ! The whole point of the probe: allocate the matching class, then load into it
        select case ( plan_kind%val )
        case ( PLAN_C2C%val ); allocate( dtfft_plan_c2c_t :: plan )
        case ( PLAN_R2C%val ); allocate( dtfft_plan_r2c_t :: plan )
        case ( PLAN_R2R%val ); allocate( dtfft_plan_r2r_t :: plan )
        endselect
        call plan%load("test_save_load_kind_r2r.dtfft", MPI_COMM_WORLD, ierr)
        call check("probe: load into probed class", ierr, DTFFT_SUCCESS)
        call plan%destroy(ierr)
        deallocate( plan )

        call delete_file("test_save_load_kind_c2c.dtfft")
        call delete_file("test_save_load_kind_r2r.dtfft")
    end subroutine test_plan_kind_probe

!------------------------------------------------------------------------------------------------
! Failure modes
!------------------------------------------------------------------------------------------------
    subroutine test_missing_file()
    !! Loading a file that is not there must fail cleanly, not abort
        type(dtfft_plan_c2c_t)  :: plan
        type(plan_kind_t) :: plan_kind

        call plan%load("test_save_load_does_not_exist.dtfft", MPI_COMM_WORLD, ierr)
        call check("missing file: load", ierr, DTFFT_ERROR_FILE_NOT_FOUND)

        ierr = get_plan_kind_from_file("test_save_load_does_not_exist.dtfft", MPI_COMM_WORLD, plan_kind)
        call check("missing file: probe", ierr, DTFFT_ERROR_FILE_NOT_FOUND)
    end subroutine test_missing_file

    subroutine test_corrupted_file()
    !! A file whose magic does not match must be rejected rather than misinterpreted
        type(dtfft_plan_c2c_t)  :: plan
        character(len=*), parameter :: filename = "test_save_load_garbage.dtfft"
        integer(int32) :: unit

        if ( comm_rank == 0 ) then
            open(newunit=unit, file=filename, access="stream", form="unformatted", status="replace")
            write(unit) repeat("x", 512)
            close(unit)
        endif
        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        call plan%load(filename, MPI_COMM_WORLD, ierr)
        call check("corrupted file: load", ierr, DTFFT_ERROR_FILE_INVALID)

        call delete_file(filename)
    end subroutine test_corrupted_file

    subroutine test_format_version_gate()
    !! Compatibility is decided by the layout version in the file, not by the library version.
    !!
    !! Only two things are refused: a layout newer than this library understands, and one old
    !! enough to have been dropped. Anything in between must stay readable, which is the whole
    !! point of stamping the file with a format version.
        type(dtfft_plan_c2c_t)  :: plan, reloaded
        character(len=*), parameter :: filename = "test_save_load_version.dtfft"
        integer(int8), parameter :: VERSION_OFFSET = 19_int8   !! 16 bytes of magic + 2 of endian marker

        call plan%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call plan%save(filename, ierr)
        call check("version gate: save", ierr, DTFFT_SUCCESS)
        call plan%destroy(ierr)

        ! Claim a layout from the future
        call patch_byte(filename, VERSION_OFFSET, 99_int8)
        call reloaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("version gate: newer layout refused", ierr, DTFFT_ERROR_FILE_VERSION_MISMATCH)

        ! Claim a layout older than anything this library ever wrote
        call patch_byte(filename, VERSION_OFFSET, 0_int8)
        call reloaded%load(filename, MPI_COMM_WORLD, ierr)
        call check("version gate: retired layout refused", ierr, DTFFT_ERROR_FILE_VERSION_MISMATCH)

        call delete_file(filename)
    end subroutine test_format_version_gate

    ! subroutine test_self_describing_sizes()
    ! !! The fixed prefix states how large the global section and each per-rank record are, and
    ! !! the reader takes record offsets from those numbers rather than from its own constants.
    ! !!
    ! !! That is what allows a section to grow in a later version without stranding older files,
    ! !! so it is worth checking that the numbers actually describe the file rather than merely
    ! !! being present in it.
    !     type(dtfft_plan_c2c_t)  :: plan
    !     character(len=*), parameter :: filename = "test_save_load_sizes.dtfft"
    !     integer(int32) :: unit, global_bytes, record_bytes, file_bytes, expected

    !     call plan%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
    !     call plan%save(filename, ierr)
    !     call check("self-describing: save", ierr, DTFFT_SUCCESS)
    !     call plan%destroy(ierr)

    !     global_bytes = 0
    !     record_bytes = 0
    !     file_bytes = 0
    !     expected = 0
    !     if ( comm_rank == 0 ) then
    !         ! 8 bytes of magic, 2 of endian marker, version, reserved, then the two sizes
    !         open(newunit=unit, file=filename, access="stream", form="unformatted", status="old")
    !         read(unit, pos=13) global_bytes
    !         read(unit, pos=17) record_bytes
    !         close(unit)
    !         inquire(file=filename, size=file_bytes)
    !         expected = global_bytes + comm_size * record_bytes
    !     endif

    !     call check("self-describing: file size matches prefix", file_bytes, expected)

    !     call delete_file(filename)
    ! end subroutine test_self_describing_sizes

    ! subroutine test_unavailable_backend()
    ! !! Which backends exist is a property of the build, not of the format, so a file may name
    ! !! one this library was not compiled with. Replaying it would fail later and far less
    ! !! clearly, so the load has to turn it down up front.
    !     type(dtfft_plan_c2c_t)  :: plan, reloaded
    !     character(len=*), parameter :: filename = "test_save_load_backend.dtfft"
    !     ! 20 bytes of fixed prefix, then plan_kind, ndims, precision, effort and flags
    !     integer(int8), parameter :: BACKEND_OFFSET = 26_int8
    !     integer(int8), parameter :: NO_SUCH_BACKEND = 99_int8

    !     call plan%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
    !     call plan%save(filename, ierr)
    !     call check("unavailable backend: save", ierr, DTFFT_SUCCESS)
    !     call plan%destroy(ierr)

    !     call patch_byte(filename, BACKEND_OFFSET, NO_SUCH_BACKEND)
    !     call reloaded%load(filename, MPI_COMM_WORLD, ierr)
    !     call check("unavailable backend: refused", ierr, DTFFT_ERROR_FILE_BACKEND_NOT_SUPPORTED)

    !     call delete_file(filename)
    ! end subroutine test_unavailable_backend

    subroutine patch_byte(filename, offset, value)
    !! Overwrites a single byte of a file, from rank 0
        character(len=*),   intent(in) :: filename   !! Name of file
        integer(int8),      intent(in) :: offset     !! 1-based byte position
        integer(int8),      intent(in) :: value      !! Byte to store
        integer(int8) :: old_value
        integer(int32) :: unit

        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        if ( comm_rank == 0 ) then
            open(newunit=unit, file=filename, access="stream", form="unformatted", status="old")
            read(unit, pos=int(offset, int64)) old_value
            write(unit, pos=int(offset, int64)) value
            close(unit)
        endif
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
    end subroutine patch_byte

    subroutine test_kind_mismatch()
    !! Loading a C2C file into an R2R variable would otherwise silently produce a plan
    !! that transforms the wrong thing
        type(dtfft_plan_c2c_t)  :: c2c
        type(dtfft_plan_r2r_t)  :: r2r
        character(len=*), parameter :: filename = "test_save_load_mismatch.dtfft"

        call c2c%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call c2c%save(filename, ierr)
        call c2c%destroy(ierr)

        call r2r%load(filename, MPI_COMM_WORLD, ierr)
        call check("kind mismatch: load", ierr, DTFFT_ERROR_FILE_PLAN_MISMATCH)

        call delete_file(filename)
    end subroutine test_kind_mismatch

    subroutine test_comm_size_mismatch()
    !! The file holds one record per rank and the process count is part of the plan
    !! identity, so replaying it on a different number of ranks must be refused
        type(dtfft_plan_c2c_t)  :: plan
        character(len=*), parameter :: filename = "test_save_load_nproc.dtfft"
        TYPE_MPI_COMM :: single_comm

        if ( comm_size == 1 ) return

        call plan%create(DIMS_3D, effort=DTFFT_ESTIMATE, executor=DTFFT_EXECUTOR_NONE, error_code=ierr)
        call plan%save(filename, ierr)
        call plan%destroy(ierr)

        ! Every rank gets its own single-rank communicator, so the collective calls inside
        ! `load` stay matched while the recorded size no longer fits
        call MPI_Comm_split(MPI_COMM_WORLD, comm_rank, 0, single_comm, ierr)
        block
            type(dtfft_plan_c2c_t) :: narrow

            call narrow%load(filename, single_comm, ierr)
            call check("comm size mismatch: load", ierr, DTFFT_ERROR_FILE_PLAN_MISMATCH)
        endblock
        call MPI_Comm_free(single_comm, ierr)

        call delete_file(filename)
    end subroutine test_comm_size_mismatch

    subroutine test_save_before_create()
    !! There is nothing to save before the plan exists
        type(dtfft_plan_c2c_t)  :: plan

        call plan%save("test_save_load_never_written.dtfft", ierr)
        call check("save before create", ierr, DTFFT_ERROR_PLAN_NOT_CREATED)
    end subroutine test_save_before_create

!------------------------------------------------------------------------------------------------
! Helpers
!------------------------------------------------------------------------------------------------
    subroutine compare_plans(tag, ndims, tuned, loaded)
    !! A loaded plan must agree with the tuned one on everything the user can observe
        character(len=*),       intent(in)    :: tag      !! Test name prefix
        integer(int32),         intent(in)    :: ndims    !! Number of dimensions of both plans
        class(dtfft_plan_t),    intent(inout) :: tuned    !! Plan that was autotuned
        class(dtfft_plan_t),    intent(inout) :: loaded   !! Plan restored from file

        integer(int32), pointer :: grid_a(:), grid_b(:)
        integer(int32) :: starts_a(3), counts_a(3), starts_b(3), counts_b(3)
        integer(int64) :: alloc_a, alloc_b
        type(dtfft_backend_t) :: backend_a, backend_b

        ! Both plans have to exist before anything can be compared: `get_grid_dims` hands back
        ! an unassociated pointer for a plan that was never created, and dereferencing that
        ! turns a reported failure into a crash.
        nullify( grid_a, grid_b )
        call tuned%get_grid_dims(grid_a, ierr)
        call loaded%get_grid_dims(grid_b, ierr)
        if ( .not. associated(grid_a) .or. .not. associated(grid_b) ) then
            call check(tag//": both plans available for comparison", 1, 0)
            return
        endif

        call check(tag//": grid size", size(grid_b), size(grid_a))
        if ( size(grid_a) == size(grid_b) ) then
            call check(tag//": grid dims", maxval(abs(grid_b - grid_a)), 0)
        endif

        call check(tag//": z-slab", merge(1, 0, loaded%get_z_slab_enabled(ierr)),  &
                                    merge(1, 0, tuned%get_z_slab_enabled(ierr)))
        call check(tag//": y-slab", merge(1, 0, loaded%get_y_slab_enabled(ierr)),  &
                                    merge(1, 0, tuned%get_y_slab_enabled(ierr)))

        backend_a = tuned%get_backend(ierr)
        backend_b = loaded%get_backend(ierr)
        call check(tag//": backend", backend_b%val, backend_a%val)
        call check(tag//": element size", int(loaded%get_element_size(ierr)), int(tuned%get_element_size(ierr)))
        call check(tag//": alloc bytes", int(loaded%get_alloc_bytes(ierr)), int(tuned%get_alloc_bytes(ierr)))
        call check(tag//": aux bytes", int(loaded%get_aux_bytes(ierr)), int(tuned%get_aux_bytes(ierr)))

        starts_a = 0; counts_a = 0; starts_b = 0; counts_b = 0
        call tuned%get_local_sizes(starts_a(:ndims), counts_a(:ndims), alloc_size=alloc_a, error_code=ierr)
        call loaded%get_local_sizes(starts_b(:ndims), counts_b(:ndims), alloc_size=alloc_b, error_code=ierr)
        call check(tag//": local starts", maxval(abs(starts_b(:ndims) - starts_a(:ndims))), 0)
        call check(tag//": local counts", maxval(abs(counts_b(:ndims) - counts_a(:ndims))), 0)
        call check(tag//": alloc size", int(alloc_b), int(alloc_a))
    end subroutine compare_plans

    subroutine check_transpose_roundtrip(tag, ndims, plan)
    !! Forward then backward transposition must return the original data.
    !! This is what proves the restored handles actually work.
        character(len=*),       intent(in)    :: tag      !! Test name prefix
        integer(int32),         intent(in)    :: ndims    !! Number of dimensions of the plan
        class(dtfft_plan_t),    intent(inout) :: plan     !! Plan to exercise

        real(real32), pointer :: in(:), out(:)
        type(c_ptr) :: in_ptr, out_ptr, aux_ptr
        integer(int64) :: n_bytes, aux_bytes, n_floats, n_data_floats, i
        integer(int32) :: n_bad
        integer(int32) :: in_counts(3)

        n_bytes = plan%get_alloc_bytes(ierr)
        aux_bytes = plan%get_aux_bytes(ierr)
        n_floats = n_bytes / 4_int64

        ! The allocation is sized for the largest layout and doubles as scratch space, so
        ! only the leading local block is user data. Everything past it is free for the
        ! backends to clobber and must not be compared.
        in_counts = 1
        call plan%get_local_sizes(in_counts=in_counts(:ndims), error_code=ierr)
        n_data_floats = product(int(in_counts(:ndims), int64)) * plan%get_element_size(ierr) / 4_int64
call plan%report()
        in_ptr  = plan%mem_alloc_ptr(n_bytes, ierr)
        out_ptr = plan%mem_alloc_ptr(n_bytes, ierr)
        aux_ptr = c_null_ptr
        if ( aux_bytes > 0_int64 ) aux_ptr = plan%mem_alloc_ptr(aux_bytes, ierr)

        call c_f_pointer(in_ptr, in, [n_floats])
        call c_f_pointer(out_ptr, out, [n_floats])

        do i = 1, n_data_floats
            in(i) = reference_value(i)
        enddo
        out(:) = -1.0_real32

        ! With DTFFT_EXECUTOR_NONE this is pure data movement, so forward followed by
        ! backward has to reproduce the input bit for bit
        call plan%execute_ptr(in_ptr, out_ptr, DTFFT_EXECUTE_FORWARD, aux_ptr, ierr)
        call check(tag//": forward execute", ierr, DTFFT_SUCCESS)
        call plan%execute_ptr(out_ptr, in_ptr, DTFFT_EXECUTE_BACKWARD, aux_ptr, ierr)
        call check(tag//": backward execute", ierr, DTFFT_SUCCESS)

        ! With DTFFT_EXECUTOR_NONE nothing but data movement happens, so the comparison is
        ! bit-exact by design. Comparing the bit patterns says that outright and keeps the
        ! compiler from warning about float equality.
        n_bad = 0
        do i = 1, n_data_floats
            ! if ( i > 3000 .and. i < 3200 ) then
            !     print*, transfer(in(i), 0_int32), transfer(reference_value(i), 0_int32)
            ! endif
            if ( transfer(in(i), 0_int32) /= transfer(reference_value(i), 0_int32) ) n_bad = n_bad + 1
        enddo
        call check(tag//": roundtrip mismatches", n_bad, 0)
        if ( n_bad > 0 ) then
            call MPI_Finalize(ierr)
            stop
        endif

        call plan%mem_free_ptr(in_ptr, ierr)
        call plan%mem_free_ptr(out_ptr, ierr)
        if ( aux_bytes > 0_int64 ) call plan%mem_free_ptr(aux_ptr, ierr)
    end subroutine check_transpose_roundtrip

    subroutine check_resave_identical(tag, filename, loaded)
    !! Saves the loaded plan again and compares the two files byte for byte.
        character(len=*),       intent(in)    :: tag      !! Test name prefix
        character(len=*),       intent(in)    :: filename !! File written by the tuned plan
        class(dtfft_plan_t),    intent(inout) :: loaded   !! Plan restored from that file

        character(len=:), allocatable :: resaved
        integer(int8),    allocatable :: bytes_a(:), bytes_b(:)
        integer(int32) :: size_a, size_b, n_diff, i

        allocate( resaved, source=filename//".again" )
        call loaded%save(resaved, ierr)
        call check(tag//": re-save", ierr, DTFFT_SUCCESS)

        call read_whole_file(filename, bytes_a, size_a)
        call read_whole_file(resaved, bytes_b, size_b)
        ! `check` is collective, so every rank has to reach it the same number of times.
        ! Only rank 0 actually read the files; the others contribute a passing zero.
        n_diff = 0
        if ( size_a /= size_b ) then
            n_diff = 1
        else if ( size_a > 0 ) then
            n_diff = count(bytes_a /= bytes_b)
        endif
        if ( n_diff /= 0 ) then
            do i = 1, size_a
                if ( bytes_a(i) == bytes_b(i) ) cycle
                call report("  note "//tag//": offset "//to_string(i - 1)//": saved "        &
                            //to_string(int(bytes_a(i), int32))//" -> re-saved "            &
                            //to_string(int(bytes_b(i), int32)))
            enddo
        endif
        call check(tag//": re-saved file identical", n_diff, 0)

        if ( allocated(bytes_a) ) deallocate( bytes_a )
        if ( allocated(bytes_b) ) deallocate( bytes_b )
        call delete_file(resaved)
        deallocate( resaved )
    end subroutine check_resave_identical

    subroutine read_whole_file(filename, bytes, n_bytes)
    !! Reads a whole file into a byte array. Rank 0 only; other ranks get an empty result.
        character(len=*),               intent(in)  :: filename  !! Name of file
        integer(int8),   allocatable,   intent(out) :: bytes(:)  !! File content
        integer(int32),                 intent(out) :: n_bytes   !! Number of bytes read

        integer(int32) :: unit, io_stat

        n_bytes = 0
        allocate( bytes(0) )
        if ( comm_rank /= 0 ) return

        inquire(file=filename, size=n_bytes)
        if ( n_bytes <= 0 ) return

        deallocate( bytes )
        allocate( bytes(n_bytes) )
        open(newunit=unit, file=filename, access="stream", form="unformatted", status="old", iostat=io_stat)
        if ( io_stat /= 0 ) then
            n_bytes = 0
            return
        endif
        read(unit) bytes
        close(unit)
    end subroutine read_whole_file

    pure real(real32) function reference_value(i)
    !! Deterministic filler, distinct per rank so a misdirected exchange shows up
        integer(int64), intent(in) :: i   !! Element index

        reference_value = real(mod(i * 7_int64, 101_int64), real32)
    end function reference_value

    logical function all_ok(local_code)
    !! Whether every rank got `DTFFT_SUCCESS`.
    !!
    !! Collective, so a test can bail out on failure without some ranks walking on into a
    !! collective call the others already left. Guarding on a bare local code would be a
    !! deadlock waiting to happen.
        integer(int32), intent(in) :: local_code    !! This rank's error code

        integer(int32) :: worst

        call MPI_Allreduce(merge(0, 1, local_code == DTFFT_SUCCESS), worst, 1, MPI_INTEGER4,  &
                           MPI_MAX, MPI_COMM_WORLD, ierr)
        all_ok = worst == 0
    end function all_ok

    subroutine check(name, actual, expected)
    !! Keeps existing call sites while sharing MPI-aware assertion handling.
        character(len=*), intent(in) :: name      !! Test name
        integer(int32),   intent(in) :: actual    !! Value produced
        integer(int32),   intent(in) :: expected  !! Value required

        call test%check(name, actual, expected)
    end subroutine check

    subroutine report(message)
    !! Prints a single line from rank 0
        character(len=*), intent(in) :: message   !! Message

        if ( comm_rank == 0 ) write(output_unit, "(a)") message
        flush(output_unit)
    end subroutine report

    function to_string(val) result(str)
    !! Left-trimmed decimal representation of `val`
        integer(int32), intent(in)    :: val   !! Value
        character(len=:), allocatable :: str   !! Result
        character(len=32) :: buf

        write(buf, "(i0)") val
        allocate( str, source=trim(buf) )
    end function to_string

    subroutine delete_file(filename)
    !! Removes a plan file, from a single rank
        character(len=*), intent(in) :: filename  !! Name of file

        integer(int32) :: unit, io_stat

        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        if ( comm_rank == 0 ) then
            open(newunit=unit, file=filename, status="old", iostat=io_stat)
            if ( io_stat == 0 ) close(unit, status="delete")
        endif
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
    end subroutine delete_file
end program test_plan_save_load
