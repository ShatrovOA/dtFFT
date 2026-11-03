program test_rtc_block_optimizer
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_interface_cuda_runtime, only: device_props
use dtfft_parameters
use dtfft_rtc_block_optimizer
use dtfft_utils, only: string
use test_suite, only: test_suite_t
#include "_dtfft_mpi.h"
implicit none
    !! Kernel cases: each row encodes (kernel_type_val, tile_dim, other_dim, needs_neighbor)
    !!   - KERNEL_PERMUTE_FORWARD             tile_dim=2, other_dim=3
    !!   - KERNEL_PERMUTE_BACKWARD            tile_dim=3, other_dim=2
    !!   - KERNEL_PERMUTE_BACKWARD_START      tile_dim=3, other_dim=2
    !!   - KERNEL_PERMUTE_BACKWARD_END_PIPELINED tile_dim=2, other_dim=3, needs_neighbor
    !!   - KERNEL_UNPACK_PIPELINED            tile_dim=2, other_dim=3, needs_neighbor
    !!   - KERNEL_PACK_PIPELINED              tile_dim=2, other_dim=3, needs_neighbor
    integer(int32), parameter :: N_KERNEL_CASES = 6
    integer(int32), parameter :: kernel_vals(N_KERNEL_CASES) = &
        [KERNEL_PERMUTE_FORWARD%val, KERNEL_PERMUTE_BACKWARD%val,         &
         KERNEL_PERMUTE_BACKWARD_START%val,                                &
         KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val,                        &
         KERNEL_UNPACK_PIPELINED%val, KERNEL_PACK_PIPELINED%val]
    integer(int32), parameter :: kernel_tile_dim(N_KERNEL_CASES) = &
        [2, 3, 3, 2, 2, 2]
    integer(int32), parameter :: kernel_other_dim(N_KERNEL_CASES) = &
        [3, 2, 2, 3, 3, 3]
    logical, parameter :: kernel_needs_neighbor(N_KERNEL_CASES) = &
        [.false., .false., .false., .true., .true., .true.]

    !! Dim cases: [nx, ny, nz]
    !!   1 - typical square large
    !!   2 - typical rectangular large
    !!   3 - asymmetric: nx >> ny (wide)
    !!   4 - asymmetric: ny >> nx (tall)
    !!   5 - very small (< warp size)
    !!   6 - degenerate z=1 (2D-like)
    !!   7 - non-power-of-two in all dims
    integer(int32), parameter :: N_DIMS_CASES = 7
    integer(int32) :: test_dims(3, N_DIMS_CASES)

    type(device_props) :: props_ampere, props_volta
    type(kernel_config) :: candidates(MAX_CANDIDATES)
    type(kernel_config) :: cfg
    integer(int32) :: num_candidates
    integer(int32) :: d, k
    integer(int32) :: sorted_indices(MAX_CANDIDATES)
    integer(int32) :: invalid_conflicts
    real(real32) :: scores(MAX_CANDIDATES)
    integer(int32) :: ierr
    type(test_suite_t) :: test

    call MPI_Init(ierr)

    props_ampere = get_ampere_architecture()
    props_volta  = get_volta_architecture()

    test_dims(:, 1) = [128, 128, 64]
    test_dims(:, 2) = [256, 96,  32]
    test_dims(:, 3) = [512, 8,   16]
    test_dims(:, 4) = [8,   512, 16]
    test_dims(:, 5) = [4,   4,   4]
    test_dims(:, 6) = [64,  64,  1]
    test_dims(:, 7) = [100, 70,  13]

    ! --- bank conflicts: invalid config (block_rows > tile_size) must return MAX_INT32 ---
    invalid_conflicts = count_bank_conflicts(16, 32, FLOAT_STORAGE_SIZE, 1)
    call test%assert_true("count_bank_conflicts must reject invalid configuration (block_rows > tile_size)", invalid_conflicts == MAX_INT32)

    ! --- run all combinations of dims x kernel_type x architecture ---
    do d = 1, N_DIMS_CASES
        do k = 1, N_KERNEL_CASES
            call run_case(test_dims(:, d), kernel_type_t(kernel_vals(k)), &
                          kernel_tile_dim(k), kernel_other_dim(k),        &
                          kernel_needs_neighbor(k), props_ampere, "ampere")
            call run_case(test_dims(:, d), kernel_type_t(kernel_vals(k)), &
                          kernel_tile_dim(k), kernel_other_dim(k),        &
                          kernel_needs_neighbor(k), props_volta,  "volta")
        end do
    end do

    call MPI_Finalize(ierr)

contains

    !! For pack/unpack kernels generate_candidates and evaluate_analytical_performance
    !! operate on neighbor dimensions, not on local dims.  We use dims itself as a
    !! simple stand-in for neighbor_data so the API contract is exercised correctly.
    subroutine run_case(dims, kernel_type, tile_dim, other_dim, needs_neighbor, props, arch_name)
        integer(int32),       intent(in) :: dims(3)
        type(kernel_type_t),  intent(in) :: kernel_type
        integer(int32),       intent(in) :: tile_dim
        integer(int32),       intent(in) :: other_dim
        logical,              intent(in) :: needs_neighbor
        type(device_props),   intent(in) :: props
        character(len=*),     intent(in) :: arch_name
        integer(int32) :: work_dims(3)  ! dims that generate_candidates operates on
        integer(int32) :: neighbor(3)
        integer(int32) :: i
        real(real32) :: score
        character(len=128) :: prefix
        type(string) :: kname

        kname = get_kernel_string(kernel_type)
        write(prefix, '(5a,i0,2(a,i0),a)') trim(arch_name), " [", trim(kname%raw), "] ", &
            "dims=[", dims(1), ",", dims(2), ",", dims(3), "]"
        call kname%destroy()

        if (needs_neighbor) then
            ! For pack/unpack kernels candidates are generated from neighbor dimensions.
            ! Use a rotated version of dims as a plausible neighbor layout.
            neighbor(1) = dims(tile_dim)
            neighbor(2) = dims(other_dim)
            neighbor(3) = dims(1)
            work_dims = neighbor
        else
            work_dims = dims
        end if

        call generate_candidates(work_dims, tile_dim, other_dim, FLOAT_STORAGE_SIZE, props, candidates, num_candidates)
        call test%assert_true(prefix//": num_candidates negative", num_candidates >= 0)
        call test%assert_true(prefix//": num_candidates > MAX_CANDIDATES", num_candidates <= MAX_CANDIDATES)
        call test%assert_true(prefix//": num_candidates positive", num_candidates > 0)

        do i = 1, num_candidates
            call test%assert_true(prefix//": tile_size < block_rows", candidates(i)%tile_size >= candidates(i)%block_rows)
            call test%assert_true(prefix//": candidate exceeds max_threads_per_block", candidates(i)%tile_size * candidates(i)%block_rows <= props%max_threads_per_block)
            call test%assert_true(prefix//": tile_size not power of two", is_power_of_two(candidates(i)%tile_size))
            call test%assert_true(prefix//": block_rows not power of two", is_power_of_two(candidates(i)%block_rows))
            call test%assert_true(prefix//": threads not multiple of warp size", mod(candidates(i)%tile_size * candidates(i)%block_rows, 32) == 0)

            if (needs_neighbor) then
                score = evaluate_analytical_performance(work_dims, tile_dim, other_dim, &
                    kernel_type, candidates(i), props, FLOAT_STORAGE_SIZE, neighbor_data=neighbor)
            else
                score = evaluate_analytical_performance(work_dims, tile_dim, other_dim, &
                    kernel_type, candidates(i), props, FLOAT_STORAGE_SIZE)
            end if
            call test%assert_true(prefix//": float score out of [0,1]", score >= 0.0_real32 .and. score <= 1.0_real32)

            if (needs_neighbor) then
                score = evaluate_analytical_performance(work_dims, tile_dim, other_dim, &
                    kernel_type, candidates(i), props, DOUBLE_STORAGE_SIZE, neighbor_data=neighbor)
            else
                score = evaluate_analytical_performance(work_dims, tile_dim, other_dim, &
                    kernel_type, candidates(i), props, DOUBLE_STORAGE_SIZE)
            end if
            call test%assert_true(prefix//": double score out of [0,1]", score >= 0.0_real32 .and. score <= 1.0_real32)

            scores(i) = score
        end do

        call sort_candidates_by_score(scores(1:num_candidates), num_candidates, sorted_indices(1:num_candidates))

        do i = 2, num_candidates
            call test%assert_true(prefix//": sort ordering violated", scores(sorted_indices(i - 1)) >= scores(sorted_indices(i)))
        end do

        cfg = candidates(sorted_indices(1))
        call test%assert_true(prefix//": best candidate has zero padding", cfg%padding >= 1)

        write(output_unit,'(2a,a,i0,a,i0,a,i0)') &
            "  OK ", trim(prefix), " best=", cfg%tile_size, "x", cfg%block_rows, "+", cfg%padding
    end subroutine run_case

    function is_power_of_two(x) result(is_power)
        integer, intent(in) :: x
        logical :: is_power

        is_power = (x > 0 .AND. POPCNT(x) == 1)
    end function is_power_of_two

end program test_rtc_block_optimizer
