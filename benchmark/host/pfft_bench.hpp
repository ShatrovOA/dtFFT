#pragma once

#include "config.hpp"

#include <complex.h>
#include <cstring>
#include <algorithm> // std::min_element
#include <pfft.h>

double run_pfft_private(ptrdiff_t *n, int *proc_grid)
{
    // ptrdiff_t n[3];
    ptrdiff_t alloc_local, l;
    ptrdiff_t local_ni[3], local_i_start[3];
    ptrdiff_t local_no[3], local_o_start[3];
    double err;
    pfft_complex *in, *out;
    pfft_plan plan_forw = NULL, plan_back = NULL;
    MPI_Comm grid_comm;
    int comm_size, comm_rank;
    int periods[3] = {0, 0, 0};
    int counter;
    double ts, t_global, t_sum;

    // /* Set size of FFT and process mesh */
    // n[0] = global_dims[2];
    // n[1] = global_dims[1];
    // n[2] = global_dims[0];

    /* Initialize MPI and PFFT */
    pfft_init();

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    // if (comm_rank == 0)
    // {
    //     printf("#####################################\n");
    //     printf("PFFT benchmark\n");
    //     printf("#####################################\n");
    // }

    double best_time = INFINITY;
    // int best_grid[3];

    // for (int p = 1; p <= comm_size; p++)
    // {
    //     if (comm_size % p)
    //         continue;
    //     int q = comm_size / p;
    //     if (n[0] < q || n[1] < p || n[2] < p)
    //         continue;
    //     int dims[2] = {q, p};

        if (comm_rank == 0)
        {
            printf("Trying process grid: [%d, %d, %d]\n", proc_grid[0], proc_grid[1], proc_grid[2]);
        }

        int ndims = 3;
        if( proc_grid[2] == 1) ndims = 2;

        MPI_Cart_create(MPI_COMM_WORLD, ndims, proc_grid, periods, 1, &grid_comm);

        /* Get parameters of data distribution */
        alloc_local = pfft_local_size_dft_3d(n, grid_comm, PFFT_TRANSPOSED_OUT,
                                             local_ni, local_i_start, local_no, local_o_start);

        /* Allocate memory */
        in = pfft_alloc_complex(alloc_local);
        out = pfft_alloc_complex(alloc_local);

        double create_time = -MPI_Wtime();
        /* Plan parallel forward FFT */
        plan_forw = pfft_plan_dft_3d(
            n, in, out, grid_comm, PFFT_FORWARD, PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);

        /* Plan parallel backward FFT */
        plan_back = pfft_plan_dft_3d(
            n, out, in, grid_comm, PFFT_BACKWARD, PFFT_TRANSPOSED_IN | PFFT_MEASURE | PFFT_DESTROY_INPUT);

        create_time += MPI_Wtime();

        if (comm_rank == 0)
        {
            printf("Plans created successfully, time spent: %f\n", create_time);
        }

        memset(in, 0, alloc_local * sizeof(fftw_complex));

        for (counter = 0; counter < WARMUP_ITERATIONS; counter++)
        {
            pfft_execute_dft(plan_forw, in, out);
            pfft_execute_dft(plan_back, out, in);
        }
        if (comm_rank == 0)
        {
            printf("Warmup\n");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();
        for (counter = 0; counter < TEST_ITERATIONS; counter++)
        {
            pfft_execute_dft(plan_forw, in, out);
            pfft_execute_dft(plan_back, out, in);
        }
        double end_time = MPI_Wtime();
        double elapsed_time = (end_time - start_time);

        double min_s, max_s, avg_s;
        MPI_Allreduce(&elapsed_time, &min_s, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &max_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &avg_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        avg_s /= (double)comm_size;

        if (comm_rank == 0)
        {
            printf("min time: %f [s]\n", min_s);
            printf("max time: %f [s]\n", max_s);
            printf("avg time: %f [s]\n", avg_s);
            printf("avg time per iteration: %f [s]\n", avg_s / (double)TEST_ITERATIONS);
            printf("----------------------------------------\n");
        }

        // if (max_s < best_time)
        // {
        //     best_time = max_s;
        //     best_grid[0] = dims[0];
        //     best_grid[1] = dims[1];

        // }
        /* free mem and finalize */
        pfft_destroy_plan(plan_forw);
        pfft_destroy_plan(plan_back);
        MPI_Comm_free(&grid_comm);
        pfft_free(in);
        pfft_free(out);
    // }

    // if (comm_rank == 0)
    // {
    //     printf("PFFT Results\n");
    //     printf("----------------------------------------\n");
    //     printf("Fastest execution time: %f\n", best_time);
    //     printf("Fastest grid: %ix%i\n", best_grid[0], best_grid[1]);
    //     printf("----------------------------------------\n");
    // }

    return max_s;
}


double run_pfft(const std::vector<int32_t> &global_dims, const std::vector<int> &grid = std::vector<int>())
{
    ptrdiff_t n[3];
    int proc_grid[3];

    /* Set size of FFT and process mesh */
    n[0] = global_dims[2];
    n[1] = global_dims[1];
    n[2] = global_dims[0];

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("PFFT benchmark\n");
        printf("#####################################\n");
    }

    double best_time = INFINITY;
    int best_grid[3];
    double test_time;
    if ( grid.size() > 0 ) {
        proc_grid[0] = grid[2];
        proc_grid[1] = grid[1];
        proc_grid[2] = grid[0];
        test_time = run_pfft_private(n, proc_grid);
        best_time = test_time;
        best_grid[0] = grid[0];
        best_grid[1] = grid[1];
        best_grid[2] = grid[2];
    } else {
        for (int p = 1; p <= comm_size; p++)
        {
            if (comm_size % p)
                continue;
            int q = comm_size / p;
            if (n[0] < q || n[1] < p || n[2] < p)
                continue;

            proc_grid[0] = q;
            proc_grid[1] = p;
            proc_grid[2] = 1;

            test_time = run_pfft_private(n, proc_grid);

            // if (comm_rank == 0)
            // {
            //     printf("Testing grid: %dx%d\n", p, q);
            // }

        // Using input configuration with pencil data format in X direction
        // and output configuration with pencil data in the Z direction.
        // This format uses only two internal reshape operation.
            // std::array<int, 3> input_grid = {1, p, q};
            // std::array<int, 3> output_grid = {p, q, 1};


            // std::array<int, 3> lower = {0, 0, 0};
            // std::array<int, 3> upper = {dims[0] - 1, dims[1] - 1, dims[2] - 1};
            // auto world = heffte::box3d<>(lower, upper);

            // std::vector<heffte::box3d<>> inboxes  = heffte::split_world(world, input_grid);
            // std::vector<heffte::box3d<>> outboxes = heffte::split_world(world, output_grid);

            // heffte::box3d<> boxin = inboxes[comm_rank];
            // heffte::box3d<> boxout = outboxes[comm_rank];

            // if (comm_rank == 0)
            // {
            //     printf("Testing grid: %dx%d\n", p, q);
            // }

            // double test_time = run_heffte_private(boxin, boxout);
            if ( test_time < best_time )
            {
                best_time = test_time;
                best_grid[0] = q;
                best_grid[1] = p;
                best_grid[2] = 1;
            }

        }
    }

    return best_time;
}