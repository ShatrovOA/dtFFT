#pragma once

#include "config.hpp"

#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <accfft.h>

double run_accfft(const std::vector<int> &dims)
{
    int comm_size, comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("AccFFT benchmark\n");
        printf("#####################################\n");
    }

    int N[3] = {dims[2], dims[1], dims[0]};

    int nthreads = 1;
    accfft_init();
    /* Create Cartesian Communicator */
    
    MPI_Comm c_comm;
    int l, counter;

    double best_time = INFINITY;
    int best_grid[2];

    for (int p = 1; p <= comm_size; p++)
    {
        if ( comm_size % p ) continue;
        int q = comm_size / p;
        if ( N[0] < p || N[1] < q ) continue;

        int c_dims[2] = {p, q};

        // if (comm_rank == 0)
        // {
        //     printf("Trying process grid: [%d, %d]\n", c_dims[0], c_dims[1]);
        // }

        accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

        Complex *in;
        Complex *out;
        double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
        int alloc_max = 0;

        int isize[3], osize[3], istart[3], ostart[3];
        /* Get the local pencil size and the allocation size */
        alloc_max = accfft_local_size_dft_c2c(N, isize, istart, osize, ostart, c_comm);

        in = (Complex *)accfft_alloc(isize[0] * isize[1] * isize[2] * 2 * sizeof(double));
        out = (Complex *)accfft_alloc(alloc_max);


        /* Create FFT plan */
        setup_time = -MPI_Wtime();
        accfft_plan *plan = accfft_plan_dft_3d_c2c(N, in, out, c_comm, ACCFFT_MEASURE);
        setup_time += MPI_Wtime();

        /*  Initialize data */
        for (l = 0; l < isize[0] * isize[1] * isize[2]; l++)
        {
            in[l][0] = 0.0;
            in[l][1] = 0.0;
        }


        for (counter = 0; counter < WARMUP_ITERATIONS; counter++)
        {
            accfft_execute_c2c(plan, ACCFFT_FORWARD, in, out);
            accfft_execute_c2c(plan, ACCFFT_BACKWARD, out, in);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();
        for (counter = 0; counter < TEST_ITERATIONS; counter++)
        {
            accfft_execute_c2c(plan, ACCFFT_FORWARD, in, out);
            accfft_execute_c2c(plan, ACCFFT_BACKWARD, out, in);
        }

        double end_time = MPI_Wtime();
        double elapsed_time = (end_time - start_time);

        double min_s, max_s, avg_s;
        MPI_Allreduce(&elapsed_time, &min_s, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &max_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &avg_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        avg_s /= (double)comm_size;

        // if(comm_rank == 0) {
        //     printf("min time: %f [s]\n", min_s);
        //     printf("max time: %f [s]\n", max_s);
        //     printf("avg time: %f [s]\n", avg_s);
        //     printf("avg time per iteration: %f [s]\n", avg_s / (double)TEST_ITERATIONS);
        //     printf("----------------------------------------\n");
        // }

        if ( max_s < best_time ) {
            best_time = max_s;
            best_grid[0] = c_dims[0];
            best_grid[1] = c_dims[1];
        }

        accfft_free(in);
        accfft_free(out);

        accfft_destroy_plan(plan);
        MPI_Comm_free(&c_comm);
    }

    if (comm_rank == 0)
    {
        printf("AccFFT Results\n");
        printf("----------------------------------------\n");
        printf("Fastest execution time: %f\n", best_time);
        printf("Fastest grid: %ix%i\n", best_grid[0], best_grid[1]);
        printf("----------------------------------------\n");
    }

    return best_time;
}