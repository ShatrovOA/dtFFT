#pragma once

#include "config.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>

extern "C"
void run_2d_decomp_bench(int nx, int ny, int nz,
                         int p_row, int p_col,
                         int warmup_iters, int measure_iters,
                         double *time);


double run_2d_decomp(const std::vector<int32_t> &global_dims)
{
    int comm_size, comm_rank;
    int periods[2] = {0, 0};
    int counter;
    double time;


    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("2d-decomp benchmark\n");
        printf("#####################################\n");
    }

    double best_time = INFINITY;
    int best_grid[2];

    for (int p = 1; p <= comm_size; p++)
    {
        if (comm_size % p)
            continue;
        if (global_dims[1] < p || global_dims[2] < comm_size / p || global_dims[0] < p || global_dims[1] < comm_size / p)
            continue;
        int dims[2] = {p, comm_size / p};

        // if (comm_rank == 0)
        // {
        //     printf("Trying process grid: [%d, %d]\n", dims[0], dims[1]);
        // }

        run_2d_decomp_bench(global_dims[0], global_dims[1], global_dims[2],
                          dims[0], dims[1],
                          WARMUP_ITERATIONS, TEST_ITERATIONS,
                          &time);

        if (time < best_time)
        {
            best_time = time;
            best_grid[0] = dims[0];
            best_grid[1] = dims[1];
        }
    }

    if (comm_rank == 0)
    {
        printf("2d-decomp Results\n");
        printf("----------------------------------------\n");
        printf("Fastest execution time: %f\n", best_time);
        printf("Fastest grid: %ix%i\n", best_grid[0], best_grid[1]);
        printf("----------------------------------------\n");
    }

    return best_time;
}
