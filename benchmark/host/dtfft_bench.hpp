#pragma once

#include "config.hpp"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <memory>
#include <algorithm>

#include <dtfft.hpp>

void setup_dtfft_config(bool enable_z_slab, bool use_datatypes)
{
    auto config = dtfft::Config()
        .set_enable_log(false)
        .set_enable_z_slab(enable_z_slab)
        .set_enable_y_slab(false)
        .set_enable_fourier_reshape(true)
        .set_enable_datatype_backend(use_datatypes)
        .set_backend(dtfft::Backend::MPI_P2P_PIPELINED)
        .set_enable_mpi_backends(!use_datatypes);
    dtfft::set_config(config);
}

// template<typename PlanType>
double run_dtfft_internal(dtfft::Plan* plan, const std::string &plan_name,
                          double create_time, dtfft::Precision precision, dtfft::Executor executor,
                          int64_t scaler, bool enable_z_slab)
{
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (enable_z_slab)
    {
        bool is_z_slab = plan->get_z_slab_enabled();
        if (!is_z_slab)
        {
            if (comm_rank == 0)
            {
                // printf("Plan is not using Z slab, skipping benchmark\n");
            }
            return -1.0;
        }
    }

    // if (comm_rank == 0)
    // {
    //     printf("Plan created successfully, time spent: %f\n", create_time);
    // }
    plan->report();

    size_t alloc_size = plan->get_alloc_size();
    size_t alloc_bytes = alloc_size * (scaler * sizeof(float));
    size_t aux_bytes = plan->get_aux_bytes();

    auto in = static_cast<float *>(plan->mem_alloc(alloc_bytes));
    auto out = static_cast<float *>(plan->mem_alloc(alloc_bytes));
    auto aux = static_cast<float *>(plan->mem_alloc(aux_bytes));

    memset(in, 0, alloc_bytes);

    for (int iter = 0; iter < WARMUP_ITERATIONS; iter++)
    {
        DTFFT_CXX_CALL(plan->execute(in, out, dtfft::Execute::FORWARD, aux));
        DTFFT_CXX_CALL(plan->execute(out, in, dtfft::Execute::BACKWARD, aux));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    for (int iter = 0; iter < TEST_ITERATIONS; iter++)
    {
        plan->execute(in, out, dtfft::Execute::FORWARD, aux);
        plan->execute(out, in, dtfft::Execute::BACKWARD, aux);
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
        printf("--------------------------------------------\n");
    }

    DTFFT_CXX_CALL(plan->mem_free(in));
    DTFFT_CXX_CALL(plan->mem_free(out));
    DTFFT_CXX_CALL(plan->mem_free(aux));

    return max_s;
}

double run_dtfft_c2c(const std::vector<int> &dims, dtfft::Precision precision, dtfft::Executor executor, bool enable_z_slab, bool use_datatypes, const std::vector<int> &grid=std::vector<int>())
{
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if ( enable_z_slab && dims[2] < comm_size ) {
        if ( comm_rank == 0 )
            printf("Skipping Z-slab benchmark as number of processes (%d) > size of 3rd dimension (%d)\n", comm_size, dims[2]);
        return -1.0;
    }
    setup_dtfft_config(enable_z_slab, use_datatypes);

    dtfft::PlanC2C *plan;

    double create_time = -MPI_Wtime();
    if ( grid.size() > 0 ) {
        // Create Cartesian communicator to get coordinates
        MPI_Comm cart_comm;
        std::vector<int> periods(grid.size(), 0); // non-periodic
        MPI_Cart_create(MPI_COMM_WORLD, grid.size(), const_cast<int*>(grid.data()), periods.data(), 0, &cart_comm);
        
        std::vector<int> coords(grid.size());
        MPI_Cart_coords(cart_comm, comm_rank, grid.size(), coords.data());
        
        // Compute starts and counts
        std::vector<int32_t> starts(dims.size()), counts(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            int total = dims[i];
            int nprocs = grid[i];
            int base_count = total / nprocs;
            int remainder = total % nprocs;
            int my_count = base_count + (coords[i] < remainder ? 1 : 0);
            int my_start = coords[i] * base_count + std::min(coords[i], remainder);
            starts[i] = my_start;
            counts[i] = my_count;
        }
        auto pencil = dtfft::Pencil(starts, counts);
        plan = new dtfft::PlanC2C(pencil, MPI_COMM_WORLD, precision, dtfft::Effort::EXHAUSTIVE, executor);
        MPI_Comm_free(&cart_comm);
    } else {
        plan = new dtfft::PlanC2C(dims, MPI_COMM_WORLD, precision, dtfft::Effort::EXHAUSTIVE, executor);
    }
    create_time += MPI_Wtime();

    int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 4 : 2;
    return run_dtfft_internal(plan, "C2C", create_time, precision, executor, scaler, enable_z_slab);
}

double run_dtfft_r2r(const std::vector<int> &dims, dtfft::Precision precision, bool enable_z_slab, bool use_datatypes)
{
    setup_dtfft_config(enable_z_slab, use_datatypes);

    double create_time = -MPI_Wtime();
    auto plan = new dtfft::PlanR2R(dims, precision, dtfft::Effort::EXHAUSTIVE);
    create_time += MPI_Wtime();

    int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 2 : 1;
    return run_dtfft_internal(plan, "R2R", create_time, precision, dtfft::Executor::NONE, scaler, enable_z_slab);
}
