#pragma once

#include "heffte.h"
#include "config.hpp"
// #include "dtfft.hpp"

// There are multiple ways to select the GPU tag

// 1. Using the default_backend trait with the tag::gpu for the location
using backend_tag = heffte::backend::fftw;

template <typename T>
// double run_heffte_private(heffte::box3d<> &boxin, heffte::box3d<> &boxout)
double run_heffte_private(const std::vector<int> &dims, const std::vector<int> &grid)
{

    MPI_Comm comm = MPI_COMM_WORLD;

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    std::array<int, 3> input_grid = {grid[0], grid[1], grid[2]};
    std::array<int, 3> output_grid = {grid[1], grid[2], grid[0]};

    std::array<int, 3> lower = {0, 0, 0};
    std::array<int, 3> upper = {dims[0] - 1, dims[1] - 1, dims[2] - 1};
    auto world = heffte::box3d<>(lower, upper);

    std::vector<heffte::box3d<>> inboxes  = heffte::split_world(world, input_grid);
    std::vector<heffte::box3d<>> outboxes = heffte::split_world(world, output_grid);

    heffte::box3d<> boxin = inboxes[comm_rank];
    heffte::box3d<> boxout = outboxes[comm_rank];

    // define the heffte class and the input and output geometry
    // heffte::plan_options can be specified just as in the backend::fftw
    heffte::fft3d<backend_tag> fft(boxin, boxout, comm);

    size_t alloc_size = std::max(fft.size_inbox(), fft.size_outbox());

    std::vector<std::complex<T>> input(alloc_size);

    // allocate memory on the device for the output
    std::vector<std::complex<T>> output(alloc_size);

    std::fill(input.begin(), input.end(), std::complex<T>(0.0, 0.0));

    // allocate scratch space, this is using the public type alias buffer_container
    // and for the cufft backend this is heffte::gpu::vector
    // for the CPU backends (fftw and mkl) the buffer_container is std::vector
    heffte::fft3d<backend_tag>::buffer_container<std::complex<T>> workspace(fft.size_workspace());


    // for (int iter = 0; iter < WARMUP_ITERATIONS; iter++)
    // {
    //     // perform forward fft using arrays and the user-created workspace
    //     fft.forward(input.data(), output.data(), workspace.data(), heffte::scale::none);
    //     fft.backward(output.data(), input.data(), workspace.data(), heffte::scale::none);
    // }

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    for (int iter = 0; iter < 1; iter++)
    {
        // perform forward fft using arrays and the user-created workspace
        // fft.forward(input.data(), output.data(), workspace.data(), heffte::scale::none);
        fft.backward(output.data(), input.data(), workspace.data(), heffte::scale::none);
    }
    double end_time = MPI_Wtime();
    double elapsed_time = (end_time - start_time);

    double min_s, max_s, avg_s;
    MPI_Allreduce(&elapsed_time, &min_s, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&elapsed_time, &max_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&elapsed_time, &avg_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_s /= (double)comm_size;

    // if (comm_rank == 0)
    // {
    //     printf("min time: %f [s]\n", min_s);
    //     printf("max time: %f [s]\n", max_s);
    //     printf("avg time: %f [s]\n", avg_s);
    //     printf("avg time per iteration: %f [s]\n", avg_s / (double)TEST_ITERATIONS);
    //     printf("----------------------------------------\n");
    // }

    return max_s;
}

template <typename T>
double run_heffte(const std::vector<int> &dims, const std::vector<int> &grid = std::vector<int>())
{
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("HeFFTe benchmark\n");
        printf("#####################################\n");
    }

        std::string root_filename = "_tracing_file";
    heffte::init_tracing(root_filename);

    double best_time = INFINITY;
    int best_grid[3];
    double test_time;
    if ( grid.size() > 0 ) {
        test_time = run_heffte_private<T>(dims, grid);
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
            if (dims[1] < p || dims[2] < q)
                continue;
            if ( dims[0] < p || dims[1] < q)
                continue;

            test_time = run_heffte_private<T>(dims, {1, p, q});

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
                best_grid[0] = 1;
                best_grid[1] = p;
                best_grid[2] = q;
            }

        }
    }

    heffte::finalize_tracing();

    if (comm_rank == 0)
    {
        printf("heFFTe Results\n");
        printf("----------------------------------------\n");
        printf("Fastest execution time: %f\n", best_time);
        printf("Fastest grid: %ix%ix%i\n", best_grid[0], best_grid[1], best_grid[2]);
        printf("----------------------------------------\n");
    }

    return best_time;
}