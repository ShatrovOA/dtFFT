/*
 * AccFFT GPU Precision Benchmark Header
 * Template-based approach for single and double precision testing
 */
#pragma once

#include <stdlib.h>
#include <math.h>
#include "config.hpp"

// AccFFT headers
#include <accfft_gpu.h>   // double precision
#include <accfft_gpuf.h>  // float precision

template<typename T>
struct AccFFTTraits;

template<>
struct AccFFTTraits<double> {
    using ComplexType = Complex;
    using PlanType = accfft_plan_gpu;

    static int local_size_c2c(int* n, int* isize, int* istart, int* osize, int* ostart, MPI_Comm c_comm) {
        return accfft_local_size_dft_c2c_gpu(n, isize, istart, osize, ostart, c_comm);
    }

    static PlanType* plan_c2c(int* n, ComplexType* data, ComplexType* data_hat, MPI_Comm c_comm) {
        return accfft_plan_dft_3d_c2c_gpu(n, data, data_hat, c_comm, ACCFFT_PATIENT);
    }

    static void execute_c2c(PlanType* plan, int direction, ComplexType* in, ComplexType* out) {
        accfft_execute_c2c_gpu(plan, direction, in, out, nullptr, 000);
    }

    static void destroy_plan(PlanType* plan) {
        accfft_destroy_plan_gpu(plan);
    }

    static void cleanup() {
        accfft_cleanup_gpu();
    }

    static const char* precision_name() { return "double"; }
};

template<>
struct AccFFTTraits<float> {
    using ComplexType = Complexf;
    using PlanType = accfft_plan_gpuf;

    static int local_size_c2c(int* n, int* isize, int* istart, int* osize, int* ostart, MPI_Comm c_comm) {
        return accfft_local_size_dft_c2c_gpuf(n, isize, istart, osize, ostart, c_comm);
    }

    static PlanType* plan_c2c(int* n, ComplexType* data, ComplexType* data_hat, MPI_Comm c_comm) {
        return accfft_plan_dft_3d_c2c_gpuf(n, data, data_hat, c_comm, ACCFFT_PATIENT);
    }

    static void execute_c2c(PlanType* plan, int direction, ComplexType* in, ComplexType* out) {
        accfft_execute_c2c_gpuf(plan, direction, in, out, nullptr, 000);
    }

    static void destroy_plan(PlanType* plan) {
        accfft_destroy_plan_gpu(plan);
    }

    static void cleanup() {
        accfft_cleanup_gpuf();
    }

    static const char* precision_name() { return "float"; }
};

template<typename T>
double run_accfft_private(const std::vector<int>& dims, const std::vector<int>& grid_dims) {
    using Traits = AccFFTTraits<T>;
    using ComplexType = typename Traits::ComplexType;
    using PlanType = typename Traits::PlanType;

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create Cartesian Communicator
    int c_dims[2] = {grid_dims[1], grid_dims[2]};
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

    int n[3] = {dims[0], dims[1], dims[2]};
    int isize[3], osize[3], istart[3], ostart[3];
    int alloc_max = Traits::local_size_c2c(n, isize, istart, osize, ostart, c_comm);


    // Allocate memory
    ComplexType *data_gpu, *data_hat_gpu;
    size_t input_size = isize[0] * isize[1] * isize[2] * sizeof(ComplexType);

    CUDA_CALL(cudaMalloc((void**)&data_gpu, input_size));
    CUDA_CALL(cudaMalloc((void**)&data_hat_gpu, input_size));

    // Create plan
    PlanType* plan = Traits::plan_c2c(n, data_gpu, data_hat_gpu, c_comm);

    // Initialize data on GPU
    CUDA_CALL(cudaMemset(data_gpu, 0, input_size));

    // Timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CALL(cudaEventCreate(&startEvent));
    CUDA_CALL(cudaEventCreate(&stopEvent));
    float ms;

    // Warmup
    // if (comm_rank == 0) {
    //     printf("Started warmup\n");
    // }
    for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) {
        Traits::execute_c2c(plan, ACCFFT_FORWARD, data_gpu, data_hat_gpu);
        Traits::execute_c2c(plan, ACCFFT_BACKWARD, data_hat_gpu, data_gpu);
    }

    MPI_Barrier(c_comm);
    // if (comm_rank == 0) {
    //     printf("Ended warmup\n");
    // }

    CUDA_CALL(cudaEventRecord(startEvent));
    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        Traits::execute_c2c(plan, ACCFFT_FORWARD, data_gpu, data_hat_gpu);
        Traits::execute_c2c(plan, ACCFFT_BACKWARD, data_hat_gpu, data_gpu);
    }
    CUDA_CALL(cudaEventRecord(stopEvent));
    CUDA_CALL(cudaEventSynchronize(stopEvent));
    CUDA_CALL(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    // Gather timing statistics
    float min_ms, max_ms, avg_ms;
    MPI_Allreduce(&ms, &min_ms, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&ms, &avg_ms, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    avg_ms /= (float)comm_size;

    // if (comm_rank == 0) {
    //     printf("min time: %f [ms]\n", min_ms);
    //     printf("max time: %f [ms]\n", max_ms);
    //     printf("avg time: %f [ms]\n", avg_ms);
    //     printf("----------------------------------------\n");
    // }

    // Cleanup
    CUDA_CALL(cudaFree(data_gpu));
    CUDA_CALL(cudaFree(data_hat_gpu));
    CUDA_CALL(cudaEventDestroy(startEvent));
    CUDA_CALL(cudaEventDestroy(stopEvent));

    Traits::destroy_plan(plan);
    Traits::cleanup();
    MPI_Comm_free(&c_comm);

    return (double)max_ms;
}

template<typename T>
double run_accfft(const std::vector<int>& dims) {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int product = 1;
    for( auto d : dims) {
        product = product * d;
    }
    product /= comm_size;
    size_t p = (size_t)product * 2 * sizeof(T);
    product *= 2 * sizeof(T);

    if ( (size_t)product != p ) {
        if (comm_rank == 0) {
            printf("Skipping AccFFT benchmark\n");
        }

        return -1.0;
    }

    if (comm_rank == 0) {
        using Traits = AccFFTTraits<T>;
        printf("----------------------------------------\n");
        printf("AccFFT GPU benchmark\n");
        printf("Precision is %s\n", Traits::precision_name());
        printf("----------------------------------------\n");
    }

    double best_time = INFINITY;
    int best_grid[3];
    double test_time;

    for (int p = 1; p <= comm_size; p++)
    {
        if (comm_size % p)
            continue;
        int q = comm_size / p;
        if (dims[1] < p || dims[2] < q)
            continue;
        if ( dims[0] < p || dims[1] < q)
            continue;

        test_time = run_accfft_private<T>(dims, {1, p, q});

        if ( test_time < best_time )
        {
            best_time = test_time;
            best_grid[0] = 1;
            best_grid[1] = p;
            best_grid[2] = q;
        }

    }

    if (comm_rank == 0)
    {
        printf("AccFFT Results\n");
        printf("----------------------------------------\n");
        printf("Fastest execution time: %f\n", best_time);
        printf("Fastest grid: %ix%ix%i\n", best_grid[0], best_grid[1], best_grid[2]);
        printf("----------------------------------------\n");
    }
    return best_time;
}

// // Wrapper functions for external usage
// double run_accfft_double(const std::vector<int>& dims) {
//     return run_accfft<double>(dims);
// }

// double run_accfft_float(const std::vector<int>& dims) {
//     return run_accfft<float>(dims);
// }
