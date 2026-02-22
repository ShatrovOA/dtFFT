#pragma once

#include "config.hpp"
#include <complex>
#include <cstring>
#include "mkl_service.h"
#include "mkl_cdft.h"


double run_mkl(const std::vector<int>& dims)
{

    std::complex<double> *in;
    std::complex<double> *out;

    DFTI_DESCRIPTOR_DM_HANDLE plan;
    MKL_LONG alloc_size;
    MKL_LONG nx_local;
    MKL_LONG status = DFTI_NO_ERROR;

    MKL_LONG sizes[3];

    double  maxerr;

    double ts, t_global, t_sum;
    int comm_size;
    int comm_rank;
    int i, counter;

    sizes[0] = dims[2];
    sizes[1] = dims[1];
    sizes[2] = dims[0];

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if ( comm_size > dims[dims.size() - 1] ) return -1.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("MKL DFTI benchmark\n");
        printf("#####################################\n");
    }

    status = DftiCreateDescriptorDM(MPI_COMM_WORLD,&plan, DFTI_DOUBLE, DFTI_COMPLEX, 3, sizes);

    status = DftiGetValueDM(plan,CDFT_LOCAL_SIZE,&alloc_size);

    in = (std::complex<double> *)mkl_malloc(alloc_size*sizeof(std::complex<double>), 16);
    out = (std::complex<double> *)mkl_malloc(alloc_size*sizeof(std::complex<double>), 16);

    status = DftiSetValueDM(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptorDM(plan);

    std::memset(in, 0, alloc_size * sizeof(std::complex<double>));

    for (counter = 0; counter < WARMUP_ITERATIONS; counter++)
    {
      status = DftiComputeForwardDM(plan, in, out);
      status = DftiComputeBackwardDM(plan, out, in);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    for (counter = 0; counter < TEST_ITERATIONS; counter++)
    {
      status = DftiComputeForwardDM(plan, in, out);
      status = DftiComputeBackwardDM(plan, out, in);
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

    if (in    != NULL) mkl_free(in);
    if (out   != NULL) mkl_free(out);
    status = DftiFreeDescriptorDM(&plan);

    return max_s;
}