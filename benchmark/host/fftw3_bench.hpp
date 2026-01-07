#include "config.hpp"
#include <complex.h>
#include <cstring>
#include <fftw3-mpi.h>

double run_fftw3(const std::vector<int> &dims)
{
    // ptrdiff_t n[3];
    ptrdiff_t alloc_local, l;
    ptrdiff_t local_ni, local_i_start;
    ptrdiff_t local_no, local_o_start;
    double err;
    fftw_complex *in, *out;
    fftw_plan plan_forw = NULL, plan_back = NULL;
    int comm_size, comm_rank;
    int counter;

    /* Set size of FFT and process mesh */
    // n[0] = dims[3]; n[1] = NY; n[2] = NZ;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if ( comm_size > dims[dims.size() - 1] ) return -1.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("FFTW3 benchmark\n");
        printf("#####################################\n");
    }

    ptrdiff_t nx, ny, nz;
    nx = (ptrdiff_t)dims[2];
    ny = (ptrdiff_t)dims[1];
    nz = (ptrdiff_t)dims[0];

    /* Get parameters of data distribution */
    alloc_local = fftw_mpi_local_size_3d_transposed(nx, ny, nz, MPI_COMM_WORLD, &local_ni, &local_i_start, &local_no, &local_o_start);

    /* Allocate memory */
    in = fftw_alloc_complex(alloc_local);
    out = fftw_alloc_complex(alloc_local);

    double create_time = -MPI_Wtime();
    /* Plan parallel forward FFT */
    plan_forw = fftw_mpi_plan_dft_3d(nx, ny, nz, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE | FFTW_DESTROY_INPUT);

    /* Plan parallel backward FFT */
    plan_back = fftw_mpi_plan_dft_3d(nx, ny, nz, out, in, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MPI_TRANSPOSED_IN | FFTW_MEASURE | FFTW_DESTROY_INPUT);
    create_time += MPI_Wtime();

    // if (comm_rank == 0) printf("Plans created successfully, time spent: %f\n", create_time);

    memset(in, 0, alloc_local * sizeof(fftw_complex));

    for (counter = 0; counter < WARMUP_ITERATIONS; counter++)
    {
        fftw_mpi_execute_dft(plan_forw, in, out);
        fftw_mpi_execute_dft(plan_back, out, in);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    for (counter = 0; counter < TEST_ITERATIONS; counter++)
    {
        fftw_mpi_execute_dft(plan_forw, in, out);
        fftw_mpi_execute_dft(plan_back, out, in);
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

    /* free mem and finalize */
    fftw_destroy_plan(plan_forw);
    fftw_destroy_plan(plan_back);
    fftw_free(in);
    fftw_free(out);

    return max_s;
}
