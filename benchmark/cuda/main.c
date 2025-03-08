#include <mpi.h>
#include "config.h"
#include "dtfft_bench.h"
#include "cudecomp_bench.h"


int main(int argc, char *argv[]) {
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CUDA_CALL( cudaSetDevice(local_rank) );
  MPI_Comm_free(&local_comm);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if ( comm_rank == 0 ) {
    printf("Nx = %d, Ny = %d, Nz = %d\n", NX, NY, NZ);
    printf("Number of GPUs: %d\n", comm_size);
  }

  run_cudecomp(CUDECOMP_DOUBLE_COMPLEX);
  // run_cudecomp(CUDECOMP_FLOAT_COMPLEX);
  run_cudecomp(CUDECOMP_DOUBLE);
  run_cudecomp(CUDECOMP_FLOAT);

  run_dtfft(1, DTFFT_DOUBLE, 1);
  run_dtfft(1, DTFFT_DOUBLE, 0);
  // run_dtfft(1, DTFFT_SINGLE, 1);
  // run_dtfft(1, DTFFT_SINGLE, 0);

  run_dtfft(0, DTFFT_DOUBLE, 1);
  run_dtfft(0, DTFFT_DOUBLE, 0);
  run_dtfft(0, DTFFT_SINGLE, 1);
  run_dtfft(0, DTFFT_SINGLE, 0);

  MPI_Finalize();
}