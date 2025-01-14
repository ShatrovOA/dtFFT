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


  run_cudecomp();
  // run_dtfft(1);
  run_dtfft(0);

  MPI_Finalize();
}