#pragma once

#define WARMUP_ITERATIONS 5
#define TEST_ITERATIONS 50

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <vector>

#define CUDA_CALL(call)                                           \
  do {                                                            \
    cudaError_t ierr = call;                                      \
    if ( ierr != cudaSuccess ) {                                 \
      fprintf(stderr, "Fatal error in CUDA: %s at %s:%d\n",       \
          cudaGetErrorString(ierr), __FILE__, __LINE__);          \
      MPI_Abort(MPI_COMM_WORLD, ierr);                            \
    }                                                             \
  } while (0)