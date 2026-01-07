#pragma once

#define WARMUP_ITERATIONS 5
#define TEST_ITERATIONS 50

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <vector>
#include <stdio.h>

#define CUDA_CALL(call)                                           \
  do {                                                            \
    cudaError_t ierr = call;                                      \
    if ( ierr != cudaSuccess ) {                                 \
      fprintf(stderr, "Fatal error in CUDA: %s at %s:%d\n",       \
          cudaGetErrorString(ierr), __FILE__, __LINE__);          \
      MPI_Abort(MPI_COMM_WORLD, ierr);                            \
    }                                                             \
  } while (0)


void release_mpi_handles(MPI_Comm comm)
{
  void *ptr1, *ptr2;

  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  size_t before, after, dummy;

  CUDA_CALL( cudaMemGetInfo(&before, &dummy) );

  if ( comm_rank == 0 ) printf("Free mem before: %zu\n", before);

  size_t alloc_bytes = 1024 * 1024 * 30;
  CUDA_CALL( cudaMalloc(&ptr1, alloc_bytes * comm_size) );
  CUDA_CALL( cudaMalloc(&ptr2, alloc_bytes * comm_size) );

  MPI_Alltoall(ptr1, (int)alloc_bytes, MPI_BYTE, ptr2, (int)alloc_bytes, MPI_BYTE, comm);
  MPI_Alltoall(ptr2, (int)alloc_bytes, MPI_BYTE, ptr1, (int)alloc_bytes, MPI_BYTE, comm);

  CUDA_CALL( cudaFree(ptr1) );
  CUDA_CALL( cudaFree(ptr2) );

  CUDA_CALL( cudaMemGetInfo(&after, &dummy) );

  if ( comm_rank == 0 ) printf("Free mem after: %zu\n", after);
}