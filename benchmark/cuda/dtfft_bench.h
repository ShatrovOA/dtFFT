#include "config.h"

#include <stdbool.h>
#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#define DTFFT_WITH_CUDA
#include <dtfft.h>
#include <cuda_runtime.h>



void run_dtfft(bool enable_z_slab) {
  dtfft_plan plan;
  int comm_rank, comm_size;
  size_t i;
  int32_t in_counts[3], out_counts[3], n[3] = {NX, NY, NZ};


  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


  if(comm_rank == 0) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free memory available before: %li\n", free);
    printf("----------------------------------------\n");
    printf(" dtFFT benchmark: c2c_3d, DTFFT_DOUBLE  \n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", NX, NY, NZ);
    printf("Number of processors: %d\n", comm_size);
    if ( enable_z_slab ) {
      printf("Using Z-slab optimization\n");
    }
    printf("----------------------------------------\n");
  }

  if ( enable_z_slab ) {
    dtfft_enable_z_slab();
  } else {
    dtfft_disable_z_slab();
  }

  dtfft_executor_t executor_type = DTFFT_EXECUTOR_NONE;
  // dtfft_disable_pipelined_backends();
  // dtfft_enable_mpi_backends();
  dtfft_set_gpu_backend(DTFFT_GPU_BACKEND_NCCL);

  double create_time = -MPI_Wtime();
  // Create plan
  DTFFT_CALL( dtfft_create_plan_c2c(3, n, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_MEASURE, executor_type, &plan) );
  int64_t alloc_size;
  DTFFT_CALL( dtfft_get_alloc_size(plan, &alloc_size) );
  create_time +=MPI_Wtime();
  if(comm_rank == 0) {
    printf("Plan creation time: %f [s]\n", create_time);
  }

  cudaStream_t stream;
  DTFFT_CALL( dtfft_get_stream(plan, &stream) );

  double2 *in, *out, *aux;
  alloc_size *= sizeof(double2);

  CUDA_CALL( cudaMalloc((void**)&in,  alloc_size) );
  CUDA_CALL( cudaMalloc((void**)&out, alloc_size) );
  CUDA_CALL( cudaMalloc((void**)&aux, alloc_size) );

  CUDA_CALL( cudaMemset(in, 0, alloc_size));

  if(comm_rank == 0) {
    printf("Started warmup\n");
  }
  for ( int iter = 0; iter < WARMUP_ITERATIONS; iter++ ) {
    DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux) );
    DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, aux) );
  }
  CUDA_CALL( cudaStreamSynchronize(stream) );

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0) {
    printf("Ended warmup\n");
  }

  cudaEvent_t startEvent, stopEvent;
  CUDA_CALL( cudaEventCreate(&startEvent) );
  CUDA_CALL( cudaEventCreate(&stopEvent) );
  float ms;


  CUDA_CALL( cudaEventRecord(startEvent, stream) );

  for ( int iter = 0; iter < TEST_ITERATIONS; iter++ ) {
    dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux);
    dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, aux);
  }

  CUDA_CALL( cudaEventRecord(stopEvent, stream) );
  CUDA_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  float min_ms, max_ms, avg_ms;

  MPI_Allreduce(&ms, &min_ms, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &avg_ms, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    printf("min time: %f [ms]\n", min_ms);
    printf("max time: %f [ms]\n", max_ms);
    printf("avg time: %f [ms]\n", avg_ms /= (float)comm_size);
    printf("----------------------------------------\n");
  }

  CUDA_CALL( cudaFree(in) );
  CUDA_CALL( cudaFree(out) );
  CUDA_CALL( cudaFree(aux) );

  CUDA_CALL( cudaEventDestroy(startEvent) );
  CUDA_CALL( cudaEventDestroy(stopEvent) );

  DTFFT_CALL( dtfft_destroy(&plan) );
  if ( comm_rank == 0 ) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free memory available after: %li\n", free);
  }
}