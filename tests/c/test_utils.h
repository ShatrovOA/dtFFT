#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <dtfft.h>
#include <float.h>

void print_timers(double time)
{
  double t_sum, t_min, t_max;
  int comm_size, comm_rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


  MPI_Reduce(&time, &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(comm_rank == 0)
  {
    printf("  avg: %f\n", t_sum / (double) comm_size);
    printf("  min: %f\n", t_min);
    printf("  max: %f\n", t_max);
    printf("----------------------------------------\n");
  }
}

void report_execution_time(double time_forward, double time_backward)
{
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if(comm_rank == 0)
  {
    printf("----------------------------------------\n");
    printf("|          Forward execution           |\n");
    printf("----------------------------------------\n");
  }
  print_timers(time_forward);
  if(comm_rank == 0)
  {
    printf("|         Backward execution           |\n");
    printf("----------------------------------------\n");
  }
  print_timers(time_backward);
}



void report_private(double local_error, double errthr, double time_forward, double time_backward) {

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  double global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < errthr) {
      printf("Test PASSED!\n");
    } else {
      fprintf(stderr, "Test FAILED, error = %e, threshold = %e\n", global_error, errthr);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
  report_execution_time(time_forward, time_backward);
}

void report_float(int *nx, int *ny, int *nz, float local_error, double time_forward, double time_backward) {
  int temp = (*nx) * (*ny);
  if (nz) temp *= (*nz);
  float errthr = 5.0f * logf((float) temp) / logf(2.0f) * FLT_EPSILON;
  report_private((double)local_error, (double)errthr, time_forward, time_backward);
}

void report_double(int *nx, int *ny, int *nz, double local_error, double time_forward, double time_backward) {
  int temp = (*nx) * (*ny);
  if (nz) temp *= (*nz);
  double errthr = 5.0 * log((double) temp) / log(2.0) * DBL_EPSILON;
  report_private(local_error, errthr, time_forward, time_backward);
}

#ifdef DTFFT_WITH_CUDA
#include <cuda_runtime_api.h>


#define CUDA_SAFE_CALL(call) do {                                \
  cudaError_t err = call;                                                   \
  if( cudaSuccess != err) {                                               \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
              __FILE__, __LINE__, cudaGetErrorString(err) );              \
      MPI_Abort(MPI_COMM_WORLD, err);                                     \
  } } while (0);
#endif

void assign_device_to_process() {
#ifdef DTFFT_WITH_CUDA
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CUDA_SAFE_CALL( cudaSetDevice(local_rank) );

  MPI_Comm_free(&local_comm);
#endif
}


#endif

