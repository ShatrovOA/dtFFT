#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <dtfft.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void scaleFloatHost(void* buffer, size_t count, size_t scale);
void scaleDoubleHost(void* buffer, size_t count, size_t scale);
void scaleComplexFloatHost(void* buffer, size_t count, size_t scale);
void scaleComplexDoubleHost(void* buffer, size_t count, size_t scale);

float checkFloat(void *check, void *buf, size_t buf_size);
double checkDouble(void *check, void *buf, size_t buf_size);
float checkComplexFloat(void *check, void *buf, size_t buf_size);
double checkComplexDouble(void *check, void *buf, size_t buf_size);

void reportSingle(double *time_forward, double *time_backward, float *local_error,
  const int32_t *nx, const int32_t *ny, const int32_t *nz);
void reportDouble(double *time_forward, double *time_backward, double *local_error,
    const int32_t *nx, const int32_t *ny, const int32_t *nz);

void attach_gpu_to_process();

#if defined(DTFFT_WITH_CUDA)

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(call) do {                                         \
  cudaError_t err = call;                                                 \
  if( cudaSuccess != err) {                                               \
      fprintf(stderr, "Cuda error in '%s:%i : %s.\n",                     \
              __FILE__, __LINE__, cudaGetErrorString(err) );              \
      MPI_Abort(MPI_COMM_WORLD, err);                                     \
  } } while (0);

void scaleFloat(void* buffer, size_t count, size_t scale, cudaStream_t stream);
void scaleDouble(void* buffer, size_t count, size_t scale, cudaStream_t stream);
void scaleComplexFloat(void* buffer, size_t count, size_t scale, cudaStream_t stream);
void scaleComplexDouble(void* buffer, size_t count, size_t scale, cudaStream_t stream);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif

