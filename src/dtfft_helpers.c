#include "dtfft_config.h"
#include <stdbool.h>

bool is_same_ptr(const void *ptr1, const void *ptr2) {
  return ptr1 == ptr2;
}

#ifdef DTFFT_WITH_CUDA

#include <cuda_runtime_api.h>
#include <cuda.h>         // cuLaunchKernel

typedef struct
{
  int n_ints;
  int ints[5];
  int n_ptrs;
  void *ptrs[3];
} kernelArgs;

int run_cuda_kernel(void *func, void *in, void *out, dim3 *blocks, dim3 *threads,
  cudaStream_t stream, kernelArgs *args) {

  void *kernelParams[2 + args->n_ints + args->n_ptrs];
  kernelParams[0] = &out;
  kernelParams[1] = &in;
  int arg_counter = 2;
  int i;
  for (i = 0; i < args->n_ints; i++, arg_counter++) {
    kernelParams[arg_counter] = &args->ints[i];
  }
  for (i = 0; i < args->n_ptrs; i++, arg_counter++) {
    kernelParams[arg_counter] = &args->ptrs[i];
  }

  return cuLaunchKernel(
    func,
    blocks->x, blocks->y, blocks->z,
    threads->x, threads->y, threads->z,
    0, stream, kernelParams, NULL);
}

bool is_device_ptr(const void *ptr) {
  struct cudaPointerAttributes attrs;

  cudaError_t ierr = cudaPointerGetAttributes(&attrs, ptr);
  if ( ierr == cudaErrorInvalidValue ) return false;
  return attrs.devicePointer != NULL;
}

#endif