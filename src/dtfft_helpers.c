#include "dtfft_config.h"
#include "dtfft_private.h"
#include <stdbool.h>
#include <stdlib.h>
#include <dlfcn.h>


void mem_alloc_host(size_t alloc_size, void **ptr) {
  size_t displ = alloc_size % ALLOC_ALIGNMENT;
  size_t alloc_size_ = alloc_size;
  if ( displ != 0 ) {
    alloc_size_ += (ALLOC_ALIGNMENT - displ);
  }
  void *ptr_ = aligned_alloc(ALLOC_ALIGNMENT, alloc_size_);
  *ptr = ptr_;
}

void mem_free_host(void *ptr) {
  free(ptr);
}

// #define DTFFT_WITH_CUDA
#ifdef DTFFT_WITH_CUDA

#include <cuda_runtime_api.h>
// #include <cuda.h>         // cuLaunchKernel
#include <mpi.h>

typedef struct
{
  int n_ints;
  int ints[5];
  int n_ptrs;
  void *ptrs[3];
} kernelArgs;

typedef int (*cuLaunchKernel_t)(
  void* f,
  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
  unsigned int sharedMemBytes, cudaStream_t hStream,
  void** kernelParams, void** extra
);

int run_cuda_kernel(void *func, void *in, void *out, dim3 *blocks, dim3 *threads,
  cudaStream_t stream, kernelArgs *args, void *func_ptr) {

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

  cuLaunchKernel_t cuLaunchKernel = (cuLaunchKernel_t)func_ptr;

  return cuLaunchKernel(
    func,
    blocks->x, blocks->y, blocks->z,
    threads->x, threads->y, threads->z,
    0, stream, kernelParams, NULL);
}

bool is_device_ptr(const void *ptr)
{
  struct cudaPointerAttributes attrs;

  cudaError_t ierr = cudaPointerGetAttributes(&attrs, ptr);
  if ( ierr == cudaErrorInvalidValue ) return false;
  return attrs.devicePointer != NULL;
}



void get_cuda_architecture(int device, int *major, int *minor)
{
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  *major = prop.major;
  *minor = prop.minor;
}

MPI_Comm
Comm_f2c(MPI_Fint fcomm)
{
  return MPI_Comm_f2c(fcomm);
}

#endif