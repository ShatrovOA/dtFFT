#include "dtfft_config.h"

#ifdef DTFFT_WITH_CUDA

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <stdbool.h>

bool is_device_ptr(const void *ptr)
{
  struct cudaPointerAttributes attrs;

  cudaError_t ierr = cudaPointerGetAttributes(&attrs, ptr);
  if ( ierr == cudaErrorInvalidValue ) return false;
  return attrs.devicePointer != NULL;
}

typedef struct {
  int sm_count;
  int max_threads_per_sm;
  int max_blocks_per_sm;
  size_t shared_mem_per_sm;
  int max_threads_per_block;
  size_t shared_mem_per_block;
  int l2_cache_size;
  int compute_capability_major;
  int compute_capability_minor;
} device_props;


void get_device_props_cuda(int device, device_props *props)
{
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  props->sm_count = prop.multiProcessorCount;
  props->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  props->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
  props->shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  props->max_threads_per_block = prop.maxThreadsPerBlock;
  props->shared_mem_per_block = prop.sharedMemPerBlock;
  props->l2_cache_size = prop.l2CacheSize;
  props->compute_capability_major = prop.major;
  props->compute_capability_minor = prop.minor;
}

MPI_Comm
Comm_f2c(MPI_Fint fcomm)
{
  return MPI_Comm_f2c(fcomm);
}

#endif