#include <stdio.h>
#include <stdint.h>

#include "vkFFT.h"
#include "mpi.h"
#include "dtfft_config.h"

#define VKFFT_CALL(call)                                          \
  do {                                                            \
    VkFFTResult ierr = call;                                      \
    if (ierr != VKFFT_SUCCESS) {                                  \
      fprintf(stderr, "Fatal error in vkFFT: %s at %s:%d\n",      \
          getVkFFTErrorString(ierr), __FILE__, __LINE__);         \
      MPI_Abort(MPI_COMM_WORLD, ierr);                            \
    }                                                             \
  } while (0)

#define CUDA_CALL(call)                                           \
  do {                                                            \
    cudaError_t ierr = call;                                      \
    if ( ierr != cudaSuccess ) {                                  \
      fprintf(stderr, "Fatal error in CUDA: %s at %s:%d\n",       \
          cudaGetErrorString(ierr), __FILE__, __LINE__);          \
      MPI_Abort(MPI_COMM_WORLD, ierr);                            \
    }                                                             \
  } while (0)


void vkfft_create(const int8_t rank, const int *dims, const int8_t precision, const int how_many,
                  const int8_t r2c, const int8_t c2r, const int8_t dct, const int8_t dst, cudaStream_t stream, VkFFTApplication **app_handle) {
  VkFFTConfiguration config = {};
  VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(VkFFTApplication));

  config.FFTdim = rank;
  int dim;
  for (dim = 0; dim < rank; dim++)
  {
    config.size[dim] = dims[dim];
  }
  config.doublePrecision = precision == CONF_DTFFT_DOUBLE ? 1 : 0;
  config.numberBatches = how_many;

  CUdevice device;
  int device_num;

  CUDA_CALL( cudaGetDevice(&device_num) );
  CUDA_CALL( cuDeviceGet(&device, device_num) );
  config.device = &device;
  config.stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  config.stream[0] = stream;
  config.num_streams = 1;

  config.isInputFormatted = 1;
  config.isOutputFormatted = 1;

  config.performDCT = dct;
  config.performDST = dst;

  if ( r2c || c2r ) {
    config.performR2C = 1;
    if ( r2c ) {
      config.inputBufferStride[0] = dims[0];
      config.outputBufferStride[0] = (dims[0] / 2) + 1;
      config.makeForwardPlanOnly = 1;
    } else {
      config.inputBufferStride[0] = (dims[0] / 2) + 1;
      config.outputBufferStride[0] = dims[0];
      config.makeInversePlanOnly = 1;
    }
    for ( dim = 1; dim < rank; dim++ ) {
      config.inputBufferStride[dim] = config.inputBufferStride[dim - 1] * dims[dim];
      config.outputBufferStride[dim] = config.outputBufferStride[dim - 1] * dims[dim];
    }
  }

  VKFFT_CALL( initializeVkFFT(app, config) );
  *app_handle = app;
}

void vkfft_execute(VkFFTApplication *app_handle, void *in, void *out, int8_t sign) {
  VkFFTLaunchParams launch_handle = {};
  launch_handle.buffer = &in;
  launch_handle.inputBuffer = &in;
  launch_handle.outputBuffer = &out;

  VKFFT_CALL( VkFFTAppend(app_handle, (int)sign, &launch_handle) );
}

void vkfft_destroy(VkFFTApplication *app_handle){
  deleteVkFFT(app_handle);
}
