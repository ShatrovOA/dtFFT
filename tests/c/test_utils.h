#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <dtfft.h>

#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_WITH_MOCK_ENABLED)
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(DTFFT_WITH_MOCK_ENABLED) && defined(DTFFT_WITH_CUDA)
// Mock CUDA declarations for CPU testing
typedef void* cudaStream_t;
typedef enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidValue = 1
} cudaError_t;

typedef enum {
    cudaMemAttachGlobal = 1
} cudaMemAttachFlags_t;

// Mock CUDA function declarations
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaMemset(void* ptr, int value, size_t count);
const char* cudaGetErrorString(cudaError_t error);
#endif


void setTestValuesComplexDouble(void *, size_t);
void setTestValuesComplexFloat(void *, size_t);
void setTestValuesDouble(void *, size_t);
void setTestValuesFloat(void *, size_t);
void createGridDims(int32_t, int32_t*, int32_t*, int32_t*, int32_t*);

void attach_gpu_to_process();

#if defined(DTFFT_WITH_CUDA)

#define CUDA_SAFE_CALL(call) do {                                         \
  cudaError_t err = call;                                                 \
  if( cudaSuccess != err) {                                               \
      fprintf(stderr, "Cuda error in '%s:%i : %s.\n",                     \
              __FILE__, __LINE__, cudaGetErrorString(err) );              \
      MPI_Abort(MPI_COMM_WORLD, err);                                     \
  } } while (0);

void scaleFloat(int32_t, void *, size_t, size_t, int32_t, dtfft_stream_t);
void scaleDouble(int32_t, void *, size_t, size_t, int32_t, dtfft_stream_t);
void scaleComplexFloat(int32_t, void *, size_t, size_t, int32_t, dtfft_stream_t);
void scaleComplexDouble(int32_t, void *, size_t, size_t, int32_t, dtfft_stream_t);

void complexDoubleH2D(void *, void *, size_t, int32_t);
void complexFloatH2D(void *, void *, size_t, int32_t);
void doubleH2D(void *, void *, size_t, int32_t);
void floatH2D(void *, void *, size_t, int32_t);

void checkAndReportComplexDouble(size_t, double, double, void *, size_t, void *, int32_t);
void checkAndReportComplexFloat(size_t, double, double, void *, size_t, void *, int32_t);
void checkAndReportDouble(size_t, double, double, void *, size_t, void *, int32_t);
void checkAndReportFloat(size_t, double, double, void *, size_t, void *, int32_t);
#else
void scaleFloat(int32_t, void *, size_t, size_t);
void scaleDouble(int32_t, void *, size_t, size_t);
void scaleComplexFloat(int32_t, void *, size_t, size_t);
void scaleComplexDouble(int32_t, void *, size_t, size_t);

void complexDoubleH2D(void *, void *, size_t);
void complexFloatH2D(void *, void *, size_t);
void doubleH2D(void *, void *, size_t);
void floatH2D(void *, void *, size_t);

void checkAndReportComplexDouble(size_t, double, double, void *, size_t, void *);
void checkAndReportComplexFloat(size_t, double, double, void *, size_t, void *);
void checkAndReportDouble(size_t, double, double, void *, size_t, void *);
void checkAndReportFloat(size_t, double, double, void *, size_t, void *);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif

