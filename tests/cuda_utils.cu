#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstddef>
#include <complex>
#include <stdio.h>

// Helper function for cuFloatComplex division by scalar
__device__ __host__ cuFloatComplex operator/(cuFloatComplex a, float b) {
    return make_cuFloatComplex(a.x / b, a.y / b);
}

// Helper function for cuDoubleComplex division by scalar
__device__ __host__ cuDoubleComplex operator/(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x / b, a.y / b);
}


template<typename T, typename T2>
__global__ void scaleKernel(T* buffer, size_t count, size_t scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        buffer[idx] = buffer[idx] / static_cast<T2>(scale);
    }
}

void launchScaleKernel(void* buffer, size_t count, size_t scale,
                      size_t elementSize, cudaStream_t stream, int typeFlag = 0) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    switch(elementSize) {
        case 4:  // float
            scaleKernel<float, float><<<blocks, threads, 0, stream>>>(
                static_cast<float*>(buffer), count, scale);
            break;
        case 8:
            if (typeFlag == 1) { // cuFloatComplex
                scaleKernel<cuFloatComplex, float><<<blocks, threads, 0, stream>>>(
                    static_cast<cuFloatComplex*>(buffer), count, scale);
            } else { // double (default для 8 байт)
                scaleKernel<double, double><<<blocks, threads, 0, stream>>>(
                    static_cast<double*>(buffer), count, scale);
            }
            break;
        case 16: // cuDoubleComplex
            scaleKernel<cuDoubleComplex, double><<<blocks, threads, 0, stream>>>(
                static_cast<cuDoubleComplex*>(buffer), count, scale);
            break;
        default:
            fprintf(stderr, "Unsupported element size: %zu\n", elementSize);
            exit(-1);
    }
}

// C/Fortran wrappers
extern "C" void scaleFloat(void* buffer, size_t count, size_t scale,
                               cudaStream_t stream) {
    launchScaleKernel(buffer, count, scale, 4, stream);
}

extern "C" void scaleDouble(void* buffer, size_t count, size_t scale,
                                cudaStream_t stream) {
    launchScaleKernel(buffer, count, scale, 8, stream, 0);
}

extern "C" void scaleComplexFloat(void* buffer, size_t count, size_t scale,
                                      cudaStream_t stream) {
    launchScaleKernel(buffer, count, scale, 8, stream, 1);
}

extern "C" void scaleComplexDouble(void* buffer, size_t count, size_t scale,
                                       cudaStream_t stream) {
    launchScaleKernel(buffer, count, scale, 16, stream);
}