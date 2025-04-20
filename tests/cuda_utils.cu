#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>


template<typename T>
__global__ void scaleKernel(T* buffer, size_t count, T scale_value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        buffer[idx] = buffer[idx] * scale_value;
    }
}

void launchScaleKernel(void* buffer, size_t count, size_t scale, bool is_double, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    double scale_value = 1.0 / static_cast<double>(scale);

    if ( is_double ) {
        scaleKernel<double><<<blocks, threads, 0, stream>>>(static_cast<double*>(buffer), count, scale_value);
    } else {
        scaleKernel<float><<<blocks, threads, 0, stream>>>(static_cast<float*>(buffer), count, static_cast<float>(scale_value));
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// C/Fortran wrappers
void scaleFloat(void* buffer, size_t count, size_t scale, cudaStream_t stream)
{
    launchScaleKernel(buffer, count, scale, 0, stream);
}

void scaleDouble(void* buffer, size_t count, size_t scale, cudaStream_t stream)
{
    launchScaleKernel(buffer, count, scale, 1, stream);
}

void scaleComplexFloat(void* buffer, size_t count, size_t scale, cudaStream_t stream)
{
    scaleFloat(buffer, 2 * count, scale, stream);
}

void scaleComplexDouble(void* buffer, size_t count, size_t scale, cudaStream_t stream)
{
    scaleDouble(buffer, 2 * count, scale, stream);
}

#ifdef __cplusplus
}
#endif
