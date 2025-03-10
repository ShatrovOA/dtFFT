# dtFFT - DataTyped Fast Fourier Transform

[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()
[![dtfft workflow](https://github.com/ShatrovOA/dtFFT/actions/workflows/main.yml/badge.svg)](https://github.com/ShatrovOA/dtFFT/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/ShatrovOA/dtFFT/graph/badge.svg?token=6BI4AQVH7Z)](https://codecov.io/gh/ShatrovOA/dtFFT)
[![License](https://img.shields.io/github/license/ShatrovOA/dtFFT?color=brightgreen&logo=License)]()

![Pencils](docs/images/pencils.png)

``dtFFT`` (DataTyped Fast Fourier Transform) is a high-performance library designed for parallel data transpositions and optional Fast Fourier Transforms (FFTs) in multidimensional computing environments.

Initially developed to perform zero-copy transpositions using custom MPI datatypes on CPU clusters, ``dtFFT`` leverages these efficient data structures to minimize memory overhead in distributed systems. However, as the demand for GPU-accelerated computing grew, it became clear that MPI datatypes were suboptimal for GPU workflows. To address this, a parallel approach was crafted for GPU execution: instead of relying on custom datatypes, ``dtFFT`` compiles CUDA kernels at runtime using ``nvrtc``, tailoring them to the specific plan and data layout.

The library supports MPI for distributed systems and GPU acceleration via CUDA, integrating seamlessly with external FFT libraries such as FFTW3, MKL DFTI, cuFFT, and VkFFT, or operating in transpose-only mode.

Whether you're working on CPU clusters or GPU-enabled nodes, ``dtFFT`` provides a flexible and efficient framework for scientific computing tasks requiring large-scale data transformations.

## Features
- R2C, C2C, and R2R transforms are supported
- Single and Double precision
- Fortran, C, and C++ interfaces
- 2D and 3D transposition plans
- Slab and Pencil decompositions
- CUDA support
- Can be linked with multiple FFT libraries simultaneously or with no library at all. The execution library can be specified during plan creation. Currently supported libraries are:
  - [FFTW3](https://www.fftw.org/)
  - [MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2/fourier-transform-functions.html)
  - [cuFFT](https://docs.nvidia.com/cuda/cufft/)
  - [VkFFT](https://github.com/DTolm/VkFFT)
- The CUDA version supports different backends for data exchange:
  - MPI
  - NCCL
  - cuFFTMp

## Documentation

The documentation consists of two parts:

**User Documentation**:  
Available at [dtFFT User Documentation](https://dtfft.readthedocs.io/latest/index.html), this section provides instructions on how to compile and use the library, as well as the public API documentation for C, C++, and Fortran. It is generated using [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://readthedocs.org/).

**Internal Documentation**:  
Available at [dtFFT Internal Documentation](https://shatrovoa.github.io/dtFFT/index.html), this section describes the internal structure of the library, including call graphs and detailed implementation insights. It is generated using [FORD](https://forddocs.readthedocs.io/en/latest/index.html).

**Usage examples** can be found in the `tests` folder of the repository.

## Next Steps
- Further optimization of CUDA NVRTC kernels
- Add support for more ``nvshmem``-based backends
- Add HIP support

## Contribution
You can help this project by reporting problems, suggesting improvements, localizing it, or contributing to the code. Go to the [issue tracker](https://github.com/ShatrovOA/dtFFT/issues) and check if your problem or suggestion has already been reported. If not, create a new issue with a descriptive title and detail your suggestion or steps to reproduce the problem.

## Licensing
The source code is licensed under GPL v3. The license is available [here](/LICENSE).