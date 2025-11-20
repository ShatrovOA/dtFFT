# dtFFT - DataTyped Fast Fourier Transform

[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://github.com/ShatrovOA/dtFFT/releases)
[![Building](https://github.com/ShatrovOA/dtFFT/actions/workflows/gnu_linux.yml/badge.svg)](https://github.com/ShatrovOA/dtFFT/actions/workflows/gnu_linux.yml)
[![codecov](https://codecov.io/gh/ShatrovOA/dtFFT/graph/badge.svg?token=6BI4AQVH7Z)](https://codecov.io/gh/ShatrovOA/dtFFT)
[![License](https://img.shields.io/github/license/ShatrovOA/dtFFT?color=brightgreen&logo=License)](https://github.com/ShatrovOA/dtFFT/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/dtfft/badge/?version=latest)](https://dtfft.readthedocs.io/latest/?badge=latest)
[![Build and Deploy Documentation](https://github.com/ShatrovOA/dtFFT/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/ShatrovOA/dtFFT/actions/workflows/gh-pages.yml)

![Pencils](docs/images/pencils.png)

dtFFT is a high-performance library for parallel data transpositions and optional Fast Fourier Transforms (FFTs) in multidimensional computing environments. It minimizes memory overhead in distributed systems by using custom MPI datatypes for CPU clusters and runtime-compiled CUDA kernels for GPU workflows. Supports integration with FFT libraries like FFTW3, MKL, cuFFT, and VkFFT, or transpose-only mode.

Key benefits: Zero-copy transpositions, GPU acceleration, and seamless MPI/CUDA integration for scientific computing.

dtFFT aims to optimize following cycles of transformations (forward and backward):

```math
X \times \dfrac{Y}{P_1} \to Y \times \dfrac{X}{P_1}
```
for 2D case, and
```math
X \times \dfrac{Y}{P_1} \times \dfrac{Z}{P_2} \to Y \times \dfrac{Z}{P_2} \times \dfrac{X}{P_1} \to Z \times \dfrac{X}{P_1} \times \dfrac{Y}{P_2}
```
for 3D case. Where $X, Y, Z$ are the spatial dimensions of the data, and $P_1, P_2$ are the number of processes in the $Y$ and $Z$ directions, respectively.

## Features
- **Transform Types**: R2C, C2C, and R2R transforms
- **Precision**: Single and double precision support
- **Interfaces**: Fortran, C, and C++ APIs
- **Decompositions**: 2D and 3D transposition plans with Slab and Pencil modes
- **Transpositions**: Custom MPI datatypes enhanced with standard host-based transpositions
- **GPU Support**: CUDA acceleration with runtime kernel compilation
- **FFT Libraries**: built-in support:
  - [FFTW3](https://www.fftw.org/)
  - [MKL DFTI](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2/fourier-transform-functions.html)
  - [cuFFT](https://docs.nvidia.com/cuda/cufft/)
  - [VkFFT](https://github.com/DTolm/VkFFT)
- **CUDA Backends**: MPI, NCCL, cuFFTMp for data exchange

## Limitations
- Memory is assumed to be contiguous. Ghost boundaries are not allowed.
- OpenMP for multicore parallelism is not supported.
- Maximum number of elements per process/GPU cannot exceed $2^{31} - 1$ - max value of int32

## Requirements
- Fortran/C/C++ compilers (GCC, Intel and NVHPC-SDK are tested)
- MPI (OpenMPI, MPICH and Intel MPI are tested)

## Installation
1. Clone the repository: `git clone https://github.com/ShatrovOA/dtFFT.git`
2. Configure with CMake: `mkdir build && cd build && cmake ..`
3. Build: `make`
4. Install: `make install`

For detailed instructions, see the [Building the Library](https://dtfft.readthedocs.io/latest/build.html).

## Quick Start
```fortran
use dtfft
use mpi
use iso_fortran_env
type(dtfft_plan_c2c_t) :: plan
complex(real64), pointer :: real_buffer(:,:)
complex(real64), pointer :: fourier_buffer(:,:)
integer(int32) :: in_counts(2), out_counts(2)
integer(int64) :: alloc_size ! Can be bigger then `product(in_counts)` and `product(out_counts)`

! Create plan
call plan%create([100, 100], comm=MPI_COMM_WORLD)
! Get memory requirements
call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size)
! Allocate memory
call plan%mem_alloc(alloc_size, real_buffer, in_counts)
call plan%mem_alloc(alloc_size, fourier_buffer, out_counts)

! Execute plan
call plan%execute(real_buffer, fourier_buffer, DTFFT_EXECUTE_FORWARD)
! Now data is aligned in Y direction
! Execute backwards
call plan%execute(fourier_buffer, real_buffer, DTFFT_EXECUTE_BACKWARD)
! Free all memory
call plan%mem_free(real_buffer)
call plan%mem_free(fourier_buffer)
! Destroy plan
call plan%destroy()
```
**More examples** can be found in the `tests` folder of the repository.

## Documentation

The documentation consists of two parts:

**User Documentation**:
Available at [dtFFT User Documentation](https://dtfft.readthedocs.io/latest/index.html), this section provides instructions on how to compile and use the library, as well as the public API documentation for C, C++, and Fortran. It is generated using [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://readthedocs.org/).

**Internal Documentation**:
Available at [dtFFT Internal Documentation](https://shatrovoa.github.io/dtFFT/index.html), this section describes the internal structure of the library, including call graphs and detailed implementation insights. It is generated using [FORD](https://forddocs.readthedocs.io/en/latest/index.html).

## API Reference
- [C API](https://dtfft.readthedocs.io/latest/api_c.html)
- [C++ API](https://dtfft.readthedocs.io/latest/api_cxx.html)
- [Fortran API](https://dtfft.readthedocs.io/latest/api_fortran.html)

## Roadmap
The following is an ambitious list of features to implement. The items are in no particular order.

- nvSHMEM-based backends
- HIP platform
- zfp compression
- 3d grid decomposition (bricks)
- Ghost boundaries and support for halo exchange
- long double/quad precision

## Contributing
We welcome contributions! Report issues or suggest improvements via the [issue tracker](https://github.com/ShatrovOA/dtFFT/issues). For code contributions, please follow standard GitHub workflows (fork, branch, PR).

## Licensing
Licensed under GPL v3. See [LICENSE](LICENSE) for details.