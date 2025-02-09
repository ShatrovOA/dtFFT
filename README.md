# dtFFT -  DataTyped Fast Fourier Transform

[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()
[![dtfft workflow](https://github.com/ShatrovOA/dtFFT/actions/workflows/main.yml/badge.svg)](https://github.com/ShatrovOA/dtFFT/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/ShatrovOA/dtFFT/graph/badge.svg?token=6BI4AQVH7Z)](https://codecov.io/gh/ShatrovOA/dtFFT)
[![License](https://img.shields.io/github/license/ShatrovOA/dtFFT?color=brightgreen&logo=License)]()

This repository contains new library to perform FFT on a distributed memory cluster. It is written in modern Fortran and uses MPI to handle communications between processes.
The main idea of this library is to implement zero-copy algorithms in 2d and 3d cases. It uses advance MPI to create send and recieve MPI datatypes in a such way that recieved data will be aligned in memory and ready to run 1d FFT.

Following Fortran column-major order consider XYZ is a three-dimensional buffer: X index varies most quickly. dtFFT will create MPI derived datatypes which will produce
- Forward transform: XYZ --> YXZ --> ZXY
- Backward transform: ZXY --> YXZ --> XYZ

Special optimization is automatically used in 3D plan in case number of MPI processes is less then number of grid points in Z direction. In such case additional MPI datatype will be created to perform direct data transposition: XYZ --> ZXY.

![Pencils](doc/pencils.png)

## Features
- R2C, C2C, R2R transforms are supported
- Single and double precision
- Fortran, C and C++ interfaces
- 2D and 3D transposition plans
- Slab and Pencil decompositions
- Can be linked with multiple FFT libraries simultaneously. Execution library can be specified during plan creation. Currenly supported libraries are:
  -  [FFTW3](https://www.fftw.org/)
  -  [MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2/fourier-transform-functions.html)


## Usage
Basic usage of dtFFT consists of 6 steps:
- Create plan
- Obtain local sizes in "real" and "Fourier" spaces
- Allocate memory
- Execute plan as many times as you want
- Destroy plan

### Plan creation
#### Fortran
3 Derived types are available in fortran interface: `dtfft_plan_c2c`, `dtfft_plan_r2c` and `dtfft_plan_r2r`. To create plan one have to call `create` method.
User is able to provide two kinds of communicators. Without grid topology, e.g. `MPI_COMM_WORLD` and with created cartesian topology. dtFFT will handle both of such cases and create needed internal communicators.

Plan creation subroutines have two common arguments:
- effort_flag - Three options are possible:
  - `DTFFT_ESTIMATE` - Will create plan as fast as possible.
  - `DTFFT_MEASURE` - Only make sense in 3D plan and MPI Communicator without attached cartesian topology. In such cases dtFFT will allocate temporal memory and run transpose routines to find the best grid decomposition.
  - `DTFFT_PATIENT` - most time consuming flag. It does same job as `DTFFT_MEASURE` plus will test different MPI Datatypes. In case of 3d plan, plan creation will take 8 times longer then passing `DTFFT_MEASURE` flag.
- executor_type - this argument specifies which external library should be used to create and execute 1d FFT plans. Default value is `DTFFT_EXECUTOR_NONE` which means that FFTs will not be executed.

### Execution
When executing plan user must provide `transpose_type` argument. Two options are available: `DTFFT_TRANSPOSE_OUT` and `DTFFT_TRANSPOSE_IN`. First one assumes that incoming data is aligned in X direction (fastest) and return data aligned in Z direction.

All plans require additional auxiliary buffer. This buffer can be passed by user to `execute` method.  If user do not provide such buffer during the first call to `execute`, necessary memory will be allocated internally and deallocated when user calls `destroy` method.



For more detaled examples check out tests provided in ```tests/fortran``` folder.
## Building library
To build this library modern (2008+) Fortran compiler is required. This library successfully builds with gfortran-7 and above, ifort-18 and above. Currenly library can only be build using CMake. List of Cmake options can be found below:

| Option   | Possible values | Default value | Description |
| -------- | ------- | -------- | ------- |
| DTFFT_WITH_FFTW | on / off | off | Build dtFFT with FFTW support. When enabled user need to set `FFTWDIR` environmental variable in order to find FFTW3 located in custom directory. Both single and double precision versions of library are required |
| DTFFT_WITH_MKL | on / off | off | Build dtFFT with MKL DFTI support |
| DTFFT_BUILD_TESTS | on / off | off | Build tests |
| DTFFT_ENABLE_COVERAGE | on / off | off | Build coverage of library. Only possible with gfortran |
| DTFFT_BUILD_SHARED | on / off | on | Build shared library |
| DTFFT_USE_MPI | on / off | on | Use Fortran `mpi` module instead of `mpi_f08` |
| DTFFT_BUILD_C_CXX_API | on / off | on | Build C/C++ API |
| DTFFT_ENABLE_PERSISTENT_COMM | on / off | off | In case you are planning to execute plan multiple times then it can be very beneficial to use persistent communications. But user must aware that such communications are created at first call to `execute` or `transpose` subroutines and pointers are saved internally inside MPI. All other plan executions will use those pointers. Take care not to free them. |
| DTFFT_WITH_CALIPER | on / off | off | Enable library profiler via Caliper. Additional parameter is required to find caliper: `caliper_DIR` |
| DTFFT_MEASURE_ITERS | positive integer | 2 | Number of iterations to run in order to find best plan when passing `DTFFT_MEASURE` or `DTFFT_PATIENT` to effort_flag parameter during plan creation |
| DTFFT_FORWARD_X_Y | 1 / 2 | 2 | Default id of transposition plan for X -> Y transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags |
| DTFFT_BACKWARD_X_Y | 1 / 2 | 2 | Default id of transposition plan for Y -> X transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags |
| DTFFT_FORWARD_Y_Z | 1 / 2 | 2 | Default id of transposition plan for Y -> Z transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags |
| DTFFT_BACKWARD_Y_Z | 1 / 2 | 2 | Default id of transposition plan for Z -> Y transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags |
| DTFFT_FORWARD_X_Z | 1 / 2 | 2 | Default id of transposition plan for X -> Z transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags in case Z-slab optimization is used |
| DTFFT_BACKWARD_X_Z | 1 / 2 | 2 | Default id of transposition plan for Z -> Y transpose which will be used if plan created with `DTFFT_ESTIMATE` and `DTFFT_MEASURE` effort_flags in case Z-slab optimization is used|

## Notes for C users
C and C++ interfaces of the library is available. Simply
```c
// C header
#include <dtfft.h>
// C++ header
#include <dtfft.hpp>
```
and tell compiler where it should search for it:
```bash
mpicc ... -I<path_to_dtfft>/include ...
```
Since C arrays are stored in row-major order which is opposite to Fortran column-major when creating the plan, user should pass the dimensions of the array to the planner in reverse order. For example, if your array is a rank three N x M x L matrix in row-major order, you should pass the dimensions of the array as if it were an L x M x N matrix. Also if you are using R2R transform and wish to perform different transform kinds on different dimensions then buffer ```kinds``` should also be reversed.

## Next Steps

- GPU Support
## Contribution

You can help this project by reporting problems, suggestions, localizing it or contributing to the code. Go to issue tracker and check if your problem/suggestion is already reported. If not, create a new issue with a descriptive title and detail your suggestion or steps to reproduce the problem.

## Licensing

The source code is licensed under GPL v3. License is available [here](/LICENSE).
