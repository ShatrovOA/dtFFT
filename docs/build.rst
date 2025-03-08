.. _building_link:

####################
Building the Library
####################

This page outlines the process of building the ``dtFFT`` library using CMake, including compiler requirements, configuration options, and integration instructions for downstream projects. 
The library supports both host and GPU environments, leveraging modern Fortran and optional dependencies like CUDA, FFTW3, MKL, cuFFT, and VkFFT.

Prerequisites
=============

Since ``dtFFT`` is primarely written in Fortran, a modern Fortran compiler (2008 standard or later) is required. The library has been successfully tested with:

- **GNU Fortran (gfortran)**: Version 12 and above
- **Intel Fortran (ifort / ifx)**: Version 18 and above
- **NVHPC Fortran (nvfortran)**: Version 24.5 and above

Currently, ``dtFFT`` can only be built using CMake (version 3.20 or higher recommended). Ensure CMake is installed and available in your PATH before proceeding.

Configuration Options
=====================

The build process is controlled via CMake options, listed below. These options enable or disable features such as GPU support, FFT library integration, and additional utilities. 
Set them using ``-D<OPTION>=<VALUE>`` during CMake configuration.

.. list-table:: CMake Configuration Options
   :widths: 20 20 15 45
   :header-rows: 1

   * - Option
     - Possible Values
     - Default
     - Description
   * - ``DTFFT_WITH_CUDA``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables CUDA support. Requires NVIDIA HPC SDK compilers (C/C++/Fortran) and ``nvcc`` in the PATH for the target CUDA version.
   * - ``DTFFT_CUDA_CC_LIST``
     - Valid CUDA CC list (e.g., ``70;80;90``)
     - ``70;80;90``
     - Specifies CUDA compute capabilities for CUDA Fortran compilation.
   * - ``DTFFT_WITH_FFTW``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables FFTW3 support. Requires the ``FFTWDIR`` environment variable to point to a directory with single- and double-precision FFTW3 libraries.
   * - ``DTFFT_WITH_MKL``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables MKL DFTI support. Requires the ``MKLROOT`` environment variable to be set to the MKL installation path.
   * - ``DTFFT_WITH_CUFFT``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables cuFFT support. Automatically sets ``DTFFT_WITH_CUDA`` to ``ON``.
   * - ``DTFFT_WITH_VKFFT``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables VkFFT support. Requires the ``VKFFT_DIR`` variable to point to ``vkFFT.h`` and automatically sets ``DTFFT_WITH_CUDA`` to ``ON``.
   * - ``DTFFT_BUILD_TESTS``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Builds the library’s test suite.
   * - ``DTFFT_ENABLE_COVERAGE``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables code coverage analysis (gfortran only).
   * - ``DTFFT_BUILD_SHARED``
     - ``ON`` / ``OFF``
     - ``ON``
     - Builds a shared library instead of a static one.
   * - ``DTFFT_USE_MPI``
     - ``ON`` / ``OFF``
     - ``ON``
     - Uses the Fortran ``mpi`` module instead of ``mpi_f08`` for MPI integration. This make possible to pass ``integer`` to ``comm``
       parameter when creating plan, but can produce integer overflow when using CUDA build and big array sizes.
   * - ``DTFFT_BUILD_C_CXX_API``
     - ``ON`` / ``OFF``
     - ``ON``
     - Builds the C and C++ APIs alongside the Fortran API.
   * - ``DTFFT_ENABLE_PERSISTENT_COMM``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables persistent MPI communications for multiple plan executions. Communications are initialized on the first call to :f:func:`execute` or :f:func:`transpose`, with pointers stored internally by MPI. 
       Users must ensure these pointers remain valid and are not freed prematurely.
   * - ``DTFFT_WITH_PROFILER``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables profiling. Uses NVTX3 with CUDA support or Caliper otherwise (requires ``caliper_DIR`` if Caliper is used).
   * - ``DTFFT_WITH_CUSTOM_NCCL``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Uses a custom NCCL build instead of the HPC SDK version. Requires the ``NCCL_ROOT`` environment variable to point to the custom NCCL directory.

Building the Library
====================

1. **Configure the Build**:
   Run CMake to generate build files, specifying the installation prefix and desired options. For example:

.. code-block:: bash

  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install -DDTFFT_WITH_CUDA=ON -DDTFFT_WITH_CUFFT=ON

Replace ``/path/to/install`` with your target installation directory.

2. **Build the Library**:
   Compile the library using:

.. code-block:: bash

  cmake --build build --target install

This compiles and installs ``dtFFT`` to the specified prefix.

Integration with CMake Projects
===============================

Once installed, ``dtFFT`` can be integrated into other CMake projects using ``find_package``. Example configuration:

.. code-block:: cmake

   find_package(dtfft REQUIRED)
   add_executable(my_prog my_prog.c)
   target_link_libraries(my_prog PRIVATE dtfft)

The ``dtfft`` target automatically sets include directories and links required libraries. Specify the installation path when configuring your project:

.. code-block:: bash

   cmake -S . -B build -Ddtfft_DIR=/path/to/install/lib[64]/cmake/dtfft ..

The installation also provides the following CMake variables for conditional compilation:

- ``DTFFT_WITH_CUDA``: Indicates CUDA support
- ``DTFFT_WITH_C_CXX_API``: Indicates C/C++ API availability
- ``DTFFT_WITH_MPI_MODULE``: Indicates use of the ``mpi`` module