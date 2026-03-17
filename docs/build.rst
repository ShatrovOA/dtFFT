.. _building_link:

####################
Building the Library
####################

This page outlines the process of building the ``dtFFT`` library using CMake, including compiler requirements, configuration options, and integration instructions for downstream projects. 
The library supports both host and GPU environments, leveraging modern Fortran and optional dependencies like CUDA, FFTW3, MKL, cuFFT, and VkFFT.

Prerequisites
=============

Since ``dtFFT`` is primarily written in Fortran, a modern Fortran compiler (2008 standard or later) is required. The library has been successfully tested with:

- **GNU Fortran (gfortran)**: Version 12 and above
- **Intel Fortran (ifort / ifx)**: Version 18 and above
- **NVHPC Fortran (nvfortran)**: Version 24.5 and above

Currently, ``dtFFT`` can only be built using CMake (version 3.25 or higher recommended). Ensure CMake is installed and available in your PATH before proceeding.

**Requirements**:

- **CMake**: Version 3.25 or higher
- **Modern Fortran compiler**: 2008 standard or later
- **MPI**: Message Passing Interface (MPI) implementation
- **Caliper** (optional): For performance profiling and analysis

**For CUDA support**:

- **CUDA-aware MPI**: Required for GPU acceleration
- **NCCL** (optional): NVIDIA Collective Communications Library (automatically linked if ``nvfortran`` is used)
- **nvfortran** (optional): NVHPC Fortran compiler (enables additional features like NCCL and cuFFTMp)
- **NVTX3** (optional): NVIDIA Tools Extension for profiling and debugging

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
     - Enables CUDA support. Requires ``nvcc`` in the PATH for the target CUDA version.
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
       parameter when creating plan instead of ``type(MPI_Comm)``, but can produce integer overflow when using CUDA build and big array sizes.
   * - ``DTFFT_BUILD_C_CXX_API``
     - ``ON`` / ``OFF``
     - ``ON``
     - Builds the C and C++ APIs alongside the Fortran API.
   * - ``DTFFT_ENABLE_PERSISTENT_COMM``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables persistent MPI communications for multiple plan executions.
       Communications are initialized on the first call to :f:func:`execute` or :f:func:`transpose`, with pointers stored internally by MPI. 
       Users must ensure these pointers remain valid and are not freed prematurely.
   * - ``DTFFT_WITH_PROFILER``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables profiling. Uses NVTX3 with CUDA support or Caliper otherwise (requires ``caliper_DIR`` if Caliper is used).
   * - ``DTFFT_WITH_NCCL``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Requires the ``NCCL_ROOT`` environment variable to point to the custom NCCL directory.
   * - ``DTFFT_WITHOUT_NCCL``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Disables use of NCCL shipped with HPC-SDK. NCCL Backends will be unavailable
   * - ``DTFFT_WITHOUT_NVSHMEM``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Disables use of NVSHMEM-based backends shipped with HPC-SDK.
   * - ``DTFFT_ENABLE_DEVICE_CHECKS``
     - ``ON`` / ``OFF``
     - ``ON``
     - Enable error checking for all GPU libraries calls. Can be turned off for best performance.
   * - ``DTFFT_WITH_RMA``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enable MPI RMA backends (currently in beta). It has been noticed that call to ``MPI_Win_create`` fails with OpenMPI 4.0.5 with UCX enabled.
   * - ``DTFFT_ENABLE_INPUT_CHECK``
     - ``ON`` / ``OFF``
     - ``ON``
     - Enables input parameter checks for plan execution functions. Should be turned off by advanced users to best performance.
   * - ``DTFFT_WITH_ZFP``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables ZFP support for compressed transposes. Requires the ``zfp_DIR`` variable to point to the ZFP installation path.
   * - ``DTFFT_WITH_MOCK_ENABLED``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables mock version of CUDA support for testing and development without a CUDA-capable GPU. This option is intended for development and testing purposes only and should not be used in production environments.
   * - ``DTFFT_WITH_OPENMP``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables OpenMP support.
   * - ``DTFFT_WITH_FFTW_THREADS``
     - ``ON`` / ``OFF``
     - ``OFF``
     - Enables threads library of FFTW instead of OpenMP. This option is only applicable if both ``DTFFT_WITH_FFTW`` and ``DTFFT_WITH_OPENMP`` are enabled.


Building the Library
====================

1. **Configure the Build**:
   Run CMake to generate build files, specifying the installation prefix and desired options. For example:

.. code-block:: bash

  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install -DDTFFT_WITH_CUDA=ON -DDTFFT_WITH_CUFFT=ON

Replace ``/path/to/install`` with your target installation directory.

.. note:: CUDA support in ``dtFFT`` does not replace the host version but extends it. For more details, refer to the guide 
  :ref:`here<config_link>` and the environment variable :ref:`DTFFT_PLATFORM<dtfft_platform_env>`.

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
- ``DTFFT_WITH_NCCL``: Indicates NCCL support
- ``DTFFT_WITH_NVSHMEM``: Indicates NVSHMEM support
- ``DTFFT_WITH_OPENMP``: Indicates OpenMP support

.. _python_install_link:

Python Package
==============

``dtFFT`` exposes a large number of CMake configuration options (see :ref:`Configuration Options<config_link>` above) that cannot all be reflected in pre-built packages. The wheels published on PyPI cover only a limited subset of the available functionality.
For most real-world use cases — especially those requiring a specific MPI implementation, FFTW3, cuFFT, or other optional backends — it is strongly recommended to build the package from source so that all required options can be enabled.

Installing from PyPI
--------------------

Pre-built wheels are available on PyPI for Linux (x86_64 and aarch64) and macOS (Apple Silicon).
Choose the variant that matches your environment:

.. list-table:: PyPI package variants
   :widths: 22 12 10 22 18 16
   :header-rows: 1

   * - Package
     - FFT backend
     - MPI
     - OS
     - Platform
     - Extra system requirements
   * - ``dtfft-openmpi``
     - none (transpose only)
     - OpenMPI
     - Linux (x86_64, aarch64), macOS (arm64)
     - CPU
     - —
   * - ``dtfft-mpich``
     - none (transpose only)
     - MPICH
     - Linux (x86_64, aarch64), macOS (arm64)
     - CPU
     - —
   * - ``dtfft-fftw-openmpi``
     - FFTW3
     - OpenMPI
     - Linux (x86_64, aarch64), macOS (arm64)
     - CPU
     - system ``libfftw3``
   * - ``dtfft-fftw-mpich``
     - FFTW3
     - MPICH
     - Linux (x86_64, aarch64)
     - CPU
     - system ``libfftw3``
   * - ``dtfft-cuda12x-openmpi``
     - cuFFT
     - OpenMPI
     - Linux (x86_64, aarch64)
     - CPU + NVIDIA GPU (CUDA 12)
     - CUDA 12 toolkit, ``cupy-cuda12x``
   * - ``dtfft-cuda12x-mpich``
     - cuFFT
     - MPICH
     - Linux (x86_64, aarch64)
     - CPU + NVIDIA GPU (CUDA 12)
     - CUDA 12 toolkit, ``cupy-cuda12x``

All packages share the same importable namespace: ``import dtfft``.
To install, simply run:

.. code-block:: bash

   # Transpose-only (no FFT backend) — OpenMPI
   pip install dtfft-openmpi
   # Transpose-only (no FFT backend) — MPICH
   pip install dtfft-mpich

   # With FFTW3 — OpenMPI
   pip install dtfft-fftw-openmpi

   # CUDA 12 — OpenMPI
   pip install dtfft-cuda12x-openmpi

Installing from Source
----------------------

A source distribution (sdist) of ``dtfft`` is published to PyPI alongside the pre-built wheels as part of the CI/CD pipeline.
This allows installation with fully custom CMake options on any supported platform — without cloning the repository.
Build-time Python dependencies (``scikit-build``, ``cmake``, ``ninja``, ``pybind11``) are declared in ``pyproject.toml`` and are installed automatically by pip.

Building from source requires:

- A Fortran compiler (GCC ≥ 10, Intel, or NVHPC)
- CMake ≥ 3.25
- An MPI implementation with development headers
- Python ≥ 3.9

.. warning::

   The Python environment must contain:

   - ``mpi4py`` **compiled against the exact MPI implementation** that dtFFT will be linked with.
     Installing a pre-built binary via ``pip install mpi4py`` may link against a different MPI and cause runtime errors.
     Always build it from source: ``pip install --no-binary mpi4py mpi4py``.
   - ``cupy`` matching your **CUDA toolkit version** (e.g. ``cupy-cuda12x`` for CUDA 12) when building with CUDA support.
     Using a mismatched CuPy version will lead to import failures or silent data corruption.

1. **Install runtime Python dependencies**:

   .. code-block:: bash

      # mpi4py must be compiled against the MPI you have installed
      pip install --no-binary mpi4py mpi4py numpy
      # for CUDA builds, install cupy matching your CUDA version, e.g.:
      # pip install cupy-cuda12x

2. **Build and install from source**:

   Use the ``CMAKE_ARGS`` environment variable to enable optional backends.
   pip will automatically fetch the source distribution from PyPI and build it locally.

   **Transpose-only (no FFT backend)**:

   .. code-block:: bash

      pip install dtfft

   **With FFTW3**:

   .. code-block:: bash

      CMAKE_ARGS="-DDTFFT_WITH_FFTW=ON" pip install dtfft

   **With cuFFT (CUDA)**:

   .. code-block:: bash

      CMAKE_ARGS="-DDTFFT_WITH_CUDA=ON -DDTFFT_WITH_CUFFT=ON" pip install dtfft

   .. note::

      If FFTW3 is not in a standard location, pass its path via ``CMAKE_ARGS`` as well:

      .. code-block:: bash

         CMAKE_ARGS="-DDTFFT_WITH_FFTW=ON -DFFTWDIR=/path/to/fftw" pip install dtfft

3. **Verify the installation**:

   .. code-block:: python

      import dtfft
      print(dtfft.is_fftw_enabled())   # True if built with FFTW3
      print(dtfft.is_cufft_enabled())  # True if built with cuFFT