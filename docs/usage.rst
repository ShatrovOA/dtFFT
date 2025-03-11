.. _usage_link:

###########
Usage Guide
###########

This guide provides a comprehensive overview of using the ``dtFFT`` library to perform parallel data transpositions and optionally
Fast Fourier Transforms (FFTs) across host and GPU environments.
Designed for high-performance computing, ``dtFFT`` simplifies the process of decomposing multidimensional data, managing memory,
and executing transformations by integrating with external FFT libraries or operating in transpose-only mode.

Whether targeting CPU clusters with MPI or GPU-accelerated systems with CUDA, this library offers flexible configuration options to
optimize performance for specific use cases. The following sections detail key aspects of working with ``dtFFT``, from plan creation to
execution and resource management, with practical examples in Fortran, C, and C++.

Error Handling and Macros
=========================

Almost all ``dtFFT`` functions return error codes to indicate whether execution was successful. These codes help users identify and handle issues during plan creation, memory allocation, execution, and finalization. The error handling mechanism differs slightly across language APIs:

- **Fortran API**: Functions include an optional ``error_code`` parameter (type ``integer(int32)``), always positioned as the last argument.
  If omitted, errors must be checked through other means, such as program termination or runtime assertions.
- **C API**: Functions return a value of type :c:type:`dtfft_error_code_t`, allowing direct inspection of the result.
- **C++ API**: Functions return :cpp:type:`dtfft::ErrorCode`, typically used with exception handling or explicit checks.

To simplify error checking, ``dtFFT`` provides predefined macros that wrap function calls and handle error codes automatically:

- **Fortran**: The ``DTFFT_CHECK`` macro, defined in ``dtfft.f03``, checks the ``error_code`` and halts execution with an informative message
  if an error occurs. Include this header with ``#include "dtfft.f03"`` to use it.
- **C**: The ``DTFFT_CALL`` macro wraps function calls, checks the returned :c:type:`dtfft_error_code_t`,
  and triggers an appropriate response (printing an error message and exiting) if the call fails.
- **C++**: The ``DTFFT_CXX_CALL`` macro similarly wraps calls, throws C++ exception and displays an error message.

Below is an example demonstrating error handling with these macros:

.. tabs::

  .. code-tab:: fortran

    #include "dtfft.f03"

    ...

    call plan%execute(a, b, DTFFT_EXECUTE_FORWARD, error_code=error_code)
    DTFFT_CHECK(error_code)  ! Halts if error_code != DTFFT_SUCCESS

    ...

  .. code-tab:: c

    #include <dtfft.h>

    ...

    DTFFT_CALL( dtfft_execute(plan, a, b, DTFFT_EXECUTE_FORWARD, NULL) )

    ...

  .. code-tab:: c++

    #include <dtfft.hpp>

    ...

    DTFFT_CXX_CALL( plan.execute(a, b, dtfft::ExecuteType::FORWARD, nullptr) );


Error codes are defined in the API sections (e.g., :f:var:`DTFFT_SUCCESS`, :f:var:`DTFFT_ERROR_INVALID_TRANSPOSE_TYPE`). Refer to the Fortran, C, and C++ API documentation for a complete list and detailed descriptions.

Plan Creation
=============

Creating a plan in ``dtFFT`` involves specifying the dimensions of the data, along with optional parameters such as the MPI communicator, precision, and FFT executor type. The library supports three plan types:

- Real-to-Real (R2R)
- Complex-to-Complex (C2C)
- Real-to-Complex (R2C)

.. note:: The Real-to-Complex plan is not available in the API if the library was not compiled with any FFT support.

Each is tailored to specific transformation needs.
Plans are created using the ``create`` method or corresponding constructor, as detailed in the Fortran, C, and C++ API sections.
For every plan type, an MPI communicator must be specified to define the process distribution (see `Grid Decomposition`_ below).
The optimization level applied during plan creation can be controlled via the effort parameter (see `Selecting plan effort`_ below). Additional parameters include:

- **Precision**: Controlled by :f:type:`dtfft_precision_t` with the following options:

  - ``DTFFT_SINGLE``: Single precision
  - ``DTFFT_DOUBLE``: Double precision

- **FFT Executor**: Specified via :f:type:`dtfft_executor_t` to determine the external FFT library or transpose-only mode, with the following options:

  - ``DTFFT_EXECUTOR_NONE``: ``Transpose-only`` (no FFT)
  - ``DTFFT_EXECUTOR_FFTW3``: FFTW3 (host only, available if compiled with FFTW3 support)
  - ``DTFFT_EXECUTOR_MKL``: MKL DFTI (host only, available if compiled with MKL support)
  - ``DTFFT_EXECUTOR_CUFFT``: cuFFT (GPU only, available if compiled with CUDA support)
  - ``DTFFT_EXECUTOR_VKFFT``: VkFFT (GPU only, available if compiled with VkFFT support)

Additional optional settings can be specified before plan creation using :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_ below),
allowing users to customize behavior such as Z-slab optimization or GPU backend selection.

The following example creates a 3D C2C double-precision transpose-only plan:

.. tabs::

  .. code-tab:: fortran

    #include "dtfft.f03"
    ! dtfft.f03 contains macro DTFFT_CHECK
    use iso_fortran_env
    use dtfft
    use mpi ! or use mpi_f08

    type(dtfft_plan_c2c_t) :: plan
    integer(int32) :: dims(3)
    integer(int32) :: error_code
    type(dtfft_effort_t) :: effort = DTFFT_PATIENT
    type(dtfft_precision_t) :: precision = DTFFT_DOUBLE
    type(dtfft_executor_t) :: executor = DTFFT_EXECUTOR_NONE

    call MPI_Init()

    ! Set dimensions
    dims = [32, 32, 32]

    ! Creating plan with create method
    call plan%create(dims, MPI_COMM_WORLD, precision, effort, executor, error_code)
    DTFFT_CHECK(error_code)

    ! OR use plan constructor method
    ! plan = dtfft_plan_c2c_t(dims, MPI_COMM_WORLD, precision, effort, executor, error_code)
    ! DTFFT_CHECK(error_code)

    ! OR use abstract plan to create C2C plan
    ! class(dtfft_plan_t), allocatable :: plan
    ! allocate(dtfft_plan_c2c_t :: plan)
    ! select type (plan)
    ! class is (dtfft_plan_c2c_t)
    !   call plan%create(dims, MPI_COMM_WORLD, precision, effort, executor, error_code)
    ! end select
    ! DTFFT_CHECK(error_code)

  .. code-tab:: c

    #include <dtfft.h>
    #include <mpi.h>

    int main(int argc, char *argv[]) {
      dtfft_plan_t plan;
      int32_t dims[3] = {32, 32, 32};

      MPI_Init(&argc, &argv);

      // Creating plan
      DTFFT_CALL( dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, DTFFT_EXECUTOR_NONE, &plan) );

      return 0;
    }

  .. code-tab:: c++

    #include <dtfft.hpp>
    #include <mpi.h>
    #include <vector>

    int main(int argc, char *argv[]) {
      MPI_Init(&argc, &argv);

      const std::vector<int32_t> dims = {32, 32, 32};
      dtfft::Precision precision = dtfft::Precision::DOUBLE;
      dtfft::Effort effort = dtfft::Effort::PATIENT;
      dtfft::Executor executor = dtfft::Executor::NONE;

      // Creating plan with constructor
      dtfft::PlanC2C plan(dims, MPI_COMM_WORLD, precision, effort, executor);

      // OR use generic interface
      // dtfft::PlanC2C plan(dims.size(), dims.data(), MPI_COMM_WORLD, precision, effort, executor);

      // OR use Plan pointer
      // dtfft::Plan *plan = new dtfft::PlanC2C(dims, MPI_COMM_WORLD, precision, effort, executor);

      return 0;
    }

Grid Decomposition
------------------

``dtFFT`` decomposes multidimensional data into a grid to distribute it across MPI processes for parallel execution.
The decomposition strategy depends on the global dimensions ``NX × NY × NZ`` (in Fortran order), the number of MPI
processes ``P``, and the provided communicator.

Default Behavior
________________

When the communicator passed during plan creation is ``MPI_COMM_WORLD`` with ``P`` processes, ``dtFFT`` attempts the following steps in order:
  - If ``P <= NZ`` (and ``NZ / P >= 16`` for the GPU version), split the grid as ``NX × NY × NZ / P``.
    This distributes the Z-dimension across ``P`` processes. Division need not be even, and the local size per process may vary slightly.
  - If the Z-split fails (e.g., ``P > NZ`` or ``NZ / P < 16`` on GPU), attempt ``NX × NY / P × NZ``.
    This distributes the Y-dimension across ``P`` processes, provided ``NX <= P`` to ensure compatibility with future transpositions (e.g., X-to-Y).
  - If both attempts fail, ``dtFFT`` constructs a 3D communicator by fixing the X-dimension split to 1 and using
    ``MPI_Dims_create(P, 2, dims)`` to balance the remaining ``P`` processes across Y and Z, resulting in ``NX × NY / P1 × NZ / P2``
    (where ``P1 × P2 = P``).
  - If this 3D decomposition is not viable (e.g., ``NY < P1`` or ``NZ < P2``), ``dtFFT`` falls back to distributing the
    Z-dimension across all ``P`` processes as ``NX × NY × NZ / P``, assigning each process a portion of ``NZ`` (rounded down),
    with any remainder distributed to earlier ranks. This ensures a valid decomposition even when ``P > NZ``.

User-Controlled Decomposition
_____________________________

Users can specify a custom MPI communicator with grid topology attached. Its grid dimensions must be defined in Fortran order (X, Y, Z):
  - **1D Communicator**: A one-dimensional communicator with ``P`` processes splits the grid as ``NX × NY × NZ / P``,
    distributing the Z-dimension across ``P`` processes.
  - **2D Communicator**: A two-dimensional communicator with topology ``P1 × P2`` (where ``P1 * P2 = P``) decomposes the grid as
    ``NX × NY / P1 × NZ / P2``, splitting Y by ``P1`` and Z by ``P2`` while keeping X indivisible.
  - **3D Communicator**: A three-dimensional communicator with topology ``P0 × P1 × P2`` (where ``P0 * P1 * P2 = P``) is supported,
    but ``P0`` (X-dimension split) must be 1 to preserve the fastest-varying dimension. This results in ``NX × NY / P1 × NZ / P2``.
    Violating this condition triggers :f:var:`DTFFT_ERROR_INVALID_COMM_FAST_DIM`.

Z-Slab Optimization
___________________

When the grid is decomposed as ``NX × NY × NZ / P`` (e.g., via a 1D communicator or the first default step), the Z-slab optimization
becomes available. If enabled, it reduces the number of network data transfers by employing a two-dimensional FFT algorithm during
calls to the :f:func:`execute` method. This also enables the use of ``DTFFT_TRANSPOSE_X_TO_Z`` and ``DTFFT_TRANSPOSE_Z_TO_X`` in
the :f:func:`transpose` method, while all other transpose types (e.g., ``DTFFT_TRANSPOSE_X_TO_Y``, ``DTFFT_TRANSPOSE_Y_TO_Z``)
remain available to the user.

This optimization can be disabled by passing the appropriate parameter in :f:type:`dtfft_config_t` (see configuration details below),
but it cannot be forcibly enabled by passing an ``MPI_COMM_WORLD`` communicator if conditions for its applicability are not met.

---------

The resulting local data extents for each process can be retrieved using :f:func:`get_local_sizes` or :f:func:`get_pencil`,
providing the necessary information for memory allocation and interfacing with external FFT libraries. The starting indices
("starts") of each process's local data portion are determined based on its coordinates within the MPI grid topology.


Selecting plan effort
---------------------

The ``effort`` parameter in ``dtFFT`` determines the level of optimization applied during plan creation,
influencing how data transposition is configured. On the host, ``dtFFT`` leverages custom MPI datatypes to perform transpositions,
tailored to the grid decomposition and data layout. On the GPU, transposition is handled by nvRTC-compiled kernels, optimized at runtime
for specific data sizes and types, with data exchange between GPUs facilitated by various backend options (e.g., NCCL, MPI P2P).
The supported effort levels defined by :f:type:`dtfft_effort_t` control the extent of this optimization as follows:

DTFFT_ESTIMATE
______________

This minimal-effort option prioritizes fast plan creation.

On the host, ``dtFFT`` selects a default grid decomposition (see `Grid Decomposition`_ above) and constructs MPI datatypes based
on environment variables such as ``DTFFT_DTYPE_X_Y`` and ``DTFFT_DTYPE_Y_Z`` (see :ref:`MPI Datatype Selection variables <datatype_selection>`),
which define the default send and receive strategies.

On the GPU, it uses a pre-selected backend specified via :f:type:`dtfft_config_t` (see configuration details below), compiling an nvRTC
kernel tailored to the chosen backend.

DTFFT_MEASURE
_____________

With this moderate-effort setting, ``dtFFT`` explores multiple grid decomposition strategies to reduce communication overhead
during transposition, cycling through possible grid layouts to find an efficient configuration. On the host, it uses the same MPI datatypes
as defined by environment variables in ``DTFFT_ESTIMATE``. On the GPU, it employs the same backend as specified in the configuration for ``DTFFT_ESTIMATE``.

If a Cartesian communicator is provided, it reverts to ``DTFFT_ESTIMATE`` behavior, relying on the user-specified topology.

DTFFT_PATIENT
_____________

This maximum-effort option extends ``DTFFT_MEASURE`` by exhaustively optimizing transposition strategies. On the host, it cycles
through various custom MPI datatype combinations (e.g., contiguous send with sparse receive, sparse send with contiguous receive) to
minimize network latency and maximize throughput. On the GPU, it cycles through available GPU backends (e.g., NCCL, MPI P2P) to select
the fastest available backend.

---------

The choice of ``effort`` impacts both plan creation time and runtime performance.
Higher effort levels (``DTFFT_MEASURE`` and ``DTFFT_PATIENT``) increase setup time but can enhance transposition efficiency,
especially for large datasets or complex grids.

If a user already knows the optimal grid decomposition, MPI datatypes, or GPU backend from a previous computation,
these can be pre-specified before plan creation: the grid via a custom ``MPI_Comm`` communicator, MPI datatypes through environment
variables (e.g., ``DTFFT_DTYPE_X_Y``), and the GPU backend through :f:type:`dtfft_config_t`.


Setting Additional Configurations
---------------------------------

The :f:type:`dtfft_config_t` type allows users to set additional configuration parameters for ``dtFFT`` before plan creation,
tailoring its behavior to specific needs. These settings are optional and can be applied using the constructor ``dtfft_config_t()``
or the :f:func:`dtfft_create_config` function, followed by a call to :f:func:`dtfft_set_config`.

Configurations must be set prior to creating a plan to take effect. The available parameters are:

- **Z-Slab Optimization** (``enable_z_slab``)

  A logical flag determining whether ``dtFFT`` uses Z-slab optimization (see `Grid Decomposition`_).
  When enabled (default: ``.true.``), it reduces network data transfers in plans decomposed as ``NX × NY × NZ / P`` by employing
  a two-dimensional FFT algorithm. Disabling it (``.false.``) may resolve :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` or
  improve performance if the underlying 2D FFT implementation is suboptimal.
  In most cases, Z-slab is faster due to fewer transpositions.

.. _dtfft_platform_conf:

- **Execution platform** (``platform``)

  A :f:type:`dtfft_platform_t` value specifying the platform for executing ``dtFFT`` plans.
  By default, set to :f:var:`DTFFT_PLATFORM_HOST`, meaning execution occurs on the host (CPU).
  Users can set it to :f:var:`DTFFT_PLATFORM_CUDA` for GPU execution, provided the build supports CUDA
  (``DTFFT_WITH_CUDA`` defined).

  Available only in CUDA-enabled builds.

- **CUDA Stream** (``stream``)

  A :f:type:`dtfft_stream_t` value specifying the main CUDA stream for GPU operations.
  By default, ``dtFFT`` manages its own stream, retrievable via :f:func:`get_stream`. Users can set a custom stream,
  taking responsibility for its destruction after the plan is destroyed with :f:func:`destroy`.

  Available only in CUDA-enabled builds.

- **GPU Backend** (``gpu_backend``)

  A :f:type:`dtfft_gpu_backend_t` value selecting the GPU backend for transposition when ``effort`` is :f:var:`DTFFT_ESTIMATE` or
  :f:var:`DTFFT_MEASURE` (see `Selecting plan effort`_). The default is :f:var:`DTFFT_GPU_BACKEND_NCCL` if NCCL is available
  in the library build; otherwise, :f:var:`DTFFT_GPU_BACKEND_MPI_P2P`. Supported options include:

  - :f:var:`DTFFT_GPU_BACKEND_MPI_DATATYPE`: Backend using MPI datatypes.
  - :f:var:`DTFFT_GPU_BACKEND_MPI_P2P`: MPI peer-to-peer backend.
  - :f:var:`DTFFT_GPU_BACKEND_MPI_A2A`: MPI backend using ``MPI_Alltoallv``.
  - :f:var:`DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED`: Pipelined MPI peer-to-peer backend.
  - :f:var:`DTFFT_GPU_BACKEND_NCCL`: NCCL backend.
  - :f:var:`DTFFT_GPU_BACKEND_NCCL_PIPELINED`: Pipelined NCCL backend.
  - :f:var:`DTFFT_GPU_BACKEND_CUFFTMP`: cuFFTMp backend.

  Available only in CUDA-enabled builds.

- **MPI Backends** (``enable_mpi_backends``)

  A logical flag controlling whether MPI-based GPU backends (e.g., MPI P2P) are tested during autotuning with ``DTFFT_PATIENT``
  effort (default: ``.false.``). Disabled by default due to an OpenMPI bug (https://github.com/open-mpi/ompi/issues/12849)
  causing GPU memory leaks during autotuning (e.g., 8 GB leak for a 1024×1024×512 C2C plan with Z-slab on a single GPU,
  or 24 GB per GPU on four GPUs without Z-slab).

  Workarounds include disabling MPI backends or using ``--mca btl_smcuda_use_cuda_ipc 0`` with ``mpiexec``,
  though the latter reduces performance.

  Available only in CUDA-enabled builds.

- **Pipelined Backends** (``enable_pipelined_backends``)

  A logical flag enabling pipelined GPU backends (e.g., overlapping data copy and unpack) during ``DTFFT_PATIENT``
  autotuning (default: ``.true.``). These require an additional internal buffer managed by ``dtFFT``.

  Available only in CUDA-enabled builds.

- **NCCL Backends** (``enable_nccl_backends``)

  A logical flag enabling NCCL backends during ``DTFFT_PATIENT`` autotuning (default: ``.true.``).

  Available only in CUDA-enabled builds.

- **NVSHMEM Backends** (``enable_nvshmem_backends``)

  A logical flag enabling ``NVSHMEM``-enabled backends support during ``DTFFT_PATIENT`` autotuning (default: ``.true.``).

  Available only in CUDA-enabled builds.

These settings allow fine-tuning of transposition strategies and GPU behavior.
For example, disabling ``enable_mpi_backends`` mitigates memory leaks, while setting a custom ``stream`` integrates ``dtFFT``
with existing CUDA workflows. Refer to the Fortran, C and C++ API pages for detailed parameter specifications.

.. note:: Almost all values can be overridden by setting the appropriate environment variable, which takes precedence if set.
  Refer to :ref:`Environment Variables<environ_link>` section.

Following example creates config object, disables Z-slab, enables MPI Backends and sets custom stream:

.. tabs::

  .. code-tab:: fortran

    use cudafor
    use dtfft

    integer(cuda_stream_kind) :: my_stream
    type(dtfft_config_t) :: config
    integer :: ierr

    ! Create config with default values
    config = dtfft_config_t()

    ! Disable Z-slab optimization
    config%enable_z_slab = .false.

    ! Enable MPI backends for autotuning
    config%enable_mpi_backends = .true.

    ! Create and set custom CUDA stream
    ierr = cudaStreamCreate(my_stream)
    config%stream = dtfft_stream_t(my_stream)

    ! Apply configuration
    call dtfft_set_config(config)

    ! Now we can create a plan

  .. code-tab:: c

    #include <cuda_runtime.h>
    #include <dtfft.h>

    cudaStream_t my_stream;
    dtfft_config_t config;

    // Create config with default values
    dtfft_create_config(&config);

    // Disable Z-slab optimization
    config.enable_z_slab = 0;

    // Enable MPI backends for autotuning
    config.enable_mpi_backends = 1;

    // Create and set custom CUDA stream
    cudaStreamCreate(&my_stream);
    config.stream = (dtfft_stream_t)my_stream;

    // Apply configuration
    dtfft_set_config(config);

    // Now we can create a plan

  .. code-tab:: c++

    #include <cuda_runtime.h>
    #include <dtfft.hpp>

    cudaStream_t my_stream;
    dtfft::Config config;  // Automatically fills with default values

    // Disable Z-slab optimization
    config.set_enable_z_slab(false);

    // Enable MPI backends for autotuning
    config.set_enable_mpi_backends(true);

    // Create and set custom CUDA stream
    cudaStreamCreate(&my_stream);
    config.set_stream((dtfft_stream_t)my_stream);

    // Apply configuration
    dtfft::set_config(config);

    // Now we can create a plan


Memory Allocation
=================

After a plan is created, users may need to determine the memory required to execute it.

The plan method :f:func:`get_local_sizes` retrieves the number of elements in "real" and "Fourier" spaces and the
minimum number of elements that must be allocated:

- **in_starts**: Start indices of the local data portion in real space (0-based)
- **in_counts**: Number of elements in the local data portion in real space
- **out_starts**: Start indices of the local data portion in Fourier space (0-based)
- **out_counts**: Number of elements in the local data portion in Fourier space
- **alloc_size**: Minimum number of elements needed for ``in``, ``out``, or ``aux`` buffers

Arrays ``in_starts``, ``in_counts``, ``out_starts``, and ``out_counts`` must have at least as many elements as the plan's dimensions.

The minimum number of bytes required for each buffer is ``alloc_size * element_size``.
The ``element_size`` can be obtained by :f:func:`get_element_size` which returns:

- **C2C**: ``2 * sizeof(double)`` (double precision) or ``2 * sizeof(float)`` (single precision)
- **R2R and R2C**: ``sizeof(double)`` (double precision) or ``sizeof(float)`` (single precision)

.. tabs::

  .. code-tab:: fortran

    integer(int64) :: alloc_size, element_size

    ! Get number of elements
    call plan%get_local_sizes(alloc_size=alloc_size)

    ! OR use convenient wrapper
    ! alloc_size = plan%get_alloc_size()

    ! Optionally get element size in bytes
    element_size = plan%get_element_size()

  .. code-tab:: c

    size_t alloc_size;

    // Get number of elements
    dtfft_get_local_sizes(plan, NULL, NULL, NULL, NULL, &alloc_size);

    // OR use convenient wrapper
    // dtfft_get_alloc_size(plan, &alloc_size);

    // Optionally get element size in bytes
    size_t element_size;
    dtfft_get_element_size(plan, &element_size);

  .. code-tab:: c++

    size_t alloc_size;

    // Get number of elements
    DTFFT_CXX_CALL( plan.get_local_sizes(nullptr, nullptr, nullptr, nullptr, &alloc_size) );

    // OR use wrapper
    // DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) );

    // Optionally get element size in bytes
    size_t element_size;
    DTFFT_CXX_CALL( plan.get_element_size(&element_size) );

For 3D plans, :f:func:`get_local_sizes` does not detail the intermediate Y-direction layout.
This information, useful for transpose-only plans or when using unsupported FFT libraries, can be retrieved via the ``pencil``
interface (see `Pencil Decomposition`_ below). Pencil IDs start from 1 in both C and Fortran.

The ``dtFFT`` library provides functions to allocate and free memory tailored to the plan:

- :f:func:`mem_alloc`: Allocates memory.
- :f:func:`mem_free`: Frees memory allocated by :f:func:`mem_alloc`.

Host Version
------------

Allocates memory based on the FFT library: ``fftw_malloc`` for FFTW3, ``mkl_malloc`` for MKL DFTI, or
C11 ``aligned_alloc`` (16-byte alignment) for transpose-only plans.

GPU Version
-----------

Allocates memory based on the :f:type:`dtfft_gpu_backend_t`. Uses ``ncclMemAlloc`` for NCCL (if available), ``nvshmem_malloc`` for NVSHMEM-based
backends or ``cudaMalloc`` otherwise. Future versions may support HIP-based allocations.

If NCCL is used and supports buffer registration via ``ncclCommRegister``, and the environment variable 
:ref:`DTFFT_NCCL_BUFFER_REGISTER<dtfft_nccl_buffer_register_env>` is set to ``1``, the allocated buffer will also be registered. 
This registration optimizes communication performance by reducing the overhead of memory operations, 
which is particularly beneficial for workloads with repeated communication patterns.

.. tabs::

  .. code-tab:: fortran

    use iso_c_binding
    integer(int64) :: alloc_bytes
    type(c_ptr) :: a_ptr, b_ptr, aux_ptr

    ! Host version
    complex(real64), pointer :: a(:), b(:), aux(:)
    ! CUDA Fortran version
    complex(real64), device, contiguous, pointer :: a(:), b(:), aux(:)

    alloc_bytes = alloc_size * element_size

    ! Allocates memory
    a_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)
    b_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)
    aux_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)

    ! Convert pointer to Fortran array
    call c_f_pointer(a_ptr, a, [alloc_size])
    call c_f_pointer(b_ptr, b, [alloc_size])
    call c_f_pointer(aux_ptr, aux, [alloc_size])

  .. code-tab:: c

    size_t alloc_bytes = alloc_size * element_size;
    double *a, *b, *aux;

    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&a) );
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&b) );
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&aux) );

  .. code-tab:: c++

    #include <complex>

    size_t alloc_bytes = alloc_size * element_size;
    std::complex<double> *a, *b, *aux;

    DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&a) );
    DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&b) );
    DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&aux) );

.. note:: Memory allocated with :f:func:`mem_alloc` must be deallocated with :f:func:`mem_free` **before** the plan is destroyed to avoid memory leaks.

Pencil Decomposition
--------------------

For detailed layout information in 3D plans (e.g., intermediate states like Y-direction distribution), use
the :f:func:`get_pencil` method. This returns a ``dtfft_pencil_t`` structure containing:

- **dim**: Aligned dimension ID (1 for X, 2 for Y, 3 for Z)
- **ndims**: Number of dimensions in the pencil (2 or 3)
- **starts**: Local start indices in natural Fortran order (0-based, array of 3 elements)
- **counts**: Local element counts in natural Fortran order (array of 3 elements, only first ``ndims`` elements are defined)

.. tabs::

  .. code-tab:: fortran

    integer(int8) :: i
    type(dtfft_pencil_t) :: pencils(3)

    do i = 1, 3
      ! Get pencil for dimension i
      call plan%get_pencil(i, pencils(i), error_code)
      DTFFT_CHECK(error_code)
      ! Access pencil properties, e.g., pencils(i)%dim, pencils(i)%starts
    end do

  .. code-tab:: c

    dtfft_pencil_t pencils[3];

    for (int8_t i = 0; i < 3; i++) {
      DTFFT_CALL( dtfft_get_pencil(plan, i + 1, &pencils[i]) );
      // Access pencil properties, e.g., pencils[i].dim, pencils[i].starts
    }

  .. code-tab:: c++

    std::vector<dtfft::Pencil> pencils;

    for (int8_t i = 0; i < 3; i++) {
      dtfft::Pencil pencil;
      DTFFT_CXX_CALL( plan.get_pencil(i + 1, pencil) );
      pencils.push_back(pencil);
      // Access pencil properties, e.g., pencils[i].get_dim(), pencils[i].get_starts()
    }

In C++, the ``dtfft::Pencil`` class provides additional methods:

- ``get_ndims()``: Returns the number of dimensions
- ``get_dim()``: Returns the aligned dimension ID
- ``get_starts()``: Returns the start indices as a ``std::vector<int32_t>``
- ``get_counts()``: Returns the element counts as a ``std::vector<int32_t>``
- ``get_size()``: Returns the total number of elements (product of counts)
- ``c_struct()``: Returns the underlying C structure (``dtfft_pencil_t``)

Plan properties
=====================================

After creating a plan, several methods are available to inspect its runtime configuration and behavior
These methods, defined in :f:type:`dtfft_plan_t`, provide valuable insights into the plan's setup and are
particularly useful for debugging or integrating with custom workflows. The following methods are supported:

- :f:func:`get_z_slab_enabled`:
  Returns a logical value indicating whether Z-slab optimization is active in the plan,
  as configured via :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_).
  This helps users confirm if the optimization is applied, especially when troubleshooting performance or compatibility issues.

- :f:func:`get_gpu_backend`:
  Retrieves the GPU backend (e.g., NCCL, MPI P2P) selected during plan creation or autotuning with ``DTFFT_PATIENT`` effort (see `Selecting plan effort`_).

  Available only in CUDA-enabled builds, this method allows users to verify the transposition strategy chosen for GPU execution.

- :f:func:`get_stream`:
  Returns the CUDA stream associated with the plan, either the default stream managed by ``dtFFT`` or a custom one set via
  :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_).

  Available only in CUDA-enabled builds, it enables integration with existing CUDA workflows by exposing the stream used for GPU operations.

- :f:func:`report`:
  Prints detailed plan information to stdout, including grid decomposition, backend selection, and optimization settings.
  This diagnostic tool aids in understanding the plan's configuration and troubleshooting unexpected behavior.

These methods provide a window into the plan's internal state, allowing users to validate settings or gather diagnostics post-creation. They remain accessible until the plan is destroyed with :f:func:`destroy`.

Plan Execution
==============

There are two primary methods to execute a plan in ``dtFFT``: ``transpose`` and ``execute``.
Below, we detail each method, including their behavior for host and GPU versions of the API.

Transpose
---------

The first method is to call the :f:func:`transpose` method of the plan.

Signature
_________

The signature is as follows:

.. tabs::

  .. code-tab:: fortran

    subroutine dtfft_plan_t%transpose(in, out, transpose_type, error_code)
      type(*)                       intent(inout) :: in(..)
      type(*)                       intent(inout) :: out(..)
      type(dtfft_transpose_type_t), intent(in)    :: transpose_type
      integer(int32),   optional,   intent(out)   :: error_code

  .. code-tab:: c

      dtfft_error_code_t
      dtfft_transpose(
        dtfft_plan_t plan,
        void *in,
        void *out,
        const dtfft_transpose_type_t transpose_type);

  .. code-tab:: c++

      dtfft::ErrorCode
      dtfft::Plan::transpose(
          void *in,
          void *out,
          const dtfft::TransposeType transpose_type);

Description
___________

This method transposes data according to the specified ``transpose_type``. Supported options include:

- ``DTFFT_TRANSPOSE_X_TO_Y``: Transpose from X to Y
- ``DTFFT_TRANSPOSE_Y_TO_X``: Transpose from Y to X
- ``DTFFT_TRANSPOSE_Y_TO_Z``: Transpose from Y to Z (valid only for 3D plans)
- ``DTFFT_TRANSPOSE_Z_TO_Y``: Transpose from Z to Y (valid only for 3D plans)
- ``DTFFT_TRANSPOSE_X_TO_Z``: Transpose from X to Z (valid only for 3D plans using Z-slab)
- ``DTFFT_TRANSPOSE_Z_TO_X``: Transpose from Z to X (valid only for 3D plans using Z-slab)

.. note::
   Passing the same pointer to both ``in`` and ``out`` is not permitted; doing so triggers the error :f:var:`DTFFT_ERROR_INPLACE_TRANSPOSE`.

**Host Version**: Executes a single ``MPI_Alltoall(w)`` call using non-contiguous MPI Datatypes and returns once the ``out`` 
buffer contains the transposed data, leaving the ``in`` buffer unchanged.

**GPU Version**: Performs a two-step transposition:

- Launches an nvRTC-compiled kernel to transpose data locally. On a single GPU, this completes the task, and control returns to the user.
- Performs data redistribution using the selected GPU backend (e.g., MPI, NCCL), followed by final processing (e.g., unpacking via nvRTC or copying to ``out``)
  Differences between backends begin at this step (see below for specifics).

In the GPU version, the ``in`` buffer may serve as intermediate storage, potentially modifying its contents,
except when operating on a single GPU, where it remains unchanged.

GPU Backend-Specific Behavior
_____________________________

- **MPI-Based Backends** (:f:var:`DTFFT_GPU_BACKEND_MPI_P2P` and :f:var:`DTFFT_GPU_BACKEND_MPI_A2A`):

  After local transposition, redistributes data using CUDA-aware MPI. Data destined for the same GPU ("self" data) is
  copied via ``cudaMemcpyAsync``.

  For **MPI Peer-to-Peer** (``MPI_P2P``), it issues non-blocking ``MPI_Irecv`` and ``MPI_Isend``
  calls (or ``MPI_Recv_init`` and ``MPI_Send_init`` with ``MPI_Startall`` if built with ``DTFFT_ENABLE_PERSISTENT_COMM``) for point-to-point
  exchanges between GPUs, completing with ``MPI_Waitall``; an nvRTC kernel then unpacks all data at once.

  For **MPI All-to-All** (``MPI_A2A``), it performs a single ``MPI_Ialltoallv`` call (or ``MPI_Alltoallv_init`` with ``MPI_Start``
  if built with ``DTFFT_ENABLE_PERSISTENT_COMM`` and supported by MPI), completing with ``MPI_Wait``; an nvRTC kernel then unpacks the data.

- **Pipelined MPI Peer-to-Peer** (:f:var:`DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED`):

  After local transposition, redistributes data similarly to ``MPI_P2P`` using CUDA-aware MPI with non-blocking ``MPI_Irecv`` and
  ``MPI_Isend`` calls (or ``MPI_Recv_init`` and ``MPI_Send_init`` with ``MPI_Startall`` if built with ``DTFFT_ENABLE_PERSISTENT_COMM``).

  Data destined for the same GPU ("self" data) is copied via ``cudaMemcpyAsync``. Unlike ``MPI_P2P``, as soon as data arrives from a
  process *i*, it is immediately unpacked by launching an nvRTC kernel specific to that process's data.

  This results in *N* nvRTC kernels (one per process) instead of a single kernel unpacking all data, enabling pipelining of
  communication and computation to reduce latency.

- **NCCL-Based Backends** (:f:var:`DTFFT_GPU_BACKEND_NCCL` and :f:var:`DTFFT_GPU_BACKEND_NCCL_PIPELINED`):

  After local transposition, redistributes data using the NCCL library for GPU-to-GPU communication.

  For **NCCL** (``DTFFT_GPU_BACKEND_NCCL``), it executes a cycle of ``ncclSend`` and ``ncclRecv`` calls within ``ncclGroupStart``
  and ``ncclGroupEnd`` to perform point-to-point exchanges between all processes, including "self" data. Once communication completes,
  an nvRTC kernel unpacks all data at once, similar to ``MPI_P2P``.

  For **Pipelined NCCL** (:f:var:`DTFFT_GPU_BACKEND_NCCL_PIPELINED`), it copies "self" data using ``cudaMemcpyAsync`` and immediately
  unpacks it with an nvRTC kernel in a parallel stream created by ``dtFFT``. Concurrently, in main stream, it runs
  the same ``ncclSend`` / ``ncclRecv`` cycle (within ``ncclGroupStart`` and ``ncclGroupEnd``) for data exchange with other
  processes, excluding "self" data. After communication completes, an nvRTC kernel unpacks the data received from all other processes.

- **cuFFTMp** (:f:var:`DTFFT_GPU_BACKEND_CUFFTMP`):

  After local transposition from the ``in`` buffer to the ``out`` buffer using an nvRTC kernel,
  redistributes data using the cuFFTMp library by calling ``cufftMpExecReshapeAsync``.
  This function performs an asynchronous all-to-all exchange across multiple GPUs, reshaping the data from the ``out`` buffer
  back into the ``in`` buffer. Since the final transposed data is required in the ``out`` buffer,
  it is then copied from ``in`` to ``out`` using ``cudaMemcpyAsync``.


.. note::

  Performance and behavior may vary based on GPU interconnects (e.g., NVLink), MPI implementation, and system configuration. 
  To automatically select the fastest GPU backend for a given system, use the ``DTFFT_PATIENT`` effort level when creating plan, 
  which tests each backend and chooses the most efficient one.

.. note::

  Pipelined backends (:f:var:`DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED` and :f:var:`DTFFT_GPU_BACKEND_NCCL_PIPELINED`) require an
  additional ``aux`` buffer, which is managed internally by ``dtFFT`` and inaccessible to the user.
  Similarly, :f:var:`DTFFT_GPU_BACKEND_CUFFTMP` may require an ``aux`` buffer if ``cufftMpGetReshapeSize`` returns a value greater than 0,
  such as when the environment variable ``CUFFT_RESHAPE_USE_PACKING=1`` is set.

  In all other cases, transposition requires only the ``in`` and ``out`` buffers.

Example
_______

Below is an example of transposing data from X to Y and back:

.. tabs::

  .. code-tab:: fortran

    ! Assuming plan is created and buffers `a` and `b` are allocated.
    call plan%transpose(a, b, DTFFT_TRANSPOSE_X_TO_Y, error_code)
    DTFFT_CHECK(error_code)  ! Checks for errors, e.g., DTFFT_ERROR_INPLACE_TRANSPOSE

    ! Process Y-aligned data in buffer `b`
    ! ... (e.g., apply scaling or analysis)

    ! Reverse transposition
    call plan%transpose(b, a, DTFFT_TRANSPOSE_Y_TO_X, error_code)
    DTFFT_CHECK(error_code)

  .. code-tab:: c

    // Assuming plan is created and buffers `a` and `b` are allocated.
    DTFFT_CALL( dtfft_transpose(plan, a, b, DTFFT_TRANSPOSE_X_TO_Y) )

    // Process Y-aligned data in buffer `b`
    // ... (e.g., apply scaling or analysis)

    // Reverse transposition
    DTFFT_CALL( dtfft_transpose(plan, b, a, DTFFT_TRANSPOSE_Y_TO_X) )

  .. code-tab:: c++

    // Assuming plan is created and buffers `a` and `b` are allocated.
    DTFFT_CXX_CALL( plan.transpose(a, b, dtfft::TransposeType::X_TO_Y) )

    // Process Y-aligned data in buffer `b`
    // ... (e.g., apply scaling or analysis)

    // Reverse transposition
    DTFFT_CXX_CALL( plan.transpose(b, a, dtfft::TransposeType::Y_TO_X) )

Execute
-------

The second method is to call the :f:func:`execute` method of the plan.

Signature
_________

The signature is as follows:

.. tabs::

  .. code-tab:: fortran

    subroutine dtfft_plan_t%execute(in, out, execute_type, aux, error_code)
      type(*)                     intent(inout) :: in(..)
      type(*)                     intent(inout) :: out(..)
      type(dtfft_execute_type_t), intent(in)    :: execute_type
      type(*),          optional, intent(inout) :: aux(..)
      integer(int32),   optional, intent(out)   :: error_code

  .. code-tab:: c

      dtfft_error_code_t
      dtfft_execute(
        dtfft_plan_t plan,
        void *in,
        void *out,
        const dtfft_execute_type_t execute_type,
        void *aux);

  .. code-tab:: c++

      dtfft::ErrorCode
      dtfft::Plan::execute(
          void *in,
          void *out,
          const dtfft::ExecuteType execute_type,
          void *aux=nullptr);

Description
___________

This method executes a plan, performing transpositions and optionally FFTs based on the specified ``execute_type``.
It supports in-place execution; the same pointer can be safely passed to both ``in`` and ``out``.
To optimize memory usage, ``dtFFT`` uses the ``in`` buffer as intermediate storage, overwriting its contents.
Users needing to preserve original data should copy it elsewhere.

The key parameter is ``execute_type``, with two options:
- ``DTFFT_EXECUTE_FORWARD``: Forward execution
- ``DTFFT_EXECUTE_BACKWARD``: Backward execution

For 3D plans, the method operates as follows:

**Forward Execution** (``DTFFT_EXECUTE_FORWARD``):

- If ``Transpose-Only``:

  - Transpose from X to Y
  - Transpose from Y to Z
- If ``Transpose-Only`` with Z-slab and distinct ``in`` and ``out``:

  - Transpose from X to Z
- If using FFT:

  - Forward FFT in X direction
  - Transpose from X to Y
  - Forward FFT in Y direction
  - Transpose from Y to Z
  - Forward FFT in Z direction
- If using FFT with Z-slab:

  - Forward 2D FFT in X-Y directions
  - Transpose from X to Z
  - Forward FFT in Z direction

**Backward Execution** (``DTFFT_EXECUTE_BACKWARD``):

- If ``Transpose-Only``:

  - Transpose from Z to Y
  - Transpose from Y to X
- If ``Transpose-Only`` with Z-slab and distinct ``in`` and ``out``:

  - Transpose from Z to X
- If using FFT:

  - Backward FFT in Z direction
  - Transpose from Z to Y
  - Backward FFT in Y direction
  - Transpose from Y to X
  - Backward FFT in X direction
- If using FFT with Z-slab:

  - Backward FFT in Z direction
  - Transpose from Z to X
  - Backward 2D FFT in X-Y directions

.. note::
   For ``Transpose-Only`` plans with a Z-slab and identical ``in`` and ``out`` pointers, execution uses a
   two-step transposition, as direct transposition is not possible with a single pointer.

An optional auxiliary buffer ``aux`` may be provided. If omitted on the first call to :f:func:`execute`,
it is allocated internally and freed when the plan is destroyed. C users can pass ``NULL`` to opt out.

Example
_______

Below is an example of executing a plan forward and backward:

.. tabs::

  .. code-tab:: fortran

    ! Assuming a 3D FFT plan is created and buffers `a`, `b`, and `aux` are allocated
    call plan%execute(a, b, DTFFT_EXECUTE_FORWARD, aux, error_code)
    DTFFT_CHECK(error_code)  ! Checks for execution errors

    ! Process Fourier-space data in buffer `b`
    ! ... (e.g., apply filtering)

    ! Backward execution
    call plan%execute(b, a, DTFFT_EXECUTE_BACKWARD, aux, error_code)
    DTFFT_CHECK(error_code)

  .. code-tab:: c

    // Assuming a 3D FFT plan is created and buffers `a`, `b`, and `aux` are allocated
    DTFFT_CALL( dtfft_execute(plan, a, b, DTFFT_EXECUTE_FORWARD, aux) )

    // Process Fourier-space data in buffer `b`
    // ... (e.g., apply filtering)

    // Backward execution
    DTFFT_CALL( dtfft_execute(plan, b, a, DTFFT_EXECUTE_BACKWARD, aux) )

  .. code-tab:: c++

    // Assuming a 3D FFT plan is created and buffers `a`, `b`, and `aux` are allocated
    DTFFT_CXX_CALL( plan.execute(a, b, dtfft::ExecuteType::FORWARD, aux) )

    // Process Fourier-space data in buffer `b`
    // ... (e.g., apply filtering)

    // Backward execution
    DTFFT_CXX_CALL( plan.execute(b, a, dtfft::ExecuteType::BACKWARD, aux) )

GPU Notes
---------

Both ``transpose`` and ``execute`` in the GPU version operate asynchronously.
When either function returns, computations are queued in a CUDA stream but may not be complete.
Full synchronization with the host requires calling ``cudaDeviceSynchronize``, ``cudaStreamSynchronize``, or ``!$acc wait`` (for OpenACC).

During execution, ``dtFFT`` may use multiple CUDA streams, but the final computation stage always occurs in the
stream returned by :f:func:`get_stream`. Thus, synchronization may be unnecessary if users submit additional kernels to that stream.

Plan Finalization
=================

To fully release all memory resources allocated by ``dtFFT`` for a plan,
the plan must be explicitly destroyed. This ensures that all internal buffers and resources associated with the plan are freed.

.. note::
   If buffers were allocated using :f:func:`mem_alloc`, they must be deallocated with :f:func:`mem_free` *before* calling the destroy method.
   Failing to do so may result in memory leaks or undefined behavior.

Example
-------

Below is an example of properly finalizing a plan and freeing allocated memory:

.. tabs::

  .. code-tab:: fortran

    ! Assuming a plan and buffers `a_ptr`, `b_ptr` and `aux_ptr` are created and allocated with `mem_alloc`
    call plan%mem_free(a_ptr, error_code)    ! Free buffer `a_ptr`
    DTFFT_CHECK(error_code)
    call plan%mem_free(b_ptr, error_code)    ! Free buffer `b_ptr`
    DTFFT_CHECK(error_code)
    call plan%mem_free(aux_ptr, error_code)  ! Free buffer `aux_ptr`
    DTFFT_CHECK(error_code)
    call plan%destroy(error_code)            ! Destroy the plan
    DTFFT_CHECK(error_code)

  .. code-tab:: c

    // Assuming a plan and buffers `a`, `b` and `aux` are created and allocated with `dtfft_mem_alloc`
    DTFFT_CALL( dtfft_mem_free(plan, a) )   // Free buffer `a`
    DTFFT_CALL( dtfft_mem_free(plan, b) )   // Free buffer `b`
    DTFFT_CALL( dtfft_mem_free(plan, aux) ) // Free buffer `aux`
    DTFFT_CALL( dtfft_destroy(&plan) )      // Destroy the plan

  .. code-tab:: c++

    // Assuming a plan and buffers `a`, `b` and `aux` are created and allocated with `mem_alloc`
    DTFFT_CXX_CALL( plan.mem_free(a) )    // Free buffer `a`
    DTFFT_CXX_CALL( plan.mem_free(b) )    // Free buffer `b`
    DTFFT_CXX_CALL( plan.mem_free(aux) )  // Free buffer `aux`
    DTFFT_CXX_CALL( plan.destroy() )      // Explicitly destroy the plan (optional if using destructor)
                                          // Automatic ~Plan() call when `plan` goes out of scope

Complete Example
================

The following example demonstrates the full lifecycle of a ``dtFFT`` complex-to-complex plan:
creating a plan, allocating memory, executing forward and backward transformations, and properly finalizing resources.

.. tabs::

  .. code-tab:: fortran

    program dtfft_sample
    #include "dtfft.f03"
    use iso_fortran_env
    use dtfft
    use mpi ! or use mpi_f08
    use iso_c_binding
    implicit none
      type(dtfft_plan_c2c_t) :: plan
      type(dtfft_config_t) :: config
      integer(int32) :: dims(3) = [64, 64, 64]  ! Example dimensions
      integer(int32) :: error_code
      integer(int64) :: alloc_size, element_size, alloc_bytes
      complex(real64), pointer :: a(:), b(:), aux(:)
      type(c_ptr) :: a_ptr, b_ptr, aux_ptr


      call MPI_Init(error_code)

      ! Create dtfft_config_t object with default values
      config = dtfft_config_t()

      ! Disable Z-slab
      config%enable_z_slab = .false.

      ! Apply configuration to dtFFT
      call dtfft_set_config(config, error_code)
      DTFFT_CHECK(error_code)

      ! Create plan
      call plan%create(dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, DTFFT_EXECUTOR_NONE, error_code)
      DTFFT_CHECK(error_code)

      ! Obtain allocation sizes
      alloc_size = plan%get_alloc_size(error_code); DTFFT_CHECK(error_code)
      element_size = plan%get_element_size(); DTFFT_CHECK(error_code)

      alloc_bytes = alloc_size * element_size
      ! Allocate memory
      a_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)
      b_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)
      aux_ptr = plan%mem_alloc(alloc_bytes, error_code); DTFFT_CHECK(error_code)

      ! Convert to Fortran arrays
      call c_f_pointer(a_ptr, a, [alloc_size])
      call c_f_pointer(b_ptr, b, [alloc_size])
      call c_f_pointer(aux_ptr, aux, [alloc_size])

      ! Forward execution
      call plan%execute(a, b, DTFFT_EXECUTE_FORWARD, aux, error_code)
      DTFFT_CHECK(error_code)

      ! Process Fourier-space data in buffer `b` (e.g., apply filtering)
      ! ...

      ! Backward execution
      call plan%execute(b, a, DTFFT_EXECUTE_BACKWARD, aux, error_code)
      DTFFT_CHECK(error_code)

      ! Free memory
      call plan%mem_free(a_ptr, error_code); DTFFT_CHECK(error_code)
      call plan%mem_free(b_ptr, error_code); DTFFT_CHECK(error_code)
      call plan%mem_free(aux_ptr, error_code); DTFFT_CHECK(error_code)

      ! Destroy the plan
      call plan%destroy(error_code)
      DTFFT_CHECK(error_code)

      call MPI_Finalize(error_code)
    end program dtfft_sample

  .. code-tab:: c

    #include <dtfft.h>
    #include <mpi.h>

    int main(int argc, char *argv[])
    {
      dtfft_plan_t plan;
      dtfft_complex *a, *b, *aux;  // Use dtfft_complex from dtfft.h
      int32_t dims[3] = {64, 64, 64};  // Example dimensions
      size_t alloc_size;

      MPI_Init(&argc, &argv);

      dtfft_config_t config;
      // Set default values to config
      dtfft_create_config(&config);
      // Disable Z-slab
      config.enable_z_slab = 0;

      // Apply configuration to dtFFT
      DTFFT_CALL( dtfft_set_config(config) );

      // Create plan
      DTFFT_CALL( dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, DTFFT_EXECUTOR_NONE, &plan) );

      // Obtain allocation size
      DTFFT_CALL( dtfft_get_alloc_size(plan, &alloc_size) );

      // Allocate memory
      DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(dtfft_complex) * alloc_size, (void**)&a) );
      DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(dtfft_complex) * alloc_size, (void**)&b) );
      DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(dtfft_complex) * alloc_size, (void**)&aux) );

      // Forward execution
      DTFFT_CALL( dtfft_execute(plan, a, b, DTFFT_EXECUTE_FORWARD, aux) );

      // Process Fourier-space data in buffer `b` (e.g., apply filtering)
      // ...

      // Backward execution
      DTFFT_CALL( dtfft_execute(plan, b, a, DTFFT_EXECUTE_BACKWARD, aux) );

      // Free memory
      DTFFT_CALL( dtfft_mem_free(plan, a) );
      DTFFT_CALL( dtfft_mem_free(plan, b) );
      DTFFT_CALL( dtfft_mem_free(plan, aux) );

      // Destroy the plan
      DTFFT_CALL( dtfft_destroy(&plan) );

      MPI_Finalize();
      return 0;
    }

  .. code-tab:: c++

    #include <dtfft.hpp>
    #include <mpi.h>
    #include <complex>
    #include <vector>

    using namespace dtfft;

    int main(int argc, char *argv[])
    {
      MPI_Init(&argc, &argv);

      std::vector<int32_t> dims = {64, 64, 64};  // Example dimensions

      // Set default values to config
      Config config;
      config.set_enable_z_slab(false);

      // Apply configuration to dtFFT
      DTFFT_CXX_CALL( set_config(config) );

      // Create plan
      PlanC2C plan(dims, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, Executor::NONE);

      size_t alloc_size, element_size;
      DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) );
      DTFFT_CXX_CALL( plan.get_element_size(&element_size) );

      size_t alloc_bytes = alloc_size * element_size;
      std::complex<double> *a, *b, *aux;

      // Allocate memory
      DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&a) );
      DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&b) );
      DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, (void**)&aux) );

      // Forward execution
      DTFFT_CXX_CALL( plan.execute(a, b, ExecuteType::FORWARD, aux) );

      // Process Fourier-space data in buffer `b` (e.g., apply filtering)
      // ...

      // Backward execution
      DTFFT_CXX_CALL( plan.execute(b, a, ExecuteType::BACKWARD, aux) );

      // Free memory
      DTFFT_CXX_CALL( plan.mem_free(a) );
      DTFFT_CXX_CALL( plan.mem_free(b) );
      DTFFT_CXX_CALL( plan.mem_free(aux) );

      // Explicitly destroy the plan
      DTFFT_CXX_CALL( plan.destroy() );

      MPI_Finalize();
      return 0;
    }