.. _usage_link:

###########
Usage Guide
###########

This guide provides a comprehensive overview of using the ``dtFFT`` library to perform parallel data transpositions and optionally
Fast Fourier Transforms (FFTs) across host and GPU environments.
Designed for high-performance computing, ``dtFFT`` simplifies the process of decomposing multidimensional data, managing memory,
and executing transformations by integrating with external FFT libraries or operating in ``Transpose-Only`` mode.

Whether targeting CPU clusters with MPI or GPU-accelerated systems with CUDA, this library offers flexible configuration options to
optimize performance for specific use cases. The following sections detail key aspects of working with ``dtFFT``, from plan creation to
execution and resource management, with practical examples in Fortran, C, and C++.

Error Handling and Macros
=========================

Almost all ``dtFFT`` functions return error codes to indicate whether execution was successful. These codes help users identify and handle issues during plan creation, memory allocation, execution, and finalization. The error handling mechanism differs slightly across language APIs:

- **Fortran API**: Functions include an optional ``error_code`` parameter (type ``integer(int32)``), always positioned as the last argument.
  If omitted, errors must be checked through other means, such as program termination or runtime assertions.
- **C API**: Functions return a value of type :cpp:type:`dtfft_error_t`, allowing direct inspection of the result.
- **C++ API**: Functions return :cpp:type:`dtfft::Error`, typically used with exception handling or explicit checks.

To simplify error checking, ``dtFFT`` provides predefined macros that wrap function calls and handle error codes automatically:

- **Fortran**: The ``DTFFT_CHECK`` macro, defined in ``dtfft.f03``, checks the ``error_code`` and halts execution with an informative message if an error occurs. Include this header with ``#include "dtfft.f03"`` to use it.
- **C**: The ``DTFFT_CALL`` macro wraps function calls, checks the returned :c:type:`dtfft_error_t`,
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

    DTFFT_CXX_CALL( plan.execute(a, b, dtfft::Execute::FORWARD, nullptr) );


Error codes are defined in the API sections (e.g., :f:var:`DTFFT_SUCCESS`, :f:var:`DTFFT_ERROR_INVALID_TRANSPOSE_TYPE`). Refer to the Fortran, C, and C++ API documentation for a complete list and detailed descriptions.

Plan Creation
=============

dtFFT supports three plan categories, each tailored to specific transformation requirements:

- Real-to-Real (R2R)
- Complex-to-Complex (C2C)
- Real-to-Complex (R2C)

.. note:: The Real-to-Complex plan is available only when the library is built with FFT support.

dtFFT provides two complementary workflows for constructing a plan:

1. **Global-dimension workflow** – supply the global lattice extents and allow ``dtFFT`` to derive the process decomposition. 
   This workflow is detailed in `Global-Dimension Workflow`_.
2. **Local-decomposition workflow** – supply the portion of the domain owned by each MPI rank via a pencil descriptor. 
   This workflow is described in `Local-Decomposition Workflow`_.

Both workflows share the same configuration surface (plan category, precision, executor, and effort level); they differ only in how the data distribution is communicated to the library.

Global-Dimension Workflow
-------------------------

This default workflow constructs a plan by providing the global array dimensions (in Fortran order) together with the MPI communicator. ``dtFFT`` deduces the process decomposition from that information, optionally complemented by the optimization effort and FFT executor parameters.

Plans are instantiated through the ``create`` method or the corresponding language-specific constructor, as described in the Fortran, C, and C++ API sections. Every plan accepts an MPI communicator that defines the process distribution. 

When the global-dimension workflow is used, ``dtFFT`` must derive how the global domain is partitioned across MPI ranks. The subsections below outline the default strategy and how to supply a custom topology. Users employing the local-decomposition workflow already provide this information explicitly through the pencil descriptor and can skim this section.

**Default Behavior**

When the communicator passed during plan creation is ``MPI_COMM_WORLD`` with :math:`P` processes, ``dtFFT`` attempts the following steps in order:

- If :math:`P <= N_z` (and :math:`N_z / P >= 32` for the GPU version), split the grid as :math:`N_x \times N_y \times N_z / P`. 
  This distributes the Z-dimension across :math:`P` processes. Division need not be even, and the local size per process may vary.
- If the Z-split fails (e.g., :math:`P > N_z` or :math:`N_z / P < 32` on GPU), attempt :math:`N_x \times N_y / P \times N_z`. 
  This distributes the Y-dimension across ``P`` processes, provided :math:`N_x <= P` to remain compatible with future transpositions (e.g., X-to-Y).
- If both attempts fail, ``dtFFT`` constructs a 3D communicator by fixing the X-dimension split to 1 and using ``MPI_Dims_create(P, 2, dims)`` 
  to balance the remaining :math:`P` processes across :math:`Y` and :math:`Z`, resulting in :math:`N_x \times N_y / P_1 \times N_z / P_2` 
  (where :math:`P_1 \times P_2 = P`).
- If this 3D decomposition is not viable (e.g., :math:`N_y < P_1` or :math:`N_z < P_2`), ``dtFFT`` proceeds but prints a warning message. 
  Ensure :ref:`DTFFT_ENABLE_LOG<dtfft_enable_log_env>` is enabled to observe it.

**User-Controlled Decomposition**

Applications may supply a communicator with an attached Cartesian topology. Grid dimensions must be provided in Fortran order (X, Y, Z).

- **1D Communicator**: A one-dimensional communicator with :math:`P` processes splits the grid as :math:`N_x \times N_y \times N_z / P`, 
  distributing the Z-dimension across :math:`P` processes.
- **2D Communicator**: A two-dimensional communicator with topology :math:`P_1 \times P_2` (where :math:`P_1 * P_2 = P`) decomposes the grid 
  as :math:`N_x \times N_y / P_1 \times N_z / P_2`, splitting :math:`Y` by :math:`P_1` and :math:`Z` by :math:`P_2` while keeping :math:`X` indivisible.
- **3D Communicator**: A three-dimensional communicator with topology :math:`P_0 \times P_1 \times P_2` (where :math:`P_0 * P_1 * P_2 = P`) 
  is supported, but :math:`P_0` (the X split) must be 1 to preserve the fastest-varying dimension. 
  Violating this constraint triggers :f:var:`DTFFT_ERROR_INVALID_COMM_FAST_DIM`.

The example below illustrates the global-dimension workflow by creating a 3D C2C double-precision ``Transpose-Only`` plan:

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

.. _plan_creation_pencil:

Local-Decomposition Workflow
----------------------------

The alternative workflow constructs a plan from a user-defined pencil decomposition. Instead of supplying global dimensions, the application provides, for each MPI rank, the starting indices and extents of the local sub-domain. This workflow affords full control over data locality and aligns ``dtFFT`` with pre-existing domain decompositions.

Use this approach when you need to:

- Reuse a decomposition generated by another solver or library.
- Guarantee specific locality constraints (for example, to co-locate data with accelerators or I/O tasks).
- Persist a previously tuned decomposition and avoid re-running autotuning logic.

Both constructors and ``create`` methods accept the :f:type:`dtfft_pencil_t` descriptor. The descriptor stores the dimensionality, the local starting indices (0-based), and the counts along each dimension.

The example below decomposes a :math:`64 \times 64 \times 64` grid by splitting only along the slowest (Z) dimension. Each rank describes its local block and then creates a plan using the pencil descriptor.

.. tabs::

  .. code-tab:: fortran

    #include "dtfft.f03"
    use iso_fortran_env
    use dtfft
    use mpi

    type(dtfft_plan_c2c_t) :: plan
    type(dtfft_pencil_t) :: my_pencil
    integer(int32) :: error_code
    integer(int32) :: starts(3), counts(3)
    integer :: rank, size, ierr

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    starts = [0, 0, rank * (64 / size)]
    counts = [64, 64, 64 / size]

    my_pencil = dtfft_pencil_t(starts, counts)

    call plan%create(my_pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, error_code)
    DTFFT_CHECK(error_code)

  .. code-tab:: c

    #include <dtfft.h>
    #include <mpi.h>

    int main(int argc, char *argv[]) {
      dtfft_plan_t plan;
      dtfft_pencil_t pencil;
      int rank, size;

      MPI_Init(&argc, &argv);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      pencil.ndims = 3;
      pencil.starts[0] = 0;
      pencil.starts[1] = 0;
      pencil.starts[2] = rank * (64 / size);
      pencil.counts[0] = 64;
      pencil.counts[1] = 64;
      pencil.counts[2] = 64 / size;

      DTFFT_CALL( dtfft_create_plan_c2c_pencil(&pencil, MPI_COMM_WORLD,
                    DTFFT_DOUBLE, DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &plan) );

      return 0;
    }

  .. code-tab:: c++

    #include <dtfft.hpp>

    int main(int argc, char *argv[]) {
      MPI_Init(&argc, &argv);

      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      std::vector<int32_t> starts = {0, 0, rank * (64 / size)};
      std::vector<int32_t> counts = {64, 64, 64 / size};

      auto pencil = dtfft::Pencil(starts, counts);

      dtfft::PlanC2C plan(pencil, MPI_COMM_WORLD, dtfft::Precision::DOUBLE,
                          dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);

      return 0;
    }

The pencil API validates that the provided communicator and local shapes collectively form a consistent global domain. Refer to the Fortran, C, and C++ API references for the full descriptor layout and helper constructors.

Slab Optimizations
------------------

dtFFT supports two slab optimizations that can reduce the number of data transpositions during FFT execution by employing two-dimensional FFT algorithms where applicable. These optimizations are controlled via the :f:type:`dtfft_config_t` structure or corresponding environment variables.

Z-Slab Optimization
___________________

When the grid is decomposed as :math:`N_x \times N_y \times N_z / P` (e.g., via a 1D communicator or the first default step), the Z-slab optimization becomes available. If enabled (default), it reduces the number of data transpositions by employing a two-dimensional FFT algorithm in X-Y directions during calls to :f:func:`execute`. This also enables ``DTFFT_TRANSPOSE_X_TO_Z`` and ``DTFFT_TRANSPOSE_Z_TO_X`` in :f:func:`transpose`, while other transpose types remain available.

This optimization can be disabled through the ``enable_z_slab`` field in :f:type:`dtfft_config_t` or the :ref:`DTFFT_ENABLE_Z_SLAB<dtfft_enable_z_slab_env>` environment variable. It cannot be forced when the decomposition is incompatible with Z-slab requirements. Consider disabling it to resolve ``DTFFT_ERROR_VKFFT_R2R_2D_PLAN`` errors or when the underlying 2D FFT implementation is too slow. In all other cases, Z-slab is considered faster.

Y-Slab Optimization
___________________

When the grid is decomposed as :math:`N_x \times N_y / P \times N_z` (e.g., via a 2D communicator splitting along Y), the Y-slab optimization can be enabled. If set (disabled by default), dtFFT will skip the transpose step between Y and Z aligned layouts during calls to :f:func:`execute`, employing a two-dimensional FFT algorithm instead.

This optimization can be enabled through the ``enable_y_slab`` field in :f:type:`dtfft_config_t`. Consider disabling it when the underlying 2D FFT implementation is too slow. In all other cases, Y-slab is considered faster.

.. note:: When Y-slab is enabled,


Precision and FFT Executor
--------------------------

Two parameters govern the numerical representation and FFT backend selection:

- **Precision** (:f:type:`dtfft_precision_t`):

  - ``DTFFT_SINGLE`` – single precision
  - ``DTFFT_DOUBLE`` – double precision

- **FFT Executor** (:f:type:`dtfft_executor_t`):

  - ``DTFFT_EXECUTOR_NONE`` – ``Transpose-Only`` (no FFT)
  - ``DTFFT_EXECUTOR_FFTW3`` – FFTW3 (host only, available when compiled with FFTW3 support)
  - ``DTFFT_EXECUTOR_MKL`` – MKL DFTI (host only, available when compiled with MKL support)
  - ``DTFFT_EXECUTOR_CUFFT`` – cuFFT (GPU only, available when compiled with CUDA support)
  - ``DTFFT_EXECUTOR_VKFFT`` – VkFFT (GPU only, available when compiled with VkFFT support)

Selecting plan effort
---------------------

The ``effort`` parameter in ``dtFFT`` determines the level of optimization applied during plan creation, influencing how data transposition is configured. On the host, ``dtFFT`` leverages custom MPI datatypes to perform transpositions, tailored to the grid decomposition and data layout. On the GPU, transposition is handled by nvRTC-compiled kernels, optimized at runtime for specific data sizes and types, with data exchange between GPUs facilitated by various backend options (e.g., NCCL, MPI P2P). The supported effort levels defined by :f:type:`dtfft_effort_t` control the extent of this optimization as follows:

DTFFT_ESTIMATE
______________

This minimal-effort option prioritizes fast plan creation.

On the host, ``dtFFT`` selects a default grid decomposition and if selected backend is ``DTFFT_BACKEND_MPI_DATATYPE`` constructs MPI datatypes based on environment variables such as ``DTFFT_DTYPE_X_Y`` and ``DTFFT_DTYPE_Y_Z`` (see :ref:`MPI Datatype Selection variables <datatype_selection>`), which define the default send and receive strategies. In a case of other backends preparations are minimal to none.

On the GPU, it uses a pre-selected backend specified via :f:type:`dtfft_config_t` (see configuration details below), compiling an nvRTC kernel tailored to the chosen backend.

DTFFT_MEASURE
_____________

With this moderate-effort setting, ``dtFFT`` explores multiple grid decomposition strategies to reduce communication overhead during transposition, cycling through possible grid layouts to find an efficient configuration. On the host, it uses the same MPI datatypes as defined by environment variables in ``DTFFT_ESTIMATE``. On the GPU, it employs the same backend as specified in the configuration for ``DTFFT_ESTIMATE``.

If a Cartesian communicator is provided or plan is being created using :f:type:`dtfft_pencil_t` structure, it reverts to ``DTFFT_ESTIMATE`` behavior, relying on the user-specified topology.

DTFFT_PATIENT
_____________

This maximum-effort option extends ``DTFFT_MEASURE`` by exhaustively optimizing transposition strategies. On the host, it cycles through various custom MPI datatype combinations, optionally including MPI backends if enabled to minimize network latency and maximize throughput. 

On the GPU, it cycles through available backends (e.g., NCCL, MPI P2P). Additionally, it performs kernel autotuning by launching multiple kernel configurations and measuring their performance to select the best one.

.. note:: Kernel optimization can be enabled with both ``DTFFT_MEASURE`` and ``DTFFT_PATIENT`` effort levels by setting field 
  ``force_kernel_optimization`` of :f:type:`dtfft_config_t` to ``true``.


---------

The choice of ``effort`` impacts both plan creation time and runtime performance. Higher effort levels (``DTFFT_MEASURE`` and ``DTFFT_PATIENT``) increase setup time but can enhance transposition efficiency, especially for large datasets or complex grids.

If a user already knows the optimal grid decomposition, MPI datatypes, or backend from a previous computation, these can be pre-specified before plan creation: the grid via a custom ``MPI_Comm`` communicator or ``dtfft_pencil_t`` structure,  MPI datatypes through environment variables (e.g., ``DTFFT_DTYPE_X_Y``), and the backend through :f:type:`dtfft_config_t`.

.. _config_link:

Setting Additional Configurations
---------------------------------

The :f:type:`dtfft_config_t` type allows users to set additional configuration parameters for ``dtFFT`` before plan creation, tailoring its behavior to specific needs. These settings are optional and can be applied using the constructor ``dtfft_config_t()`` or the :f:func:`dtfft_create_config` function, followed by a call to :f:func:`dtfft_set_config`.

Configurations must be set prior to creating a plan to take effect. The available parameters are summarized below:

.. list-table:: Configuration parameters
   :header-rows: 1
   :widths: 16 18 10 6 50

   * - Field
     - Type / Enum
     - Default
     - CUDA
     - Description
   * - ``enable_log``
     - logical
     - ``.false.``
     - 
     - Enable autotuning / selection logging (errors are always printed regardless).
   * - ``enable_z_slab``
     - logical
     - ``.true.``
     - 
     - Enable Z-slab optimization (fewer transfers, enables X↔Z transpose path). Disable to work around 2D FFT issues (e.g. ``DTFFT_ERROR_VKFFT_R2R_2D_PLAN``).
   * - ``enable_y_slab``
     - logical
     - ``.false.``
     - 
     - Enable Y-slab optimization (fewer transfers). Disable to work around 2D FFT issues.
   * - ``n_measure_warmup_iters``
     - integer
     - ``2``
     -
     - Number of warmup iterations during autotune when effort > ``DTFFT_ESTIMATE``.
   * - ``n_measure_iters``
     - integer
     - ``5``
     -
     - Number of measured iterations during autotune when effort > ``DTFFT_ESTIMATE``.
   * - ``platform``
     - :f:type:`dtfft_platform_t`
     - ``DTFFT_PLATFORM_HOST``
     - ✓
     - Execution platform (HOST / CUDA). Available only when built with CUDA. When ``dtFFT`` is built with CUDA support, user
   * - ``stream``
     - :f:type:`dtfft_stream_t`
     - (internal)
     - ✓
     - Custom CUDA stream override (user destroys it after plan). Otherwise internally managed.
   * - ``backend``
     - :f:type:`dtfft_backend_t`
     - differs between HOST / CUDA
     - 
     - Backend used for ``DTFFT_ESTIMATE`` / ``DTFFT_MEASURE``. Default is ``DTFFT_BACKEND_MPI_DATATYPE`` when executed on host. When executed on GPU default is ``DTFFT_BACKEND_NCCL`` if available, otherwise falls back to ``DTFFT_BACKEND_MPI_P2P``.
   * - ``enable_datatype_backend``
     - logical
     - ``.true.``
     - 
     - Allow MPI datatype backend during autotuning on host.
   * - ``enable_mpi_backends``
     - logical
     - ``.false.``
     - 
     - Allow MPI backends (tested in ``DTFFT_PATIENT``). Disabled by default due to OpenMPI leak (see docs)
   * - ``enable_pipelined_backends``
     - logical
     - ``.true.``
     - 
     - Try pipelined variants (overlap copy/unpack); may need internal aux buffer.
   * - ``enable_nccl_backends``
     - logical
     - ``.true.``
     - ✓
     - Allow NCCL-based backends during autotuning.
   * - ``enable_nvshmem_backends``
     - logical
     - ``.true.``
     - ✓
     - Include NVSHMEM-enabled backends (if library built with NVSHMEM support).
   * - ``enable_kernel_optimization``
     - logical
     - ``.true.``
     -
     - Autotune transpose kernels (only in ``DTFFT_PATIENT`` unless forced).
   * - ``n_configs_to_test``
     - integer
     - ``5``
     -
     - Number of kernel configs actually launched after scoring (max 25). ``0`` or ``1`` disables kernel optimization.
   * - ``force_kernel_optimization``
     - logical
     - ``.false.``
     -
     - Force kernel autotuning even for lower effort levels (no extra comm, small overhead).

.. note::
   Fields marked “CUDA” are available only if the library was compiled with CUDA (``DTFFT_WITH_CUDA``).

.. note:: Almost all values can be overridden by setting the appropriate environment variable, which takes precedence if set. 
  Refer to :ref:`Environment Variables<environ_link>` section.

These settings allow fine-tuning of transposition strategies and GPU behavior. For example, disabling ``enable_mpi_backends`` mitigates memory leaks, while setting a custom ``stream`` integrates ``dtFFT`` with existing CUDA workflows. Refer to the Fortran, C and C++ API pages for detailed parameter specifications.


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
    dtfft_set_config(&config);

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

The plan method :f:func:`get_local_sizes` retrieves the number of elements in "real" and "Fourier" spaces and the minimum number of elements that must be allocated:

- **in_starts**: Start indices of the local data portion in real space (0-based)
- **in_counts**: Number of elements in the local data portion in real space
- **out_starts**: Start indices of the local data portion in Fourier space (0-based)
- **out_counts**: Number of elements in the local data portion in Fourier space
- **alloc_size**: Minimum number of elements needed for ``in``, ``out``, or ``aux`` buffers

.. note:: If Y-slab optimization is enabled (see :f:func:`get_y_slab_enabled`), the Fourier space layout is Y-aligned instead of Z-aligned, and ``out_*`` values reflect the Y-aligned layout.

Arrays ``in_starts``, ``in_counts``, ``out_starts``, and ``out_counts`` must have at least as many elements as the plan's dimensions.

The minimum number of bytes required for each buffer is ``alloc_size * element_size``. The ``element_size`` can be obtained by :f:func:`get_element_size` which returns:

- **C2C**: ``2 * sizeof(double) = 16 bytes`` (double precision) or ``2 * sizeof(float) = 8 bytes`` (single precision)
- **R2R and R2C**: ``sizeof(double) = 8 bytes`` (double precision) or ``sizeof(float) = 4 bytes`` (single precision)

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
    DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) );

    // OR use even more convenient wrapper
    auto alloc_size = plan.get_alloc_size();

    // Optionally get element size in bytes
    size_t element_size;
    DTFFT_CXX_CALL( plan.get_element_size(&element_size) );

    // OR use convenient wrapper
    auto element_size = plan.get_element_size();

For 3D plans, :f:func:`get_local_sizes` does not detail the intermediate Y-direction layout. This information, useful for transpose-only plans or when using unsupported FFT libraries, can be retrieved via the ``pencil`` interface (see `Pencil Decomposition`_ below). Pencil IDs start from 1 in both C and Fortran.

The ``dtFFT`` library provides functions to allocate and free memory tailored to the plan:

- :f:func:`mem_alloc`: Allocates memory.
- :f:func:`mem_free`: Frees memory allocated by :f:func:`mem_alloc`.

Fortran interface provides additional methods for memory allocation and deallocation:

- :f:func:`mem_alloc_ptr`: Allocates memory and returns a pointer of type ``c_ptr``.
- :f:func:`mem_free_ptr`: Frees memory allocated by :f:func:`mem_alloc_ptr`.

Host Version
------------

Allocates memory based on the :f:type:`dtfft_executor_t`: 

- ``fftw_malloc`` for FFTW3
- ``mkl_malloc`` for MKL DFT
- ``aligned_alloc`` (16-byte alignment) from C11 Standard library for transpose-only plans.

GPU Version
-----------

Allocates memory based on the :f:type:`dtfft_backend_t`:

- ``ncclMemAlloc`` for NCCL (if available)
- ``nvshmem_malloc`` for NVSHMEM-based backends
- ``cudaMalloc`` otherwise.

If NCCL is used and supports buffer registration via ``ncclCommRegister``, and the environment variable  :ref:`DTFFT_NCCL_BUFFER_REGISTER<dtfft_nccl_buffer_register_env>` is not set to ``0``, the allocated buffer will also be registered.  This registration optimizes communication performance by reducing the overhead of memory operations, which is particularly beneficial for workloads with repeated communication patterns.

.. tabs::

  .. code-tab:: fortran

    use iso_fortran_env

    ! Host version
    complex(real64), pointer :: a(:), b(:), aux(:)
    ! CUDA Fortran version
    complex(real64), device, contiguous, pointer :: a(:), b(:), aux(:)

    ! Allocates memory
    call plan%mem_alloc(alloc_size, a, error_code=error_code); DTFFT_CHECK(error_code)
    call plan%mem_alloc(alloc_size, b, error_code=error_code); DTFFT_CHECK(error_code)
    call plan%mem_alloc(alloc_size, aux, error_code=error_code); DTFFT_CHECK(error_code)

    ! or use pointers of type c_ptr
    use iso_c_binding

    type(c_ptr) :: a_ptr, b_ptr, aux_ptr
    integer(int64) :: alloc_bytes

    alloc_bytes = alloc_size * element_size
    a_ptr = plan%mem_alloc_ptr(alloc_bytes, error_code=error_code); DTFFT_CHECK(error_code)
    b_ptr = plan%mem_alloc_ptr(alloc_bytes, error_code=error_code); DTFFT_CHECK(error_code)
    aux_ptr = plan%mem_alloc_ptr(alloc_bytes, error_code=error_code); DTFFT_CHECK(error_code)


  .. code-tab:: c

    size_t alloc_bytes = alloc_size * element_size;
    double *a, *b, *aux;

    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&a) );
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&b) );
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&aux) );

  .. code-tab:: c++

    #include <complex>

    size_t alloc_bytes = alloc_size * element_size;
    std::complex<double> *a;

    // C-like way of memory allocation
    DTFFT_CXX_CALL( plan.mem_alloc(alloc_bytes, reinterpret_cast<void**>(&a)) );

    // C++ way, note that this way may throw dtfft::Exception on error
    // Note that number of elements is passed here instead of bytes
    // Size of each element is defined by template argument
    auto b = plan.mem_alloc<std::complex<double>>(alloc_size);
    auto aux = plan.mem_alloc<std::complex<double>>(alloc_size);

.. note:: Memory allocated with :f:func:`mem_alloc` must be deallocated with :f:func:`mem_free` **before** the plan is destroyed to avoid memory leaks.

Pencil Decomposition
--------------------

For detailed layout information in 3D plans (e.g., intermediate states like Y-direction distribution), use the :f:func:`get_pencil` method. This returns a ``dtfft_pencil_t`` structure containing:

- **dim**: Aligned dimension ID (1 for X, 2 for Y, 3 for Z).
- **ndims**: Number of dimensions in the pencil (2 or 3)
- **starts**: Local start indices in natural Fortran order. (Allocatable array of size ``ndims``)
- **counts**: Local element counts in natural Fortran order (Allocatable array of size ``ndims``)
- **size**: Total number of elements in a pencil

.. tabs::

  .. code-tab:: fortran

    integer(int8) :: i
    type(dtfft_pencil_t) :: pencils(3)

    do i = 1, 3
      ! Get pencil for dimension i
      pencils(i) = plan%get_pencil(i, error_code)
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
      dtfft::Pencil pencil = plan.get_pencil(i + 1); // This call will throw an exception if an error occurs
      pencils.push_back(pencil);
      // Access pencil properties, e.g., pencils[i].get_dim(), pencils[i].get_starts()
    }

In C++, the ``dtfft::Pencil`` class provides properties via getter methods:

- ``get_ndims()``: Returns the number of dimensions
- ``get_dim()``: Returns the aligned dimension ID
- ``get_starts()``: Returns the start indices as a ``std::vector<int32_t>``
- ``get_counts()``: Returns the element counts as a ``std::vector<int32_t>``
- ``get_size()``: Returns the total number of elements.
- ``c_struct()``: Returns the underlying C structure (``dtfft_pencil_t``)

Plan properties
=====================================

After creating a plan, several methods are available to inspect its runtime configuration and behavior. These methods, defined in :f:type:`dtfft_plan_t`, provide valuable insights into the plan's setup and are particularly useful for debugging or integrating with custom workflows. The following methods are supported:

- :f:func:`get_z_slab_enabled`: Returns a logical value indicating whether Z-slab optimization is active in the plan, as configured via :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_). This helps users confirm if the optimization is applied, especially when troubleshooting performance or compatibility issues.

- :f:func:`get_y_slab_enabled`: Returns a logical value indicating whether Y-slab optimization is active in the plan, as configured via :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_). This allows users to verify the optimization status, which can impact performance and data layout during execution.

- :f:func:`get_backend`: Retrieves the backend (e.g., NCCL, MPI P2P) selected during plan creation or autotuning with ``DTFFT_PATIENT`` effort (see `Selecting plan effort`_).

- :f:func:`get_stream`: Returns the CUDA stream associated with the plan, either the default stream managed by ``dtFFT`` or a custom one set via :f:type:`dtfft_config_t` (see `Setting Additional Configurations`_).

  Available only in CUDA-enabled builds, it enables integration with existing CUDA workflows by exposing the stream used for GPU operations.

- :f:func:`report`: Prints detailed plan information to stdout, including grid decomposition, backend selection, and optimization settings. This diagnostic tool aids in understanding the plan's configuration and troubleshooting unexpected behavior.

- :f:func:`get_executor`: Returns the executor type (e.g., NONE, VKFFT, CUFFT) used for FFT computations within the plan.

- :f:func:`get_precision`: Returns the numerical precision (:f:var:`DTFFT_SINGLE` or :f:var:`DTFFT_DOUBLE`) of the plan.

- :f:func:`get_dims`: Returns global dimensions of the plan. This can be useful for validating the plan's setup against expected sizes.

- :f:func:`get_grid_dims`: Returns the grid decomposition dimensions used in the plan, reflecting how the global domain is partitioned across MPI ranks.

- :f:func:`get_platform`: Returns the execution platform (:f:var:`DTFFT_PLATFORM_HOST` or :f:var:`DTFFT_PLATFORM_CUDA`) of the plan.

  Available only in CUDA-enabled builds

These methods provide a window into the plan's internal state, allowing users to validate settings or gather diagnostics post-creation. They remain accessible until the plan is destroyed with :f:func:`destroy`.

Plan Execution
==============

There are two primary methods to execute a plan in ``dtFFT``: ``transpose`` and ``execute``. Below, we detail each method, including their behavior for host and GPU versions of the API.

Transpose
---------

The first method is to call the :f:func:`transpose` method of the plan. There are two ways to invoke it: asynchronously by executing :f:func:`transpose_start` followed by :f:func:`transpose_end`, or synchronously by calling :f:func:`transpose` directly.

Asynchronous transpose is only useful for host plans when backend is set to one of the :f:var:`DTFFT_BACKEND_MPI_DATATYPE`, :f:var:`DTFFT_BACKEND_MPI_P2P` or :f:var:`DTFFT_BACKEND_MPI_A2A` options, allowing overlap of communication with computation. All CUDA backends are executed asynchronously by default.

Signature
_________

The signature is as follows:

.. tabs::

  .. code-tab:: fortran

    subroutine dtfft_plan_t%transpose(in, out, transpose_type, error_code)
      type(*)                     intent(inout) :: in(..)
      type(*)                     intent(inout) :: out(..)
      type(dtfft_transpose_t),    intent(in)    :: transpose_type
      integer(int32),   optional, intent(out)   :: error_code
    end subroutine

    subroutine dtfft_plan_t%transpose_ptr(in, out, transpose_type, error_code)
      type(c_ptr)                 intent(in)    :: in
      type(c_ptr)                 intent(in)    :: out
      type(dtfft_transpose_t),    intent(in)    :: transpose_type
      integer(int32),   optional, intent(out)   :: error_code
    end subroutine

    type(dtfft_request_t) function dtfft_plan_t%transpose_start(in, out, transpose_type, error_code)
      type(*)                     intent(inout) :: in(..)
      type(*)                     intent(inout) :: out(..)
      type(dtfft_transpose_t),    intent(in)    :: transpose_type
      integer(int32),   optional, intent(out)   :: error_code
    end function

    type(dtfft_request_t) function dtfft_plan_t%transpose_start_ptr(in, out, transpose_type, error_code)
      type(c_ptr)                 intent(in)    :: in
      type(c_ptr)                 intent(in)    :: out
      type(dtfft_transpose_t),    intent(in)    :: transpose_type
      integer(int32),   optional, intent(out)   :: error_code
    end function

    subroutine dtfft_plan_t%transpose_end(request, error_code)
      type(dtfft_request_t),      intent(inout) :: request
      integer(int32),   optional, intent(out)   :: error_code
    end subroutine

  .. code-tab:: c

      dtfft_error_t
      dtfft_transpose(
        dtfft_plan_t plan,
        void *in,
        void *out,
        const dtfft_transpose_t transpose_type);

      dtfft_error_t
      dtfft_transpose_start(
        dtfft_plan_t plan,
        void *in,
        void *out,
        const dtfft_transpose_t transpose_type,
        dtfft_request_t *request);

      dtfft_error_t
      dtfft_transpose_end(
        dtfft_plan_t plan,
        dtfft_request_t request);

  .. code-tab:: c++

      dtfft::Error
      dtfft::Plan::transpose(
          void *in,
          void *out,
          const dtfft::Transpose transpose_type);

      dtfft::Error
      dtfft::Plan::transpose_start(
          void *in,
          void *out,
          const dtfft::Transpose transpose_type,
          dtfft_request_t* request);

      dtfft::Error
      dtfft::Plan::transpose_end(
          dtfft_request_t request);

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

.. note::

  Calling :f:func:`transpose` for R2C plan is not allowed.

**Datatype Backend Version**: When the backend is ``DTFFT_BACKEND_MPI_DATATYPE``, calling :f:func:`transpose` executes a single ``MPI_Ialltoall(w)`` call followed by ``MPI_Wait`` to complete the operation. In contrast :f:func:`transpose_start` initiates an asynchronous ``MPI_Ialltoall(w)`` call, creating the corresponding MPI request and returning a ``dtfft_request_t`` object containing information about the started transposition. In both cases, non-contiguous MPI Datatypes are used, and the ``out`` buffer contains the transposed data once the operation completes, leaving the ``in`` buffer unchanged.

**Generic Version**: Performs a three-step transposition:

- Executes transposition kernel to transpose data locally. On a single process, this completes the task, and control returns to the user.
- Performs data redistribution using the selected backend (e.g., MPI, NCCL). Differences between backends begin at this step (see below for specifics).
- Final step executes data unpacking kernel to rearrange received data into the ``out`` buffer.

In the Generic version, the ``in`` buffer may serve as intermediate storage, potentially modifying its contents, except when operating on a single process, where it remains unchanged.

GPU Backend-Specific Behavior
_____________________________

- **MPI-Based Backends** (:f:var:`DTFFT_BACKEND_MPI_P2P`, :f:var:`DTFFT_BACKEND_MPI_A2A` and :f:var:`DTFFT_BACKEND_MPI_P2P_SCHEDULED`):

  After local transposition, redistributes data using CUDA-aware MPI. Data destined for the same GPU ("self" data) is copied via ``cudaMemcpyAsync``.

  For **MPI Peer-to-Peer** (``MPI_P2P``), it issues non-blocking ``MPI_Irecv`` and ``MPI_Isend`` calls (or ``MPI_Recv_init`` and ``MPI_Send_init`` with ``MPI_Startall`` if built with ``DTFFT_ENABLE_PERSISTENT_COMM``) for point-to-point exchanges between GPUs, completing with ``MPI_Waitall``; an nvRTC kernel then unpacks all data at once.

  For **MPI All-to-All** (``MPI_A2A``), it performs a single ``MPI_Ialltoall(v)`` call (or ``MPI_Alltoall(v)_init`` with ``MPI_Start`` if built with ``DTFFT_ENABLE_PERSISTENT_COMM`` and supported by MPI), completing with ``MPI_Wait``; an nvRTC kernel then unpacks the data.

  For **MPI Peer-to-Peer with explicit scheduling** (``MPI_P2P_SCHEDULED``), it uses a precomputed round-robin schedule to issue blocking MPI_Sendrecv call sequence.

- **Pipelined MPI Peer-to-Peer** (:f:var:`DTFFT_BACKEND_MPI_P2P_PIPELINED`):

  After local transposition, redistributes data similarly to ``MPI_P2P`` using CUDA-aware MPI with non-blocking ``MPI_Irecv`` and ``MPI_Isend`` calls (or ``MPI_Recv_init`` and ``MPI_Send_init`` with ``MPI_Startall`` if built with ``DTFFT_ENABLE_PERSISTENT_COMM``).

  Data destined for the same GPU ("self" data) is copied via ``cudaMemcpyAsync``. Unlike ``MPI_P2P``, as soon as data arrives from a process *i*, it is immediately unpacked by launching an nvRTC kernel specific to that process's data.

  This results in *N* nvRTC kernels (one per process) instead of a single kernel unpacking all data, enabling pipelining of communication and computation to reduce latency.

- **NCCL-Based Backends** (:f:var:`DTFFT_BACKEND_NCCL` and :f:var:`DTFFT_BACKEND_NCCL_PIPELINED`):

  After local transposition, redistributes data using the NCCL library for GPU-to-GPU communication.

  For **NCCL** (``DTFFT_BACKEND_NCCL``), it executes a cycle of ``ncclSend`` and ``ncclRecv`` calls within ``ncclGroupStart`` and ``ncclGroupEnd`` to perform point-to-point exchanges between all processes, including "self" data. Once communication completes, an nvRTC kernel unpacks all data at once, similar to ``MPI_P2P``.

  For **Pipelined NCCL** (:f:var:`DTFFT_BACKEND_NCCL_PIPELINED`), it copies "self" data using ``cudaMemcpyAsync`` and immediately unpacks it with an nvRTC kernel in a parallel stream created by ``dtFFT``. Concurrently, in main stream, it runs the same ``ncclSend`` / ``ncclRecv`` cycle (within ``ncclGroupStart`` and ``ncclGroupEnd``) for data exchange with other processes, excluding "self" data. After communication completes, an nvRTC kernel unpacks the data received from all other processes.

- **cuFFTMp** (:f:var:`DTFFT_BACKEND_CUFFTMP`):

  After local transposition from the ``in`` buffer to the ``out`` buffer using an nvRTC kernel, redistributes data using the cuFFTMp library by calling ``cufftMpExecReshapeAsync``. This function performs an asynchronous all-to-all exchange across multiple GPUs, reshaping the data from the ``out`` buffer back into the ``in`` buffer. Since the final transposed data is required in the ``out`` buffer, it is then copied from ``in`` to ``out`` using ``cudaMemcpyAsync``.

- **Pipelined cuFFTMp** (:f:var:`DTFFT_BACKEND_CUFFTMP_PIPELINED`):

  This backend optimizes the standard ``cuFFTMp`` approach by eliminating the final ``cudaMemcpyAsync`` step. It begins with a local transposition from the ``in`` buffer to an auxiliary (``aux``) buffer using an nvRTC kernel. Then, it calls ``cufftMpExecReshapeAsync`` to perform the all-to-all exchange, reshaping the data directly from the ``aux`` buffer into the final ``out`` buffer. This approach avoids the extra copy required by the standard ``cuFFTMp`` backend, potentially reducing latency, but requires an additional ``aux`` buffer for its operation.


.. note::

  Performance and behavior may vary based on GPU interconnects (e.g., NVLink), MPI implementation, and system configuration. To automatically select the fastest GPU backend for a given system, use the ``DTFFT_PATIENT`` effort level when creating plan, which tests each backend and chooses the most efficient one.

.. note::

  Pipelined backends (:f:var:`DTFFT_BACKEND_MPI_P2P_PIPELINED` and :f:var:`DTFFT_BACKEND_NCCL_PIPELINED`) require an additional ``aux`` buffer, which is managed internally by ``dtFFT`` and inaccessible to the user. Similarly, :f:var:`DTFFT_BACKEND_CUFFTMP` may require an ``aux`` buffer if ``cufftMpGetReshapeSize`` returns a value greater than 0, such as when the environment variable ``CUFFT_RESHAPE_USE_PACKING=1`` is set.

  In all other cases, transposition requires only the ``in`` and ``out`` buffers.

.. note::

  Host version of MPI-based backends does practically the same as GPU version, but uses host memory buffers and performs transpositions and data unpacking using precompiled host kernels.

Example
_______

Below is an example of transposing data from X to Y and back:

.. tabs::

  .. code-tab:: fortran

    ! Assuming plan is created and buffers `a` and `b` are allocated.
    call plan%transpose(a, b, DTFFT_TRANSPOSE_X_TO_Y, error_code)
    DTFFT_CHECK(error_code)  ! Checks for errors

    ! Process Y-aligned data in buffer `b`
    ! ... (e.g., apply scaling or analysis)

    ! Reverse transposition
    call plan%transpose(b, a, DTFFT_TRANSPOSE_Y_TO_X, error_code)
    DTFFT_CHECK(error_code)

    ! Alternatively, using pointers of type c_ptr
    call plan%transpose_ptr(a_ptr, b_ptr, DTFFT_TRANSPOSE_X_TO_Y, error_code)
    DTFFT_CHECK(error_code)

    ! ...

    call plan%transpose_ptr(b_ptr, a_ptr, DTFFT_TRANSPOSE_Y_TO_X, error_code)
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
    DTFFT_CXX_CALL( plan.transpose(a, b, dtfft::Transpose::X_TO_Y) )

    // Process Y-aligned data in buffer `b`
    // ... (e.g., apply scaling or analysis)

    // Reverse transposition
    DTFFT_CXX_CALL( plan.transpose(b, a, dtfft::Transpose::Y_TO_X) )

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
      type(dtfft_execute_t),      intent(in)    :: execute_type
      type(*),          optional, intent(inout) :: aux(..)
      integer(int32),   optional, intent(out)   :: error_code
    end subroutine

    subroutine dtfft_plan_t%execute_ptr(in, out, execute_type, aux, error_code)
      type(c_ptr)                 intent(in)    :: in
      type(c_ptr)                 intent(in)    :: out
      type(dtfft_execute_t),      intent(in)    :: execute_type
      type(c_ptr)                 intent(in)    :: aux
      integer(int32),   optional, intent(out)   :: error_code
    end subroutine

  .. code-tab:: c

      dtfft_error_t
      dtfft_execute(
        dtfft_plan_t plan,
        void *in,
        void *out,
        const dtfft_execute_t execute_type,
        void *aux);

  .. code-tab:: c++

      dtfft::Error
      dtfft::Plan::execute(
          void *in,
          void *out,
          const dtfft::Execute execute_type,
          void *aux=nullptr);

      template<typename Tr>
      Tr *
      dtfft::Plan::execute(
          void *inout, 
          const Execute execute_type, 
          void *aux=nullptr);

      template<typename T, typename Tr = T>
      Tr *
      dtfft::Plan::execute(
          T *inout, 
          const dtfft::Execute execute_type, 
          void *aux=nullptr);

      dtfft::Error
      dtfft::Plan::forward(
          void *in, 
          void *out, 
          void *aux);

      template<typename Tr>
      Tr *
      dtfft::Plan::forward(
          void *inout, 
          void *aux=nullptr);

      template<typename T, typename Tr = T>
      Tr *
      dtfft::Plan::forward(
          T *inout, 
          void *aux=nullptr);

      dtfft::Error
      dtfft::Plan::backward(
          void *in, 
          void *out, 
          void *aux);

      template<typename Tr>
      Tr *
      dtfft::Plan::backward(
          void *inout, 
          void *aux=nullptr);

      template<typename T, typename Tr = T>
      Tr *
      dtfft::Plan::backward(
          T *inout, 
          void *aux=nullptr);

Description
___________

This method executes a plan, performing transpositions and optionally FFTs based on the specified ``execute_type``. It supports in-place execution; the same pointer can be safely passed to both ``in`` and ``out``. To optimize memory usage, ``dtFFT`` uses the ``in`` buffer as intermediate storage, overwriting its contents. Users needing to preserve original data should copy it elsewhere.

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
- If ``Transpose-Only`` with Y-slab and distinct ``in`` and ``out``:

  - Transpose from X to Y
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
- If using FFT with Y-slab:

  - Forward FFT in X direction
  - Transpose from X to Y
  - Forward 2D FFT in Y-Z directions

**Backward Execution** (``DTFFT_EXECUTE_BACKWARD``):

- If ``Transpose-Only``:

  - Transpose from Z to Y
  - Transpose from Y to X
- If ``Transpose-Only`` with Z-slab and distinct ``in`` and ``out``:

  - Transpose from Z to X
- If ``Transpose-Only`` with Y-slab and distinct ``in`` and ``out``:

  - Transpose from Y to X
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
- If using FFT with Y-slab:

  - Backward 2D FFT in Y-Z directions
  - Transpose from Y to X
  - Backward FFT in X direction

.. note::

  For ``Transpose-Only`` plans with a Z-slab and identical ``in`` and ``out`` pointers, execution uses a
  two-step transposition, as direct transposition is not possible with a single pointer.

.. note::

  There are two cases when in-place execution is not allowed:

    - 2D ``Transpose-Only`` plan
    - 3D ``Transpose-Only`` with Y-slab optimization enabled.

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

    ! Alternatively, using pointers of type c_ptr. If aux is not needed, pass c_null_ptr
    call plan%execute_ptr(a_ptr, b_ptr, DTFFT_EXECUTE_FORWARD, aux_ptr, error_code)
    DTFFT_CHECK(error_code)

    ! ...

    call plan%execute_ptr(b_ptr, a_ptr, DTFFT_EXECUTE_BACKWARD, c_null_ptr, error_code)
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
    DTFFT_CXX_CALL( plan.execute(a, b, dtfft::Execute::FORWARD, aux) )

    // Process Fourier-space data in buffer `b`
    // ... (e.g., apply filtering)

    // Backward execution
    DTFFT_CXX_CALL( plan.execute(b, a, dtfft::Execute::BACKWARD, aux) )

GPU Notes
---------

Both ``transpose`` and ``execute`` in the GPU version operate asynchronously. When either function returns, computations are queued in a CUDA stream but may not be complete. Full synchronization with the host requires calling ``cudaDeviceSynchronize``, ``cudaStreamSynchronize``, or ``!$acc wait`` (for OpenACC).

During execution, ``dtFFT`` may use multiple CUDA streams, but the final computation stage always occurs in the stream returned by :f:func:`get_stream`. Thus, synchronization may be unnecessary if users submit additional kernels to that stream.

Plan Finalization
=================

To fully release all memory resources allocated by ``dtFFT`` for a plan, the plan must be explicitly destroyed. This ensures that all internal buffers and resources associated with the plan are freed.

.. note::
   If buffers were allocated using :f:func:`mem_alloc`, they must be deallocated with :f:func:`mem_free` *before* calling the destroy method. Failing to do so may result in memory leaks or undefined behavior.

Example
-------

Below is an example of properly finalizing a plan and freeing allocated memory:

.. tabs::

  .. code-tab:: fortran

    ! Assuming a plan and buffers ``a``, ``b`` and ``aux`` are created and allocated with ``mem_alloc``
    call plan%mem_free(a, error_code);   DTFFT_CHECK(error_code)
    call plan%mem_free(b, error_code);   DTFFT_CHECK(error_code)
    call plan%mem_free(aux, error_code); DTFFT_CHECK(error_code)

    ! Pointers allocated via mem_alloc_ptr must be freed with ``mem_free_ptr``
    call plan%mem_free_ptr(a_ptr, error_code);   DTFFT_CHECK(error_code)
    call plan%mem_free_ptr(b_ptr, error_code);   DTFFT_CHECK(error_code)
    call plan%mem_free_ptr(aux_ptr, error_code); DTFFT_CHECK(error_code)

    call plan%destroy(error_code)            ! Destroy the plan
    DTFFT_CHECK(error_code)

  .. code-tab:: c

    // Assuming a plan and buffers ``a``, ``b`` and ``aux`` are created and allocated with `dtfft_mem_alloc`
    DTFFT_CALL( dtfft_mem_free(plan, a) )   // Free buffer ``a``
    DTFFT_CALL( dtfft_mem_free(plan, b) )   // Free buffer ``b``
    DTFFT_CALL( dtfft_mem_free(plan, aux) ) // Free buffer ``aux``
    DTFFT_CALL( dtfft_destroy(&plan) )      // Destroy the plan

  .. code-tab:: c++

    // Assuming a plan and buffers ``a``, ``b`` and ``aux`` are created and allocated with `mem_alloc`
    DTFFT_CXX_CALL( plan.mem_free(a) )    // Free buffer ``a``
    DTFFT_CXX_CALL( plan.mem_free(b) )    // Free buffer ``b``
    DTFFT_CXX_CALL( plan.mem_free(aux) )  // Free buffer ``aux``
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

      ! Allocate memory
      call plan%mem_alloc(alloc_size, a, error_code); DTFFT_CHECK(error_code)
      call plan%mem_alloc(alloc_size, b, error_code); DTFFT_CHECK(error_code)
      call plan%mem_alloc(alloc_size, aux, error_code); DTFFT_CHECK(error_code)

      ! Forward execution
      call plan%execute(a, b, DTFFT_EXECUTE_FORWARD, aux, error_code)
      DTFFT_CHECK(error_code)

      ! Process Fourier-space data in buffer `b` (e.g., apply filtering)
      ! ...

      ! Backward execution
      call plan%execute(b, a, DTFFT_EXECUTE_BACKWARD, aux, error_code)
      DTFFT_CHECK(error_code)

      ! Free memory
      call plan%mem_free(a, error_code); DTFFT_CHECK(error_code)
      call plan%mem_free(b, error_code); DTFFT_CHECK(error_code)
      call plan%mem_free(aux, error_code); DTFFT_CHECK(error_code)

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
      DTFFT_CALL( dtfft_set_config(&config) );

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
      DTFFT_CXX_CALL( plan.execute(a, b, Execute::FORWARD, aux) );

      // Process Fourier-space data in buffer `b` (e.g., apply filtering)
      // ...

      // Backward execution
      DTFFT_CXX_CALL( plan.execute(b, a, Execute::BACKWARD, aux) );

      // Free memory
      DTFFT_CXX_CALL( plan.mem_free(a) );
      DTFFT_CXX_CALL( plan.mem_free(b) );
      DTFFT_CXX_CALL( plan.mem_free(aux) );

      // Explicitly destroy the plan
      DTFFT_CXX_CALL( plan.destroy() );

      MPI_Finalize();
      return 0;
    }