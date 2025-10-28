.. _environ_link:

#####################
Environment Variables
#####################

This page lists all environment variables that can modify ``dtFFT`` behavior at runtime, providing users with granular control over logging, performance measurement, and data transposition strategies.

Most of these variables override settings specified in the :f:type:`dtfft_config_t` structure, allowing users to adjust configurations without modifying code.

.. _dtfft_enable_log_env:

DTFFT_ENABLE_LOG
================

Enables logging within the ``dtFFT`` library. By default, the library runs silently with no output. When enabled, it provides detailed insights into internal processes, aiding analysis and debugging.

Purpose
-------

Logging enables monitoring of key operations, including:

- **Selected Datatype IDs**: Reports the :ref:`datatype IDs<datatype_selection>` or GPU backend chosen during autotuning when ``effort`` is :f:var:`DTFFT_PATIENT`.
- **Execution Times During Autotune**: Logs timing data for autotuning stages.

Accepted Values
---------------

- **Type**: Integer
- **Supported Values:**:

  - ``0`` (disabled)
  - ``1`` (enabled)

- **Default**: ``0``

DTFFT_MEASURE_WARMUP_ITERS
==========================

Defines the number of warmup iterations performed when ``effort`` exceeds :f:var:`DTFFT_ESTIMATE`. Warmup iterations ensure stable performance measurements in parallel environments.

Purpose
-------

Warmup iterations stabilize system performance by preloading caches, establishing communication channels, and mitigating initial overhead. This is crucial for accurate benchmarking, especially in distributed setups, preventing skewed results from cold starts.

Accepted Values
---------------

- **Type**: Non-negative integer
- **Recommended Range**: 2–10 (values > 0 are advised for reliable results)
- **Default**: ``2``

DTFFT_MEASURE_ITERS
===================

Specifies the number of measurement iterations for transposition and data exchange when ``effort`` exceeds :f:var:`DTFFT_ESTIMATE`. Multiple iterations enhance measurement reliability.

Purpose
-------

Single measurements may be inconsistent due to system noise or cache effects. By repeating transpositions, ``dtFFT`` averages performance data, ensuring robust selection of optimal backends or MPI datatypes during autotuning.

Accepted Values
---------------

- **Type**: Positive integer
- **Recommended Range**: 5–20 (values > 1 balance accuracy and runtime)
- **Default**: ``5``

.. _dtfft_platform_env:

DTFFT_PLATFORM
==============

Specifies the execution platform for ``dtFFT`` plans.
This environment variable allows users to override the platform set via the ``dtfft_config_t`` structure,
taking precedence over API configuration.

Purpose
-------

The ``DTFFT_PLATFORM`` variable provides a flexible way to control whether ``dtFFT`` executes on the host (CPU) or a CUDA-enabled GPU
without modifying code or API calls. It ensures that runtime platform selection aligns with user preferences or system capabilities,
prioritizing environment settings over programmatic defaults.

Accepted Values
---------------

- **Type**: String
- **Supported Values**:

  - ``host``: Execute on the host (CPU).
  - ``cuda``: Execute on a CUDA device (GPU).

- **Default**: ``host``

.. note::
   - Case-insensitive (e.g., ``HOST`` is equivalent to ``host``).
   - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined). In non-CUDA builds, it is ignored, and execution
     defaults to the host.
   - If an unsupported value is provided, it is silently ignored, and the default (``host``) is used.

DTFFT_BACKEND
=============

Specifies the backend used by ``dtFFT`` for data transposition and communication when executing plans.
This environment variable allows users to override the backend selected through the ``dtfft_config_t`` structure,
taking precedence over API configuration.

Purpose
-------

The ``DTFFT_BACKEND`` variable enables users to select a specific backend for optimizing data movement and computation in ``dtFFT`` plans.
Different backends offer varying performance characteristics depending on the system configuration, workload, and MPI implementation,
allowing fine-tuned control over execution without modifying code.

Accepted Values
---------------

- **Type**: String
- **Supported Values**:

  - ``mpi_dt``: Backend using MPI datatypes.
  - ``mpi_p2p``: MPI peer-to-peer backend.
  - ``mpi_a2a``: MPI backend using ``MPI_Alltoallv``.
  - ``mpi_p2p_pipe``: Pipelined MPI peer-to-peer backend with overlapping data copying and unpacking.
  - ``mpi_rma``: MPI RMA backend that uses MPI_Rget for data transfers.
  - ``mpi_rma_pipe``: Pipelined MPI RMA backend with overlapping data copying and unpacking.
  - ``nccl``: NCCL backend.
  - ``nccl_pipe``: Pipelined NCCL backend with overlapping data copying and unpacking.
  - ``cufftmp``: cuFFTMp backend.
  - ``cufftmp_pipe``: cuFFTMp backend that uses additional buffer to avoid extra copy and gain performance.

- **Default**: When built with CUDA Support: ``nccl`` if NCCL is available in the library build; otherwise, ``mpi_p2p``.
  When built without CUDA Support: ``mpi_dt``.

.. note::
   - Case-insensitive (e.g., ``MPI_DT`` is equivalent to ``mpi_dt``).
   - If an unsupported value is provided, it is silently ignored, and the default backend (``nccl`` or ``mpi_p2p``, depending on build) is used.
   - Availability of some backends (e.g., ``nccl``, ``cufftmp``) depends on additional library
     support (e.g., NCCL, cuFFTMp) during compilation.

.. _dtfft_nccl_buffer_register_env:

DTFFT_NCCL_BUFFER_REGISTER
==========================

Specifies whether to enable buffer registration for NCCL operations.
When enabled, NCCL buffers are registered, which can improve performance for certain workloads.

Purpose
-------

Buffer registration can reduce the overhead of memory operations in NCCL by pre-registering memory regions.
This is particularly useful for workloads with repeated communication patterns. However, in some cases, disabling registration may
be beneficial, depending on the specific system configuration or workload characteristics.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable NCCL buffer registration.
  - ``1``: Enable NCCL buffer registration.

- **Default**: ``1``

.. _dtfft_enable_z_slab_env:

DTFFT_ENABLE_Z_SLAB
===================

Specifies whether to enable Z-slab optimization for ``dtFFT`` plans.
When enabled, Z-slab optimization reduces network data transfers by employing a two-dimensional FFT algorithm.

Purpose
-------

Z-slab optimization is designed to improve performance for plans decomposed as ``NX × NY × NZ / P``.
Disabling it may resolve issues like :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` or improve performance if the underlying 2D FFT implementation is suboptimal.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable Z-slab optimization.
  - ``1``: Enable Z-slab optimization.

- **Default**: ``1``

DTFFT_ENABLE_Y_SLAB
===================

Specifies whether to enable Y-slab optimization for ``dtFFT`` plans.
When enabled, Y-slab optimization reduces network data transfers by employing a two-dimensional FFT algorithm.

Purpose
-------

Y-slab optimization is designed to improve performance for plans decomposed as ``NX × NY / P × NZ``.
Disabling it may resolve issues like :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` or improve performance if the underlying 2D FFT implementation is suboptimal.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable Y-slab optimization.
  - ``1``: Enable Y-slab optimization.

- **Default**: ``0``

DTFFT_ENABLE_MPI_DT
===================

Specifies whether to enable MPI datatype backend when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, the MPI datatype backend is tested during autotuning.

Purpose
-------

The MPI datatype backend is a simple and robust method for data transposition using MPI derived datatypes.
However, it may not be the most efficient option for large-scale systems or specific data layouts.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable MPI datatype backend.
  - ``1``: Enable MPI datatype backend.

- **Default**: ``1``

.. _dtfft_enable_mpi_env:

DTFFT_ENABLE_MPI
================

Specifies whether to enable MPI-based backends for ``dtFFT`` when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, MPI backends (e.g., MPI P2P) are tested during autotuning.

Purpose
-------

The following applies only to CUDA builds: 
MPI backends are useful for distributed GPU systems but may cause GPU memory leaks in certain OpenMPI versions.
Disabling this option can prevent such issues.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable MPI-based backends.
  - ``1``: Enable MPI-based backends.

- **Default**: ``0``

.. _dtfft_enable_nccl:

DTFFT_ENABLE_NCCL
=================

Specifies whether to enable NCCL backends when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, NCCL backends are tested during autotuning.

Purpose
-------

NCCL backends are optimized for GPU-to-GPU communication and can significantly improve performance in multi-GPU systems.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable NCCL backends.
  - ``1``: Enable NCCL backends.

- **Default**: ``1``

.. note::

  - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set
    to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).

.. _dtfft_enable_nvshmem:

DTFFT_ENABLE_NVSHMEM
====================

Specifies whether to enable NVSHMEM backends when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, NVSHMEM backends are tested during autotuning.

Purpose
-------

NVSHMEM backends provide efficient communication for GPU clusters, leveraging shared memory capabilities.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable NVSHMEM backends.
  - ``1``: Enable NVSHMEM backends.

- **Default**: ``1``

.. note::

  - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set
    to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).


DTFFT_ENABLE_PIPE
=================

Specifies whether to enable pipelined backends when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, pipelined backends (e.g., overlapping data copy and unpack) are tested during autotuning.

Purpose
-------

Pipelined backends improve performance by overlapping communication and computation, but they require additional internal buffers.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable pipelined backends.
  - ``1``: Enable pipelined backends.

- **Default**: ``1``


.. _enable_kernel_optimization:

DTFFT_ENABLE_KERNEL_OPTIMIZATION
================================

Specifies whether to enable transposition kernels optimizations when ``effort`` is :f:var:`DTFFT_PATIENT`.
When enabled, optimized CUDA kernels are used for data transposition on GPUs.

Purpose
-------

Kernel optimizations can significantly improve performance for various data layouts and sizes.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Disable kernel optimizations.
  - ``1``: Enable kernel optimizations.

- **Default**: ``1``


.. note::

  - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set
    to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).


DTFFT_CONFIGS_TO_TEST
=====================

Specifies number of kernel configurations to test when effort is :f:var:`DTFFT_PATIENT` and kernel optimizations are enabled.
This variable allows users to control the extent of autotuning for kernel optimizations.

Purpose
-------

Testing multiple configurations helps identify the best-performing kernel for specific data layouts and sizes.

Accepted Values
---------------

- **Type**: Positive integer
- **Recommended Range**: 3–10 (higher values increase tuning time but may yield better performance. Theoretical maximum is 25)
- **Default**: ``5``

.. note::

  - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set
    to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).
  - Setting this variable to zero or one disables kernel optimizations, equivalent to setting
    :ref:`DTFFT_ENABLE_KERNEL_OPTIMIZATION<enable_kernel_optimization>` to ``0``.


DTFFT_FORCE_KERNEL_OPTIMIZATION
===============================

Forces to run kernel optimizations when effort is NOT :f:var:`DTFFT_PATIENT`.

Purpose
-------

Since kernel optimization is performed without data transfers, the overall autotuning time increase should not be significant.

Accepted Values
---------------

- **Type**: Integer
- **Accepted Values**:

  - ``0``: Do not force kernel optimizations.
  - ``1``: Force kernel optimizations.

- **Default**: ``0``

.. note::

  - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set
    to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).


.. _datatype_selection:

MPI Datatype Selection Variables
================================

These environment variables control how MPI derived datatypes are constructed for global data transpositions in the host version of ``dtFFT``. They apply only when ``effort`` is :f:var:`DTFFT_ESTIMATE` or :f:var:`DTFFT_MEASURE`; for :f:var:`DTFFT_PATIENT`, the library autotunes the best datatype automatically.

Purpose
-------

MPI derived datatypes define the memory layout for data exchanged between processes during transposition. Two construction methods are supported:

- **Method 1** (``1``): Contiguous send datatype with sparse receive datatype.
- **Method 2** (``2``): Sparse send datatype with contiguous receive datatype (default).

These variables allow manual selection based on data characteristics or system requirements.

Accepted Values
---------------

- **Type**: Integer
- **Values**:

  - ``1`` (Method 1)
  - ``2`` (Method 2)

DTFFT_DTYPE_X_Y
_______________

Controls datatype construction for X-to-Y transposition.

DTFFT_DTYPE_Y_Z
_______________

Controls datatype construction for Y-to-Z transposition.

DTFFT_DTYPE_X_Z
_______________

Controls datatype construction for X-to-Z transposition.

DTFFT_DTYPE_Y_X
_______________

Controls datatype construction for Y-to-X transposition.

DTFFT_DTYPE_Z_Y
_______________

Controls datatype construction for Z-to-Y transposition.

DTFFT_DTYPE_Z_X
_______________

Controls datatype construction for Z-to-X transposition.
