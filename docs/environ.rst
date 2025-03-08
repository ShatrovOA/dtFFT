.. _environ_link:

#####################
Environment Variables
#####################

This page lists all environment variables that can modify ``dtFFT`` behavior at runtime, offering users granular control over logging, performance measurement, and data transposition strategies.

DTFFT_ENABLE_LOG
================

Enables logging within the ``dtFFT`` library. By default, the library runs silently with no output. When enabled, it provides detailed insights into internal processes, aiding analysis and debugging.

Purpose
-------

Logging enables monitoring of key operations, including:

- **Z-Slab Usage**: Indicates whether the plan utilizes Z-slab optimization.
- **Selected Datatype IDs**: Reports the :ref:`datatype IDs<datatype_selection>` or GPU backend chosen during autotuning when ``effort`` is :f:var:`DTFFT_PATIENT`.
- **Execution Times During Autotune**: Logs timing data for autotuning stages.
- **Detected Input Errors**: Highlights errors from invalid user input for easier diagnosis.

Accepted Values
---------------

- **Type**: Integer
- **Values**: ``0`` (disabled), ``1`` (enabled)
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

Single measurements may be inconsistent due to system noise or cache effects. By repeating transpositions, ``dtFFT`` averages performance data, ensuring robust selection of optimal GPU backends or MPI datatypes during autotuning.

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

.. _dtfft_gpu_backend_env:

DTFFT_GPU_BACKEND
=================

Specifies the GPU backend used by ``dtFFT`` for data transposition and communication when executing plans on a CUDA device. 
This environment variable allows users to override the backend selected through the ``dtfft_config_t`` structure, 
taking precedence over API configuration.

Purpose
-------

The ``DTFFT_GPU_BACKEND`` variable enables users to select a specific GPU backend for optimizing data movement and computation in ``dtFFT`` plans. Different backends offer varying performance characteristics depending on the system configuration, workload, and MPI implementation, allowing fine-tuned control over GPU execution without modifying code.

Accepted Values
---------------

- **Type**: String
- **Supported Values**:

  - ``mpi_dt``: Backend using MPI datatypes.
  - ``mpi_p2p``: MPI peer-to-peer backend.
  - ``mpi_a2a``: MPI backend using ``MPI_Alltoallv``.
  - ``mpi_p2p_pipe``: Pipelined MPI peer-to-peer backend with overlapping data copying and unpacking.
  - ``nccl``: NCCL backend.
  - ``nccl_pipe``: Pipelined NCCL backend with overlapping data copying and unpacking.
  - ``cufftmp``: cuFFTMp backend.
  
- **Default**: ``nccl`` if NCCL is available in the library build; otherwise, ``mpi_p2p``.

.. note::
   - Case-insensitive (e.g., ``MPI_DT`` is equivalent to ``mpi_dt``).
   - Only applicable in builds with CUDA support (``DTFFT_WITH_CUDA`` defined) and when the execution platform is set 
     to ``cuda`` (via :ref:`DTFFT_PLATFORM<dtfft_platform_env>` or :f:type:`dtfft_config_t`).
   - If an unsupported value is provided, it is silently ignored, and the default backend (``nccl`` or ``mpi_p2p``, depending on build) is used.
   - Availability of some backends (e.g., ``nccl``, ``nccl_pipe``, ``cufftmp``) depends on additional library 
     support (e.g., NCCL, cuFFTMp) during compilation.



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
- **Values**: ``1`` (Method 1), ``2`` (Method 2)

DTFFT_DTYPE_X_Y
_______________

Controls datatype construction for X-to-Y transposition.
- **Default**: ``2``

DTFFT_DTYPE_Y_Z
_______________

Controls datatype construction for Y-to-Z transposition.
- **Default**: ``2``

DTFFT_DTYPE_X_Z
_______________

Controls datatype construction for X-to-Z transposition.
- **Default**: ``2``

DTFFT_DTYPE_Y_X
_______________

Controls datatype construction for Y-to-X transposition.
- **Default**: ``2``

DTFFT_DTYPE_Z_Y
_______________

Controls datatype construction for Z-to-Y transposition.
- **Default**: ``2``

DTFFT_DTYPE_Z_X
_______________

Controls datatype construction for Z-to-X transposition.
- **Default**: ``2``