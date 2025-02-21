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