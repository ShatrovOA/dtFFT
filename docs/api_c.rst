:tocdepth: 3

.. _c_link:

###############
C API Reference
###############

This page describes all types, functions and macros available in ``dtFFT`` C API.
In order to use them user have to ``#include <dtfft.h>``.

.. note::
  Not all of the API listed below can be accessible in runtime.
  For example :cpp:enum:`dtfft_platform_t` can only be used if ``dtFFT`` compiled with CUDA support.

Predefined Macros
=================

.. doxygendefine:: DTFFT_VERSION_MAJOR
.. doxygendefine:: DTFFT_VERSION_MINOR
.. doxygendefine:: DTFFT_VERSION_PATCH
.. doxygendefine:: DTFFT_VERSION_CODE
.. doxygendefine:: DTFFT_VERSION
.. doxygendefine:: DTFFT_CALL


Enumerators
===========

.. doxygenenum:: dtfft_error_t

---------

.. doxygenenum:: dtfft_execute_t

---------

.. doxygenenum:: dtfft_transpose_t

---------

.. doxygenenum:: dtfft_precision_t

---------

.. doxygenenum:: dtfft_effort_t

---------

.. doxygenenum:: dtfft_executor_t

---------

.. doxygenenum:: dtfft_r2r_kind_t

---------

.. doxygenenum:: dtfft_backend_t

---------

.. doxygenenum:: dtfft_transpose_mode_t

---------

.. doxygenenum:: dtfft_access_mode_t

---------

.. doxygenenum:: dtfft_platform_t

---------

.. doxygenenum:: dtfft_reshape_t

---------

.. doxygenenum:: dtfft_layout_t

---------

.. doxygenenum:: dtfft_compression_lib_t

---------

.. doxygenenum:: dtfft_compression_mode_t


Types
=====

.. doxygentypedef:: dtfft_plan_t

---------

.. doxygenstruct:: dtfft_pencil_t
  :members:

---------

.. doxygenstruct:: dtfft_config_t
  :members:

---------

.. doxygentypedef:: dtfft_stream_t

---------

.. doxygentypedef:: dtfft_request_t

---------

.. doxygenstruct:: dtfft_compression_config_t
  :members:

Functions
=========

.. doxygenfunction:: dtfft_get_version

---------

.. doxygenfunction:: dtfft_get_error_string

---------

.. doxygenfunction:: dtfft_get_backend_string

---------

.. doxygenfunction:: dtfft_get_precision_string

---------

.. doxygenfunction:: dtfft_get_executor_string

---------

.. doxygenfunction:: dtfft_create_config

---------

.. doxygenfunction:: dtfft_set_config

---------

.. doxygenfunction:: dtfft_get_backend_pipelined

Plan constructors
======================

All plan constructors must be called after ``MPI_Init``. Plan must be destroyed before call to ``MPI_Finalize``.

.. doxygenfunction:: dtfft_create_plan_r2r

---------

.. doxygenfunction:: dtfft_create_plan_r2r_pencil

---------

.. doxygenfunction:: dtfft_create_plan_c2c

---------

.. doxygenfunction:: dtfft_create_plan_c2c_pencil

---------

.. doxygenfunction:: dtfft_create_plan_r2c

---------

.. doxygenfunction:: dtfft_create_plan_r2c_pencil

Plan destructor
======================

.. doxygenfunction:: dtfft_destroy

Memory allocation
======================

.. doxygenfunction:: dtfft_mem_alloc

---------

.. doxygenfunction:: dtfft_mem_free

Plan execution
======================

.. doxygenfunction:: dtfft_execute

---------

.. doxygenfunction:: dtfft_transpose

---------

.. doxygenfunction:: dtfft_transpose_start

---------

.. doxygenfunction:: dtfft_transpose_end

---------

.. doxygenfunction:: dtfft_reshape

---------

.. doxygenfunction:: dtfft_reshape_start

---------

.. doxygenfunction:: dtfft_reshape_end

Saving and loading plans
========================

A plan created with a high ``effort`` level spends most of its creation time benchmarking.
:c:func:`dtfft_save_plan` writes those results to a file, and :c:func:`dtfft_load_plan`
rebuilds the very same plan from it without measuring anything.

The file records the number of MPI processes and refuses to load on a different one. It
does **not** record how ranks were placed across nodes and devices: keeping that identical
between runs is the caller's responsibility. A plan replayed onto a different placement
stays correct, but the choices in it were measured for a layout that no longer holds, so it
may no longer be the fastest one.

A file also carries the layout version of the format itself. A later ``dtFFT`` reads a file
written by an earlier one as long as it still understands that layout; only a file newer than
the reader, or one whose layout has been retired, is refused. The library version and build
configuration are recorded too, but for diagnostics -- what is actually enforced is that every
backend named in the file exists in this build, since which backends are compiled in varies.

.. doxygenfunction:: dtfft_save_plan

---------

.. doxygenfunction:: dtfft_load_plan

Plan information
======================

.. doxygenfunction:: dtfft_report

---------

.. doxygenfunction:: dtfft_report_compression

---------

.. doxygenfunction:: dtfft_get_local_sizes

---------

.. doxygenfunction:: dtfft_get_alloc_size

---------

.. doxygenfunction:: dtfft_get_alloc_bytes

---------

.. doxygenfunction:: dtfft_get_aux_size

---------

.. doxygenfunction:: dtfft_get_aux_bytes

---------

.. doxygenfunction:: dtfft_get_aux_size_reshape

---------

.. doxygenfunction:: dtfft_get_aux_bytes_reshape

---------

.. doxygenfunction:: dtfft_get_aux_size_transpose

---------

.. doxygenfunction:: dtfft_get_aux_bytes_transpose

---------

.. doxygenfunction:: dtfft_get_element_size

---------

.. doxygenfunction:: dtfft_get_pencil

---------

.. doxygenfunction:: dtfft_get_z_slab_enabled

---------

.. doxygenfunction:: dtfft_get_y_slab_enabled

---------

.. doxygenfunction:: dtfft_get_stream

---------

.. doxygenfunction:: dtfft_get_backend

---------

.. doxygenfunction:: dtfft_get_reshape_backend

---------

.. doxygenfunction:: dtfft_get_platform

---------

.. doxygenfunction:: dtfft_get_executor

---------

.. doxygenfunction:: dtfft_get_precision

---------

.. doxygenfunction:: dtfft_get_dims

---------

.. doxygenfunction:: dtfft_get_grid_dims
