.. _c_link:

###############
C API Reference
###############

This page describes all types, functions and macros available in ``dtFFT`` C API. 
In order to use them user have to ``#include <dtfft.h>``.

.. note::
  Not all of the API listed below can be accessible in runtime.
  For example :cpp:func:`dtfft_create_plan_r2c` can only be used if ``dtFFT`` compiled with any FFT

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

.. doxygenenum:: dtfft_error_code_t

---------

.. doxygenenum:: dtfft_execute_type_t

---------

.. doxygenenum:: dtfft_transpose_type_t

---------

.. doxygenenum:: dtfft_precision_t

---------

.. doxygenenum:: dtfft_effort_t

---------

.. doxygenenum:: dtfft_executor_t

---------

.. doxygenenum:: dtfft_r2r_kind_t

---------

.. doxygenenum:: dtfft_gpu_backend_t

---------

.. doxygenenum:: dtfft_platform_t


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

Functions
=========

.. doxygenfunction:: dtfft_get_version

---------

.. doxygenfunction:: dtfft_get_error_string

---------

.. doxygenfunction:: dtfft_get_gpu_backend_string

---------

.. doxygenfunction:: dtfft_create_config

---------

.. doxygenfunction:: dtfft_set_config


Plan constructors
======================

All plan constructors must be called after ``MPI_Init``. Plan must be destroyed before call to ``MPI_Finalize``.

.. doxygenfunction:: dtfft_create_plan_r2r

---------

.. doxygenfunction:: dtfft_create_plan_c2c

---------

.. doxygenfunction:: dtfft_create_plan_r2c

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

Plan information
======================

.. doxygenfunction:: dtfft_report

---------

.. doxygenfunction:: dtfft_get_local_sizes

---------

.. doxygenfunction:: dtfft_get_alloc_size

---------

.. doxygenfunction:: dtfft_get_element_size

---------

.. doxygenfunction:: dtfft_get_pencil

---------

.. doxygenfunction:: dtfft_get_z_slab_enabled

---------

.. doxygenfunction:: dtfft_get_stream

---------

.. doxygenfunction:: dtfft_get_gpu_backend

---------

.. doxygenfunction:: dtfft_get_platform