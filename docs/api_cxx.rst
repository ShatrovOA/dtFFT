.. _cpp_link:

#################
C++ API Reference
#################

This page describes all classes, enumerators and functions available in ``dtFFT`` C++ API.
In order to use them user have to ``#include <dtfft.hpp>``. All API is contained within ``dtfft`` namespace.


Predefined Macros
=================

.. doxygendefine:: DTFFT_CXX_CALL

Enumerators
===========

.. doxygenenum:: dtfft::Error

.. doxygenenum:: dtfft::Execute

.. doxygenenum:: dtfft::Transpose

.. doxygenenum:: dtfft::Precision

.. doxygenenum:: dtfft::Effort

.. doxygenenum:: dtfft::Executor

.. doxygenenum:: dtfft::R2RKind

.. doxygenenum:: dtfft::Backend

.. doxygenenum:: dtfft::Platform

.. doxygenenum:: dtfft::Layout

.. doxygenenum:: dtfft::Reshape

.. doxygenenum:: dtfft::TransposeMode

.. doxygenenum:: dtfft::AccessMode

.. doxygenenum:: dtfft::CompressionLib

.. doxygenenum:: dtfft::CompressionMode

Functions
=========

.. doxygenfunction:: dtfft::get_backend_string

.. doxygenfunction:: dtfft::get_error_string

.. doxygenfunction:: dtfft::get_precision_string

.. doxygenfunction:: dtfft::get_executor_string

.. doxygenfunction:: dtfft::set_config

.. doxygenfunction:: dtfft::get_backend_pipelined

Structs
=======

.. doxygenstruct:: dtfft::Version
  :members:

.. doxygenstruct:: dtfft::Pencil
  :members:

.. doxygenstruct:: dtfft::Config
  :members:

.. doxygenstruct:: dtfft::CompressionConfig
  :members:

Classes
=======

.. doxygenclass:: dtfft::Exception
  :members:

.. doxygenclass:: dtfft::Plan
  :members:

.. doxygenclass:: dtfft::PlanC2C
  :members:

.. doxygenclass:: dtfft::PlanR2C
  :members:

.. doxygenclass:: dtfft::PlanR2R
  :members:

Saving and loading plans
========================

``Plan::save`` writes the results of autotuning to a file, and the from-file constructors of
:cpp:class:`dtfft::PlanC2C`, :cpp:class:`dtfft::PlanR2C` and :cpp:class:`dtfft::PlanR2R`
rebuild the very same plan from it without measuring anything.

The C++ plan classes are thin wrappers around an opaque handle and carry no state of their
own, so any of them can load any plan file: the plan that gets built is the one the file
describes, whichever wrapper was used to open it.

The file records the number of MPI processes and refuses to load on a different one. It does
**not** record how ranks were placed across nodes and devices: keeping that identical between
runs is the caller's responsibility. A plan replayed onto a different placement stays correct,
but the choices in it were measured for a layout that no longer holds.
