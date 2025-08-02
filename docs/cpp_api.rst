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

Functions
=========

.. doxygenfunction:: dtfft::get_backend_string

.. doxygenfunction:: dtfft::get_error_string

.. doxygenfunction:: dtfft::get_precision_string

.. doxygenfunction:: dtfft::get_executor_string

.. doxygenfunction:: dtfft::set_config

Classes
=======

.. doxygenclass:: dtfft::Version
  :members:

.. doxygenclass:: dtfft::Exception
  :members:

.. doxygenclass:: dtfft::Pencil
  :members:

.. doxygenclass:: dtfft::Config
  :members:

.. doxygenclass:: dtfft::Plan
  :members:

.. doxygenclass:: dtfft::PlanC2C
  :members:

.. doxygenclass:: dtfft::PlanR2C
  :members:

.. doxygenclass:: dtfft::PlanR2R
  :members:
