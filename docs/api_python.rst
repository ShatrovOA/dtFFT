####################
Python API Reference
####################

This page provides the generated Python API reference for ``dtfft`` with explicit per-object documentation.

.. currentmodule:: dtfft

Exceptions
==========

.. autoclass:: dtfft_Exception
   :show-inheritance:

Enumerations and Constants
==========================

.. autoclass:: Version
   :members:
   :undoc-members:

.. autoenum:: Execute

.. autoenum:: Transpose

.. autoenum:: Reshape

.. autoenum:: Layout

.. autoenum:: Precision

.. autoenum:: Effort

.. autoenum:: Executor

.. autoenum:: R2RKind

.. autoenum:: TransposeMode

.. autoenum:: AccessMode

.. autoenum:: Backend

.. autoenum:: Platform

.. autoenum:: CompressionMode

.. autoenum:: CompressionLib


Functions
=========

.. autofunction:: get_backend_string
.. autofunction:: is_fftw_enabled
.. autofunction:: is_mkl_enabled
.. autofunction:: is_cufft_enabled
.. autofunction:: is_vkfft_enabled
.. autofunction:: is_cuda_enabled
.. autofunction:: is_transpose_only_enabled
.. autofunction:: is_nccl_enabled
.. autofunction:: is_nvshmem_enabled
.. autofunction:: is_compression_enabled

Core Classes
============

.. autoclass:: Config
   :members:
   :undoc-members:

.. autoclass:: Pencil
   :members:
   :undoc-members:

.. autoclass:: Request

.. autoclass:: CompressionConfig
   :members:
   :undoc-members:

Plan Classes
============

.. autoclass:: Plan
   :members:
   :undoc-members:

.. autoclass:: PlanC2C
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PlanR2C
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PlanR2R
   :members:
   :undoc-members:
   :show-inheritance:

