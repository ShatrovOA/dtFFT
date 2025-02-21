##############################
Welcome to dtFFT Documentation
##############################

``dtFFT`` is a high-performance library designed for parallel data transpositions and optional Fast Fourier Transforms (FFTs)
in multidimensional computing environments.

It leverages MPI for distributed systems and supports GPU acceleration via CUDA, integrating seamlessly with external FFT libraries such as FFTW3, MKL DFTI, cuFFT, and VkFFT, or operating in transpose-only mode.
Whether you're working on CPU clusters or GPU-enabled nodes, ``dtFFT`` provides a flexible and efficient framework for scientific computing tasks requiring large-scale data transformations.

This documentation covers the essentials of using, building, and extending ``dtFFT``. Explore the sections below to get started.

Getting Started
===============

To begin using ``dtFFT``:

1. **Build the Library**: Follow the instructions in :ref:`Building the Library<building_link>` to compile ``dtFFT`` with your desired features (e.g., CUDA or FFTW3 support).
2. **Configure Runtime**: Set environment variables as needed (see :ref:`Environment Variables<environ_link>`) to tweak logging or datatype selection.
3. **Use the Library**: Refer to the :ref:`Usage Guide<usage_link>` for step-by-step examples of creating plans, allocating memory, and executing transformations.

Detailed API specifications are available in the Fortran, C, and C++ sections.

Contributing
============

Feedback, bug reports, and contributions are welcome! Please submit issues or pull requests via the project's `repository <https://github.com/ShatrovOA/dtFFT>`_. 
For API-specific details, consult the respective language sections.


Table of Contents
=================
.. toctree::
   :maxdepth: 2

   build
   usage
   f_api
   c_api
   cpp_api
   environ