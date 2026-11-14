# dtfft

[![License](https://img.shields.io/github/license/ShatrovOA/dtFFT?color=brightgreen)](https://github.com/ShatrovOA/dtFFT/blob/master/LICENSE)
[![Documentation](https://readthedocs.org/projects/dtfft/badge/?version=latest)](https://dtfft.readthedocs.io/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dtfft)](https://pypi.org/project/dtfft/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dtfft)

Python bindings for **[dtFFT](https://github.com/ShatrovOA/dtFFT)** — a high-performance library for parallel data transpositions and Fast Fourier Transforms via MPI.

This package is distributed as a **source distribution (sdist)** and is compiled locally during installation.
It gives you full control over all build options via CMake flags.

> If you need a pre-built wheel (no compiler required), use one of the variant packages:
> `dtfft-openmpi`, `dtfft-mpich`, `dtfft-fftw-openmpi`, `dtfft-cuda12x-openmpi`, etc.
> See the [full list on PyPI](https://pypi.org/search/?q=dtfft) or the
> [documentation](https://dtfft.readthedocs.io/latest/build.html#installing-from-pypi).

## Installation

Building from source requires a Fortran compiler (GCC ≥ 10, Intel, or NVHPC), CMake ≥ 3.25,
and an MPI implementation with development headers.
Build-time Python dependencies (`scikit-build`, `cmake`, `ninja`, `pybind11`) are installed
automatically by pip.

> **Important**: `mpi4py` in your environment must be compiled against the same MPI implementation
> that dtFFT will be linked with. Install it from source:
>
> ```bash
> pip install --no-binary mpi4py mpi4py
> ```

**Transpose-only (no FFT backend)**:

```bash
pip install dtfft
```

**With FFTW3**:

```bash
CMAKE_ARGS="-DDTFFT_WITH_FFTW=ON" pip install dtfft
```

**With cuFFT (CUDA)**:

> Also install `cupy` matching your CUDA toolkit version first, e.g. `pip install cupy-cuda12x`.

```bash
CMAKE_ARGS="-DDTFFT_WITH_CUDA=ON -DDTFFT_WITH_CUFFT=ON" pip install dtfft
```

Any CMake option documented on the
[build page](https://dtfft.readthedocs.io/latest/build.html#configuration-options)
can be passed via `CMAKE_ARGS`.

## Quick start

```python
import numpy as np
from mpi4py import MPI
import dtfft

# Create a 3-D complex-to-complex plan
plan = dtfft.PlanC2C([256, 256, 256], comm=MPI.COMM_WORLD)

# Allocate MPI-decomposed buffers
x = plan.get_ndarray(plan.alloc_size)
y = plan.get_ndarray(plan.alloc_size)

x[...] = np.random.random(x.shape) + 1j * np.random.random(x.shape)

# Forward transform  (pencil decomposition applied automatically)
plan.execute(x, y, dtfft.Execute.FORWARD)

# Backward transform
plan.execute(y, x, dtfft.Execute.BACKWARD)
```

## Documentation

- [API reference](https://dtfft.readthedocs.io/latest/api_python.html)
- [Building from source](https://dtfft.readthedocs.io/latest/build.html)
- [Examples](https://github.com/ShatrovOA/dtFFT/tree/master/tests/python)

## License

GPL v3 — see [LICENSE](https://github.com/ShatrovOA/dtFFT/blob/master/LICENSE).
