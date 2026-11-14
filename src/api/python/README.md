# {PACKAGE_TITLE}

[![License](https://img.shields.io/github/license/ShatrovOA/dtFFT?color=brightgreen)](https://github.com/ShatrovOA/dtFFT/blob/master/LICENSE)
[![Documentation](https://readthedocs.org/projects/dtfft/badge/?version=latest)](https://dtfft.readthedocs.io/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/{PACKAGE_NAME})](https://pypi.org/project/{PACKAGE_NAME}/)
[![Python](https://img.shields.io/pypi/pyversions/{PACKAGE_NAME})](https://pypi.org/project/{PACKAGE_NAME}/)

Python bindings for **[dtFFT](https://github.com/ShatrovOA/dtFFT)** — a high-performance library for parallel data transpositions and Fast Fourier Transforms via MPI.

> This is the **`{PACKAGE_NAME}`** variant.
{VARIANT_DESCRIPTION}

## Included backends

{BACKENDS_TABLE}

## Installation

```bash
pip install {PACKAGE_NAME}
```

{EXTRA_INSTALL_NOTE}

### Requirements

- MPI runtime (OpenMPI or MPICH)
- Python ≥ 3.9
{EXTRA_REQUIREMENTS}

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

## Package variants

Multiple PyPI distributions are provided so you can declare an exact,
conflict-free dependency for your environment:

| PyPI package | FFT backend | MPI | Platform | Extra deps |
|---|---|---|---|---|
| `dtfft-openmpi` | none (transpose-only) | OpenMPI | CPU | — |
| `dtfft-mpich` | none (transpose-only) | MPICH | CPU | — |
| `dtfft-fftw-openmpi` | FFTW3 | OpenMPI | CPU | system libfftw3 |
| `dtfft-fftw-mpich` | FFTW3 | MPICH | CPU | system libfftw3 |
| `dtfft-cuda12x-openmpi` | cuFFT | OpenMPI | CPU + NVIDIA GPU (CUDA 12) | `cupy-cuda12x` |
| `dtfft-cuda12x-mpich` | cuFFT | MPICH | CPU + NVIDIA GPU (CUDA 12) | `cupy-cuda12x` |

All packages share the same importable namespace: `import dtfft`.

## Documentation

- [API reference](https://dtfft.readthedocs.io/latest/api_python.html)
- [Building from source](https://dtfft.readthedocs.io/latest/build.html)
- [Examples](https://github.com/ShatrovOA/dtFFT/tree/master/tests/python)

## License

GPL v3 — see [LICENSE](https://github.com/ShatrovOA/dtFFT/blob/master/LICENSE).
