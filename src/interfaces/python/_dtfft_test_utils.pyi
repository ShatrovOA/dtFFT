"""
Type stubs for the native _dtfft_test_utils C extension module (pybind11).

Conditional features:
  - ``attach_gpu_to_process`` — attaches a GPU when built with CUDA.
  - ``platform`` and ``stream`` parameters on ``scale*``, ``*H2D``, and
    ``checkAndReport*`` functions — available only when built with CUDA.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def createGridDims(
    dims: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Return ``(grid, starts, counts)`` for the provided global dimensions.

    Parameters
    ----------
    dims:
        Global dimensions in natural Fortran order.

    Returns
    -------
    grid:
        MPI process grid dimensions.
    starts:
        Local starts for this process in natural Fortran order.
    counts:
        Local counts for this process in natural Fortran order.
    """
    ...

def attach_gpu_to_process() -> None:
    """Bind the current MPI process rank to the corresponding GPU device."""
    ...

# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def scaleFloat(
    executor: int,
    buffer: Any,
    scale: int,
    platform: int = 0,
    stream: int = 0,
) -> None:
    """Divide every element of a single-precision real *buffer* by *scale*.

    Parameters
    ----------
    executor:
        Integer identifying the FFT executor (maps to ``dtfft::Executor``).
    buffer:
        Array exposing ``__array_interface__`` (host) or
        ``__cuda_array_interface__`` (device).
    scale:
        Divisor applied to every element.
    platform:
        Integer identifying the runtime platform (``0`` = host). CUDA builds only.
    stream:
        Raw integer CUDA stream pointer. CUDA builds only.
    """
    ...

def scaleDouble(
    executor: int,
    buffer: Any,
    scale: int,
    platform: int = 0,
    stream: int = 0,
) -> None:
    """Divide every element of a double-precision real *buffer* by *scale*.

    See :func:`scaleFloat` for parameter descriptions.
    """
    ...

def scaleComplexFloat(
    executor: int,
    buffer: Any,
    scale: int,
    platform: int = 0,
    stream: int = 0,
) -> None:
    """Divide every element of a single-precision complex *buffer* by *scale*.

    See :func:`scaleFloat` for parameter descriptions.
    """
    ...

def scaleComplexDouble(
    executor: int,
    buffer: Any,
    scale: int,
    platform: int = 0,
    stream: int = 0,
) -> None:
    """Divide every element of a double-precision complex *buffer* by *scale*.

    See :func:`scaleFloat` for parameter descriptions.
    """
    ...

# ---------------------------------------------------------------------------
# Host-to-Device copy helpers
# ---------------------------------------------------------------------------

def complexDoubleH2D(
    src: Any,
    dst: Any,
    platform: int = 0,
) -> None:
    """Copy a double-precision complex array from host *src* to device *dst*.

    Parameters
    ----------
    src:
        Source array exposing ``__array_interface__``.
    dst:
        Destination array exposing ``__cuda_array_interface__``.
    platform:
        Integer identifying the runtime platform (``0`` = host). CUDA builds only.
    """
    ...

def complexFloatH2D(
    src: Any,
    dst: Any,
    platform: int = 0,
) -> None:
    """Copy a single-precision complex array from host *src* to device *dst*.

    See :func:`complexDoubleH2D` for parameter descriptions.
    """
    ...

def doubleH2D(
    src: Any,
    dst: Any,
    platform: int = 0,
) -> None:
    """Copy a double-precision real array from host *src* to device *dst*.

    See :func:`complexDoubleH2D` for parameter descriptions.
    """
    ...

def floatH2D(
    src: Any,
    dst: Any,
    platform: int = 0,
) -> None:
    """Copy a single-precision real array from host *src* to device *dst*.

    See :func:`complexDoubleH2D` for parameter descriptions.
    """
    ...

# ---------------------------------------------------------------------------
# Validation and reporting helpers
# ---------------------------------------------------------------------------

def checkAndReportComplexDouble(
    n_global: int,
    tf: float,
    tb: float,
    buffer: Any,
    check: Any,
    platform: int = 0,
) -> None:
    """Validate *buffer* against *check* and print a timing report.

    Parameters
    ----------
    n_global:
        Total number of global elements (used for throughput calculation).
    tf:
        Elapsed time for the forward transform (seconds).
    tb:
        Elapsed time for the backward transform (seconds).
    buffer:
        Result array exposing ``__array_interface__`` or
        ``__cuda_array_interface__``.
    check:
        Reference array to compare against.
    platform:
        Integer identifying the runtime platform (``0`` = host). CUDA builds only.
    """
    ...

def checkAndReportComplexFloat(
    n_global: int,
    tf: float,
    tb: float,
    buffer: Any,
    check: Any,
    platform: int = 0,
) -> None:
    """Validate single-precision complex *buffer* against *check* and report.

    See :func:`checkAndReportComplexDouble` for parameter descriptions.
    """
    ...

def checkAndReportDouble(
    n_global: int,
    tf: float,
    tb: float,
    buffer: Any,
    check: Any,
    platform: int = 0,
) -> None:
    """Validate double-precision real *buffer* against *check* and report.

    See :func:`checkAndReportComplexDouble` for parameter descriptions.
    """
    ...

def checkAndReportFloat(
    n_global: int,
    tf: float,
    tb: float,
    buffer: Any,
    check: Any,
    platform: int = 0,
) -> None:
    """Validate single-precision real *buffer* against *check* and report.

    See :func:`checkAndReportComplexDouble` for parameter descriptions.
    """
    ...
