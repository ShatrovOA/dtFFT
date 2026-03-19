"""
Python interface for dtFFT library.

This module provides a high-level interface to the dtFFT library, allowing
users to create and execute FFT plans using numpy arrays. It includes
convenience classes for different types of FFT plans and handles memory
management and type checking automatically.
"""

from __future__ import annotations

import math
import weakref
from abc import ABC
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from mpi4py import MPI

from . import _dtfft  # type: ignore


def is_fftw_enabled() -> bool:
    """Return ``True`` if dtFFT was built with FFTW3 support."""
    return _dtfft.is_fftw_enabled()


def is_mkl_enabled() -> bool:
    """Return ``True`` if dtFFT was built with Intel MKL support."""
    return _dtfft.is_mkl_enabled()


def is_cufft_enabled() -> bool:
    """Return ``True`` if dtFFT was built with cuFFT support."""
    return _dtfft.is_cufft_enabled()


def is_vkfft_enabled() -> bool:
    """Return ``True`` if dtFFT was built with VkFFT support."""
    return _dtfft.is_vkfft_enabled()


def is_cuda_enabled() -> bool:
    """Return ``True`` if CUDA platform support is enabled in dtFFT."""
    return _dtfft.is_cuda_enabled()


def is_transpose_only_enabled() -> bool:
    """Return ``True`` if dtFFT was built in transpose-only mode."""
    return _dtfft.is_transpose_only_enabled()


def is_nccl_enabled() -> bool:
    """Return ``True`` if NCCL support is enabled in dtFFT."""
    return _dtfft.is_nccl_enabled()


def is_nvshmem_enabled() -> bool:
    """Return ``True`` if NVSHMEM support is enabled in dtFFT."""
    return _dtfft.is_nvshmem_enabled()


def is_compression_enabled() -> bool:
    """Return ``True`` if compression backends are enabled in dtFFT."""
    return _dtfft.is_compression_enabled()


def get_backend_string(backend: Backend) -> str:
    """Return string representation of a ``Backend`` value."""
    return _dtfft.get_backend_string(backend)


if TYPE_CHECKING:
    AccessMode: TypeAlias = Any
    Backend: TypeAlias = Any
    Effort: TypeAlias = Any
    Execute: TypeAlias = Any
    Executor: TypeAlias = Any
    Layout: TypeAlias = Any
    Precision: TypeAlias = Any
    R2RKind: TypeAlias = Any
    Reshape: TypeAlias = Any
    Transpose: TypeAlias = Any
    TransposeMode: TypeAlias = Any
    if is_cuda_enabled():
        Platform: TypeAlias = Any
    if is_compression_enabled():
        CompressionConfig: TypeAlias = Any
        CompressionLib: TypeAlias = Any
        CompressionMode: TypeAlias = Any
else:
    AccessMode = _dtfft.AccessMode
    Backend = _dtfft.Backend
    Effort = _dtfft.Effort
    Execute = _dtfft.Execute
    Executor = _dtfft.Executor
    Layout = _dtfft.Layout
    Precision = _dtfft.Precision
    R2RKind = _dtfft.R2RKind
    Reshape = _dtfft.Reshape
    Transpose = _dtfft.Transpose
    TransposeMode = _dtfft.TransposeMode
    if is_cuda_enabled():
        Platform = _dtfft.Platform
    if is_compression_enabled():
        CompressionConfig = _dtfft.CompressionConfig
        CompressionLib = _dtfft.CompressionLib
        CompressionMode = _dtfft.CompressionMode

dtfft_Exception = _dtfft.Exception
Version = _dtfft.Version

__version__ = f"{Version.MAJOR}.{Version.MINOR}.{Version.PATCH}"

_ELEMENT_SIZE_C128 = 16  # complex128: 2 × float64
_ELEMENT_SIZE_F64 = 8  # float64 / complex64: 8 bytes

if is_cuda_enabled():
    import cupy as cp  # type: ignore


class UnsupportedMethod(AttributeError):
    def __init__(self, name: str):
        super().__init__(f"Method `{name}` is not supported for current build")


def _check_method_supported(cls, func: str):
    if not hasattr(cls, func):
        raise UnsupportedMethod(func)


def _mpi_handle(comm: MPI.Comm) -> int:
    return comm.handle if hasattr(comm, "handle") else int(comm)


class Pencil:
    """
    Create a Pencil decomposition object that can be passed to plan constructors.

    Parameters
    ----------
    starts : list[int], optional
        Starting indices of the local data block in each dimension.
    counts : list[int], optional
        Number of elements in the local data block in each dimension.
    pencil : _dtfft.Pencil, optional
        An existing _dtfft.Pencil object to wrap. If provided, `starts` and `counts` are ignored.
        This parameter is intended for internal use when a Pencil object is returned by a plan method,
        and allows wrapping an existing C++ Pencil without copying.
        For user-created Pencil objects, it is more common to specify `starts` and `counts` directly.
    """

    _p: _dtfft.Pencil

    def __init__(
        self,
        starts: list[int] = None,
        counts: list[int] = None,
        pencil: _dtfft.Pencil = None,
    ):
        if pencil is not None:
            self._p = pencil
        else:
            if starts is None or counts is None:
                raise ValueError("Both `starts` and `counts` or `pencil` must be provided")
            self._p = _dtfft.Pencil(starts, counts)

    @classmethod
    def from_pencil(cls, pencil: _dtfft.Pencil):
        """
        Creates Pencil wrapper from an existing _dtfft.Pencil instance.
        This is used internally when a plan method returns a Pencil object.
        """

        return cls(pencil=pencil)

    def __getattr__(self, name):
        if name in ("ndims", "dim", "starts", "counts", "size"):
            return getattr(self._p, f"get_{name}")()
        raise AttributeError(f"`Pencil` object has no attribute '{name}'")


class Config:
    """Configuration object applied before plan creation.

    Parameters
    ----------
    enable_log : bool, optional
        Whether dtFFT should print additional information (default: ``False``).
    enable_z_slab : bool, optional
        Whether dtFFT uses Z-slab optimisation (default: ``True``).
    enable_y_slab : bool, optional
        Whether dtFFT uses Y-slab optimisation (default: ``False``).
    measure_warmup_iters : int, optional
        Number of warmup iterations used during autotuning.
    measure_iters : int, optional
        Number of measurement iterations used during autotuning.
    backend : Backend, optional
        Backend used when ``effort`` is ``ESTIMATE`` or ``MEASURE``.
    reshape_backend : Backend, optional
        Backend used for reshape operations when ``effort`` is ``ESTIMATE`` or ``MEASURE``.
    enable_datatype_backend : bool, optional
        Whether MPI datatype backend is enabled for autotuning (default: ``True``).
    enable_mpi_backends : bool, optional
        Whether MPI backends are enabled for ``PATIENT`` effort (default: ``False``).
    enable_pipelined_backends : bool, optional
        Whether pipelined backends are enabled for ``PATIENT`` effort (default: ``True``).
    enable_rma_backends : bool, optional
        Whether RMA backends are enabled for ``PATIENT`` effort (default: ``True``).
    enable_fused_backends : bool, optional
        Whether fused backends are enabled for ``PATIENT`` effort (default: ``True``).
    enable_kernel_autotune : bool, optional
        Whether kernel launch parameter autotuning is enabled below ``EXHAUSTIVE`` (default: ``False``).
    enable_fourier_reshape : bool, optional
        Whether Fourier-space reshapes are executed during ``Plan.execute`` (default: ``False``).
    transpose_mode : TransposeMode, optional
        At which stage local transposition is performed for generic backends.
    access_mode : AccessMode, optional
        Memory access mode for local transposition in generic backends.
    platform : Platform, optional
        Execution platform (default: ``Platform.HOST``). CUDA builds only.
    stream : cp.cuda.Stream, optional
        CUDA stream to use for all operations. If not specified, dtFFT will create stream internally.
        Such stream can be retrieved via the `stream` property of the plan after creation. CUDA builds only.
    enable_nccl_backends : bool, optional
        Whether NCCL backends are enabled for ``PATIENT`` effort (default: ``True``). CUDA builds only.
    enable_nvshmem_backends : bool, optional
        Whether NVSHMEM backends are enabled for ``PATIENT`` effort (default: ``True``). CUDA builds only.
    enable_compressed_backends : bool, optional
        Whether compressed backends are enabled for autotuning. Compression builds only.
    compression_config_transpose : CompressionConfig, optional
        Compression configuration for transpose operations. Compression builds only.
    compression_config_reshape : CompressionConfig, optional
        Compression configuration for reshape operations. Compression builds only.

    Raises
    ------
    AttributeError
        If an option is not available in the current build.

    Notes
    -----
    Values are passed to dtFFT via :func:`set_config` right before plan
    construction when a config is provided to a plan constructor.
    """

    def __init__(
        self,
        *,
        enable_log: bool | None = None,
        enable_z_slab: bool | None = None,
        enable_y_slab: bool | None = None,
        measure_warmup_iters: int | None = None,
        measure_iters: int | None = None,
        backend: Backend | None = None,
        reshape_backend: Backend | None = None,
        enable_datatype_backend: bool | None = None,
        enable_mpi_backends: bool | None = None,
        enable_pipelined_backends: bool | None = None,
        enable_rma_backends: bool | None = None,
        enable_fused_backends: bool | None = None,
        enable_kernel_autotune: bool | None = None,
        enable_fourier_reshape: bool | None = None,
        transpose_mode: TransposeMode | None = None,
        access_mode: AccessMode | None = None,
        platform: Platform | None = None,
        stream: cp.cuda.Stream | None = None,
        enable_nccl_backends: bool | None = None,
        enable_nvshmem_backends: bool | None = None,
        enable_compressed_backends: bool | None = None,
        compression_config_transpose: CompressionConfig | None = None,
        compression_config_reshape: CompressionConfig | None = None,
    ):
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        self._cfg = _dtfft.Config()
        for name, value in params.items():
            self._call_setter(f"set_{name}", value)

    def _commit(self):
        _dtfft.set_config(self._cfg)

    def _call_setter(self, name: str, *args) -> Config:
        _check_method_supported(self._cfg, name)
        fn = getattr(self._cfg, name)
        fn(*args)

    def __setattr__(self, name, value):
        if name == "_cfg":
            super().__setattr__(name, value)
            return
        if name == "stream" and not isinstance(value, int):
            if not hasattr(value, "ptr"):
                raise ValueError(
                    f"Expected an integer or a CUDA stream object with `ptr` attribute for `stream`, got {type(value)}"
                )
            value = value.ptr

        self._call_setter(f"set_{name}", value)


def _normalize_shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    if not isinstance(shape, Iterable):
        return (shape,)
    return tuple(int(v) for v in shape)


class _DtfftAllocation:
    def __init__(self, plan_wrapper: Any, ptr: int):
        self._plan_wrapper = plan_wrapper
        self._ptr = int(ptr)
        self._alive = True
        # weakref.finalize holds a strong reference to args, keeping plan_wrapper alive
        self._finalizer = weakref.finalize(self, self._cleanup, self._plan_wrapper, self._ptr)

    @staticmethod
    def _cleanup(plan_wrapper: Any, ptr: int):
        try:
            if plan_wrapper._plan is not None:
                plan_wrapper._plan.free_ptr(int(ptr))
            plan_wrapper._alloc_count -= 1

            # If a delayed destroy() was requested, finalize it now
            if (
                plan_wrapper._destroyed
                and plan_wrapper._alloc_count == 0
                and plan_wrapper._plan is not None
            ):
                plan_wrapper._plan.destroy()
                plan_wrapper._plan = None
        except Exception:
            pass

    def close(self) -> None:
        if self._alive:
            self._finalizer()
            self._alive = False


class _NumpyArrayInterfaceOwner:
    def __init__(self, allocation: _DtfftAllocation, shape: tuple[int, ...], dtype: np.dtype):
        self._allocation = allocation
        self._shape = shape
        self._dtype = np.dtype(dtype)

    @property
    def __array_interface__(self):
        return {
            "shape": self._shape,
            "typestr": self._dtype.str,
            "data": (int(self._allocation._ptr), False),
            "strides": None,
            "version": 3,
        }


class Request:
    """Python wrapper for internal async request handle (`dtfft_request_t`)."""

    def __init__(self, handle: int, kind: str):
        self._handle = int(handle)
        self._kind = kind

    @property
    def handle(self) -> int:
        """Opaque integer request handle returned by async operations."""
        return self._handle

    @property
    def kind(self) -> str:
        """Human-readable operation kind associated with this request."""
        return self._kind

    def __int__(self) -> int:
        return self._handle

    def __repr__(self) -> str:
        return f"Request(kind={self._kind!r}, handle=0x{self._handle:x})"


class _dtfftStream:
    """
    Stream class that implements CUDA stream protocol.
    """

    def __init__(self, ptr: int):
        self._ptr = ptr

    def __cuda_stream__(self):
        return (0, self._ptr)


class Plan(ABC):
    """Python wrapper around a concrete dtFFT plan instance.

    This class delegates most of its informational properties to the underlying C++ plan.
    The following properties are dynamically available via ``__getattr__``:

    Properties
    ----------
    z_slab_enabled : bool
        Return whether Z-slab optimization is enabled for this plan.
    y_slab_enabled : bool
        Return whether Y-slab optimization is enabled for this plan.
    alloc_size : int
        Return minimum number of elements required for main buffers.
    aux_size : int
        Return auxiliary buffer size (in elements) for ``execute``.
    aux_bytes : int
        Return auxiliary buffer size (in bytes) for ``execute``.
    aux_size_reshape : int
        Return auxiliary buffer size (in elements) for ``reshape``.
    aux_bytes_reshape : int
        Return auxiliary buffer size (in bytes) for ``reshape``.
    aux_size_transpose : int
        Return auxiliary buffer size (in elements) for ``transpose``.
    aux_bytes_transpose : int
        Return auxiliary buffer size (in bytes) for ``transpose``.
    local_sizes : tuple[list[int], list[int], list[int], list[int], int]
        Return local decomposition: ``(in_starts, in_counts, out_starts, out_counts, alloc_size)``.
    dims : list[int]
        Return global plan dimensions in natural Fortran order.
    grid_dims : list[int]
        Return MPI grid dimensions in natural Fortran order.
    element_size : int
        Return size (in bytes) of one data element for this plan.
    alloc_bytes : int
        Return minimum bytes required for input/output buffers.
    executor : Executor
        Return executor selected for this plan.
    precision : Precision
        Return precision selected for this plan.
    backend : Backend
        Return communication backend selected for this plan.
    reshape_backend : Backend
        Return backend selected for reshape operations.
    stream : cupy.cuda.Stream | cupy.cuda.ExternalStream
        Return CUDA stream associated with this plan.
    platform : Platform
        Return execution platform (HOST/CUDA) for this plan.
    """

    _plan: _dtfft.Plan
    _alloc_count: int
    _destroyed: bool
    _dtype: np.dtype

    def __getattr__(self, name: str) -> Any:
        if not hasattr(self._plan, f"get_{name}"):
            if name in ("stream", "platform"):
                raise AttributeError(f"Property `{name}` is only available in CUDA builds")
            raise AttributeError(f"`{self.__class__.__name__}` object has no attribute '{name}'")

        value = getattr(self._plan, f"get_{name}")()
        if name == "stream":
            if hasattr(cp.cuda.Stream, "from_external"):
                value = cp.cuda.Stream.from_external(_dtfftStream(value))
            else:
                value = cp.cuda.ExternalStream(value)
        return value

    def __init__(self, plan: _dtfft.Plan):
        """Initialize Python wrapper around a concrete dtFFT plan instance."""

        if self.__class__.__name__ == "Plan":
            raise TypeError("Plan is an abstract base class and cannot be instantiated directly")
        self._plan = plan
        self._alloc_count = 0
        self._destroyed = False
        element_size = self.element_size
        if isinstance(self, PlanC2C):
            self._dtype = np.complex128 if element_size == _ELEMENT_SIZE_C128 else np.complex64
        else:
            self._dtype = np.float64 if element_size == _ELEMENT_SIZE_F64 else np.float32

    def execute(
        self,
        input: np.ndarray | cp.ndarray,
        output: np.ndarray | cp.ndarray,
        execute_type: Execute,
        aux: np.ndarray | cp.ndarray = None,
    ) -> None:
        """Execute full plan in ``FORWARD`` or ``BACKWARD`` direction.

        Parameters
        ----------
        input : np.ndarray | cp.ndarray
            Input array containing the source data (e.g., numpy.ndarray or cupy.ndarray).
        output : np.ndarray | cp.ndarray
            Output array where the result will be written.
        execute_type : Execute
            Direction of execution (e.g., ``Execute.FORWARD`` or ``Execute.BACKWARD``).
        aux : np.ndarray | cp.ndarray, optional
            Optional auxiliary array for intermediate data.
        """
        self._plan.execute(input, output, execute_type, aux)

    def transpose(
        self,
        input: np.ndarray | cp.ndarray,
        output: np.ndarray | cp.ndarray,
        transpose_type: Transpose,
        aux: np.ndarray | cp.ndarray = None,
    ) -> None:
        """Transpose data in a single dimension (e.g., X->Y, Y->Z).

        Parameters
        ----------
        input : np.ndarray | cp.ndarray
            Input array containing the source data.
        output : np.ndarray | cp.ndarray
            Output array where transposed data will be written. Must be different from ``input``.
        transpose_type : Transpose
            Type of transpose operation to perform.
        aux : np.ndarray | cp.ndarray, optional
            Optional auxiliary array for intermediate data.
        """
        self._plan.transpose(input, output, transpose_type, aux)

    def reshape(
        self,
        input: np.ndarray | cp.ndarray,
        output: np.ndarray | cp.ndarray,
        reshape_type: Reshape,
        aux: np.ndarray | cp.ndarray = None,
    ) -> None:
        """Reshape data between brick and pencil layouts and vice versa.

        Parameters
        ----------
        input : np.ndarray | cp.ndarray
            Input array containing the source data.
        output : np.ndarray | cp.ndarray
            Output array where reshaped data will be written. Must be different from ``input``.
        reshape_type : Reshape
            Type of reshape operation to perform.
        aux : np.ndarray | cp.ndarray, optional
            Optional auxiliary array for intermediate data.
        """
        self._plan.reshape(input, output, reshape_type, aux)

    def transpose_start(
        self,
        input: np.ndarray | cp.ndarray,
        output: np.ndarray | cp.ndarray,
        transpose_type: Transpose,
        aux: np.ndarray | cp.ndarray = None,
    ) -> Request:
        """Start asynchronous transpose and return request handle.

        Parameters
        ----------
        input : np.ndarray | cp.ndarray
            Input array containing the source data.
        output : np.ndarray | cp.ndarray
            Output array where transposed data will be written. Must be different from ``input``.
        transpose_type : Transpose
            Type of transpose operation to perform.
        aux : np.ndarray | cp.ndarray, optional
            Optional auxiliary array for intermediate data.

        Returns
        -------
        Request
            A handle to the asynchronous operation, which must be passed to ``transpose_end``.
        """
        return Request(
            self._plan.transpose_start(input, output, transpose_type, aux),
            str(transpose_type),
        )

    def transpose_end(self, request: Request) -> None:
        """Finalize asynchronous transpose operation.

        Parameters
        ----------
        request : Request
            The operation handle previously returned by ``transpose_start``.
        """
        handle = request.handle if isinstance(request, Request) else int(request)
        self._plan.transpose_end(handle)

    def reshape_start(
        self,
        input: np.ndarray | cp.ndarray,
        output: np.ndarray | cp.ndarray,
        reshape_type: Reshape,
        aux: np.ndarray | cp.ndarray = None,
    ) -> Request:
        """Start asynchronous reshape and return request handle.

        Parameters
        ----------
        input : np.ndarray | cp.ndarray
            Input array containing the source data.
        output : np.ndarray | cp.ndarray
            Output array where reshaped data will be written. Must be different from ``input``.
        reshape_type : Reshape
            Type of reshape operation to perform.
        aux : np.ndarray | cp.ndarray, optional
            Optional auxiliary array for intermediate data.

        Returns
        -------
        Request
            A handle to the asynchronous operation, which must be passed to ``reshape_end``.
        """
        return Request(
            self._plan.reshape_start(input, output, reshape_type, aux),
            str(reshape_type),
        )

    def reshape_end(self, request: Request) -> None:
        """Finalize asynchronous reshape operation.

        Parameters
        ----------
        request : Request
            The operation handle previously returned by ``reshape_start``.
        """
        handle = request.handle if isinstance(request, Request) else int(request)
        self._plan.reshape_end(handle)

    def report(self) -> None:
        """Print plan-related information to stdout."""
        self._plan.report()

    def report_compression(self) -> None:
        """Print compression-related information to stdout (if available)."""
        _check_method_supported(self._plan, "report_compression")
        self._plan.report_compression()

    def get_pencil(self, layout: Layout) -> Pencil:
        """Return ``Pencil`` metadata for the requested ``Layout``."""
        return Pencil.from_pencil(self._plan.get_pencil(layout))

    def destroy(self) -> None:
        """Destroy underlying dtFFT plan resources explicitly."""
        if self._destroyed:
            return

        # Destroy immediately if there are no active allocations
        if self._alloc_count == 0 and self._plan is not None:
            self._plan.destroy()
            self._plan = None
            self._destroyed = True

    def _resolve_array_backend(self) -> str:
        if not is_cuda_enabled():
            return "numpy"
        return "cupy" if self._plan.get_platform() == Platform.CUDA else "numpy"

    def get_ndarray(
        self,
        size: int,
        shape: int | tuple[int, ...] | None = None,
        dtype: Any = None,
        order: str = "C",
    ) -> np.ndarray | cp.ndarray:
        """Allocate an ndarray backed by plan-managed memory.

        Parameters
        ----------
        size : int
            Number of elements to allocate.
        shape : int | tuple[int, ...] | None, optional
            Desired array shape. Note, that product of shape might be smaller then `size`.
        dtype : numpy.dtype | type | None, optional
            Element type. If ``None``, uses plan-native dtype.
        order : str, optional
            Memory order passed to NumPy/CuPy ndarray constructor.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Array view that uses dtFFT allocation and is automatically freed
            when no longer referenced.
        """
        if shape is None:
            shape = (size,)
        shape = _normalize_shape(shape)

        if math.prod(shape) > size:
            raise ValueError(f"Shape {shape} is too large for requested size {size}")

        np_dtype = np.dtype(self._dtype if dtype is None else dtype)
        nbytes = size * int(np_dtype.itemsize)
        ptr = self._plan.alloc_ptr(nbytes)

        self._alloc_count += 1
        allocation = _DtfftAllocation(self, ptr)

        xp_name = self._resolve_array_backend()
        if xp_name == "numpy":
            owner = _NumpyArrayInterfaceOwner(allocation, shape, np_dtype)
            return np.asarray(owner, order=order)
        if xp_name == "cupy":
            mem = cp.cuda.UnownedMemory(ptr, nbytes, allocation)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            return cp.ndarray(shape=shape, dtype=np_dtype, memptr=memptr, order=order)

        allocation.close()
        raise RuntimeError("Failed to resolve array backend")

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __del__(self) -> None:
        with suppress(Exception):
            self.destroy()

    @staticmethod
    def _apply_config(config: Config | None = None):
        if config is not None:
            config._commit()


class PlanC2C(Plan):
    """Create a complex-to-complex dtFFT plan.

    Parameters
    ----------
    dims_or_pencil : list[int] | Pencil
        Global dimensions (natural Fortran order) or already prepared input
        ``Pencil``.
    comm : mpi4py.MPI.Comm | int, optional
        MPI communicator object or raw handle.
    precision : Precision, optional
        Scalar precision used by the plan.
    effort : Effort, optional
        Planning effort level.
    executor : Executor, optional
        Preferred FFT executor.
    config : Config | None, optional
        Optional configuration applied before plan creation.
    """

    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        comm: MPI.Comm = MPI.COMM_WORLD,
        precision: Precision = Precision.DOUBLE,
        effort: Effort = Effort.ESTIMATE,
        executor: Executor = Executor.NONE,
        config: Config | None = None,
    ) -> None:
        Plan._apply_config(config)
        comm_handle = _mpi_handle(comm)
        plan = _dtfft.PlanC2C(
            dims_or_pencil._p if isinstance(dims_or_pencil, Pencil) else dims_or_pencil,
            comm_handle,
            precision,
            effort,
            executor,
        )
        super().__init__(plan)


class PlanR2C(Plan):
    """Create a real-to-complex dtFFT plan.

    Parameters
    ----------
    dims_or_pencil : list[int] | Pencil
        Global dimensions (natural Fortran order) or already prepared input
        ``Pencil``.
    executor : Executor
        FFT executor for local real/complex transforms.
    comm : mpi4py.MPI.Comm | int, optional
        MPI communicator object or raw handle.
    precision : Precision, optional
        Scalar precision used by the plan.
    effort : Effort, optional
        Planning effort level.
    config : Config | None, optional
        Optional configuration applied before plan creation.
    """

    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        comm: MPI.Comm = MPI.COMM_WORLD,
        precision: Precision = Precision.DOUBLE,
        effort: Effort = Effort.ESTIMATE,
        executor: Executor = Executor.NONE,
        config: Config | None = None,
    ) -> None:
        Plan._apply_config(config)
        comm_handle = _mpi_handle(comm)
        plan = _dtfft.PlanR2C(
            dims_or_pencil._p if isinstance(dims_or_pencil, Pencil) else dims_or_pencil,
            comm_handle,
            precision,
            effort,
            executor,
        )
        super().__init__(plan)


class PlanR2R(Plan):
    """Create a real-to-real dtFFT plan.

    Parameters
    ----------
    dims_or_pencil : list[int] | Pencil
        Global dimensions (natural Fortran order) or already prepared input
        ``Pencil``.
    kinds : list[R2RKind] | None, optional
        Transform kinds for each transformed axis.
    comm : mpi4py.MPI.Comm | int, optional
        MPI communicator object or raw handle.
    precision : Precision, optional
        Scalar precision used by the plan.
    effort : Effort, optional
        Planning effort level.
    executor : Executor, optional
        Preferred FFT executor.
    config : Config | None, optional
        Optional configuration applied before plan creation.
    """

    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        kinds: list[R2RKind] | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        precision: Precision = Precision.DOUBLE,
        effort: Effort = Effort.ESTIMATE,
        executor: Executor = Executor.NONE,
        config: Config | None = None,
    ) -> PlanR2R:
        if kinds is None:
            kinds = []
        Plan._apply_config(config)
        comm_handle = _mpi_handle(comm)
        plan = _dtfft.PlanR2R(
            dims_or_pencil._p if isinstance(dims_or_pencil, Pencil) else dims_or_pencil,
            kinds,
            comm_handle,
            precision,
            effort,
            executor,
        )
        super().__init__(plan)
