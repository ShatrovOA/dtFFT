"""
Type stubs for the native _dtfft C extension module (pybind11).

Conditional features:
  - ``Platform``, ``Config.set_platform``, ``Config.set_stream``,
    ``Config.set_enable_nccl_backends``, ``Config.set_enable_nvshmem_backends``,
    ``Plan.get_stream``, ``Plan.get_platform`` — available only when built with CUDA.
  - ``CompressionMode``, ``CompressionLib``, ``CompressionConfig``,
    ``Config.set_enable_compressed_backends``,
    ``Config.set_compression_config_transpose``,
    ``Config.set_compression_config_reshape``,
    ``Plan.report_compression`` — available only when built with compression support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, overload

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class Exception(RuntimeError):
    """dtFFT runtime exception raised on non-SUCCESS error codes."""

    ...

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Execute(Enum):
    """Valid ``execute_type`` parameters for :meth:`Plan.execute`."""

    FORWARD: int
    """Perform XYZ --> YZX --> ZXY plan execution (Forward)."""
    BACKWARD: int
    """Perform ZXY --> YZX --> XYZ plan execution (Backward)."""

class Transpose(Enum):
    """Valid ``transpose_type`` parameters for :meth:`Plan.transpose`."""

    X_TO_Y: int
    """Transpose from Fortran X aligned to Fortran Y aligned."""
    Y_TO_X: int
    """Transpose from Fortran Y aligned to Fortran X aligned."""
    Y_TO_Z: int
    """Transpose from Fortran Y aligned to Fortran Z aligned."""
    Z_TO_Y: int
    """Transpose from Fortran Z aligned to Fortran Y aligned."""
    X_TO_Z: int
    """Transpose from Fortran X aligned to Fortran Z aligned (3-D Z-slab plans only)."""
    Z_TO_X: int
    """Transpose from Fortran Z aligned to Fortran X aligned (3-D Z-slab plans only)."""

class Layout(Enum):
    """Valid ``layout`` parameters for :meth:`Plan.get_pencil`."""

    X_BRICKS: int
    """X-brick layout."""
    X_PENCILS: int
    """X-pencil layout."""
    X_PENCILS_FOURIER: int
    """X-pencil Fourier layout for R2C plans."""
    Y_PENCILS: int
    """Y-pencil layout."""
    Z_PENCILS: int
    """Z-pencil layout."""
    Z_BRICKS: int
    """Z-brick layout."""

class Reshape(Enum):
    """Valid ``reshape_type`` parameters for :meth:`Plan.reshape`."""

    X_BRICKS_TO_PENCILS: int
    """Reshape from X bricks to X pencils."""
    X_PENCILS_TO_BRICKS: int
    """Reshape from X pencils to X bricks."""
    Z_BRICKS_TO_PENCILS: int
    """Reshape from Z bricks to Z pencils."""
    Z_PENCILS_TO_BRICKS: int
    """Reshape from Z pencils to Z bricks."""
    Y_BRICKS_TO_PENCILS: int
    """Reshape from Y bricks to Y pencils."""
    Y_PENCILS_TO_BRICKS: int
    """Reshape from Y pencils to Y bricks."""

class TransposeMode(Enum):
    """Local transposition mode for generic backends."""

    PACK: int
    """Perform transposition during packing."""
    UNPACK: int
    """Perform transposition during unpacking."""

class AccessMode(Enum):
    """Memory access mode for local transposition in generic backends."""

    WRITE: int
    """Write-aligned access."""
    READ: int
    """Read-aligned access."""

class Precision(Enum):
    """Floating-point precision for plan constructors."""

    SINGLE: int
    """Single precision."""
    DOUBLE: int
    """Double precision."""

class Effort(Enum):
    """Autotuning effort level for plan constructors."""

    ESTIMATE: int
    """Create plan as fast as possible."""
    MEASURE: int
    """Attempt to find best MPI grid decomposition."""
    PATIENT: int
    """MEASURE + cycle through MPI datatypes / GPU backend autotune."""
    EXHAUSTIVE: int
    """PATIENT + autotune all kernels and reshape backends."""

class Executor(Enum):
    """Available FFT executors."""

    NONE: int
    """No FFT plans — transpose-only plan."""
    FFTW3: int
    """FFTW3 Executor (host only)."""
    MKL: int
    """MKL DFTI Executor (host only)."""
    CUFFT: int
    """cuFFT Executor (GPU only)."""
    VKFFT: int
    """VkFFT Executor (GPU only)."""

class R2RKind(Enum):
    """Real-to-Real FFT kinds."""

    DCT_1: int
    """DCT-I  (Logical N=2*(n-1), inverse: DCT_1)."""
    DCT_2: int
    """DCT-II  (Logical N=2*n, inverse: DCT_3)."""
    DCT_3: int
    """DCT-III (Logical N=2*n, inverse: DCT_2)."""
    DCT_4: int
    """DCT-IV  (Logical N=2*n, inverse: DCT_4)."""
    DST_1: int
    """DST-I   (Logical N=2*(n+1), inverse: DST_1)."""
    DST_2: int
    """DST-II  (Logical N=2*n, inverse: DST_3)."""
    DST_3: int
    """DST-III (Logical N=2*n, inverse: DST_2)."""
    DST_4: int
    """DST-IV  (Logical N=2*n, inverse: DST_4)."""

class Backend(Enum):
    """Communication backends available in dtFFT."""

    MPI_DATATYPE: int
    """Backend that uses MPI datatypes."""
    MPI_P2P: int
    """MPI peer-to-peer algorithm."""
    MPI_P2P_PIPELINED: int
    """MPI peer-to-peer algorithm with overlapping copy and unpack."""
    MPI_A2A: int
    """MPI backend using MPI_Alltoall[v]."""
    MPI_RMA: int
    """MPI backend using one-sided communications."""
    MPI_RMA_PIPELINED: int
    """MPI backend using pipelined one-sided communications."""
    MPI_P2P_SCHEDULED: int
    """MPI peer-to-peer algorithm with scheduled communication."""
    MPI_P2P_FUSED: int
    """MPI P2P pipelined with overlapping pack/exchange/unpack and scheduled communication."""
    MPI_RMA_FUSED: int
    """MPI RMA pipelined with overlapping pack/exchange/unpack and scheduled communication."""
    MPI_P2P_COMPRESSED: int
    """MPI peer-to-peer fused backend with compression."""
    MPI_RMA_COMPRESSED: int
    """MPI RMA fused backend with compression."""
    NCCL: int
    """NCCL backend."""
    NCCL_PIPELINED: int
    """NCCL backend with overlapping copy and unpack."""
    NCCL_COMPRESSED: int
    """NCCL backend with compression."""
    CUFFTMP: int
    """cuFFTMp backend."""
    CUFFTMP_PIPELINED: int
    """cuFFTMp backend with an additional internal buffer."""
    ADAPTIVE: int
    """Adaptive backend selection."""

# CUDA-only enum
class Platform(Enum):
    """Runtime execution platform (available when built with CUDA)."""

    HOST: int
    """Host (CPU) platform."""
    CUDA: int
    """CUDA (GPU) platform."""

# Compression-only enums
class CompressionMode(Enum):
    """Compression mode (available when built with compression support)."""

    LOSSLESS: int
    """Lossless compression."""
    FIXED_RATE: int
    """Fixed rate compression."""
    FIXED_PRECISION: int
    """Fixed precision compression."""
    FIXED_ACCURACY: int
    """Fixed accuracy compression."""

class CompressionLib(Enum):
    """Compression library (available when built with compression support)."""

    ZFP: int
    """ZFP compression library."""

# ---------------------------------------------------------------------------
# CompressionConfig (compression-only)
# ---------------------------------------------------------------------------

class CompressionConfig:
    """Compression configuration (available when built with compression support)."""

    compression_lib: CompressionLib
    compression_mode: CompressionMode
    rate: float
    precision: int
    tolerance: float

    def __init__(
        self,
        compression_lib: CompressionLib = ...,
        compression_mode: CompressionMode = ...,
        rate: float = 0.0,
        precision: int = 0,
        tolerance: float = 0.0,
    ) -> None:
        """Create a ``CompressionConfig`` with the specified parameters."""
        ...

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

class Version:
    """dtFFT version information."""

    MAJOR: int
    """dtFFT major version."""
    MINOR: int
    """dtFFT minor version."""
    PATCH: int
    """dtFFT patch version."""
    CODE: int
    """dtFFT version code (for comparison)."""

    @staticmethod
    @overload
    def get() -> int:
        """Return version code defined at compile time."""
        ...

    @staticmethod
    @overload
    def get(major: int, minor: int, patch: int) -> int:
        """Return version code computed from the given components."""
        ...

# ---------------------------------------------------------------------------
# Pencil
# ---------------------------------------------------------------------------

class Pencil:
    """Describes a pencil decomposition — local starts and counts."""

    def __init__(self, starts: list[int], counts: list[int]) -> None:
        """Create a Pencil from local starts and counts in natural Fortran order."""
        ...

    def get_ndims(self) -> int:
        """Number of dimensions in the pencil."""
        ...

    def get_dim(self) -> int:
        """Aligned dimension id (1-based)."""
        ...

    def get_starts(self) -> list[int]:
        """Local starts in natural Fortran order."""
        ...

    def get_counts(self) -> list[int]:
        """Local counts in natural Fortran order."""
        ...

    def get_size(self) -> int:
        """Total number of elements in the pencil."""
        ...

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """Additional configuration parameters passed to dtFFT before plan creation."""

    def __init__(self) -> None:
        """Create a ``Config`` with default values."""
        ...

    def set_enable_log(self, enable_log: bool) -> Config:
        """Set whether dtFFT prints additional information (default: ``False``)."""
        ...

    def set_enable_z_slab(self, enable_z_slab: bool) -> Config:
        """Set whether dtFFT uses Z-slab optimisation (default: ``True``)."""
        ...

    def set_enable_y_slab(self, enable_y_slab: bool) -> Config:
        """Set whether dtFFT uses Y-slab optimisation (default: ``False``)."""
        ...

    def set_measure_warmup_iters(self, n_measure_warmup_iters: int) -> Config:
        """Set number of warmup iterations used during autotuning."""
        ...

    def set_measure_iters(self, n_measure_iters: int) -> Config:
        """Set number of measurement iterations used during autotuning."""
        ...

    def set_backend(self, backend: Backend) -> Config:
        """Set backend used when ``effort`` is ``ESTIMATE`` or ``MEASURE``."""
        ...

    def set_reshape_backend(self, backend: Backend) -> Config:
        """Set backend used for reshape operations when ``effort`` is ``ESTIMATE`` or ``MEASURE``."""
        ...

    def set_enable_datatype_backend(self, enable_datatype_backend: bool) -> Config:
        """Set whether MPI datatype backend is enabled for autotuning (default: ``True``)."""
        ...

    def set_enable_mpi_backends(self, enable_mpi_backends: bool) -> Config:
        """Set whether MPI backends are enabled for ``PATIENT`` effort (default: ``False``)."""
        ...

    def set_enable_pipelined_backends(self, enable_pipelined_backends: bool) -> Config:
        """Set whether pipelined backends are enabled for ``PATIENT`` effort (default: ``True``)."""
        ...

    def set_enable_rma_backends(self, enable_rma_backends: bool) -> Config:
        """Set whether RMA backends are enabled for ``PATIENT`` effort (default: ``True``)."""
        ...

    def set_enable_fused_backends(self, enable_fused_backends: bool) -> Config:
        """Set whether fused backends are enabled for ``PATIENT`` effort (default: ``True``)."""
        ...

    def set_enable_kernel_autotune(self, enable_kernel_autotune: bool) -> Config:
        """Set whether kernel launch parameter autotuning is enabled below ``EXHAUSTIVE`` (default: ``False``)."""
        ...

    def set_enable_fourier_reshape(self, enable_fourier_reshape: bool) -> Config:
        """Set whether Fourier-space reshapes are executed during ``Plan.execute`` (default: ``False``)."""
        ...

    def set_transpose_mode(self, transpose_mode: TransposeMode) -> Config:
        """Set at which stage local transposition is performed for generic backends."""
        ...

    def set_access_mode(self, access_mode: AccessMode) -> Config:
        """Set memory access mode for local transposition in generic backends."""
        ...

    # CUDA-only methods
    def set_platform(self, platform: Platform) -> Config:
        """Set execution platform (default: ``Platform.HOST``). CUDA builds only."""
        ...

    def set_stream(self, stream_ptr: int) -> Config:
        """Set main CUDA stream (as raw integer pointer). CUDA builds only."""
        ...

    def set_enable_nccl_backends(self, enable_nccl_backends: bool) -> Config:
        """Set whether NCCL backends are enabled for ``PATIENT`` effort (default: ``True``). CUDA builds only."""
        ...

    def set_enable_nvshmem_backends(self, enable_nvshmem_backends: bool) -> Config:
        """Set whether NVSHMEM backends are enabled for ``PATIENT`` effort (default: ``True``). CUDA builds only."""
        ...

    # Compression-only methods
    def set_enable_compressed_backends(self, enable_compressed_backends: bool) -> Config:
        """Set whether compressed backends are enabled for autotuning. Compression builds only."""
        ...

    def set_compression_config_transpose(self, compression_config: CompressionConfig) -> Config:
        """Set compression configuration for transpose operations. Compression builds only."""
        ...

    def set_compression_config_reshape(self, compression_config: CompressionConfig) -> Config:
        """Set compression configuration for reshape operations. Compression builds only."""
        ...

# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def is_fftw_enabled() -> bool:
    """Return ``True`` if dtFFT was built with FFTW3 support."""
    ...

def is_mkl_enabled() -> bool:
    """Return ``True`` if dtFFT was built with Intel MKL support."""
    ...

def is_cufft_enabled() -> bool:
    """Return ``True`` if dtFFT was built with cuFFT support."""
    ...

def is_vkfft_enabled() -> bool:
    """Return ``True`` if dtFFT was built with VkFFT support."""
    ...

def is_cuda_enabled() -> bool:
    """Return ``True`` if CUDA platform support is enabled."""
    ...

def is_transpose_only_enabled() -> bool:
    """Return ``True`` if dtFFT was built in transpose-only mode."""
    ...

def is_nccl_enabled() -> bool:
    """Return ``True`` if NCCL support is enabled."""
    ...

def is_nvshmem_enabled() -> bool:
    """Return ``True`` if NVSHMEM support is enabled."""
    ...

def is_compression_enabled() -> bool:
    """Return ``True`` if compression backends are enabled."""
    ...

def get_backend_string(backend: Backend) -> str:
    """Return a human-readable name for the given ``Backend``."""
    ...

def set_config(config: Config) -> None:
    """Apply *config* to dtFFT. Must be called before plan creation to take effect."""
    ...

# ---------------------------------------------------------------------------
# Plan (base class)
# ---------------------------------------------------------------------------

class Plan:
    """Abstract base class for all dtFFT plans. Has no public constructors."""

    # --- Introspection ---

    def get_z_slab_enabled(self) -> bool:
        """Return ``True`` if the plan uses the Z-slab optimisation."""
        ...

    def get_y_slab_enabled(self) -> bool:
        """Return ``True`` if the plan uses the Y-slab optimisation."""
        ...

    def get_pencil(self, layout: Layout) -> Pencil:
        """Return pencil information for the given *layout*."""
        ...

    def get_alloc_size(self) -> int:
        """Minimum number of elements to allocate for ``in``, ``out``, or ``aux``."""
        ...

    def get_aux_size(self) -> int:
        """Number of elements required for the auxiliary buffer during ``execute``."""
        ...

    def get_aux_bytes(self) -> int:
        """Number of bytes required for the auxiliary buffer during ``execute``."""
        ...

    def get_aux_size_reshape(self) -> int:
        """Number of elements required for the auxiliary buffer during ``reshape``."""
        ...

    def get_aux_bytes_reshape(self) -> int:
        """Number of bytes required for the auxiliary buffer during ``reshape``."""
        ...

    def get_aux_size_transpose(self) -> int:
        """Number of elements required for the auxiliary buffer during ``transpose``."""
        ...

    def get_aux_bytes_transpose(self) -> int:
        """Number of bytes required for the auxiliary buffer during ``transpose``."""
        ...

    def get_local_sizes(self) -> tuple[list[int], list[int], list[int], list[int], int]:
        """Return ``(in_starts, in_counts, out_starts, out_counts, alloc_size)``."""
        ...

    def get_dims(self) -> list[int]:
        """Return global dimensions in natural Fortran order."""
        ...

    def get_grid_dims(self) -> list[int]:
        """Return grid decomposition dimensions in natural Fortran order."""
        ...

    def get_element_size(self) -> int:
        """Number of bytes required to store a single element."""
        ...

    def get_alloc_bytes(self) -> int:
        """Minimum bytes required to execute the plan (``alloc_size * element_size``)."""
        ...

    def get_executor(self) -> Executor:
        """Return the FFT executor used by this plan."""
        ...

    def get_precision(self) -> Precision:
        """Return the floating-point precision of this plan."""
        ...

    def get_backend(self) -> Backend:
        """Return the selected communication backend."""
        ...

    def get_reshape_backend(self) -> Backend:
        """Return the backend used for reshape operations."""
        ...

    # CUDA-only methods
    def get_stream(self) -> int:
        """Return the CUDA stream associated with the plan as a raw integer pointer. CUDA builds only."""
        ...

    def get_platform(self) -> Platform:
        """Return the execution platform of this plan. CUDA builds only."""
        ...

    # --- Execution ---

    def execute(
        self,
        input: Any,
        out: Any,
        execute_type: Execute,
        aux: Any = ...,
    ) -> None:
        """Execute the FFT plan.

        *in*, *out*, and *aux* must expose ``__array_interface__`` (host) or
        ``__cuda_array_interface__`` (device).
        """
        ...

    def transpose(
        self,
        input: Any,
        out: Any,
        transpose_type: Transpose,
        aux: Any = ...,
    ) -> None:
        """Transpose data in a single dimension. *in* and *out* must differ."""
        ...

    def transpose_start(
        self,
        input: Any,
        out: Any,
        transpose_type: Transpose,
        aux: Any = ...,
    ) -> int:
        """Start an asynchronous transpose and return an opaque request handle (integer)."""
        ...

    def transpose_end(self, request: int) -> None:
        """Complete an asynchronous transpose identified by *request*."""
        ...

    def reshape(
        self,
        input: Any,
        out: Any,
        reshape_type: Reshape,
        aux: Any = ...,
    ) -> None:
        """Reshape data between bricks and pencils."""
        ...

    def reshape_start(
        self,
        input: Any,
        out: Any,
        reshape_type: Reshape,
        aux: Any = ...,
    ) -> int:
        """Start an asynchronous reshape and return an opaque request handle (integer)."""
        ...

    def reshape_end(self, request: int) -> None:
        """Complete an asynchronous reshape identified by *request*."""
        ...

    # --- Memory helpers ---

    def alloc_ptr(self, alloc_bytes: int) -> int:
        """Allocate platform-specific memory and return raw pointer as integer."""
        ...

    def free_ptr(self, ptr: int) -> None:
        """Free memory previously allocated with :meth:`alloc_ptr`."""
        ...

    # --- Diagnostics ---

    def report(self) -> None:
        """Print plan-related information to stdout."""
        ...

    def report_compression(self) -> None:
        """Print compression-related information to stdout. Compression builds only."""
        ...

    # --- Lifecycle ---

    def destroy(self) -> None:
        """Destroy the plan.  Must be called before ``MPI_Finalize`` for proper cleanup."""
        ...

# ---------------------------------------------------------------------------
# Concrete plan classes
# ---------------------------------------------------------------------------

class PlanC2C(Plan):
    """Complex-to-Complex FFT plan."""

    @overload
    def __init__(
        self,
        dims: list[int],
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...
    @overload
    def __init__(
        self,
        pencil: Pencil,
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...

class PlanR2C(Plan):
    """Real-to-Complex FFT plan. ``executor`` cannot be ``Executor.NONE``."""

    @overload
    def __init__(
        self,
        dims: list[int],
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...
    @overload
    def __init__(
        self,
        pencil: Pencil,
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...

class PlanR2R(Plan):
    """Real-to-Real FFT plan."""

    @overload
    def __init__(
        self,
        dims: list[int],
        kinds: list[R2RKind],
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...
    @overload
    def __init__(
        self,
        pencil: Pencil,
        kinds: list[R2RKind],
        comm_handle: int,
        precision: Precision,
        effort: Effort,
        executor: Executor,
    ) -> None: ...
