"""Type stubs for dtFFT Python interface."""

from abc import ABC
from enum import Enum
from typing import Any, overload

from mpi4py.MPI import Comm

__version__: str

class dtfft_Exception(RuntimeError): ...

class Execute(Enum):
    FORWARD: int
    BACKWARD: int

class Transpose(Enum):
    X_TO_Y: int
    Y_TO_X: int
    Y_TO_Z: int
    Z_TO_Y: int
    X_TO_Z: int
    Z_TO_X: int

class Reshape(Enum):
    X_BRICKS_TO_PENCILS: int
    X_PENCILS_TO_BRICKS: int
    Z_BRICKS_TO_PENCILS: int
    Z_PENCILS_TO_BRICKS: int
    Y_BRICKS_TO_PENCILS: int
    Y_PENCILS_TO_BRICKS: int

class Layout(Enum):
    X_BRICKS: int
    X_PENCILS: int
    X_PENCILS_FOURIER: int
    Y_PENCILS: int
    Z_PENCILS: int
    Z_BRICKS: int

class TransposeMode(Enum):
    PACK: int
    UNPACK: int

class AccessMode(Enum):
    WRITE: int
    READ: int

class Precision(Enum):
    SINGLE: int
    DOUBLE: int

class Effort(Enum):
    ESTIMATE: int
    MEASURE: int
    PATIENT: int
    EXHAUSTIVE: int

class Executor(Enum):
    NONE: int
    FFTW3: int
    MKL: int
    CUFFT: int
    VKFFT: int

class R2RKind(Enum):
    DCT_1: int
    DCT_2: int
    DCT_3: int
    DCT_4: int
    DST_1: int
    DST_2: int
    DST_3: int
    DST_4: int

class Backend(Enum):
    MPI_DATATYPE: int
    MPI_P2P: int
    MPI_P2P_PIPELINED: int
    MPI_A2A: int
    MPI_RMA: int
    MPI_RMA_PIPELINED: int
    MPI_P2P_SCHEDULED: int
    MPI_P2P_FUSED: int
    MPI_RMA_FUSED: int
    MPI_P2P_COMPRESSED: int
    MPI_RMA_COMPRESSED: int
    NCCL: int
    NCCL_PIPELINED: int
    NCCL_COMPRESSED: int
    CUFFTMP: int
    CUFFTMP_PIPELINED: int
    ADAPTIVE: int
    NONE: int

class Platform(Enum):
    HOST: int
    CUDA: int

class CompressionMode(Enum):
    LOSSLESS: int
    FIXED_RATE: int
    FIXED_PRECISION: int
    FIXED_ACCURACY: int

class CompressionLib(Enum):
    ZFP: int

class CompressionConfig:
    def __init__(
        self,
        compression_lib: CompressionLib = CompressionLib.ZFP,
        compression_mode: CompressionMode = CompressionMode.LOSSLESS,
        rate: float = 0.0,
        precision: int = 0,
        tolerance: float = 0.0,
    ) -> None: ...

class Version:
    MAJOR: int
    MINOR: int
    PATCH: int
    CODE: int
    @staticmethod
    @overload
    def get() -> int: ...
    @staticmethod
    @overload
    def get(major: int, minor: int, patch: int) -> int: ...

class Pencil:
    def __init__(self, starts: list[int], counts: list[int]) -> None: ...
    @property
    def ndims(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def starts(self) -> list[int]: ...
    @property
    def counts(self) -> list[int]: ...
    @property
    def size(self) -> int: ...

class Config:
    enable_log: bool
    enable_z_slab: bool
    enable_y_slab: bool
    measure_warmup_iters: int
    measure_iters: int
    backend: Backend
    reshape_backend: Backend
    enable_datatype_backend: bool
    enable_mpi_backends: bool
    enable_pipelined_backends: bool
    enable_rma_backends: bool
    enable_fused_backends: bool
    enable_kernel_autotune: bool
    enable_fourier_reshape: bool
    transpose_mode: TransposeMode
    access_mode: AccessMode
    platform: Platform
    stream: int
    enable_nccl_backends: bool
    enable_nvshmem_backends: bool
    enable_compressed_backends: bool
    compression_config_transpose: CompressionConfig
    compression_config_reshape: CompressionConfig

    def __init__(
        self,
        *,
        enable_log: bool | None = ...,
        enable_z_slab: bool | None = ...,
        enable_y_slab: bool | None = ...,
        measure_warmup_iters: int | None = ...,
        measure_iters: int | None = ...,
        backend: Backend | None = ...,
        reshape_backend: Backend | None = ...,
        enable_datatype_backend: bool | None = ...,
        enable_mpi_backends: bool | None = ...,
        enable_pipelined_backends: bool | None = ...,
        enable_rma_backends: bool | None = ...,
        enable_fused_backends: bool | None = ...,
        enable_kernel_autotune: bool | None = ...,
        enable_fourier_reshape: bool | None = ...,
        transpose_mode: TransposeMode | None = ...,
        access_mode: AccessMode | None = ...,
        platform: Platform | None = ...,
        stream: int | None = ...,
        enable_nccl_backends: bool | None = ...,
        enable_nvshmem_backends: bool | None = ...,
        enable_compressed_backends: bool | None = ...,
        compression_config_transpose: CompressionConfig | None = ...,
        compression_config_reshape: CompressionConfig | None = ...,
    ) -> None: ...

class Request:
    def __repr__(self) -> str: ...

def is_fftw_enabled() -> bool: ...
def is_mkl_enabled() -> bool: ...
def is_cufft_enabled() -> bool: ...
def is_vkfft_enabled() -> bool: ...
def is_cuda_enabled() -> bool: ...
def is_transpose_only_enabled() -> bool: ...
def is_nccl_enabled() -> bool: ...
def is_nvshmem_enabled() -> bool: ...
def is_compression_enabled() -> bool: ...
def get_backend_string(backend: Backend) -> str: ...

class Plan(ABC):
    def execute(self, input: Any, output: Any, execute_type: Execute, aux: Any = ...) -> None: ...
    def transpose(
        self, input: Any, output: Any, transpose_type: Transpose, aux: Any = ...
    ) -> None: ...
    def transpose_start(
        self, input: Any, output: Any, transpose_type: Transpose, aux: Any = ...
    ) -> Request: ...
    def transpose_end(self, request: Request | int) -> None: ...
    def reshape(self, input: Any, output: Any, reshape_type: Reshape, aux: Any = ...) -> None: ...
    def reshape_start(
        self, input: Any, output: Any, reshape_type: Reshape, aux: Any = ...
    ) -> Request: ...
    def reshape_end(self, request: Request | int) -> None: ...
    def report(self) -> None: ...
    def report_compression(self) -> None: ...
    def get_pencil(self, layout: Layout) -> Pencil: ...
    @property
    def z_slab_enabled(self) -> bool: ...
    @property
    def y_slab_enabled(self) -> bool: ...
    @property
    def alloc_size(self) -> int: ...
    @property
    def aux_size(self) -> int: ...
    @property
    def aux_bytes(self) -> int: ...
    @property
    def aux_size_reshape(self) -> int: ...
    @property
    def aux_bytes_reshape(self) -> int: ...
    @property
    def aux_size_transpose(self) -> int: ...
    @property
    def aux_bytes_transpose(self) -> int: ...
    @property
    def local_sizes(self) -> tuple[list[int], list[int], list[int], list[int], int]: ...
    @property
    def dims(self) -> list[int]: ...
    @property
    def grid_dims(self) -> list[int]: ...
    @property
    def element_size(self) -> int: ...
    @property
    def alloc_bytes(self) -> int: ...
    @property
    def executor(self) -> Executor: ...
    @property
    def precision(self) -> Precision: ...
    @property
    def backend(self) -> Backend: ...
    @property
    def reshape_backend(self) -> Backend: ...
    @property
    def stream(self) -> int: ...
    @property
    def platform(self) -> Platform: ...
    def alloc_ptr(self, alloc_bytes: int) -> int: ...
    def free_ptr(self, ptr: int) -> None: ...
    def get_ndarray(
        self, size: int, shape: tuple[int, ...] = ..., dtype: Any = ..., order: str = ...
    ) -> Any: ...
    def destroy(self) -> None: ...
    def __del__(self) -> None: ...
    @property
    def dtype(self) -> Any: ...

class PlanC2C(Plan):
    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        comm: Comm | int = ...,
        precision: Precision = ...,
        effort: Effort = ...,
        executor: Executor = ...,
        config: Config | None = ...,
    ) -> None: ...

class PlanR2C(Plan):
    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        comm: Comm | int = ...,
        precision: Precision = ...,
        effort: Effort = ...,
        executor: Executor = ...,
        config: Config | None = ...,
    ) -> None: ...

class PlanR2R(Plan):
    def __init__(
        self,
        dims_or_pencil: list[int] | Pencil,
        kinds: list[R2RKind] = ...,
        comm: Comm | int = ...,
        precision: Precision = ...,
        effort: Effort = ...,
        executor: Executor = ...,
        config: Config | None = ...,
    ) -> None: ...
