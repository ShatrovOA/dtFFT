"""
Tests for Plan property accessors, manual memory management, and module-level
query functions. Does not execute any FFT — uses Executor.NONE to stay
executor-agnostic.
"""

import dtfft
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

nx, ny, nz = 34, 57, 23

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: plan_properties    |")
    print("---------------------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm.Get_size()}")
    print("---------------------------------------------------")

# ---------------------------------------------------------------------------
# Double-precision C2C plan (Executor.NONE — no FFT, just structure)
# ---------------------------------------------------------------------------
plan = dtfft.PlanC2C([nx, ny, nz], executor=dtfft.Executor.NONE)

# dtype
assert plan.dtype == np.complex128, f"Expected complex128, got {plan.dtype}"

# dims: length must match the number of FFT dimensions
assert len(plan.dims) == 3, f"Expected 3 dims, got {len(plan.dims)}"

# grid_dims: one value per decomposed dimension
assert len(plan.grid_dims) == 3, f"Expected 3 grid_dims, got {len(plan.grid_dims)}"

# element_size and alloc_bytes consistency
assert plan.element_size == 16, f"Expected 16 bytes (complex128), got {plan.element_size}"
assert plan.alloc_bytes == plan.alloc_size * plan.element_size, (
    f"alloc_bytes ({plan.alloc_bytes}) != alloc_size ({plan.alloc_size}) * element_size ({plan.element_size})"
)

# slab flags are bool
assert isinstance(plan.z_slab_enabled, bool)
assert isinstance(plan.y_slab_enabled, bool)

# precision enum
assert plan.precision == dtfft.Precision.DOUBLE

# executor / backend do not raise and return something
_ = plan.executor
_ = plan.backend
# reshape_backend is only meaningful with a bricks (2D) decomposition
if plan.grid_dims[0] > 1:
    _ = plan.reshape_backend

# get_backend_string accepts a Backend value and returns a non-empty string
s = dtfft.get_backend_string(plan.backend)
assert isinstance(s, str) and len(s) > 0, f"get_backend_string returned: {s!r}"

# aux size properties are non-negative integers
for attr in ("aux_size", "aux_bytes", "aux_size_transpose", "aux_bytes_transpose"):
    v = getattr(plan, attr)
    assert isinstance(v, int) and v >= 0, f"{attr} = {v!r} is not a non-negative int"
# reshape aux is only relevant with a bricks (2D) decomposition
if plan.grid_dims[0] > 1:
    for attr in ("aux_size_reshape", "aux_bytes_reshape"):
        v = getattr(plan, attr)
        assert isinstance(v, int) and v >= 0, f"{attr} = {v!r} is not a non-negative int"

# local_sizes returns a 5-tuple: (in_starts, in_counts, out_starts, out_counts, alloc_size)
ls = plan.local_sizes
assert len(ls) == 5
in_starts, in_counts, out_starts, out_counts, alloc_size = ls
assert len(in_starts) == 3 and len(in_counts) == 3
assert len(out_starts) == 3 and len(out_counts) == 3
assert alloc_size > 0

# report() must not raise
plan.report()

# ---------------------------------------------------------------------------
# get_ndarray — plan-managed array allocation
# ---------------------------------------------------------------------------
_, in_counts, _, out_counts, alloc_size = plan.local_sizes

# Determine expected array type: cupy on CUDA platform, numpy otherwise
if dtfft.is_cuda_enabled() and plan.platform == dtfft.Platform.CUDA:
    import cupy as cp
    expected_array_type = cp.ndarray
else:
    expected_array_type = np.ndarray

# default dtype matches plan.dtype
a = plan.get_ndarray(alloc_size, in_counts)
assert isinstance(a, expected_array_type), f"Expected {expected_array_type}, got {type(a)}"
assert a.dtype == plan.dtype, f"dtype mismatch: {a.dtype} != {plan.dtype}"
assert list(a.shape) == list(in_counts), f"shape mismatch: {a.shape} vs {in_counts}"

# explicit dtype override
b = plan.get_ndarray(alloc_size, out_counts, dtype=np.complex128)
assert b.dtype == np.complex128

# 1-D flat allocation (no shape argument)
flat = plan.get_ndarray(alloc_size)
assert flat.ndim == 1 and flat.size == plan.alloc_size

# ---------------------------------------------------------------------------
# stream / platform — CUDA-only properties
# ---------------------------------------------------------------------------
if dtfft.is_cuda_enabled():
    import cupy as cp
    plat = plan.platform
    assert plat in (dtfft.Platform.HOST, dtfft.Platform.CUDA)
    if plat == dtfft.Platform.CUDA:
        stream = plan.stream
        assert isinstance(stream, cp.cuda.ExternalStream)

# ---------------------------------------------------------------------------
# destroy() is deferred while live ndarray allocations exist
# ---------------------------------------------------------------------------
# Call destroy() while `a`, `b`, `flat` are still alive.
# The underlying plan must NOT be released yet — _plan stays non-None
# and _destroyed stays False because destroy() is a no-op when _alloc_count > 0.
plan.destroy()
assert plan._plan is not None, "Plan was destroyed prematurely while ndarrays are still live"
assert plan._destroyed is False, "_destroyed should be False while allocations exist"
# Properties must still be accessible (plan is alive)
plan.report()

# Release all allocations — now the plan can be cleaned up on GC
del a, b, flat
# ---------------------------------------------------------------------------
# explicit destroy() when no allocations are live — actually releases the plan
# ---------------------------------------------------------------------------
plan.destroy()
assert plan._plan is None, "Plan should be destroyed now (no live allocations)"
assert plan._destroyed is True
plan.destroy()  # second call must be safe (idempotent)

# ---------------------------------------------------------------------------
# Single-precision C2C plan — check dtype and element_size
# ---------------------------------------------------------------------------
plan_f = dtfft.PlanC2C([nx, ny], precision=dtfft.Precision.SINGLE, executor=dtfft.Executor.NONE)

assert plan_f.dtype == np.complex64, f"Expected complex64, got {plan_f.dtype}"
assert plan_f.element_size == 8, f"Expected 8 bytes (complex64), got {plan_f.element_size}"
assert plan_f.precision == dtfft.Precision.SINGLE

plan_f.destroy()

# ---------------------------------------------------------------------------
# Module-level boolean query functions
# ---------------------------------------------------------------------------
assert isinstance(dtfft.is_fftw_enabled(), bool)
assert isinstance(dtfft.is_mkl_enabled(), bool)
assert isinstance(dtfft.is_cufft_enabled(), bool)
assert isinstance(dtfft.is_vkfft_enabled(), bool)
assert isinstance(dtfft.is_cuda_enabled(), bool)
assert isinstance(dtfft.is_transpose_only_enabled(), bool)
assert isinstance(dtfft.is_nccl_enabled(), bool)
assert isinstance(dtfft.is_nvshmem_enabled(), bool)
assert isinstance(dtfft.is_compression_enabled(), bool)

if comm_rank == 0:
    print("-" * 51)
    print("Test 'plan_properties_py' PASSED!")
    print("-" * 51)
