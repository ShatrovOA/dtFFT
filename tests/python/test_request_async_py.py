"""
Tests for the asynchronous transpose/reshape API and the Request wrapper object.

Verifies:
  - transpose_start() returns a dtfft.Request instance
  - Request.handle, Request.kind, repr(), int() semantics
  - transpose_end() accepts both a Request object and a raw int handle
  - reshape_start() / reshape_end() with Request and raw int (when Pencil path active)
"""

import numpy as np
from mpi4py import MPI

import dtfft
from dtfft import _dtfft_test_utils as tu

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

nx, ny, nz = 41, 63, 17

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: request_async      |")
    print("---------------------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm.Get_size()}")
    print("---------------------------------------------------")

tu.attach_gpu_to_process()
# ---------------------------------------------------------------------------
# C2C 3D plan — test transpose_start / transpose_end
# ---------------------------------------------------------------------------
plan = dtfft.PlanC2C([nx, ny, nz])
_, in_counts, _, out_counts, alloc_size = plan.local_sizes


check = np.random.uniform(-1, 1, np.prod(in_counts)) + 1.0j * np.random.uniform(
    -1, 1, np.prod(in_counts)
)

a = plan.get_ndarray(alloc_size, in_counts)
b = plan.get_ndarray(alloc_size, out_counts)

platform = 0
stream = None
running_cuda = False
if dtfft.is_cuda_enabled():
    platform = plan.platform
    running_cuda = platform == dtfft.Platform.CUDA
    if running_cuda:
        stream = plan.stream
    platform = platform.value

tu.complexDoubleH2D(check, a, platform)

# --- transpose_start returns a dtfft.Request ---
request = plan.transpose_start(a, b, dtfft.Transpose.X_TO_Y)
assert isinstance(request, dtfft.Request), f"Expected dtfft.Request, got {type(request)}"

# Request.handle is a non-zero int
assert isinstance(request.handle, int), f"handle is not int: {type(request.handle)}"
assert request.handle != 0, "handle should not be 0"

# Request.kind is a non-empty string
assert isinstance(request.kind, str), f"kind is not str: {type(request.kind)}"
assert len(request.kind) > 0, "kind must not be empty"

# repr() contains the kind string
assert request.kind in repr(request), f"kind {request.kind!r} not found in repr: {repr(request)}"

# int(request) equals request.handle
assert int(request) == request.handle, (
    f"int(request)={int(request)} != request.handle={request.handle}"
)

# Complete via Request object
plan.transpose_end(request)
if running_cuda:
    stream.synchronize()

# --- transpose_end also accepts a raw int handle ---
request2 = plan.transpose_start(b, a, dtfft.Transpose.Y_TO_X)
raw_handle = request2.handle
plan.transpose_end(raw_handle)  # pass int directly

if running_cuda:
    stream.synchronize()

# ---------------------------------------------------------------------------
# R2R 3D plan with Pencil decomposition — test reshape_start / reshape_end
# ---------------------------------------------------------------------------
dims = [nx, ny, nz]
grid_dims, starts, counts = tu.createGridDims(dims)

if grid_dims[0] > 1:
    # Only exercise reshape path when there is actually a 3D MPI decomposition
    decomp = dtfft.Pencil(starts, counts)
    kinds = [dtfft.R2RKind.DCT_2] * 3 if dtfft.is_fftw_enabled() else None
    executor = dtfft.Executor.FFTW3 if dtfft.is_fftw_enabled() else dtfft.Executor.NONE
    plan_r = dtfft.PlanR2R(decomp, kinds=kinds, executor=executor)

    _, in_c, _, out_c, alloc_size = plan_r.local_sizes
    brick = plan_r.get_pencil(dtfft.Layout.X_BRICKS)
    pencil = plan_r.get_pencil(dtfft.Layout.X_PENCILS)
    p = plan_r.get_ndarray(alloc_size, brick.counts)
    q = plan_r.get_ndarray(alloc_size, pencil.counts)

    # reshape_start returns Request
    req = plan_r.reshape_start(p, q, dtfft.Reshape.X_BRICKS_TO_PENCILS)
    assert isinstance(req, dtfft.Request), f"reshape_start: expected dtfft.Request, got {type(req)}"
    assert isinstance(req.kind, str) and len(req.kind) > 0
    assert req.handle != 0

    # Complete via Request object
    plan_r.reshape_end(req)

    # reshape_end also accepts raw int
    req2 = plan_r.reshape_start(q, p, dtfft.Reshape.X_PENCILS_TO_BRICKS)
    plan_r.reshape_end(req2.handle)  # int path

    plan_r.destroy()

plan.destroy()

if comm_rank == 0:
    print("-" * 51)
    print("Test 'request_async_py' PASSED!")
    print("-" * 51)
