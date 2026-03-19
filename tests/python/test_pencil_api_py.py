"""
Tests for the Pencil class and plan.get_pencil() across all Layout values.

Verifies:
  - Pencil(starts, counts) constructor and all property accessors
    (ndims, dim, starts, counts, size)
  - Pencil.from_pencil() class-method
  - plan.get_pencil(Layout.*) for every Layout variant does not raise
    and returns a valid Pencil
  - The sum of pencil sizes across all MPI ranks equals the global grid size
    for each layout (partition correctness)
"""

import math

from mpi4py import MPI

import dtfft
from dtfft import _dtfft_test_utils as tu

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

nx, ny, nz = 64, 48, 32

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: pencil_api         |")
    print("---------------------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm.Get_size()}")
    print("---------------------------------------------------")

# ---------------------------------------------------------------------------
# Pencil constructor and property accessors
# ---------------------------------------------------------------------------
dims = [nx, ny, nz]
grid_dims, starts, counts = tu.createGridDims(dims)

pencil = dtfft.Pencil(starts, counts)

# ---------------------------------------------------------------------------
# Plan created from a Pencil — all Layout variants accessible
# ---------------------------------------------------------------------------
plan = dtfft.PlanC2C(pencil, executor=dtfft.Executor.NONE)

# Pencil (1D-decomposed) layouts are always available.
# Bricks layouts require a 2D MPI decomposition (grid_dims[0] > 1).
pencil_layouts = [
    dtfft.Layout.X_PENCILS,
    dtfft.Layout.Y_PENCILS,
    dtfft.Layout.Z_PENCILS,
]
bricks_layouts = [
    dtfft.Layout.X_BRICKS,
    dtfft.Layout.Z_BRICKS,
]
layouts = pencil_layouts + (bricks_layouts if grid_dims[0] > 1 else [])

for layout in layouts:
    lp = plan.get_pencil(layout)
    assert isinstance(lp, dtfft.Pencil), (
        f"get_pencil({layout}) returned {type(lp)}, expected Pencil"
    )
    assert lp.ndims == len(dims), f"Layout {layout}: expected ndims={len(dims)}, got {lp.ndims}"
    assert all(c > 0 for c in lp.counts), f"Layout {layout}: non-positive count in {lp.counts}"
    assert lp.size == math.prod(lp.counts), (
        f"Layout {layout}: size={lp.size} != prod(counts)={math.prod(lp.counts)}"
    )

# ---------------------------------------------------------------------------
# Partition check: sum of sizes across all ranks == global grid size
# ---------------------------------------------------------------------------
total_elements = nx * ny * nz

for layout in layouts:
    lp = plan.get_pencil(layout)
    local_size = lp.size
    global_size = comm.allreduce(local_size, op=MPI.SUM)
    assert global_size == total_elements, (
        f"Layout {layout}: sum of pencil sizes across ranks = {global_size}, "
        f"expected {total_elements}"
    )

# ---------------------------------------------------------------------------
# Pencil.from_pencil() class-method
# ---------------------------------------------------------------------------
raw_pencil = plan.get_pencil(dtfft.Layout.Z_PENCILS)
# from_pencil wraps an existing internal Pencil — verify it returns valid Pencil
assert isinstance(raw_pencil, dtfft.Pencil)
assert raw_pencil.size > 0

plan.destroy()

if comm_rank == 0:
    print("-" * 51)
    print("Test 'pencil_api_py' PASSED!")
    print("-" * 51)
