"""
Tests for CompressionConfig across all CompressionMode variants.

The config itself holds no state that can be verified without a plan, so each
variant is validated by creating a plan with that config, executing a
forward+backward R2R round-trip through it, and verifying the data survives.
Also checks report_compression() and the aux_*_reshape properties.

Skipped entirely when the library was built without compression support.
"""

import os
import sys

import numpy as np
from mpi4py import MPI

import dtfft
from dtfft import _dtfft_test_utils as tu

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: compression_config |")
    print("---------------------------------------------------")

if not dtfft.is_compression_enabled():
    if comm_rank == 0:
        print("Compression not enabled — skipping test")
    sys.exit(0)

# if not dtfft.is_fftw_enabled():
#     if comm_rank == 0:
#         print("FFTW not enabled — skipping test (R2R requires FFTW)")
#     sys.exit(0)
    
if comm_size == 1:
    if comm_rank == 0:
        print("Single-process run — skipping test (compression only relevant with MPI)")
    sys.exit(0)

if (
    os.getenv("DTFFT_PLATFORM") is not None
    and os.getenv("DTFFT_PLATFORM").upper() == "CUDA"
    and dtfft.is_cuda_enabled()
):
    if comm_rank == 0:
        print("CUDA platform detected — skipping test")
    sys.exit(0)

nx, ny, nz = 128, 128, 16

if comm_rank == 0:
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm.Get_size()}")
    print("---------------------------------------------------")

dims = [nx, ny, nz]
grid_dims, starts, counts = tu.createGridDims(dims)
pencil = dtfft.Pencil(starts, counts)
kinds = [dtfft.R2RKind.DCT_2] * 3


def make_plan(compression_config):
    """Create a PlanR2R with the given CompressionConfig on both channels."""
    config = dtfft.Config(
        compression_config_transpose=compression_config,
        compression_config_reshape=compression_config,
        backend=dtfft.Backend.MPI_P2P_COMPRESSED,
        reshape_backend=dtfft.Backend.MPI_P2P_COMPRESSED,
    )
    return dtfft.PlanR2R(pencil, kinds=kinds, executor=dtfft.Executor.NONE, config=config)


def run_plan(plan: dtfft.Plan, label):
    """Execute a forward+backward DCT_2 round-trip and verify the result."""
    _, in_counts, _, out_counts, alloc_size = plan.local_sizes
    
    plan.report()

    # check = np.random.uniform(-1.0, 1.0, in_counts).astype(np.float64)
    check = np.ndarray(np.prod(in_counts), dtype=np.float64)
    for i in range(check.size):
        check[i] = float(comm_rank * 1000 + i)
    a = plan.get_ndarray(alloc_size, np.prod(in_counts), dtype=np.float64)
    b = plan.get_ndarray(alloc_size, np.prod(out_counts), dtype=np.float64)    
    a[:] = check

    plan.execute(a, b, dtfft.Execute.FORWARD)
    plan.execute(b, a, dtfft.Execute.BACKWARD)

    # DCT_2 fwd / DCT_3 bwd — scale = 4 * nx * ny * nz (per dimension, FFTW3)
    # For 3D: total scale = (2*nx) * (2*ny) * (2*nz) = 8 * nx * ny * nz
    # a /= 8.0 * nx * ny * nz

    # Lossy modes allow some error; use a relaxed tolerance
    atol = 1e-6 if label == "LOSSLESS" else 0.5
    if not np.allclose(a, check, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: round-trip failed, max_err={np.max(np.abs(a - check)):.3e}")

    plan.report_compression()

    if comm_rank == 0:
        print(f"  {label}: PASSED")


tu.attach_gpu_to_process()

run_plan(
    make_plan(dtfft.CompressionConfig(
        compression_mode=dtfft.CompressionMode.LOSSLESS,
    )),
    "LOSSLESS",
)

run_plan(
    make_plan(
        dtfft.CompressionConfig(
            compression_mode=dtfft.CompressionMode.FIXED_RATE,
            rate=20.0,
        )
    ),
    "FIXED_RATE",
)

run_plan(
    make_plan(
        dtfft.CompressionConfig(
            compression_mode=dtfft.CompressionMode.FIXED_PRECISION,
            precision=32,
        )
    ),
    "FIXED_PRECISION",
)

run_plan(
    make_plan(dtfft.CompressionConfig(
        compression_mode=dtfft.CompressionMode.FIXED_ACCURACY,
        tolerance=1e-3,
    )),
    "FIXED_ACCURACY",
)

if comm_rank == 0:
    print("-" * 51)
    print("Test 'compression_config_py' PASSED!")
    print("-" * 51)
