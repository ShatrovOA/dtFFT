"""
Tests for PlanR2R with multiple DCT/DST transform kinds.

For each kind (or inverse pair) a forward+backward round-trip is performed and
the result is compared to the original input after applying the correct FFTW
normalization factor.

Scale conventions (FFTW3, for a 2D grid [nx, ny]):
  DCT_2 fwd / DCT_3 bwd  — pair   : scale = 4 * nx * ny
  DCT_3 fwd / DCT_2 bwd  — pair   : scale = 4 * nx * ny
  DCT_1 fwd / DCT_1 bwd  — self   : scale = 4 * (nx - 1) * (ny - 1)
  DCT_4 fwd / DCT_4 bwd  — self   : scale = 4 * nx * ny
  DST_1 fwd / DST_1 bwd  — self   : scale = 4 * (nx + 1) * (ny + 1)
  DST_2 fwd / DST_3 bwd  — pair   : scale = 4 * nx * ny
  DST_4 fwd / DST_4 bwd  — self   : scale = 4 * nx * ny

Skipped entirely when FFTW is not available.
Also exercises Effort.MEASURE to verify non-default effort levels are accepted.
"""

import sys
import dtfft
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: r2r_kinds          |")
    print("---------------------------------------------------")

if not dtfft.is_fftw_enabled():
    if comm_rank == 0:
        print("FFTW not enabled — skipping R2R kinds test")
    sys.exit(0)

# Small primes: reduces likelihood of accidental pass with wrong scale
nx, ny = 17, 23

if comm_rank == 0:
    print(f" Nx = {nx}, Ny = {ny}")
    print(f" Number of processors: {comm.Get_size()}")
    print("---------------------------------------------------")


def run_roundtrip(kinds: list, scale: float, label: str) -> None:
    """Create a PlanR2R, execute FORWARD then BACKWARD, verify round-trip."""
    plan = dtfft.PlanR2R(
        [nx, ny],
        kinds=kinds,
        executor=dtfft.Executor.FFTW3,
        effort=dtfft.Effort.ESTIMATE,
    )
    _, in_counts, _, out_counts, alloc_size = plan.local_sizes

    check = np.random.uniform(-1.0, 1.0, in_counts).astype(np.float64)
    a = plan.get_ndarray(alloc_size, in_counts, dtype=np.float64)
    b = plan.get_ndarray(alloc_size, out_counts, dtype=np.float64)
    a[:] = check

    plan.execute(a, b, dtfft.Execute.FORWARD)
    plan.execute(b, a, dtfft.Execute.BACKWARD)
    a /= scale

    if not np.allclose(a, check, atol=1e-10):
        raise AssertionError(
            f"Round-trip failed for {label}: "
            f"max_err={np.max(np.abs(a - check)):.3e}"
        )

    plan.destroy()
    if comm_rank == 0:
        print(f"  {label}: PASSED")


# DCT_2 / DCT_3 (inverse pair)
run_roundtrip(
    [dtfft.R2RKind.DCT_2, dtfft.R2RKind.DCT_2],
    scale=4.0 * nx * ny,
    label="DCT_2 fwd → DCT_3 bwd",
)

# DCT_3 / DCT_2 (inverse pair, reverse direction)
run_roundtrip(
    [dtfft.R2RKind.DCT_3, dtfft.R2RKind.DCT_3],
    scale=4.0 * nx * ny,
    label="DCT_3 fwd → DCT_2 bwd",
)

# DCT_1 (self-inverse)
run_roundtrip(
    [dtfft.R2RKind.DCT_1, dtfft.R2RKind.DCT_1],
    scale=4.0 * (nx - 1) * (ny - 1),
    label="DCT_1 (self-inverse)",
)

# DCT_4 (self-inverse)
run_roundtrip(
    [dtfft.R2RKind.DCT_4, dtfft.R2RKind.DCT_4],
    scale=4.0 * nx * ny,
    label="DCT_4 (self-inverse)",
)

# DST_1 (self-inverse)
run_roundtrip(
    [dtfft.R2RKind.DST_1, dtfft.R2RKind.DST_1],
    scale=4.0 * (nx + 1) * (ny + 1),
    label="DST_1 (self-inverse)",
)

# DST_2 / DST_3 (inverse pair)
run_roundtrip(
    [dtfft.R2RKind.DST_2, dtfft.R2RKind.DST_2],
    scale=4.0 * nx * ny,
    label="DST_2 fwd → DST_3 bwd",
)

# DST_4 (self-inverse)
run_roundtrip(
    [dtfft.R2RKind.DST_4, dtfft.R2RKind.DST_4],
    scale=4.0 * nx * ny,
    label="DST_4 (self-inverse)",
)

# Effort.MEASURE — verify non-default effort level is accepted without error
plan_m = dtfft.PlanR2R(
    [nx, ny],
    kinds=[dtfft.R2RKind.DCT_2, dtfft.R2RKind.DCT_2],
    executor=dtfft.Executor.FFTW3,
    effort=dtfft.Effort.MEASURE,
)
plan_m.destroy()
if comm_rank == 0:
    print("  Effort.MEASURE: PASSED")

if comm_rank == 0:
    print("-" * 51)
    print("Test 'r2r_kinds_py' PASSED!")
    print("-" * 51)
