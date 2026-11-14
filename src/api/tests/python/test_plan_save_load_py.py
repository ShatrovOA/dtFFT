"""Tests for saving and loading dtFFT plans through the Python API."""

import os
import uuid

from mpi4py import MPI

import dtfft

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

dims = [32, 24, 16]
r2r_kinds = [dtfft.R2RKind.DCT_2, dtfft.R2RKind.DCT_3, dtfft.R2RKind.DCT_2]

if comm_rank == 0:
    print("---------------------------------------------------")
    print("| dtFFT test Python interface: plan_save_load     |")
    print("---------------------------------------------------")
    print(f" Number of processors: {comm.Get_size()}")


def roundtrip(label, plan, load_plan):
    filename = comm.bcast(
        f"test_save_load_py_{uuid.uuid4().hex}.dtfft" if comm_rank == 0 else None,
        root=0,
    )
    loaded = None
    try:
        plan.save(filename)
        loaded = load_plan(filename)

        for attr in (
            "dims",
            "grid_dims",
            "precision",
            "executor",
            "dtype",
            "alloc_size",
            "alloc_bytes",
            "element_size",
            "local_sizes",
        ):
            actual = getattr(loaded, attr)
            expected = getattr(plan, attr)
            assert actual == expected, f"{label}: {attr} differs: {actual!r} != {expected!r}"

        if comm_rank == 0:
            print(f"  {label}: PASSED")
    finally:
        if loaded is not None:
            loaded.destroy()
        plan.destroy()
        comm.Barrier()
        if comm_rank == 0 and os.path.exists(filename):
            os.remove(filename)
        comm.Barrier()


roundtrip(
    "C2C",
    dtfft.PlanC2C(dims, executor=dtfft.Executor.NONE),
    lambda filename: dtfft.PlanC2C(dims, filename=filename),
)

roundtrip(
    "R2R",
    dtfft.PlanR2R(dims, kinds=r2r_kinds, executor=dtfft.Executor.NONE),
    lambda filename: dtfft.PlanR2R(dims, kinds=r2r_kinds, filename=filename),
)

if dtfft.is_fftw_enabled():
    roundtrip(
        "R2C",
        dtfft.PlanR2C(dims, executor=dtfft.Executor.FFTW3),
        lambda filename: dtfft.PlanR2C(dims, filename=filename),
    )

if comm_rank == 0:
    print("-" * 51)
    print("Test 'plan_save_load_py' PASSED!")
    print("-" * 51)
