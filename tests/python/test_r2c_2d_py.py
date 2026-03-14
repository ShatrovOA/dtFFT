import os

import numpy as np
from mpi4py import MPI

import dtfft
import dtfft._dtfft_test_utils as tu

nx = 49
ny = 321

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    print("----------------------------------------")
    print("| dtFFT test Python interface: r2c_2d  |")
    print("----------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}")
    print(f" Number of processors: {comm_size}")
    print("----------------------------------------")


myshape = np.array([nx, ny], dtype=np.int32)

executor = dtfft.Executor.NONE
if dtfft.is_fftw_enabled():
    executor = dtfft.Executor.FFTW3
elif dtfft.is_mkl_enabled():
    executor = dtfft.Executor.MKL

if (
    os.getenv("DTFFT_PLATFORM") is not None
    and os.getenv("DTFFT_PLATFORM").upper() == "CUDA"
    and dtfft.is_cuda_enabled()
):
    executor = dtfft.Executor.CUFFT
    if dtfft.is_vkfft_enabled():
        executor = dtfft.Executor.VKFFT

tu.attach_gpu_to_process()
plan = dtfft.PlanR2C(myshape[::-1].copy(), executor=executor)
_, in_counts, _, out_counts, alloc_size = plan.local_sizes

platform = 0
stream = None
running_cuda = False
if dtfft.is_cuda_enabled():
    platform = plan.platform
    running_cuda = platform == dtfft.Platform.CUDA
    if running_cuda:
        stream = plan.stream
    platform = platform.value

check = np.random.uniform(-1, 1, np.prod(in_counts))

a = plan.get_ndarray(alloc_size, shape=in_counts)
b = plan.get_ndarray(alloc_size, shape=out_counts)
tu.doubleH2D(check, a, platform)

ts = MPI.Wtime()
plan.execute(a, b, dtfft.Execute.FORWARD)
if running_cuda:
    stream.synchronize()
te = MPI.Wtime()
tf = te - ts

ts = MPI.Wtime()
plan.execute(b, a, dtfft.Execute.BACKWARD)
if running_cuda:
    stream.synchronize()
te = MPI.Wtime()
tb = te - ts

tu.scaleDouble(executor.value, a, nx * ny, platform, stream if stream is not None else 0)
tu.checkAndReportDouble(nx * ny, tf, tb, a, check, platform)
