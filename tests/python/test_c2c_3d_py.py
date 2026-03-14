import os

import sys
executable_path = sys.executable
executable_dir = os.path.dirname(executable_path)
print(f"Executable path: {executable_path}")
print(f"Executable directory: {executable_dir}")

import numpy as np
from mpi4py import MPI

import dtfft
import dtfft._dtfft_test_utils as tu

nx = 34
ny = 256
nz = 333



comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    print("----------------------------------------")
    print("| dtFFT test Python intenface: c2c_3d  |")
    print("----------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm_size}")
    print("----------------------------------------")


myshape = np.array([nx, ny, nz])


print("FFTW is enabled:", dtfft.is_fftw_enabled())
print("MKL is enabled:", dtfft.is_mkl_enabled())
print("cuFFT is enabled:", dtfft.is_cufft_enabled())
print("VKFFT is enabled:", dtfft.is_vkfft_enabled())
print("Transpose-only backend is enabled:", dtfft.is_transpose_only_enabled())
print("NCCL is enabled:", dtfft.is_nccl_enabled())
print("NVSHMEM is enabled:", dtfft.is_nvshmem_enabled())
print("Compression is enabled:", dtfft.is_compression_enabled())

conf = dtfft.Config(enable_log=True, enable_z_slab=False, backend=dtfft.Backend.ADAPTIVE)

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
print(executor)
plan = dtfft.PlanC2C(myshape, executor=executor, config=conf, effort=dtfft.Effort.EXHAUSTIVE)
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

check = np.random.uniform(-1, 1, np.prod(in_counts)) + 1.0j * np.random.uniform(
    -1, 1, np.prod(in_counts)
)

a = plan.get_ndarray(alloc_size)
b = plan.get_ndarray(alloc_size)
tu.complexDoubleH2D(check, a, platform)

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

tu.scaleComplexDouble(executor.value, a, nx * ny * nz, platform, stream.ptr if stream else 0)
tu.checkAndReportComplexDouble(nx * ny * nz, tf, tb, a, check, platform)

request = plan.transpose_start(a, b, dtfft.Transpose.X_TO_Y)
print(request)
plan.transpose_end(request)

request = plan.transpose_start(b, a, dtfft.Transpose.Y_TO_X)
print(request)
plan.transpose_end(request)

tu.checkAndReportComplexDouble(nx * ny * nz, tf, tb, a, check, platform)
