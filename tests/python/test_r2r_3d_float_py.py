import numpy as np
from mpi4py import MPI

import dtfft
from dtfft import _dtfft_test_utils as tu

nx = 128
ny = 32
nz = 16

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


if comm_rank == 0:
    print("---------------------------------------------")
    print("| dtFFT test Python interface: r2r_3d_float |")
    print("---------------------------------------------")
    print(f" Nx = {nx}, Ny = {ny}, Nz = {nz}")
    print(f" Number of processors: {comm_size}")
    print("---------------------------------------------")


dims = [nx, ny, nz]
grid_dims, starts, counts = tu.createGridDims(dims)
pencil = dtfft.Pencil(starts, counts)

config = dtfft.Config(enable_datatype_backend=False, enable_mpi_backends=True)
config.enable_log = True
config.access_mode = dtfft.AccessMode.READ
config.transpose_mode = dtfft.TransposeMode.UNPACK
config.reshape_backend = dtfft.Backend.MPI_P2P_FUSED

if dtfft.is_compression_enabled():
    compression_config = dtfft.CompressionConfig(
        compression_mode=dtfft.CompressionMode.FIXED_RATE, rate=16.0
    )
    config.compression_config_transpose = compression_config

tu.attach_gpu_to_process()
plan = dtfft.PlanR2R(
    pencil, precision=dtfft.Precision.SINGLE, config=config, effort=dtfft.Effort.PATIENT
)
print("Plan created with executor:", plan.executor)
plan.report()
_, in_counts, _, out_counts, alloc_size = plan.local_sizes


p = plan.get_pencil(dtfft.Layout.Z_PENCILS)
print(
    f"Z-pencil layout on rank {comm_rank}: dim = {p.dim}, starts = {p.starts}, counts = {p.counts}, size = {p.size}"
)

platform = 0
stream = None
running_cuda = False
if dtfft.is_cuda_enabled():
    platform = plan.platform
    running_cuda = platform == dtfft.Platform.CUDA
    if running_cuda:
        stream = plan.stream
    platform = platform.value

check = np.random.uniform(-1, 1, in_counts).astype(np.float32)
a = plan.get_ndarray(alloc_size, in_counts)
b = plan.get_ndarray(alloc_size, out_counts)
aux = plan.get_ndarray(plan.aux_size)
tu.floatH2D(check, a, platform)

ts = MPI.Wtime()
if grid_dims[0] > 1:
    plan.reshape(a, b, dtfft.Reshape.X_BRICKS_TO_PENCILS, aux=aux)
    plan.transpose(b, a, dtfft.Transpose.X_TO_Y, aux=aux)
    plan.transpose(a, b, dtfft.Transpose.Y_TO_Z, aux=aux)
    plan.reshape(b, a, dtfft.Reshape.Z_PENCILS_TO_BRICKS, aux=aux)
else:
    plan.transpose(a, b, dtfft.Transpose.X_TO_Y)
    plan.transpose(b, a, dtfft.Transpose.Y_TO_Z)
if running_cuda:
    stream.synchronize()
te = MPI.Wtime()
tf = te - ts

ts = MPI.Wtime()
if grid_dims[0] > 1:
    plan.reshape(a, b, dtfft.Reshape.Z_BRICKS_TO_PENCILS, aux=aux)
    plan.transpose(b, a, dtfft.Transpose.Z_TO_Y, aux=aux)
    plan.transpose(a, b, dtfft.Transpose.Y_TO_X, aux=aux)
    plan.reshape(b, a, dtfft.Reshape.X_PENCILS_TO_BRICKS, aux=aux)
else:
    plan.transpose(a, b, dtfft.Transpose.Z_TO_Y)
    plan.transpose(b, a, dtfft.Transpose.Y_TO_X)
if running_cuda:
    stream.synchronize()
te = MPI.Wtime()
tb = te - ts

tu.checkAndReportFloat(nx * ny * nz, tf, tb, a, check, platform)
