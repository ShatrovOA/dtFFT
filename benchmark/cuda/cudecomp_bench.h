#include "config.h"
#include <stdlib.h>
#include <math.h>
#include <cudecomp.h>
#include <cuda_runtime.h>


#define CHECK_CUDECOMP_EXIT(call)                                                                                      \
  do {                                                                                                                 \
    cudecompResult_t err = call;                                                                                       \
    if (CUDECOMP_RESULT_SUCCESS != err) {                                                                              \
      fprintf(stderr, "%s:%d CUDECOMP error. (error code %d)\n", __FILE__, __LINE__, err);                             \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define MAX(a,b) (((a)>(b))?(a):(b))

void run_cudecomp(cudecompDataType_t dtype) {
  cudecompHandle_t handle;
  CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("cuDECOMP benchmark\n");
    printf("Plan type is ");
    switch (dtype) {
    case CUDECOMP_FLOAT: printf("CUDECOMP_FLOAT"); break;
    case CUDECOMP_DOUBLE: printf("CUDECOMP_DOUBLE"); break;
    case CUDECOMP_FLOAT_COMPLEX: printf("CUDECOMP_FLOAT_COMPLEX"); break;
    case CUDECOMP_DOUBLE_COMPLEX: printf("CUDECOMP_DOUBLE_COMPLEX"); break;
    }
    printf("\n");
    printf("----------------------------------------\n");
  }


  // Create cuDecomp grid descriptor (with autotuning enabled)
  cudecompGridDescConfig_t config;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));

  // Set pdims entries to 0 to enable process grid autotuning
  config.pdims[0] = 0; // P_rows
  config.pdims[1] = 0; // P_cols

  config.gdims[0] = NX; // X
  config.gdims[1] = NY; // Y
  config.gdims[2] = NZ; // Z

  config.transpose_axis_contiguous[0] = true;
  config.transpose_axis_contiguous[1] = true;
  config.transpose_axis_contiguous[2] = true;
  config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NCCL;

    // Set up autotune options structure
  cudecompGridDescAutotuneOptions_t options;
  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));

    // General options
  options.n_warmup_trials = 3;
  options.n_trials = 5;
  options.dtype = dtype;
  options.disable_nccl_backends = false;
  options.disable_nvshmem_backends = true;
  options.skip_threshold = 0.0;

  // Process grid autotuning options
  options.grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;
  options.allow_uneven_decompositions = true;

  // Transpose communication backend autotuning options
  options.autotune_transpose_backend = false;
  options.transpose_use_inplace_buffers[0] = true;
  options.transpose_use_inplace_buffers[1] = true;
  options.transpose_use_inplace_buffers[2] = true;
  options.transpose_use_inplace_buffers[3] = true;

  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  // Print information on configuration (updated by autotuner)
  if (comm_rank == 0) {
    printf("running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
    printf("running using %s transpose backend...\n",
           cudecompTransposeCommBackendToString(config.transpose_comm_backend));
  }

  // Allocating pencil memory

  // Get X-pencil information (with halo elements).
  cudecompPencilInfo_t pinfo_x;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_x, 0, NULL));

  // Get Y-pencil information
  cudecompPencilInfo_t pinfo_y;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_y, 1, NULL));

  // Get Z-pencil information
  cudecompPencilInfo_t pinfo_z;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo_z, 2, NULL));

  // Allocate pencil memory
  int64_t data_num_elements = MAX(MAX(pinfo_x.size, pinfo_y.size), pinfo_z.size);

    // Get workspace sizes
  int64_t work_size;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &work_size));

  // Allocate using cudecompMalloc
  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(dtype, &dtype_size));

  // Allocate device buffer
  // Using inplace, since out-of-place resulted in autotune.cc:248 CUDA error. (out of memory)
  // Running on a single GPU Tesla V100, 32Gb
  float *inout;
  CUDA_CALL(cudecompMalloc(handle, grid_desc, (void**)&inout, data_num_elements * dtype_size));
  CUDA_CALL(cudaMemset(inout, 0, data_num_elements * dtype_size));

  float* work;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, (void**)(&work),
                                     work_size * dtype_size));

  cudaStream_t stream;
  CUDA_CALL( cudaStreamCreate(&stream) );

  if(comm_rank == 0) {
    printf("Started warmup\n");
  }
  for ( int iter = 0; iter < WARMUP_ITERATIONS; iter++ ) {
    // Transpose from X-pencils to Y-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeXToY(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Y-pencils to Z-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeYToZ(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Z-pencils to Y-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeZToY(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Y-pencils to X-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeYToX(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));
  }
  CUDA_CALL( cudaStreamSynchronize(stream) );

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0) {
    printf("Ended warmup\n");
  }
  cudaEvent_t startEvent, stopEvent;
  CUDA_CALL( cudaEventCreate(&startEvent) );
  CUDA_CALL( cudaEventCreate(&stopEvent) );
  float ms;

  CUDA_CALL( cudaEventRecord(startEvent, stream) );

  for ( int iter = 0; iter < TEST_ITERATIONS; iter++ ) {
    // Transpose from X-pencils to Y-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeXToY(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Y-pencils to Z-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeYToZ(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Z-pencils to Y-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeZToY(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));

    // Transpose from Y-pencils to X-pencils.
    CHECK_CUDECOMP_EXIT(
      cudecompTransposeYToX(handle, grid_desc, inout, inout,
                            work, dtype,
                            NULL, NULL, stream));
  }

  CUDA_CALL( cudaEventRecord(stopEvent, stream) );
  CUDA_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  float min_ms, max_ms, avg_ms;

  MPI_Allreduce(&ms, &min_ms, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &avg_ms, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    printf("min time: %f [ms]\n", min_ms);
    printf("max time: %f [ms]\n", max_ms);
    printf("avg time: %f [ms]\n", avg_ms /= (float)comm_size);
    printf("----------------------------------------\n");
  }

  CUDA_CALL(cudaEventDestroy(startEvent) );
  CUDA_CALL(cudaEventDestroy(stopEvent) );
  CUDA_CALL(cudaStreamDestroy(stream));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, inout));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work));
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
}