#pragma once

#include "config.hpp"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>

#include <dtfft.hpp>

void setup_dtfft_config(bool enable_z_slab) {
  dtfft::Config conf;
  conf.set_enable_z_slab(enable_z_slab);
#ifdef DTFFT_WITH_CUDA
  conf.set_backend(dtfft::Backend::NCCL)
    .set_enable_log(false)
    .set_measure_iters(15)
    .set_measure_warmup_iters(3)
    .set_force_kernel_optimization(true)
    .set_n_configs_to_test(20)
    .set_platform(dtfft::Platform::CUDA);
#endif
  DTFFT_CXX_CALL( dtfft::set_config(conf) );
}


void run_dtfft_internal(dtfft::Plan& plan, const std::string& plan_name, 
  double create_time, dtfft::Precision precision, dtfft::Executor executor, 
  int64_t scaler, bool enable_z_slab) 
{
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if ( enable_z_slab ) {
    bool is_z_slab = plan.get_z_slab_enabled();
    if ( !is_z_slab ) {
      if(comm_rank == 0) {
        printf("Plan is not using Z slab, skipping benchmark\n");
      }
      return;
    }
  }

  if(comm_rank == 0) {
    printf("Plan created successfully, time spent: %f\n", create_time);
  }
  plan.report();

#ifdef DTFFT_WITH_CUDA
  cudaStream_t stream = nullptr;
  dtfft_stream_t dtfftStream = plan.get_stream();
  stream = (cudaStream_t)dtfftStream;
#endif

  size_t alloc_size = plan.get_alloc_size();
  size_t alloc_bytes = alloc_size * (scaler * sizeof(float));

  auto in = static_cast<float*>(plan.mem_alloc(alloc_bytes));
  auto out = static_cast<float*>(plan.mem_alloc(alloc_bytes));
  auto aux = static_cast<float*>(plan.mem_alloc(alloc_bytes));

#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( cudaMemset(in, 0, alloc_bytes));

  cudaEvent_t startEvent, stopEvent;
  CUDA_CALL( cudaEventCreate(&startEvent) );
  CUDA_CALL( cudaEventCreate(&stopEvent) );
  float ms;
#else
  memset(in, 0, alloc_bytes);
#endif

  if(comm_rank == 0) {
    printf("Started warmup\n");
  }
  for ( int iter = 0; iter < WARMUP_ITERATIONS; iter++ ) {
    DTFFT_CXX_CALL( plan.execute(in, out, dtfft::Execute::FORWARD, aux) );
    DTFFT_CXX_CALL( plan.execute(out, in, dtfft::Execute::BACKWARD, aux) );
  }
#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( cudaStreamSynchronize(stream) );
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0) {
    printf("Ended warmup\n");
  }

#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( cudaEventRecord(startEvent, stream) );
#else
  double start_time = MPI_Wtime();
#endif

  for ( int iter = 0; iter < TEST_ITERATIONS; iter++ ) {
    plan.execute(in, out, dtfft::Execute::FORWARD, aux);
    plan.execute(out, in, dtfft::Execute::BACKWARD, aux);
  }

#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( cudaEventRecord(stopEvent, stream) );
  CUDA_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  float min_ms, max_ms, avg_ms;
  MPI_Allreduce(&ms, &min_ms, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ms, &avg_ms, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  avg_ms /= (float)comm_size;
#else
  double end_time = MPI_Wtime();
  double elapsed_time = (end_time - start_time) * 1000.0; // Convert to ms

  double min_ms, max_ms, avg_ms;
  MPI_Allreduce(&elapsed_time, &min_ms, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&elapsed_time, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&elapsed_time, &avg_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avg_ms /= (double)comm_size;
#endif

  if(comm_rank == 0) {
    printf("min time: %f [ms]\n", min_ms);
    printf("max time: %f [ms]\n", max_ms);
    printf("avg time: %f [ms]\n", avg_ms);
    printf("----------------------------------------\n");
  }

  DTFFT_CXX_CALL( plan.mem_free(in) );
  DTFFT_CXX_CALL( plan.mem_free(out) );
  DTFFT_CXX_CALL( plan.mem_free(aux) );

#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( cudaEventDestroy(startEvent) );
  CUDA_CALL( cudaEventDestroy(stopEvent) );
#endif
}

void run_dtfft_c2c(const std::vector<int>& dims, dtfft::Precision precision, dtfft::Executor executor, bool enable_z_slab) {
  double create_time = -MPI_Wtime();
  dtfft::PlanC2C plan(dims, MPI_COMM_WORLD, precision, dtfft::Effort::MEASURE, executor);
  create_time += MPI_Wtime();

  int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 4 : 2;
  run_dtfft_internal(plan, "C2C", create_time, precision, executor, scaler, enable_z_slab);
}

void run_dtfft_r2r(const std::vector<int>& dims, dtfft::Precision precision, bool enable_z_slab) {
  double create_time = -MPI_Wtime();
  dtfft::PlanR2R plan(dims, precision, dtfft::Effort::MEASURE);
  create_time += MPI_Wtime();

  int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 2 : 1;
  run_dtfft_internal(plan, "R2R", create_time, precision, dtfft::Executor::NONE, scaler, enable_z_slab);
}