#pragma once

#include "config.hpp"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>

#include <dtfft.hpp>

void setup_dtfft_config(bool enable_z_slab, dtfft::Backend backend = dtfft::Backend::MPI_P2P) {
  dtfft::Config conf;
  conf.set_enable_z_slab(enable_z_slab)
    .set_backend(backend)
    .set_reshape_backend(backend)
    .set_enable_log(true)
    .set_measure_iters(10)
    .set_measure_warmup_iters(3)
    .set_enable_kernel_autotune(true)
    .set_enable_fourier_reshape(true)
    .set_enable_pipelined_backends(true)
    .set_enable_mpi_backends(false)
    .set_enable_nccl_backends(true)
    .set_enable_nvshmem_backends(true)
    .set_platform(dtfft::Platform::CUDA);

  DTFFT_CXX_CALL( dtfft::set_config(conf) );
}


double run_dtfft_internal(dtfft::Plan& plan, const std::string& plan_name, 
  dtfft::Precision precision, dtfft::Executor executor, 
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
      return -1.0;
    }
  }

  if(comm_rank == 0) {
    printf("Plan created successfully\n");
  }
  plan.report();

#ifdef DTFFT_WITH_CUDA
  cudaStream_t stream = nullptr;
  dtfft_stream_t dtfftStream = plan.get_stream();
  stream = (cudaStream_t)dtfftStream;
#endif

  size_t alloc_size = plan.get_alloc_size();
  size_t alloc_bytes = alloc_size * (scaler * sizeof(float));
  size_t aux_bytes = plan.get_aux_bytes();

  auto in = static_cast<float*>(plan.mem_alloc(alloc_bytes));
  auto out = static_cast<float*>(plan.mem_alloc(alloc_bytes));
  auto aux = static_cast<float*>(plan.mem_alloc(aux_bytes));

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

  return (double)max_ms;
}

double run_dtfft_c2c(const std::vector<int>& dims, dtfft::Precision precision, dtfft::Executor executor, bool enable_z_slab, 
  const dtfft::Effort effort = dtfft::Effort::EXHAUSTIVE,
  const std::vector<int> &grid=std::vector<int>()) {
  int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 4 : 2;
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if ( grid.size() > 0 ) {
      // Create Cartesian communicator to get coordinates
      MPI_Comm cart_comm;
      std::vector<int> periods(grid.size(), 0); // non-periodic
      MPI_Cart_create(MPI_COMM_WORLD, grid.size(), const_cast<int*>(grid.data()), periods.data(), 0, &cart_comm);
      
      std::vector<int> coords(grid.size());
      MPI_Cart_coords(cart_comm, comm_rank, grid.size(), coords.data());
      
      // Compute starts and counts
      std::vector<int32_t> starts(dims.size()), counts(dims.size());
      for (size_t i = 0; i < dims.size(); ++i) {
          int total = dims[i];
          int nprocs = grid[i];
          int base_count = total / nprocs;
          int remainder = total % nprocs;
          int my_count = base_count + (coords[i] < remainder ? 1 : 0);
          int my_start = coords[i] * base_count + std::min(coords[i], remainder);
          starts[i] = my_start;
          counts[i] = my_count;
      }
      auto pencil = dtfft::Pencil(starts, counts);
      dtfft::PlanC2C plan(pencil, MPI_COMM_WORLD, precision, effort, executor);
      MPI_Comm_free(&cart_comm);
      return run_dtfft_internal(plan, "C2C", precision, executor, scaler, enable_z_slab);
    } else {
      dtfft::PlanC2C plan(dims, MPI_COMM_WORLD, precision, effort, executor);
      return run_dtfft_internal(plan, "C2C", precision, executor, scaler, enable_z_slab);
    }
}

double run_dtfft_r2r(const std::vector<int>& dims, dtfft::Precision precision, bool enable_z_slab) {
  dtfft::PlanR2R plan(dims, precision, dtfft::Effort::EXHAUSTIVE);

  int64_t scaler = (precision == dtfft::Precision::DOUBLE) ? 2 : 1;
  return run_dtfft_internal(plan, "R2R", precision, dtfft::Executor::NONE, scaler, enable_z_slab);
}