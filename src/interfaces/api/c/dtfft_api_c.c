/*
  Copyright (c) 2021, Oleg Shatrov
  All rights reserved.
  This file is part of dtFFT library.

  dtFFT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  dtFFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <mpi.h>
#include <stdlib.h>

#include "dtfft_config.h"
#include <dtfft.h>
#include <dtfft_api.h>

#define _DTFFT_PLAN_ALLOCATED 334455

// Structure for holding plan pointer
struct dtfft_plan_private_t
{
  void *_plan_ptr;      // Pointer to fortran handle
  uint32_t _is_allocated;
};


dtfft_error_code_t
dtfft_create_plan_c2c(const int8_t ndims, const int32_t *dims,
                      MPI_Comm comm,
                      const dtfft_precision_t precision,
                      const dtfft_effort_t effort_flag,
                      const dtfft_executor_t executor_type,
                      dtfft_plan_t *plan)
{
  dtfft_plan_t plan_ = malloc(sizeof(*plan_));
  plan_ -> _is_allocated = _DTFFT_PLAN_ALLOCATED;
  int32_t error_code = dtfft_create_plan_c2c_c(&ndims, dims, MPI_Comm_c2f(comm), (int8_t*)&precision, (int8_t*)&effort_flag, (int8_t*)&executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  return (dtfft_error_code_t)error_code;
}

#ifndef DTFFT_TRANSPOSE_ONLY
dtfft_error_code_t
dtfft_create_plan_r2c(const int8_t ndims, const int32_t *dims,
                      MPI_Comm comm,
                      const dtfft_precision_t precision,
                      const dtfft_effort_t effort_flag,
                      const dtfft_executor_t executor_type,
                      dtfft_plan_t *plan)
{
  dtfft_plan_t plan_ = malloc(sizeof(*plan_));
  plan_ -> _is_allocated = _DTFFT_PLAN_ALLOCATED;
  int32_t error_code = dtfft_create_plan_r2c_c(&ndims, dims, MPI_Comm_c2f(comm), (int8_t*)&precision, (int8_t*)&effort_flag, (int8_t*)&executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  return (dtfft_error_code_t)error_code;
}
#endif

dtfft_error_code_t
dtfft_create_plan_r2r(const int8_t ndims, const int32_t *dims,
                      const dtfft_r2r_kind_t *kinds,
                      MPI_Comm comm,
                      const dtfft_precision_t precision,
                      const dtfft_effort_t effort_flag,
                      const dtfft_executor_t executor_type,
                      dtfft_plan_t *plan)
{
  dtfft_plan_t plan_ = malloc(sizeof(*plan_));
  int8_t* kinds_ = NULL;
  if ( kinds ) {
    kinds_ = (int8_t*)malloc(ndims * sizeof(int8_t));
    int8_t d;
    for (d = 0; d < ndims; d++)
    {
      kinds_[d] = (int8_t)kinds[d];
    }
  }
  plan_ -> _is_allocated = _DTFFT_PLAN_ALLOCATED;
  int32_t error_code = dtfft_create_plan_r2r_c(&ndims, dims, kinds_, MPI_Comm_c2f(comm), (int8_t*)&precision, (int8_t*)&effort_flag, (int8_t*)&executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  if ( kinds ) {
    free(kinds_);
  }
  return (dtfft_error_code_t)error_code;
}

static inline
void *
get_plan_handle(dtfft_plan_t plan) {
  return plan ? ((plan -> _is_allocated == _DTFFT_PLAN_ALLOCATED) ? plan -> _plan_ptr : NULL) : NULL;
}

dtfft_error_code_t
dtfft_get_z_slab(dtfft_plan_t plan, bool *is_z_slab)
{
  return (dtfft_error_code_t)dtfft_get_z_slab_c(get_plan_handle(plan), is_z_slab);
}

dtfft_error_code_t
dtfft_execute(dtfft_plan_t plan, void *in, void *out, const dtfft_execute_type_t execute_type, void *aux)
{
  return (dtfft_error_code_t)dtfft_execute_c(get_plan_handle(plan), in, out, (int8_t*)&execute_type, aux);
}

dtfft_error_code_t
dtfft_transpose(dtfft_plan_t plan, void *in, void *out, const dtfft_transpose_type_t transpose_type)
{
  return (dtfft_error_code_t)dtfft_transpose_c(get_plan_handle(plan), in, out, (int8_t*)&transpose_type);
}

dtfft_error_code_t
dtfft_destroy(dtfft_plan_t *plan)
{
  if (!*plan) return DTFFT_ERROR_PLAN_NOT_CREATED;
  void *_plan = get_plan_handle(*plan);
  int32_t error_code = dtfft_destroy_c(&_plan);
  (*plan) -> _is_allocated = 0;
  (*plan) -> _plan_ptr = NULL;
  free(*plan);
  *plan = NULL;
  return (dtfft_error_code_t)error_code;
}

dtfft_error_code_t
dtfft_get_local_sizes(dtfft_plan_t plan, int32_t *in_starts, int32_t *in_counts, int32_t *out_starts, int32_t *out_counts, size_t *alloc_size) {
  return (dtfft_error_code_t)dtfft_get_local_sizes_c(get_plan_handle(plan), in_starts, in_counts, out_starts, out_counts, alloc_size);
}

dtfft_error_code_t
dtfft_get_alloc_size(dtfft_plan_t plan, size_t *alloc_size) {
  return dtfft_get_local_sizes(plan, NULL, NULL, NULL, NULL, alloc_size);
}

const char *
dtfft_get_error_string(const dtfft_error_code_t error_code)
{
  char *error_string = malloc(250 * sizeof(char));
  size_t error_string_size;
  dtfft_get_error_string_c(&error_code, error_string, &error_string_size);
  return realloc(error_string, sizeof(char) * error_string_size);
}

dtfft_error_code_t
dtfft_get_pencil(dtfft_plan_t plan, int8_t dim, dtfft_pencil_t *pencil) {
  return dtfft_get_pencil_c(get_plan_handle(plan), &dim, (void*)pencil);
}

#ifdef DTFFT_WITH_CUDA

dtfft_error_code_t
dtfft_set_stream(const cudaStream_t stream) {
  return (dtfft_error_code_t)dtfft_set_stream_c(&stream);
}

dtfft_error_code_t
dtfft_set_gpu_backend(const dtfft_gpu_backend_t backend_id) {
  return (dtfft_error_code_t)dtfft_set_gpu_backend_c((int8_t*)&backend_id);
}

dtfft_error_code_t
dtfft_get_stream(dtfft_plan_t plan, cudaStream_t *stream) {
  return (dtfft_error_code_t)dtfft_get_stream_c(get_plan_handle(plan), stream);
}

dtfft_error_code_t
dtfft_get_gpu_backend(dtfft_plan_t plan, dtfft_gpu_backend_t *backend_id) {
  return (dtfft_error_code_t)dtfft_get_gpu_backend_c(get_plan_handle(plan), (int8_t*)backend_id);
}

const char *
dtfft_get_gpu_backend_string(const dtfft_gpu_backend_t backend_id) {
  char *backend_string = malloc(250 * sizeof(char));
  size_t backend_string_size;
  dtfft_get_gpu_backend_string_c((int8_t*)&backend_id, backend_string, &backend_string_size);
  return realloc(backend_string, sizeof(char) * backend_string_size);
}

#endif
