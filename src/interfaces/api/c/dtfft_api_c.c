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
#include <stdio.h>

#include <dtfft.h>
#include <dtfft_api.h>

#define PLAN_ALLOCATED 334455

// Structure for holding plan pointer
struct dtfft_plan_t
{
  void *_plan_ptr;      // Pointer to fortran handle
  int  is_allocated;
};


int
dtfft_create_plan_c2c(const int ndims, const int *dims,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type,
                      dtfft_plan *plan)
{
  dtfft_plan plan_ = malloc(sizeof(*plan_));
  plan_ -> is_allocated = PLAN_ALLOCATED;
  int error_code = dtfft_create_plan_c2c_c(&ndims, dims, MPI_Comm_c2f(comm), &precision, &effort_flag, &executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  return error_code;
}


int
dtfft_create_plan_r2c(const int ndims, const int *dims,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type,
                      dtfft_plan *plan)
{
  dtfft_plan plan_ = malloc(sizeof(*plan_));
  plan_ -> is_allocated = PLAN_ALLOCATED;
  int error_code = dtfft_create_plan_r2c_c(&ndims, dims, MPI_Comm_c2f(comm), &precision, &effort_flag, &executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  return error_code;
}

int
dtfft_create_plan_r2r(const int ndims, const int *dims,
                      const int *kinds,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type,
                      dtfft_plan *plan)
{
  dtfft_plan plan_ = malloc(sizeof(*plan_));
  plan_ -> is_allocated = PLAN_ALLOCATED;
  int error_code = dtfft_create_plan_r2r_c(&ndims, dims, kinds, MPI_Comm_c2f(comm), &precision, &effort_flag, &executor_type, &plan_ -> _plan_ptr);
  *plan = plan_;
  return error_code;
}

static inline
void *
get_plan_handle(dtfft_plan plan) {
  return (plan -> is_allocated == PLAN_ALLOCATED) ? plan -> _plan_ptr : NULL;
}

int
dtfft_execute(dtfft_plan plan, void *in, void *out, const int transpose_type, void *aux)
{
  return dtfft_execute_c(get_plan_handle(plan), in, out, &transpose_type, aux);
}

int
dtfft_transpose(dtfft_plan plan, const void *in, void *out, const int transpose_type)
{
  return dtfft_transpose_c(get_plan_handle(plan), in, out, &transpose_type);
}

void
dtfft_destroy(dtfft_plan *plan)
{
  if (!*plan) return;
  void *_plan = get_plan_handle(*plan);
  dtfft_destroy_c(&_plan);
  (*plan) -> is_allocated = 0;
  (*plan) -> _plan_ptr = NULL;
  *plan = NULL;
}

int
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, size_t *alloc_size) {
  return dtfft_get_local_sizes_c(get_plan_handle(plan), in_starts, in_counts, out_starts, out_counts, alloc_size);
}

int
dtfft_get_alloc_size(dtfft_plan plan, size_t *alloc_size) {
  return dtfft_get_local_sizes(plan, NULL, NULL, NULL, NULL, alloc_size);
}

const char *
dtfft_get_error_string(const int error_code)
{
  char *error_string = malloc(250 * sizeof(char));
  size_t error_string_size;
  dtfft_get_error_string_c(&error_code, error_string, &error_string_size);
  return realloc(error_string, sizeof(char) * error_string_size);
}
