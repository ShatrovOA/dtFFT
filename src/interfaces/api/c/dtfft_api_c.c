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
#include <dtfft_private.h>

// Structure for holding plan pointer
struct dtfft_plan_t
{
  void *_plan_ptr;      // Pointer to fortran handle
};


MPI_Fint
DTFFT_Comm_c2f(MPI_Comm comm) {
  return MPI_Comm_c2f(comm);
}


dtfft_plan
dtfft_create_plan_c2c(const int ndims, const int *dims,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type)
{
  dtfft_plan plan = malloc(sizeof(*plan));
  dtfft_create_plan_c2c_c(&ndims, dims, comm, &precision, &effort_flag, &executor_type, &plan -> _plan_ptr);
  return plan;
}


dtfft_plan
dtfft_create_plan_r2c(const int ndims, const int *dims,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type)
{
  dtfft_plan plan = malloc(sizeof(*plan));
  dtfft_create_plan_r2c_c(&ndims, dims, comm, &precision, &effort_flag, &executor_type, &plan -> _plan_ptr);
  return plan;
}

dtfft_plan
dtfft_create_plan_r2r(const int ndims, const int *dims,
                      const int *in_kinds, const int *out_kinds,
                      MPI_Comm comm,
                      const int precision,
                      const int effort_flag, const int executor_type)
{
  dtfft_plan plan = malloc(sizeof(*plan));
  dtfft_create_plan_r2r_c(&ndims, dims, in_kinds, out_kinds, comm, &precision, &effort_flag, &executor_type, &plan -> _plan_ptr);
  return plan;
}

void
dtfft_execute(dtfft_plan plan, void *in, void *out, const int transpose_type, void *aux)
{
  dtfft_execute_c(plan -> _plan_ptr, in, out, &transpose_type, aux);
}

void
dtfft_transpose(dtfft_plan plan, const void *in, void *out, const int transpose_type)
{
  dtfft_transpose_c(plan -> _plan_ptr, in, out, &transpose_type);
}

void
dtfft_destroy(dtfft_plan plan)
{
  dtfft_destroy_c(&plan -> _plan_ptr);
}

size_t
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts) {
  size_t alloc_size;
  dtfft_get_local_sizes_c(plan -> _plan_ptr, in_starts, in_counts, out_starts, out_counts, &alloc_size);
  return alloc_size;
}

size_t
dtfft_get_alloc_size(dtfft_plan plan) {
  return dtfft_get_local_sizes(plan, NULL, NULL, NULL, NULL);
}

size_t
dtfft_get_aux_size(dtfft_plan plan)
{
  size_t aux_size;
  dtfft_get_aux_size_c(plan -> _plan_ptr, &aux_size);
  return aux_size;
}