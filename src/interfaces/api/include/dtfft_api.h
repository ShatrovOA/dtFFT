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

#ifndef DTFFT_API_H
#define DTFFT_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

extern
int
dtfft_create_plan_c2c_c(const int*, const int*, MPI_Fint, const int*, const int*, const int*, void**);

extern
int
dtfft_create_plan_r2c_c(const int*, const int*, MPI_Fint, const int*, const int*, const int*, void**);

extern
int
dtfft_create_plan_r2r_c(const int*, const int*, const int*, MPI_Fint, const int*, const int*, const int*, void**);

extern
int
dtfft_execute_c(const void*, void*, void*, const int*, void*);

extern
int
dtfft_transpose_c(const void*, const void*, void*, const int*);

extern
int
dtfft_destroy_c(void**);

extern
int
dtfft_get_local_sizes_c(const void*, int*, int*, int*, int*, size_t*);

extern
void
dtfft_get_error_string_c(const int*, char*, size_t*);

#ifdef __cplusplus
}
#endif

#endif