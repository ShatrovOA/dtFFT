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
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

extern
int32_t
dtfft_get_version_current();

extern
int32_t
dtfft_create_plan_c2c_c(const int8_t*, const int32_t*, MPI_Fint, const int32_t*, const int32_t*, const int32_t*, void**);

extern
int32_t
dtfft_create_plan_r2c_c(const int8_t*, const int32_t*, MPI_Fint, const int32_t*, const int32_t*, const int32_t*, void**);

extern
int32_t
dtfft_create_plan_r2r_c(const int8_t*, const int32_t*, const int32_t*, MPI_Fint, const int32_t*, const int32_t*, const int32_t*, void**);

extern
int32_t
dtfft_get_z_slab_enabled_c(const void*, bool*);

extern
int32_t
dtfft_execute_c(const void*, void*, void*, const int32_t*, void*);

extern
int32_t
dtfft_transpose_c(const void*, void*, void*, const int32_t*);

extern
int32_t
dtfft_destroy_c(void**);

extern
int32_t
dtfft_get_local_sizes_c(const void*, int32_t*, int32_t*, int32_t*, int32_t*, size_t*);

extern
int32_t
dtfft_get_alloc_size_c(const void*, size_t*);

extern
int32_t
dtfft_mem_alloc_c(const void*, size_t, void**);

extern
int32_t
dtfft_mem_free_c(const void*, void*);

extern
void
dtfft_get_error_string_c(const int32_t*, char*, size_t*);

extern
int32_t
dtfft_get_pencil_c(const void*, int8_t*, void *);

extern
int32_t
dtfft_get_element_size_c(const void *, size_t*);

extern
void*
dtfft_create_config_c();

extern
int32_t
dtfft_set_config_c(const void*);

extern
int32_t
dtfft_report_c(const void*);

#ifdef DTFFT_WITH_CUDA

#include <cuda_runtime.h> // cudaStream_t

extern
int32_t
dtfft_set_stream_c(const cudaStream_t*);

extern
int32_t
dtfft_set_gpu_backend_c(const int32_t*);

extern
int32_t
dtfft_get_stream_c(const void*, cudaStream_t*);

extern
int32_t
dtfft_get_gpu_backend_c(const void*, int32_t*);

extern
void
dtfft_get_gpu_backend_string_c(const int32_t*, char*, size_t*);

#endif

#ifdef __cplusplus
}
#endif

#endif