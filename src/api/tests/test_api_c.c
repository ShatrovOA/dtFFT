/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
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

#include <dtfft.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int comm_rank = 0;
static int n_failed = 0;

static void
check(const char* name, int actual, int expected)
{
    int failed = (actual != expected) ? 1 : 0;
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed == 0)
        return;

    n_failed++;
    if (comm_rank == 0)
        printf("  FAIL %s: got %d, expected %d\n", name, actual, expected);
}

int
main(int argc, char* argv[])
{
    const int32_t dims[3] = { 32, 24, 16 };
    const int32_t invalid_dims[2] = { 0, 16 };
    dtfft_plan_t plan = NULL;
    dtfft_plan_t invalid_plan = NULL;
    dtfft_plan_t alternate_plan = NULL;
    dtfft_error_t ierr;
    size_t alloc_size = 0;
    int8_t ndims = 0;
    const int32_t* actual_dims = NULL;
    const int32_t* grid_dims = NULL;
    int32_t in_starts[3] = { 0 }, in_counts[3] = { 0 };
    int32_t out_starts[3] = { 0 }, out_counts[3] = { 0 };
    int buffer = 0;
    size_t aux_size = 0, aux_bytes = 0, elem_size = 0, alloc_bytes = 0;
    dtfft_executor_t executor;
    dtfft_precision_t precision;
    dtfft_backend_t backend;
    dtfft_pencil_t pencil;
    dtfft_pencil_t custom_pencil = { 0 };
    dtfft_pencil_t invalid_pencil = { 0 };
    dtfft_r2r_kind_t kinds[3] = { DTFFT_DCT_2, DTFFT_DCT_2, DTFFT_DCT_2 };
    dtfft_config_t config;
    bool slab_enabled = false, pipelined = false;
    void* allocated = NULL;
    void* second_buffer = NULL;
    dtfft_request_t request = NULL;
    int comm_size = 1;
#ifdef DTFFT_WITH_CUDA
    dtfft_stream_t stream = NULL;
    dtfft_platform_t platform = DTFFT_PLATFORM_HOST;
#endif
    const char* string = NULL;

#ifdef DTFFT_WITH_CUDA
    if (setenv("DTFFT_PLATFORM", "host", 1) != 0) {
        perror("Failed to set DTFFT_PLATFORM=host");
        return EXIT_FAILURE;
    }
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_rank == 0)
        printf("test_api_c\n");

    check("version is positive", dtfft_get_version() > 0 ? 1 : 0, 1);
    check("create config", (int)dtfft_create_config(&config), (int)DTFFT_SUCCESS);
#ifdef DTFFT_WITH_CUDA
    config.platform = DTFFT_PLATFORM_HOST;
#endif
    check("set config", (int)dtfft_set_config(&config), (int)DTFFT_SUCCESS);
    check("null config output", (int)dtfft_create_config(NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null config input", (int)dtfft_set_config(NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);

    string = dtfft_get_error_string(DTFFT_SUCCESS);
    check("error string is valid", string && string[0] ? 1 : 0, 1);
    free((void*)string);
    string = dtfft_get_precision_string(DTFFT_DOUBLE);
    check("precision string is valid", string && string[0] ? 1 : 0, 1);
    free((void*)string);
    string = dtfft_get_executor_string(DTFFT_EXECUTOR_NONE);
    check("executor string is valid", string && string[0] ? 1 : 0, 1);
    free((void*)string);

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &plan);
    check("create valid C2C plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("valid plan handle", plan != NULL ? 1 : 0, 1);

    ierr = dtfft_get_dims(plan, &ndims, &actual_dims);
    check("get plan dimensions", (int)ierr, (int)DTFFT_SUCCESS);
    check("get plan dimension count", (int)ndims, 3);
    check("get plan dimension values",
          (actual_dims && actual_dims[0] == dims[0] &&
           actual_dims[1] == dims[1] && actual_dims[2] == dims[2]) ? 1 : 0,
          1);

    ierr = dtfft_get_alloc_size(plan, &alloc_size);
    check("get allocation size", (int)ierr, (int)DTFFT_SUCCESS);
    check("allocation size is positive", alloc_size > 0 ? 1 : 0, 1);
    check("get dimensions without dimension output",
          (int)dtfft_get_dims(plan, NULL, &actual_dims), (int)DTFFT_SUCCESS);
    check("get dimensions without values output",
          (int)dtfft_get_dims(plan, &ndims, NULL), (int)DTFFT_SUCCESS);
    check("get grid dimensions", (int)dtfft_get_grid_dims(plan, &ndims, &grid_dims),
          (int)DTFFT_SUCCESS);
    check("grid dimensions are available", grid_dims && ndims == 3 ? 1 : 0, 1);

    check("get local sizes",
          (int)dtfft_get_local_sizes(plan, in_starts, in_counts, out_starts,
                                     out_counts, &alloc_size),
          (int)DTFFT_SUCCESS);
    check("get auxiliary size", (int)dtfft_get_aux_size(plan, &aux_size),
          (int)DTFFT_SUCCESS);
    check("get auxiliary bytes", (int)dtfft_get_aux_bytes(plan, &aux_bytes),
          (int)DTFFT_SUCCESS);
    check("reshape auxiliary size unsupported without custom pencil",
          (int)dtfft_get_aux_size_reshape(plan, &aux_size),
          (int)DTFFT_ERROR_RESHAPE_NOT_SUPPORTED);
    check("reshape auxiliary bytes unsupported without custom pencil",
          (int)dtfft_get_aux_bytes_reshape(plan, &aux_bytes),
          (int)DTFFT_ERROR_RESHAPE_NOT_SUPPORTED);
    check("get transpose auxiliary size",
          (int)dtfft_get_aux_size_transpose(plan, &aux_size), (int)DTFFT_SUCCESS);
    check("get transpose auxiliary bytes",
          (int)dtfft_get_aux_bytes_transpose(plan, &aux_bytes), (int)DTFFT_SUCCESS);
    check("get element size", (int)dtfft_get_element_size(plan, &elem_size),
          (int)DTFFT_SUCCESS);
    check("element size is positive", elem_size > 0 ? 1 : 0, 1);
    check("get allocation bytes", (int)dtfft_get_alloc_bytes(plan, &alloc_bytes),
          (int)DTFFT_SUCCESS);
    check("allocation bytes are positive", alloc_bytes > 0 ? 1 : 0, 1);

    check("get executor", (int)dtfft_get_executor(plan, &executor), (int)DTFFT_SUCCESS);
    check("transpose-only executor", (int)executor, (int)DTFFT_EXECUTOR_NONE);
    check("get precision", (int)dtfft_get_precision(plan, &precision), (int)DTFFT_SUCCESS);
    check("plan precision", (int)precision, (int)DTFFT_DOUBLE);
    check("get z-slab flag", (int)dtfft_get_z_slab_enabled(plan, &slab_enabled),
          (int)DTFFT_SUCCESS);
    check("get y-slab flag", (int)dtfft_get_y_slab_enabled(plan, &slab_enabled),
          (int)DTFFT_SUCCESS);
    check("get backend", (int)dtfft_get_backend(plan, &backend), (int)DTFFT_SUCCESS);
    check("reshape backend unsupported without custom pencil",
          (int)dtfft_get_reshape_backend(plan, &backend),
          (int)DTFFT_ERROR_RESHAPE_NOT_SUPPORTED);
    string = dtfft_get_backend_string(backend);
    check("backend string is valid", string && string[0] ? 1 : 0, 1);
    free((void*)string);
    check("get backend pipelined",
          (int)dtfft_get_backend_pipelined(backend, &pipelined), (int)DTFFT_SUCCESS);
#ifdef DTFFT_WITH_CUDA
    check("host plan has no CUDA stream",
          (int)dtfft_get_stream(plan, &stream), (int)DTFFT_ERROR_INVALID_USAGE);
    check("get platform", (int)dtfft_get_platform(plan, &platform), (int)DTFFT_SUCCESS);
    check("plan platform is host", (int)platform, (int)DTFFT_PLATFORM_HOST);
    check("null stream output", (int)dtfft_get_stream(plan, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null platform output", (int)dtfft_get_platform(plan, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
#endif

    check("get x pencil", (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_X_PENCILS, &pencil),
          (int)DTFFT_SUCCESS);
    check("get y pencil", (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_Y_PENCILS, &pencil),
          (int)DTFFT_SUCCESS);
    check("get z pencil", (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_Z_PENCILS, &pencil),
          (int)DTFFT_SUCCESS);
    check("get x pencil for custom plan",
          (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_X_PENCILS, &pencil),
          (int)DTFFT_SUCCESS);

    custom_pencil.ndims = 3;
    custom_pencil.dim = 1;
    {
        int32_t local_x = dims[0] / comm_size;
        int32_t remainder_x = dims[0] % comm_size;
        custom_pencil.starts[0] =
            comm_rank * local_x + (comm_rank < remainder_x ? comm_rank : remainder_x);
        custom_pencil.starts[1] = 0;
        custom_pencil.starts[2] = 0;
        custom_pencil.counts[0] = local_x + (comm_rank < remainder_x ? 1 : 0);
        custom_pencil.counts[1] = dims[1];
        custom_pencil.counts[2] = dims[2];
        custom_pencil.size = (size_t)custom_pencil.counts[0] *
                             (size_t)custom_pencil.counts[1] *
                             (size_t)custom_pencil.counts[2];
    }

    ierr = dtfft_mem_alloc(plan, alloc_bytes, &allocated);
    check("allocate plan memory", (int)ierr, (int)DTFFT_SUCCESS);
    check("allocated plan memory is non-null", allocated != NULL ? 1 : 0, 1);
    if (allocated)
        memset(allocated, 0, alloc_bytes);
    check("null memory allocation output",
          (int)dtfft_mem_alloc(plan, alloc_bytes, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null memory to free", (int)dtfft_mem_free(plan, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    ierr = dtfft_mem_alloc(plan, alloc_bytes, &second_buffer);
    check("allocate second plan buffer", (int)ierr, (int)DTFFT_SUCCESS);
    if (allocated && second_buffer) {
        memset(second_buffer, 0, alloc_bytes);
        check("synchronous transpose",
              (int)dtfft_transpose(plan, allocated, second_buffer,
                                   DTFFT_TRANSPOSE_X_TO_Y, NULL),
              (int)DTFFT_SUCCESS);
        check("reverse synchronous transpose",
              (int)dtfft_transpose(plan, second_buffer, allocated,
                                   DTFFT_TRANSPOSE_Y_TO_X, NULL),
              (int)DTFFT_SUCCESS);
        check("start asynchronous transpose",
              (int)dtfft_transpose_start(plan, allocated, second_buffer,
                                         DTFFT_TRANSPOSE_X_TO_Y, NULL, &request),
              (int)DTFFT_SUCCESS);
        check("finish asynchronous transpose",
              (int)dtfft_transpose_end(plan, request), (int)DTFFT_SUCCESS);
    }
    if (second_buffer)
        check("free second plan buffer", (int)dtfft_mem_free(plan, second_buffer),
              (int)DTFFT_SUCCESS);
    if (allocated)
        check("free plan memory", (int)dtfft_mem_free(plan, allocated),
              (int)DTFFT_SUCCESS);

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, NULL);
    check("null plan output", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    ierr = dtfft_create_plan_c2c(3, NULL, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("null C2C dimensions", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    ierr = dtfft_create_plan_r2c(3, NULL, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("null R2C dimensions", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    ierr = dtfft_create_plan_r2r(3, NULL, kinds, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("null R2R dimensions", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);

    ierr = dtfft_create_plan_c2c(1, dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("invalid number of dimensions", (int)ierr,
          (int)DTFFT_ERROR_INVALID_N_DIMENSIONS);

    ierr = dtfft_create_plan_c2c(2, invalid_dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("non-positive dimension", (int)ierr, (int)DTFFT_ERROR_INVALID_DIMENSION_SIZE);

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD,
                                 (dtfft_precision_t)999, DTFFT_ESTIMATE,
                                 DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("invalid precision", (int)ierr, (int)DTFFT_ERROR_INVALID_PRECISION);

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 (dtfft_effort_t)999, DTFFT_EXECUTOR_NONE, &invalid_plan);
    check("invalid effort", (int)ierr, (int)DTFFT_ERROR_INVALID_EFFORT);

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                 DTFFT_ESTIMATE, (dtfft_executor_t)999, &invalid_plan);
    check("invalid executor", (int)ierr, (int)DTFFT_ERROR_INVALID_EXECUTOR);

    ierr = dtfft_execute(plan, NULL, &buffer, DTFFT_EXECUTE_FORWARD, NULL);
    check("null execution input", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    ierr = dtfft_execute(plan, &buffer, NULL, DTFFT_EXECUTE_FORWARD, NULL);
    check("null execution output", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose input",
          (int)dtfft_transpose(plan, NULL, &buffer, DTFFT_TRANSPOSE_X_TO_Y, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose output",
          (int)dtfft_transpose(plan, &buffer, NULL, DTFFT_TRANSPOSE_X_TO_Y, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose-start buffers",
          (int)dtfft_transpose_start(plan, NULL, &buffer, DTFFT_TRANSPOSE_X_TO_Y,
                                     NULL, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose-start request",
          (int)dtfft_transpose_start(plan, &buffer, &buffer, DTFFT_TRANSPOSE_X_TO_Y,
                                     NULL, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose-end request",
          (int)dtfft_transpose_end(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape input",
          (int)dtfft_reshape(plan, NULL, &buffer, DTFFT_RESHAPE_X_BRICKS_TO_PENCILS,
                             NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape output",
          (int)dtfft_reshape(plan, &buffer, NULL, DTFFT_RESHAPE_X_BRICKS_TO_PENCILS,
                             NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape-start buffers",
          (int)dtfft_reshape_start(plan, NULL, &buffer,
                                   DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, NULL, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape-start request",
          (int)dtfft_reshape_start(plan, &buffer, &buffer,
                                   DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, NULL, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape-end request",
          (int)dtfft_reshape_end(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null pencil output", (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_X_PENCILS, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("invalid pencil layout",
          (int)dtfft_get_pencil(plan, (dtfft_layout_t)999, &pencil),
          (int)DTFFT_ERROR_INVALID_LAYOUT);
    check("restore x pencil after invalid layout",
          (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_X_PENCILS, &pencil),
          (int)DTFFT_SUCCESS);

    ierr = dtfft_get_alloc_size(plan, NULL);
    check("null allocation-size output", (int)ierr, (int)DTFFT_ERROR_INVALID_USAGE);
    check("null local-size output",
          (int)dtfft_get_local_sizes(plan, NULL, NULL, NULL, NULL, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null z-slab output",
          (int)dtfft_get_z_slab_enabled(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null y-slab output",
          (int)dtfft_get_y_slab_enabled(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null auxiliary-size output", (int)dtfft_get_aux_size(plan, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null auxiliary-bytes output", (int)dtfft_get_aux_bytes(plan, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape auxiliary-size output",
          (int)dtfft_get_aux_size_reshape(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape auxiliary-bytes output",
          (int)dtfft_get_aux_bytes_reshape(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose auxiliary-size output",
          (int)dtfft_get_aux_size_transpose(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null transpose auxiliary-bytes output",
          (int)dtfft_get_aux_bytes_transpose(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null element-size output",
          (int)dtfft_get_element_size(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null allocation-bytes output",
          (int)dtfft_get_alloc_bytes(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null executor output",
          (int)dtfft_get_executor(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null precision output",
          (int)dtfft_get_precision(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null dimensions outputs",
          (int)dtfft_get_dims(plan, NULL, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null grid-dimensions outputs",
          (int)dtfft_get_grid_dims(plan, NULL, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null backend output",
          (int)dtfft_get_backend(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null reshape-backend output",
          (int)dtfft_get_reshape_backend(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null backend-pipelined output",
          (int)dtfft_get_backend_pipelined(backend, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null destroy argument",
          (int)dtfft_destroy(NULL), (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("null C2C pencil input",
          (int)dtfft_create_plan_c2c_pencil(NULL, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                             DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                             &alternate_plan),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null R2C pencil input",
          (int)dtfft_create_plan_r2c_pencil(NULL, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                             DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                             &alternate_plan),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null R2R pencil input",
          (int)dtfft_create_plan_r2r_pencil(NULL, kinds, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                             DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                             &alternate_plan),
          (int)DTFFT_ERROR_INVALID_USAGE);

    ierr = dtfft_destroy(&plan);
    check("destroy plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("destroy clears plan handle", plan == NULL ? 1 : 0, 1);
    ierr = dtfft_destroy(&plan);
    check("destroy missing plan", (int)ierr, (int)DTFFT_ERROR_PLAN_NOT_CREATED);

    ierr = dtfft_get_alloc_size(plan, &alloc_size);
    check("query missing plan", (int)ierr, (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    ndims = -1;
    actual_dims = dims;
    check("get dimensions from missing plan",
          (int)dtfft_get_dims(plan, &ndims, &actual_dims),
          (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("missing plan dimensions are cleared",
          ndims == 0 && actual_dims == NULL ? 1 : 0, 1);
    ndims = -1;
    grid_dims = dims;
    check("get grid dimensions from missing plan",
          (int)dtfft_get_grid_dims(plan, &ndims, &grid_dims),
          (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("missing plan grid dimensions are cleared",
          ndims == 0 && grid_dims == NULL ? 1 : 0, 1);
    check("get pencil from missing plan",
          (int)dtfft_get_pencil(plan, DTFFT_LAYOUT_X_PENCILS, &pencil),
          (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("missing plan pencil is cleared",
          pencil.ndims == 0 && pencil.size == 0 ? 1 : 0, 1);
    check("report missing plan", (int)dtfft_report(plan),
          (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("save missing plan", (int)dtfft_save_plan(plan, "unused.dtfft"),
          (int)DTFFT_ERROR_PLAN_NOT_CREATED);
    check("null save-plan filename",
          (int)dtfft_save_plan(plan, NULL), (int)DTFFT_ERROR_INVALID_USAGE);
    check("null load-plan filename",
          (int)dtfft_load_plan(NULL, MPI_COMM_WORLD, &alternate_plan),
          (int)DTFFT_ERROR_INVALID_USAGE);
    check("null load-plan output",
          (int)dtfft_load_plan("unused.dtfft", MPI_COMM_WORLD, NULL),
          (int)DTFFT_ERROR_INVALID_USAGE);

    ierr = dtfft_create_plan_c2c_pencil(&invalid_pencil, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                        DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                        &alternate_plan);
    check("uninitialized C2C pencil returns error", (int)ierr,
          (int)DTFFT_ERROR_PENCIL_NOT_INITIALIZED);
    ierr = dtfft_create_plan_r2c_pencil(&invalid_pencil, MPI_COMM_WORLD, DTFFT_SINGLE,
                                        DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                        &alternate_plan);
    check("uninitialized R2C pencil returns error", (int)ierr,
          (int)DTFFT_ERROR_PENCIL_NOT_INITIALIZED);
    ierr = dtfft_create_plan_r2r_pencil(&invalid_pencil, kinds, MPI_COMM_WORLD,
                                        DTFFT_SINGLE, DTFFT_ESTIMATE,
                                        DTFFT_EXECUTOR_NONE, &alternate_plan);
    check("uninitialized R2R pencil returns error", (int)ierr,
          (int)DTFFT_ERROR_PENCIL_NOT_INITIALIZED);

    ierr = dtfft_create_plan_r2c(3, dims, MPI_COMM_WORLD, DTFFT_SINGLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &alternate_plan);
    check("create R2C plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("destroy R2C plan", (int)dtfft_destroy(&alternate_plan),
          (int)DTFFT_SUCCESS);
    ierr = dtfft_create_plan_r2r(3, dims, kinds, MPI_COMM_WORLD, DTFFT_SINGLE,
                                 DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE, &alternate_plan);
    check("create R2R plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("destroy R2R plan", (int)dtfft_destroy(&alternate_plan),
          (int)DTFFT_SUCCESS);

    ierr = dtfft_create_plan_c2c_pencil(&custom_pencil, MPI_COMM_WORLD, DTFFT_DOUBLE,
                                        DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                        &alternate_plan);
    check("create C2C pencil plan", (int)ierr, (int)DTFFT_SUCCESS);
    if (ierr == DTFFT_SUCCESS && comm_size > 1) {
        void* reshape_in = NULL;
        void* reshape_out = NULL;
        void* reshape_aux = NULL;
        size_t reshape_alloc_bytes = 0;
        size_t reshape_aux_bytes = 0;
        check("get reshape auxiliary bytes on X-distributed pencil",
              (int)dtfft_get_aux_bytes_reshape(alternate_plan, &reshape_aux_bytes),
              (int)DTFFT_SUCCESS);
        check("get reshape backend on X-distributed pencil",
              (int)dtfft_get_reshape_backend(alternate_plan, &backend),
              (int)DTFFT_SUCCESS);
        check("get X-distributed pencil allocation bytes",
              (int)dtfft_get_alloc_bytes(alternate_plan, &reshape_alloc_bytes),
              (int)DTFFT_SUCCESS);
        check("allocate X-distributed pencil input",
              (int)dtfft_mem_alloc(alternate_plan, reshape_alloc_bytes, &reshape_in),
              (int)DTFFT_SUCCESS);
        check("allocate X-distributed pencil output",
              (int)dtfft_mem_alloc(alternate_plan, reshape_alloc_bytes, &reshape_out),
              (int)DTFFT_SUCCESS);
        if (reshape_aux_bytes > 0) {
            check("allocate X-distributed pencil reshape auxiliary",
                  (int)dtfft_mem_alloc(alternate_plan, reshape_aux_bytes, &reshape_aux),
                  (int)DTFFT_SUCCESS);
        }
        if (reshape_in && reshape_out) {
            check("reshape X bricks to pencils",
                  (int)dtfft_reshape(alternate_plan, reshape_in, reshape_out,
                                     DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, reshape_aux),
                  (int)DTFFT_SUCCESS);
            check("reshape X pencils to bricks",
                  (int)dtfft_reshape(alternate_plan, reshape_out, reshape_in,
                                     DTFFT_RESHAPE_X_PENCILS_TO_BRICKS, reshape_aux),
                  (int)DTFFT_SUCCESS);
        }
        if (reshape_aux)
            check("free X-distributed pencil reshape auxiliary",
                  (int)dtfft_mem_free(alternate_plan, reshape_aux), (int)DTFFT_SUCCESS);
        if (reshape_out)
            check("free X-distributed pencil output",
                  (int)dtfft_mem_free(alternate_plan, reshape_out), (int)DTFFT_SUCCESS);
        if (reshape_in)
            check("free X-distributed pencil input",
                  (int)dtfft_mem_free(alternate_plan, reshape_in), (int)DTFFT_SUCCESS);
    } else if (ierr == DTFFT_SUCCESS) {
        check("single-rank pencil reshape is unsupported",
              (int)dtfft_get_aux_bytes_reshape(alternate_plan, &aux_bytes),
              (int)DTFFT_ERROR_RESHAPE_NOT_SUPPORTED);
    }
    check("destroy C2C pencil plan", (int)dtfft_destroy(&alternate_plan),
          (int)DTFFT_SUCCESS);
    ierr = dtfft_create_plan_r2c_pencil(&custom_pencil, MPI_COMM_WORLD, DTFFT_SINGLE,
                                        DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE,
                                        &alternate_plan);
    check("create R2C pencil plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("destroy R2C pencil plan", (int)dtfft_destroy(&alternate_plan),
          (int)DTFFT_SUCCESS);
    ierr = dtfft_create_plan_r2r_pencil(&custom_pencil, kinds, MPI_COMM_WORLD,
                                        DTFFT_SINGLE, DTFFT_ESTIMATE,
                                        DTFFT_EXECUTOR_NONE, &alternate_plan);
    check("create R2R pencil plan", (int)ierr, (int)DTFFT_SUCCESS);
    check("destroy R2R pencil plan", (int)dtfft_destroy(&alternate_plan),
          (int)DTFFT_SUCCESS);

    if (comm_rank == 0)
        printf("%s\n", n_failed == 0 ? "All tests PASSED" : "Some tests FAILED");

    MPI_Finalize();
    return n_failed == 0 ? 0 : 1;
}
