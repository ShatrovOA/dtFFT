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

/*
  Checks the C entry points of plan save/load.

  The interesting part is the plan kind: `dtfft_load_plan` has to allocate the right concrete
  plan class before it can load into it, and only the file says which one that is. It works
  that out internally, so there is nothing for the caller to pass -- which is exactly why an
  R2R plan is needed here: a C2C-only implementation would still pass a C2C round trip.

  Every `check` here reduces across MPI_COMM_WORLD, so the control flow below is kept free
  of rank-dependent branches: a rank that skipped a check would leave the others waiting.
*/
#include <dtfft.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int comm_rank = 0;
static int n_failed = 0;

static void
report(const char* message)
{
    if (comm_rank == 0) {
        printf("%s\n", message);
        fflush(stdout);
    }
}

/* Collective: every rank must reach this the same number of times. */
static void
check(const char* name, int actual, int expected)
{
    int code = (actual == expected) ? 0 : 1;
    int worst = 0;
    MPI_Allreduce(&code, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (worst == 0)
        return;

    n_failed++;
    if (comm_rank == 0) {
        printf("  FAIL %s: got %d, expected %d\n", name, actual, expected);
        fflush(stdout);
    }
}

static void
delete_file(const char* filename)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0)
        remove(filename);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* Saves a plan, loads it back through the C API and compares what the two agree on. */
static void
roundtrip(const char* tag, dtfft_plan_t tuned, const char* filename)
{
    dtfft_plan_t loaded = NULL;
    dtfft_error_t ierr;
    const int32_t* grid_tuned = NULL;
    const int32_t* grid_loaded = NULL;
    int8_t ndims_tuned = 0, ndims_loaded = 0;
    size_t bytes_tuned = 0, bytes_loaded = 0;
    int grid_diff = 0;
    char name[160];

    snprintf(name, sizeof(name), "%s: save", tag);
    ierr = dtfft_save_plan(tuned, filename);
    check(name, (int)ierr, (int)DTFFT_SUCCESS);

    snprintf(name, sizeof(name), "%s: load", tag);
    ierr = dtfft_load_plan(filename, MPI_COMM_WORLD, &loaded);
    check(name, (int)ierr, (int)DTFFT_SUCCESS);

    /* `loaded` is either non-NULL on every rank or NULL on every rank: `dtfft_load_plan`
       reduces its error code internally. The comparisons below are therefore unconditional,
       falling back to values that match when there is no plan to inspect. */
    if (loaded) {
        dtfft_get_grid_dims(tuned, &ndims_tuned, &grid_tuned);
        dtfft_get_grid_dims(loaded, &ndims_loaded, &grid_loaded);
        if (ndims_tuned == ndims_loaded && grid_tuned && grid_loaded)
            grid_diff = memcmp(grid_tuned, grid_loaded, (size_t)ndims_tuned * sizeof(int32_t));
        dtfft_get_alloc_bytes(tuned, &bytes_tuned);
        dtfft_get_alloc_bytes(loaded, &bytes_loaded);
    }

    snprintf(name, sizeof(name), "%s: grid ndims", tag);
    check(name, (int)ndims_loaded, (int)ndims_tuned);
    snprintf(name, sizeof(name), "%s: grid dims", tag);
    check(name, grid_diff, 0);
    snprintf(name, sizeof(name), "%s: alloc bytes", tag);
    check(name, (int)bytes_loaded, (int)bytes_tuned);

    if (loaded)
        dtfft_destroy(&loaded);
    delete_file(filename);
}

int
main(int argc, char* argv[])
{
    const int32_t dims[3] = { 32, 24, 16 };
    const dtfft_r2r_kind_t kinds[3] = { DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_2 };
    dtfft_plan_t plan = NULL;
    dtfft_error_t ierr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    report("test_plan_save_load_c");

    ierr = dtfft_create_plan_c2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_MEASURE,
                                 DTFFT_EXECUTOR_NONE, &plan);
    check("c2c: create", (int)ierr, (int)DTFFT_SUCCESS);
    roundtrip("c2c", plan, "test_save_load_c_c2c.dtfft");
    dtfft_destroy(&plan);

    /* R2R proves that `dtfft_load_plan` no longer assumes C2C */
    ierr = dtfft_create_plan_r2r(3, dims, kinds, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_MEASURE,
                                 DTFFT_EXECUTOR_NONE, &plan);
    check("r2r: create", (int)ierr, (int)DTFFT_SUCCESS);
    roundtrip("r2r", plan, "test_save_load_c_r2r.dtfft");
    dtfft_destroy(&plan);

#ifdef DTFFT_WITH_FFTW
    /* An R2C plan needs a real FFT executor: transpose-only R2C is not a thing */
    ierr = dtfft_create_plan_r2c(3, dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE,
                                 DTFFT_EXECUTOR_FFTW3, &plan);
    check("r2c: create", (int)ierr, (int)DTFFT_SUCCESS);
    roundtrip("r2c", plan, "test_save_load_c_r2c.dtfft");
    dtfft_destroy(&plan);
#endif

    /* A missing file must fail cleanly rather than abort or hang */
    plan = NULL;
    ierr = dtfft_load_plan("test_save_load_c_missing.dtfft", MPI_COMM_WORLD, &plan);
    check("missing file: load", (int)ierr, (int)DTFFT_ERROR_FILE_NOT_FOUND);
    check("missing file: no plan handed back", plan == NULL ? 0 : 1, 0);

    if (n_failed == 0)
        report("All tests PASSED");
    else
        report("Some tests FAILED");

    MPI_Finalize();
    return n_failed == 0 ? 0 : 1;
}
