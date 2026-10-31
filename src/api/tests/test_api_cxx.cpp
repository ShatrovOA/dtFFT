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

#include <dtfft.hpp>
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

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
        std::printf("  FAIL %s: got %d, expected %d\n", name, actual, expected);
}

static void
check_constructor_error(const char* name, int8_t ndims, const int32_t* dims,
                        dtfft::Precision precision, dtfft::Effort effort,
                        dtfft::Executor executor, dtfft::Error expected)
{
    int actual = (int)dtfft::Error::SUCCESS;
    try {
        dtfft::PlanC2C invalid_plan(ndims, dims, MPI_COMM_WORLD, precision,
                                    effort, executor);
    } catch (const dtfft::Exception& error) {
        actual = (int)error.get_error_code();
    }
    check(name, actual, (int)expected);
}

template <typename Operation>
static void
check_pencil_error(const char* name, Operation operation, int expected)
{
    dtfft::Pencil pencil;
    int actual = (int)dtfft::Error::SUCCESS;
    try {
        operation(pencil);
    } catch (const dtfft::Exception& error) {
        actual = (int)error.get_error_code();
    }
    check(name, actual, expected);
}

static void
check_pencil_constructor_error(const char* name, int32_t ndims,
                               const int32_t* starts, const int32_t* counts,
                               int expected)
{
    int actual = (int)dtfft::Error::SUCCESS;
    try {
        dtfft::Pencil invalid_pencil(ndims, starts, counts);
    } catch (const dtfft::Exception& error) {
        actual = (int)error.get_error_code();
    }
    check(name, actual, expected);
}

int
main(int argc, char* argv[])
{
    const std::vector<int32_t> dims = { 32, 24, 16 };
    const int32_t invalid_dims[2] = { 0, 16 };
    const std::vector<dtfft::R2RKind> kinds(3, dtfft::R2RKind::DCT_2);
    int buffer = 0;
    int32_t in_starts[3] = { 0 }, in_counts[3] = { 0 };
    int32_t out_starts[3] = { 0 }, out_counts[3] = { 0 };
    size_t alloc_size = 0, aux_size = 0, aux_bytes = 0, element_size = 0;
#ifdef DTFFT_WITH_CUDA
    dtfft_stream_t stream = nullptr;
    dtfft::Platform platform = dtfft::Platform::HOST;
#endif

#ifdef DTFFT_WITH_CUDA
    if (setenv("DTFFT_PLATFORM", "host", 1) != 0) {
        std::perror("Failed to set DTFFT_PLATFORM=host");
        return EXIT_FAILURE;
    }
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_rank == 0)
        std::printf("test_api_cxx\n");

    check("version is positive", dtfft::Version::get() > 0 ? 1 : 0, 1);
    check("version encoding", dtfft::Version::get(1, 2, 3) == DTFFT_VERSION(1, 2, 3) ? 1 : 0,
          1);
    check("error string is valid",
          dtfft::get_error_string(dtfft::Error::SUCCESS).empty() ? 0 : 1, 1);
    check("precision string is valid",
          dtfft::get_precision_string(dtfft::Precision::DOUBLE).empty() ? 0 : 1, 1);
    check("executor string is valid",
          dtfft::get_executor_string(dtfft::Executor::NONE).empty() ? 0 : 1, 1);
    check("backend string is valid",
          dtfft::get_backend_string(dtfft::Backend::MPI_DATATYPE).empty() ? 0 : 1, 1);
    check("backend pipeline classification",
          dtfft::get_backend_pipelined(dtfft::Backend::MPI_DATATYPE) ? 1 : 0, 0);

    {
        dtfft::Exception exception(dtfft::Error::INVALID_USAGE, "test", __FILE__, __LINE__);
        check("exception code", (int)exception.get_error_code(),
              (int)dtfft::Error::INVALID_USAGE);
        check("exception message", exception.get_message() == "test" ? 1 : 0, 1);
        check("exception file", exception.get_file().empty() ? 0 : 1, 1);
        check("exception line", exception.get_line() > 0 ? 1 : 0, 1);
        check("exception what", exception.what() != nullptr ? 1 : 0, 1);
    }

    {
        dtfft::Config config;
        config.set_enable_log(false)
            .set_enable_z_slab(true)
            .set_enable_y_slab(false)
            .set_measure_warmup_iters(1)
            .set_measure_iters(1)
            .set_backend(dtfft::Backend::MPI_DATATYPE)
            .set_reshape_backend(dtfft::Backend::MPI_DATATYPE)
            .set_enable_datatype_backend(true)
            .set_enable_mpi_backends(false)
            .set_enable_pipelined_backends(true)
            .set_enable_rma_backends(true)
            .set_enable_fused_backends(true)
            .set_enable_kernel_autotune(false)
            .set_enable_fourier_reshape(false)
            .set_transpose_mode(dtfft::TransposeMode::PACK)
            .set_access_mode(dtfft::AccessMode::WRITE);
#ifdef DTFFT_WITH_CUDA
        config.set_platform(dtfft::Platform::HOST).set_stream(nullptr);
#endif
        check("set C++ config", (int)dtfft::set_config(config), (int)dtfft::Error::SUCCESS);
    }

    {
        dtfft::PlanC2C plan(dims, MPI_COMM_WORLD, dtfft::Precision::DOUBLE,
                            dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);
        const auto actual_dims = plan.get_dims();
        const auto actual_grid = plan.get_grid_dims();
        auto backend = plan.get_backend();
        dtfft::Pencil x_pencil;
        check("create valid C2C plan", plan.c_struct() != nullptr ? 1 : 0, 1);
        check("get plan dimensions", actual_dims == dims ? 1 : 0, 1);
        check("get grid dimensions", actual_grid.size() == dims.size() ? 1 : 0, 1);
        int8_t raw_ndims = 0;
        const int32_t* raw_dims = nullptr;
        const int32_t* raw_grid = nullptr;
        check("get dimensions through output arguments",
              (int)plan.get_dims(&raw_ndims, &raw_dims), (int)dtfft::Error::SUCCESS);
        check("raw dimensions agree",
              raw_dims && raw_ndims == 3 && raw_dims[0] == dims[0] ? 1 : 0, 1);
        check("get grid through output arguments",
              (int)plan.get_grid_dims(&raw_ndims, &raw_grid), (int)dtfft::Error::SUCCESS);
        check("raw grid is available", raw_grid && raw_ndims == 3 ? 1 : 0, 1);
        check("get executor", (int)plan.get_executor(), (int)dtfft::Executor::NONE);
        check("get precision", (int)plan.get_precision(), (int)dtfft::Precision::DOUBLE);
        check("get backend", (int)backend >= 0 ? 1 : 0, 1);
        dtfft::Backend reshape_backend = dtfft::Backend::NONE;
        dtfft::Backend selected_backend = dtfft::Backend::NONE;
        check("get backend through output argument",
              (int)plan.get_backend(selected_backend),
              (int)dtfft::Error::SUCCESS);
        check("get reshape backend through output argument",
              (int)plan.get_reshape_backend(reshape_backend),
              (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);

        check("get allocation size", plan.get_alloc_size() > 0 ? 1 : 0, 1);
        check("get auxiliary size", (int)plan.get_aux_size(&aux_size),
              (int)dtfft::Error::SUCCESS);
        check("get auxiliary size value", plan.get_aux_size() == aux_size ? 1 : 0, 1);
        check("get auxiliary bytes", (int)plan.get_aux_bytes(&aux_bytes),
              (int)dtfft::Error::SUCCESS);
        check("get auxiliary bytes value", plan.get_aux_bytes() == aux_bytes ? 1 : 0, 1);
        check("get reshape auxiliary size",
              (int)plan.get_aux_size_reshape(&aux_size),
              (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);

        check("get reshape auxiliary bytes",
              (int)plan.get_aux_bytes_reshape(&aux_bytes),
              (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);
        check("get transpose auxiliary size",
              (int)plan.get_aux_size_transpose(&aux_size), (int)dtfft::Error::SUCCESS);
        check("get transpose auxiliary size value",
              plan.get_aux_size_transpose() == aux_size ? 1 : 0, 1);
        check("get transpose auxiliary bytes",
              (int)plan.get_aux_bytes_transpose(&aux_bytes), (int)dtfft::Error::SUCCESS);
        check("get transpose auxiliary bytes value",
              plan.get_aux_bytes_transpose() == aux_bytes ? 1 : 0, 1);
        check("get local sizes",
              (int)plan.get_local_sizes(in_starts, in_counts, out_starts, out_counts,
                                        &alloc_size),
              (int)dtfft::Error::SUCCESS);
        std::vector<int32_t> vector_in_starts(3), vector_in_counts(3);
        std::vector<int32_t> vector_out_starts(3), vector_out_counts(3);
        check("get local sizes through vectors",
              (int)plan.get_local_sizes(vector_in_starts, vector_in_counts,
                                        vector_out_starts, vector_out_counts,
                                        &alloc_size),
              (int)dtfft::Error::SUCCESS);
        check("get local sizes with null outputs",
              (int)plan.get_local_sizes(), (int)dtfft::Error::INVALID_USAGE);
        check("local allocation size is positive", alloc_size > 0 ? 1 : 0, 1);
        check("get element size", (int)plan.get_element_size(&element_size),
              (int)dtfft::Error::SUCCESS);
        check("get element size value", plan.get_element_size() == element_size ? 1 : 0, 1);
        check("get allocation bytes", plan.get_alloc_bytes() > 0 ? 1 : 0, 1);
        size_t alloc_bytes = 0;
        check("get allocation bytes through output argument",
              (int)plan.get_alloc_bytes(&alloc_bytes), (int)dtfft::Error::SUCCESS);
        check("allocation byte values agree",
              plan.get_alloc_bytes() == alloc_bytes ? 1 : 0, 1);
        bool z_slab_enabled = false, y_slab_enabled = false;
        check("get z-slab flag",
              (int)plan.get_z_slab_enabled(&z_slab_enabled), (int)dtfft::Error::SUCCESS);
        check("get y-slab flag",
              (int)plan.get_y_slab_enabled(&y_slab_enabled), (int)dtfft::Error::SUCCESS);
        check("get z-slab flag value", plan.get_z_slab_enabled() == z_slab_enabled ? 1 : 0, 1);
        check("get y-slab flag value", plan.get_y_slab_enabled() == y_slab_enabled ? 1 : 0, 1);
#ifdef DTFFT_WITH_CUDA
        check("host plan has no CUDA stream",
              (int)plan.get_stream(&stream), (int)dtfft::Error::INVALID_USAGE);
        check("get platform", (int)plan.get_platform(platform), (int)dtfft::Error::SUCCESS);
        check("platform output is host", (int)platform, (int)dtfft::Platform::HOST);
        check("null stream output", (int)plan.get_stream(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("get platform by value", (int)plan.get_platform(),
              (int)dtfft::Platform::HOST);
#endif

        check("get x pencil",
              (int)plan.get_pencil(dtfft::Layout::X_PENCILS, x_pencil),
              (int)dtfft::Error::SUCCESS);
        check("pencil dimensions", (int)x_pencil.get_ndims(), 3);
        check("pencil aligned dimension", x_pencil.get_dim() > 0 ? 1 : 0, 1);
        check("pencil starts", x_pencil.get_starts().size() == dims.size() ? 1 : 0, 1);
        check("pencil counts", x_pencil.get_counts().size() == dims.size() ? 1 : 0, 1);
        check("pencil size", x_pencil.get_size() > 0 ? 1 : 0, 1);
        check("pencil C structure", x_pencil.c_struct().ndims == 3 ? 1 : 0, 1);
        check("get pencil by value",
              plan.get_pencil(dtfft::Layout::Y_PENCILS).get_ndims(), 3);
        const int32_t local_x = dims[0] / comm_size;
        const int32_t remainder_x = dims[0] % comm_size;
        std::vector<int32_t> custom_starts = {
            comm_rank * local_x + std::min(comm_rank, remainder_x), 0, 0
        };
        std::vector<int32_t> custom_counts = {
            local_x + (comm_rank < remainder_x ? 1 : 0), dims[1], dims[2]
        };
        dtfft::Pencil custom_pencil(custom_starts, custom_counts);
        dtfft::PlanC2C custom_plan(custom_pencil, MPI_COMM_WORLD,
                                   dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE,
                                   dtfft::Executor::NONE);
        check("create C2C pencil plan", custom_plan.c_struct() != nullptr ? 1 : 0, 1);
        if (comm_size > 1) {
            check("get reshape backend on X-distributed pencil",
                  (int)custom_plan.get_reshape_backend(reshape_backend),
                  (int)dtfft::Error::SUCCESS);
            check("get reshape backend value on X-distributed pencil",
                  (int)custom_plan.get_reshape_backend(), (int)reshape_backend);
            check("get reshape auxiliary size on X-distributed pencil",
                  (int)custom_plan.get_aux_size_reshape(&aux_size),
                  (int)dtfft::Error::SUCCESS);
            check("get reshape auxiliary bytes on X-distributed pencil",
                  (int)custom_plan.get_aux_bytes_reshape(&aux_bytes),
                  (int)dtfft::Error::SUCCESS);
            check("get reshape auxiliary bytes on X-distributed pencil",
                  (int)custom_plan.get_aux_bytes_reshape(),
                  int(aux_bytes));
        } else {
            check("reshape unsupported for single-rank pencil",
                  (int)custom_plan.get_reshape_backend(reshape_backend),
                  (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);
            check("reshape auxiliary size unsupported for single-rank pencil",
                  (int)custom_plan.get_aux_size_reshape(&aux_size),
                  (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);
            check("reshape auxiliary bytes unsupported for single-rank pencil",
                  (int)custom_plan.get_aux_bytes_reshape(&aux_bytes),
                  (int)dtfft::Error::RESHAPE_NOT_SUPPORTED);
        }
        if (comm_size > 1) {
            void* reshape_in = nullptr;
            void* reshape_out = nullptr;
            void* reshape_aux = nullptr;
            const auto reshape_alloc_bytes = custom_plan.get_alloc_bytes();
            check("allocate X-distributed pencil input",
                  (int)custom_plan.mem_alloc(reshape_alloc_bytes, &reshape_in),
                  (int)dtfft::Error::SUCCESS);
            check("allocate X-distributed pencil output",
                  (int)custom_plan.mem_alloc(reshape_alloc_bytes, &reshape_out),
                  (int)dtfft::Error::SUCCESS);
            if (aux_bytes > 0) {
                check("allocate X-distributed pencil reshape auxiliary",
                      (int)custom_plan.mem_alloc(aux_bytes, &reshape_aux),
                      (int)dtfft::Error::SUCCESS);
            }
            if (reshape_in && reshape_out) {
                check("reshape X bricks to pencils",
                      (int)custom_plan.reshape(reshape_in, reshape_out,
                                               dtfft::Reshape::X_BRICKS_TO_PENCILS,
                                               reshape_aux),
                      (int)dtfft::Error::SUCCESS);
                check("reshape X pencils to bricks",
                      (int)custom_plan.reshape(reshape_out, reshape_in,
                                               dtfft::Reshape::X_PENCILS_TO_BRICKS,
                                               reshape_aux),
                      (int)dtfft::Error::SUCCESS);
            }
            if (reshape_aux)
                check("free X-distributed pencil reshape auxiliary",
                      (int)custom_plan.mem_free(reshape_aux), (int)dtfft::Error::SUCCESS);
            if (reshape_out)
                check("free X-distributed pencil output",
                      (int)custom_plan.mem_free(reshape_out), (int)dtfft::Error::SUCCESS);
            if (reshape_in)
                check("free X-distributed pencil input",
                      (int)custom_plan.mem_free(reshape_in), (int)dtfft::Error::SUCCESS);
        }

        void* memory = nullptr;
        check("plan memory allocation",
              (int)plan.mem_alloc(plan.get_alloc_bytes(), &memory),
              (int)dtfft::Error::SUCCESS);
        check("allocated plan memory", memory != nullptr ? 1 : 0, 1);
        check("plan memory free", (int)plan.mem_free(memory), (int)dtfft::Error::SUCCESS);
        memory = plan.mem_alloc(plan.get_alloc_bytes());
        check("plan memory allocation by value", memory != nullptr ? 1 : 0, 1);
        check("plan memory free by value", (int)plan.mem_free(memory),
              (int)dtfft::Error::SUCCESS);
        check("null plan memory output",
              (int)plan.mem_alloc(plan.get_alloc_bytes(), nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null plan memory free", (int)plan.mem_free(nullptr),
              (int)dtfft::Error::INVALID_USAGE);

        check("null execution input",
              (int)plan.execute(nullptr, &buffer, dtfft::Execute::FORWARD),
              (int)dtfft::Error::INVALID_USAGE);
        check("null execution output",
              (int)plan.execute(&buffer, nullptr, dtfft::Execute::FORWARD),
              (int)dtfft::Error::INVALID_USAGE);
        check("null forward input",
              (int)plan.forward(nullptr, &buffer, nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null backward output",
              (int)plan.backward(&buffer, nullptr, nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null allocation-size output",
              (int)plan.get_alloc_size(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null auxiliary-size output", (int)plan.get_aux_size(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null auxiliary-bytes output", (int)plan.get_aux_bytes(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null element-size output", (int)plan.get_element_size(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null allocation-bytes output", (int)plan.get_alloc_bytes(nullptr),
              (int)dtfft::Error::INVALID_USAGE);
        check("null precision output",
              (int)plan.get_precision(nullptr), (int)dtfft::Error::INVALID_USAGE);
        check("null executor output",
              (int)plan.get_executor(nullptr), (int)dtfft::Error::INVALID_USAGE);
        check("null dimensions outputs",
              (int)plan.get_dims(nullptr, nullptr), (int)dtfft::Error::INVALID_USAGE);
        check("null grid-dimensions outputs",
              (int)plan.get_grid_dims(nullptr, nullptr), (int)dtfft::Error::INVALID_USAGE);
        dtfft::Pencil invalid_layout_pencil;
        check("invalid pencil layout",
              (int)plan.get_pencil(static_cast<dtfft::Layout>(999), invalid_layout_pencil),
              (int)dtfft::Error::INVALID_LAYOUT);
        check("null transpose input",
              (int)plan.transpose(nullptr, &buffer, dtfft::Transpose::X_TO_Y),
              (int)dtfft::Error::INVALID_USAGE);
        check("null transpose request",
              (int)plan.transpose_start(&buffer, &buffer, dtfft::Transpose::X_TO_Y,
                                        static_cast<dtfft_request_t*>(nullptr)),
              (int)dtfft::Error::INVALID_USAGE);
        dtfft_request_t request;
        check("inplace transpose start request",
              (int)plan.transpose_start(&buffer, &buffer, dtfft::Transpose::X_TO_Y, nullptr, &request),
              (int)dtfft::Error::INPLACE_TRANSPOSE);
        check("null transpose end request",
              (int)plan.transpose_end(nullptr), (int)dtfft::Error::INVALID_USAGE);
        check("null reshape input",
              (int)plan.reshape(nullptr, &buffer, dtfft::Reshape::X_BRICKS_TO_PENCILS),
              (int)dtfft::Error::INVALID_USAGE);
        check("null reshape output",
              (int)plan.reshape(&buffer, nullptr, dtfft::Reshape::X_BRICKS_TO_PENCILS),
              (int)dtfft::Error::INVALID_USAGE);
        check("null reshape request",
              (int)plan.reshape_start(&buffer, &buffer, dtfft::Reshape::X_BRICKS_TO_PENCILS,
                                      nullptr, static_cast<dtfft_request_t*>(nullptr)),
              (int)dtfft::Error::INVALID_USAGE);
       check("inplace reshape start request",
              (int)plan.reshape_start(&buffer, &buffer, dtfft::Reshape::X_BRICKS_TO_PENCILS,
                                      nullptr, &request),
              (int)dtfft::Error::INPLACE_RESHAPE);

        check("null reshape end request",
              (int)plan.reshape_end(nullptr), (int)dtfft::Error::INVALID_USAGE);

        const std::string plan_file = "test_api_cxx.dtfft";
        check("save C++ plan", (int)plan.save(plan_file), (int)dtfft::Error::SUCCESS);
        {
            dtfft::PlanC2C loaded_plan(plan_file, MPI_COMM_WORLD);
            check("load C++ plan", loaded_plan.get_dims() == dims ? 1 : 0, 1);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (comm_rank == 0)
            std::remove(plan_file.c_str());
        MPI_Barrier(MPI_COMM_WORLD);

        check("destroy plan", (int)plan.destroy(), (int)dtfft::Error::SUCCESS);
        check("destroy missing plan", (int)plan.destroy(),
              (int)dtfft::Error::PLAN_NOT_CREATED);
        int8_t missing_ndims = -1;
        const int32_t* missing_dims = dims.data();
        check("query dimensions from missing plan",
              (int)plan.get_dims(&missing_ndims, &missing_dims),
              (int)dtfft::Error::PLAN_NOT_CREATED);
        check("missing plan dimensions are cleared",
              missing_ndims == 0 && missing_dims == nullptr ? 1 : 0, 1);
        dtfft::Pencil missing_pencil;
        check("query pencil from missing plan",
              (int)plan.get_pencil(dtfft::Layout::X_PENCILS, missing_pencil),
              (int)dtfft::Error::PLAN_NOT_CREATED);
        check("report missing plan", (int)plan.report(),
              (int)dtfft::Error::PLAN_NOT_CREATED);
        check("save missing plan", (int)plan.save("unused.dtfft"),
              (int)dtfft::Error::PLAN_NOT_CREATED);
    }

    check_pencil_error("uninitialized pencil dimensions",
                       [](dtfft::Pencil& pencil) { (void)pencil.get_ndims(); },
                       (int)dtfft::Error::INVALID_USAGE);
    check_pencil_error("uninitialized pencil alignment",
                       [](dtfft::Pencil& pencil) { (void)pencil.get_dim(); },
                       (int)dtfft::Error::INVALID_USAGE);
    check_pencil_error("uninitialized pencil starts",
                       [](dtfft::Pencil& pencil) { (void)pencil.get_starts(); },
                       (int)dtfft::Error::INVALID_USAGE);
    check_pencil_error("uninitialized pencil counts",
                       [](dtfft::Pencil& pencil) { (void)pencil.get_counts(); },
                       (int)dtfft::Error::INVALID_USAGE);
    check_pencil_error("uninitialized pencil size",
                       [](dtfft::Pencil& pencil) { (void)pencil.get_size(); },
                       (int)dtfft::Error::INVALID_USAGE);
    check_pencil_error("uninitialized pencil C structure",
                       [](dtfft::Pencil& pencil) { (void)pencil.c_struct(); },
                       (int)dtfft::Error::INVALID_USAGE);

    {
        const std::vector<int32_t> starts = { 0, 0, 0 };
        const std::vector<int32_t> counts = dims;
        dtfft::Pencil pencil(starts, counts);
        check("constructed pencil dimensions", (int)pencil.get_ndims(), 3);
        check("constructed pencil C structure", pencil.c_struct().ndims == 3 ? 1 : 0, 1);
    }
    check_pencil_constructor_error("null pencil starts", 3, nullptr, dims.data(),
                                   (int)dtfft::Error::INVALID_USAGE);
    check_pencil_constructor_error("invalid pencil dimension count", 1, dims.data(),
                                   dims.data(), (int)dtfft::Error::INVALID_USAGE);

    {
        dtfft::PlanC2C plan(dims, dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE);
        check("create default-communicator C2C plan",
              plan.c_struct() != nullptr ? 1 : 0, 1);
    }
    {
        dtfft::PlanC2C plan(dims, MPI_COMM_WORLD, dtfft::Precision::DOUBLE,
                            dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);
        dtfft::Pencil pencil;
        check("get pencil for C++ pencil plan",
              (int)plan.get_pencil(dtfft::Layout::X_PENCILS, pencil),
              (int)dtfft::Error::SUCCESS);
        dtfft::PlanC2C pencil_plan(pencil, MPI_COMM_WORLD, dtfft::Precision::DOUBLE,
                                   dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);
        check("create C2C pencil plan", pencil_plan.c_struct() != nullptr ? 1 : 0, 1);
    }
    {
        dtfft::PlanR2C plan(dims, MPI_COMM_WORLD, dtfft::Precision::SINGLE,
                            dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);
        check("create R2C plan", plan.c_struct() != nullptr ? 1 : 0, 1);
    }
    {
        dtfft::PlanR2R plan(dims, kinds, MPI_COMM_WORLD, dtfft::Precision::SINGLE,
                            dtfft::Effort::ESTIMATE, dtfft::Executor::NONE);
        check("create R2R plan", plan.c_struct() != nullptr ? 1 : 0, 1);
    }
    {
        dtfft::PlanR2R plan(dims, dtfft::Precision::SINGLE, dtfft::Effort::ESTIMATE);
        check("create R2R plan without kinds",
              plan.c_struct() != nullptr ? 1 : 0, 1);
    }

    check_constructor_error("invalid number of dimensions", 1, dims.data(),
                            dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE,
                            dtfft::Executor::NONE,
                            dtfft::Error::INVALID_N_DIMENSIONS);
    check_constructor_error("non-positive dimension", 2, invalid_dims,
                            dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE,
                            dtfft::Executor::NONE,
                            dtfft::Error::INVALID_DIMENSION_SIZE);
    check_constructor_error("invalid precision", 3, dims.data(),
                            static_cast<dtfft::Precision>(999), dtfft::Effort::ESTIMATE,
                            dtfft::Executor::NONE, dtfft::Error::INVALID_PRECISION);
    check_constructor_error("invalid effort", 3, dims.data(),
                            dtfft::Precision::DOUBLE, static_cast<dtfft::Effort>(999),
                            dtfft::Executor::NONE, dtfft::Error::INVALID_EFFORT);
    check_constructor_error("invalid executor", 3, dims.data(),
                            dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE,
                            static_cast<dtfft::Executor>(999), dtfft::Error::INVALID_EXECUTOR);
    check_constructor_error("null dimensions", 3, nullptr, dtfft::Precision::DOUBLE,
                            dtfft::Effort::ESTIMATE, dtfft::Executor::NONE,
                            dtfft::Error::INVALID_USAGE);
    {
        const std::string missing_file = "missing_test_api_cxx.dtfft";
        MPI_Barrier(MPI_COMM_WORLD);
        if (comm_rank == 0)
            std::remove(missing_file.c_str());
        MPI_Barrier(MPI_COMM_WORLD);
        int actual = (int)dtfft::Error::SUCCESS;
        try {
            dtfft::PlanC2C missing_plan(missing_file, MPI_COMM_WORLD);
        } catch (const dtfft::Exception& error) {
            actual = (int)error.get_error_code();
        }
        check("missing plan file reports error", actual,
              (int)dtfft::Error::FILE_NOT_FOUND);
    }

    if (comm_rank == 0)
        std::printf("%s\n", n_failed == 0 ? "All tests PASSED" : "Some tests FAILED");

    MPI_Finalize();
    return n_failed == 0 ? 0 : 1;
}
