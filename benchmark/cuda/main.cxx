#include "config.hpp"
#include "dtfft_bench.hpp"
#include "cudecomp_bench.hpp"
#include "accfft_bench.hpp"
#include "heffte_bench.hpp"
#include "2d_decomp_bench.hpp"

#include <map>
#include <iomanip>
#include <limits>
#include <exception>


void check_mem()
{
    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    printf("Available memory = %zu\n", mf);
}

void print_benchmark_results(std::map<std::string, double> &benchmark_results, std::vector<int> &dims)
{
    int comm_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (comm_rank != 0)
        return;

    if (benchmark_results.empty())
    {
        printf("No benchmark results to display.\n");
        return;
    }

    printf("\n");
    printf("===============================================\n");
    printf("    BENCHMARK RESULTS [%ix%ix%i]\n", dims[0], dims[1], dims[2]);
    printf("===============================================\n");
    printf("%-30s %12s\n", "Benchmark", "Time (ms)");
    printf("-----------------------------------------------\n");

    // Найдем лучший результат
    double best_time = std::numeric_limits<double>::max();
    std::string best_benchmark;

    for (const auto &result : benchmark_results)
    {
        if (result.second > 0 && result.second < best_time)
        {
            best_time = result.second;
            best_benchmark = result.first;
        }
    }

    for (const auto &result : benchmark_results)
    {
        const char *marker = (result.first == best_benchmark && result.second > 0) ? " <-- WINNER" : "";
        printf("%-30s %12.3f%s\n",
               result.first.c_str(),
               result.second,
               marker);
    }

    printf("===============================================\n");
    printf("\n");
}


std::vector<int> scale_dims_balanced(const std::vector<int>& base_dims, int comm_size) {
    std::vector<int> scaled_dims = base_dims;
    int remaining_procs = comm_size;

    if ( remaining_procs == 1 ) return scaled_dims;

    while (remaining_procs > 1) {
        for (int dim_idx = 2; dim_idx >= 0 && remaining_procs > 1; dim_idx--) {
            if (remaining_procs % 2 == 0) {
                scaled_dims[dim_idx] *= 2;
                remaining_procs /= 2;
            } else {
                scaled_dims[dim_idx] *= remaining_procs;
                remaining_procs = 1;
            }
        }
    }

    return scaled_dims;
}

void run_all(std::vector<int>&dims, bool weak_scaling)
{
    int comm_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    std::map<std::string, double> benchmark_results;

    dtfft::Effort effort = dtfft::Effort::EXHAUSTIVE;
    if( !weak_scaling ) {

        double cudecomp_time =  run_cudecomp(dims, CUDECOMP_DOUBLE_COMPLEX, CUDECOMP_TRANSPOSE_COMM_MPI_P2P);
        if (cudecomp_time > 0) benchmark_results["cuDecomp"] = cudecomp_time;

        double heffte_time = run_heffte<double>(dims);
        if (heffte_time > 0) benchmark_results["HeFFTe"] = heffte_time;

        setup_dtfft_config(false);
        double dtfft_time = run_dtfft_c2c(dims, dtfft::Precision::DOUBLE, dtfft::Executor::CUFFT, false);
        if (dtfft_time > 0) benchmark_results["dtFFT CUFFT"] = dtfft_time;

        double decomp_time = run_2d_decomp(dims);
        if (decomp_time > 0) benchmark_results["2D-decomp"] = decomp_time;

        // double accfft_time = run_accfft<double>(dims);
        // if (accfft_time > 0) benchmark_results["AccFFT"] = accfft_time;
        // check_mem();
    } else {
        dtfft::Backend backend = dtfft::Backend::NCCL;
        effort = dtfft::Effort::MEASURE;
        std::vector<bool> z_slab_opts = {true, false};
        for ( auto enable_z_slab : z_slab_opts ) {
            setup_dtfft_config(enable_z_slab, backend);
            double dtfft_time = run_dtfft_c2c(dims, dtfft::Precision::DOUBLE, dtfft::Executor::NONE, enable_z_slab, effort);

            if (dtfft_time > 0) {
                auto bench_name = std::string("dtFFT transpose-only");
                if (enable_z_slab)
                    bench_name += " Z-slab";
                benchmark_results[bench_name] = dtfft_time;
            }
            // run_dtfft_r2r(dims, dtfft::Precision::DOUBLE, enable_z_slab);
            // run_dtfft_r2r(dims, dtfft::Precision::SINGLE, enable_z_slab);
        }
        check_mem();
    }

    print_benchmark_results(benchmark_results, dims);
}


int main(int argc, char *argv[]) {

    // MPI_Init must be called before calling dtFFT
    MPI_Init(&argc, &argv);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CALL( cudaSetDevice(local_rank) );
    MPI_Comm_free(&local_comm);

    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if( comm_rank == 0 ) {
        printf("*******************************\n");
        printf("Number of GPUs: %d\n", comm_size);
        printf("*******************************\n");
    }

    std::vector<std::vector<int>> base_dims_sets = {
        // {256, 256, 256},    // Basic test
        {1024, 1024, 1024},     // Narrow Z dimension
        // {2048, 32, 2048},     // Narrow Y dimension
        // {32, 2048, 2048},     // Narrow X dimension
        // {1999, 1047, 215}     // Pure evil dimensions
    };

    for (auto& base_dims : base_dims_sets) {

        std::vector<bool> weak_scaling_opts = {false};

        for( auto weak_scaling : weak_scaling_opts )
        {
            std::vector<int> final_dims;
            if (weak_scaling) {
                final_dims = scale_dims_balanced(base_dims, comm_size);

                if (comm_rank == 0) {
                    printf("Base dims: [%d, %d, %d] -> Scaled dims: [%d, %d, %d]\n",
                            base_dims[0], base_dims[1], base_dims[2],
                            final_dims[0], final_dims[1], final_dims[2]);
                    printf("Scale factors: [x%.1f, x%.1f, x%.1f]\n",
                            (double)final_dims[0] / base_dims[0],
                            (double)final_dims[1] / base_dims[1],
                            (double)final_dims[2] / base_dims[2]);
                    printf("Elements per process: %lld\n",
                            static_cast<long long>(base_dims[0]) * base_dims[1] * base_dims[2]);
                }
            } else {
                // Strong scaling
                final_dims = base_dims;

                if (comm_rank == 0) {
                    printf("Fixed dims: [%d, %d, %d]\n",
                            final_dims[0], final_dims[1], final_dims[2]);
                    printf("Total elements: %lld\n",
                            static_cast<long long>(final_dims[0]) * final_dims[1] * final_dims[2]);
                    printf("Elements per process: %lld\n",
                            static_cast<long long>(final_dims[0]) * final_dims[1] * final_dims[2] / comm_size);
                }
            }

            run_all(final_dims, weak_scaling);
        }
    }

    MPI_Finalize();
}