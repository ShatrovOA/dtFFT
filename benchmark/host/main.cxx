#include "config.hpp"
#include "dtfft_bench.hpp"
#include "fftw3_bench.hpp"
#include <omp.h>
#include "pfft_bench.hpp"
#include "p3dfft_bench.hpp"
#include "heffte_bench.hpp"
#include "accfft_bench.hpp"
#include "2d_decomp_bench.hpp"
// #include "mkl_bench.hpp"

#include <map>
#include <iomanip>
#include <limits>
#include <exception>


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
    printf("%-30s %12s\n", "Benchmark", "Time (s)");
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

std::vector<int> scale_dims_balanced(const std::vector<int> &base_dims, int comm_size)
{
    std::vector<int> scaled_dims = base_dims;
    int remaining_procs = comm_size;

    while (remaining_procs > 1)
    {
        for (int dim_idx = 2; dim_idx >= 0 && remaining_procs > 1; dim_idx--)
        {
            if (remaining_procs % 2 == 0)
            {
                scaled_dims[dim_idx] *= 2;
                remaining_procs /= 2;
            }
            else
            {
                scaled_dims[dim_idx] *= remaining_procs;
                remaining_procs = 1;
            }
        }
    }

    return scaled_dims;
}

void run_all(std::vector<int> &dims, bool weak_scaling)
{
    int comm_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    std::map<std::string, double> benchmark_results;

    double fftw3_time = run_fftw3(dims);
    if (fftw3_time > 0) benchmark_results["FFTW3 "] = fftw3_time;

    // double mkl_time = run_mkl(dims);
    // if (mkl_time > 0) benchmark_results["MKL DFTI "] = mkl_time;

    std::vector<bool> dtype_opts = {true, false};
    std::vector<bool> z_slab_opts = {true, false};

    for (auto enable_z_slab : z_slab_opts)
    {
        for (auto use_dtype : dtype_opts)
        {
            double dtfft_time = run_dtfft_c2c(dims, dtfft::Precision::DOUBLE, dtfft::Executor::NONE, 
                enable_z_slab, use_dtype);
            if (dtfft_time > 0)
            {
                auto bench_name = std::string("dtFFT");
                if (use_dtype)
                    bench_name += " Datatype backend";
                if (enable_z_slab)
                    bench_name += " Z-slab";
                benchmark_results[bench_name] = dtfft_time;
            }
        }
    }

    double pfft_time = run_pfft(dims);
    if (pfft_time > 0) benchmark_results["PFFT"] = pfft_time;

    // P3DFFT бенчмарк
    double p3dfft_time = run_p3dfft(dims);
    if (p3dfft_time > 0) benchmark_results["P3DFFT++"] = p3dfft_time;

    double heffte_time = run_heffte<double>(dims);
    if (heffte_time > 0) benchmark_results["HeFFTe"] = heffte_time;

    try {
        fftw_mpi_init();
        double accfft_time = run_accfft(dims);
        if (accfft_time > 0) benchmark_results["AccFFT"] = accfft_time;
    } catch (const std::exception& e) {
        int comm_rank;

        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

        if ( comm_rank == 0 ) e.what();
    }

    fftw_mpi_init();
    double decomp2d_time = run_2d_decomp(dims);
    if (decomp2d_time > 0) benchmark_results["2d-decomp"] = decomp2d_time;

    print_benchmark_results(benchmark_results, dims);
}

int main(int argc, char *argv[])
{

    // MPI_Init must be called before calling dtFFT
    MPI_Init(&argc, &argv);

    omp_set_num_threads(1);

    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (comm_rank == 0)
    {
        printf("*******************************\n");
        printf("Number of processes: %d\n", comm_size);
        printf("*******************************\n");
    }

    std::vector<std::vector<int>> base_dims_sets = {
        // {1024, 1024, 2},
        {128, 128, 128}, // Basic test
        // {2048, 2048, 64},     // Narrow Z dimension
        // {64, 2048, 2048},     // Narrow Y dimension
        // {1024, 1024, 4},     // Narrow X dimension
        // {1999, 1047, 215}     // Pure evil dimensions
    };

    for (auto &base_dims : base_dims_sets)
    {

        std::vector<bool> weak_scaling_opts = {false};

        for (auto weak_scaling : weak_scaling_opts)
        {
            std::vector<int> final_dims;
            // if (weak_scaling && comm_size == 1)
            // {
            //     continue;
            // }
            // else 
            if (weak_scaling)
            {
                final_dims = scale_dims_balanced(base_dims, comm_size);

                if (comm_rank == 0)
                {
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
            }
            else
            {
                // Strong scaling
                final_dims = base_dims;

                if (comm_rank == 0)
                {
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
    // fftw_mpi_cleanup();

    MPI_Finalize();
}