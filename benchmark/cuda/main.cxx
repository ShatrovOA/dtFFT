    #include "config.hpp"
    #include "dtfft_bench.hpp"
    #include "cudecomp_bench.hpp"
    #include "accfft_bench.hpp"
    #include "heffte_bench.hpp"

std::vector<int> scale_dims_balanced(const std::vector<int>& base_dims, int comm_size) {
    std::vector<int> scaled_dims = base_dims;
    int remaining_procs = comm_size;

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

    // run_cudecomp(dims, CUDECOMP_DOUBLE_COMPLEX);
    run_cudecomp(dims, CUDECOMP_DOUBLE);
    // run_cudecomp(dims, CUDECOMP_FLOAT);

    std::vector<bool> z_slab_opts = {true, false};
    for ( auto enable_z_slab : z_slab_opts ) {
        setup_dtfft_config(enable_z_slab);
        // run_dtfft_c2c(dims, dtfft::Precision::DOUBLE, dtfft::Executor::NONE, enable_z_slab);
        run_dtfft_r2r(dims, dtfft::Precision::DOUBLE, enable_z_slab);
        // run_dtfft_r2r(dims, dtfft::Precision::SINGLE, enable_z_slab);
    }

    if( !weak_scaling ) {
        // run_heffte<double>(dims, false);
        run_heffte<float>(dims, false);
        setup_dtfft_config(false);
        // run_dtfft_c2c(dims, dtfft::Precision::DOUBLE, dtfft::Executor::CUFFT, false);
        run_dtfft_c2c(dims, dtfft::Precision::SINGLE, dtfft::Executor::CUFFT, false);
        if ( comm_size > 1 ) {
            // run_accfft<double>(dims);
            run_accfft<float>(dims);
        }
    }
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
        {1024, 1024, 512},    // Basic test
        // {2048, 2048, 32},     // Narrow Z dimension
        // {2048, 32, 2048},     // Narrow Y dimension
        // {32, 2048, 2048},     // Narrow X dimension
        // {1999, 1047, 215}     // Pure evil dimensions
    };

    for (auto& base_dims : base_dims_sets) {

        std::vector<bool> weak_scaling_opts = {true, false};

        for( auto weak_scaling : weak_scaling_opts )
        {
            std::vector<int> final_dims;
            if ( weak_scaling && comm_size == 1 ) {
                continue;
            } else if (weak_scaling) {
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