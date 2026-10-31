#include <mpi.h>
#include <nvshmem.h>

extern "C" {
    void
    init_nvshmem(MPI_Fint fcomm, bool *is_init)
    {
        int init_status = nvshmemx_init_status();
        *is_init = init_status != NVSHMEM_STATUS_NOT_INITIALIZED;
        if ( *is_init ) return;

        MPI_Comm comm = MPI_Comm_f2c(fcomm);

        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }

    void finalize_nvshmem(void)
    {
        nvshmem_finalize();
    }
}