#if defined(DTFFT_USE_MPI)
use mpi
#define TYPE_MPI_COMM integer(IP)
#define TYPE_MPI_DATATYPE integer(IP)
#define TYPE_MPI_REQUEST integer(IP)
#else
use mpi_f08
#define TYPE_MPI_COMM type(MPI_Comm)
#define TYPE_MPI_DATATYPE type(MPI_Datatype)
#define TYPE_MPI_REQUEST type(MPI_Request)
#endif
