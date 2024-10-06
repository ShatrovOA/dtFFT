#if defined(DTFFT_USE_MPI)
use mpi
#define TYPE_MPI_COMM integer(IP)
#define TYPE_MPI_DATATYPE integer(IP)
#define TYPE_MPI_REQUEST integer(IP)
#define DTFFT_GET_MPI_VALUE( val ) val
#else
use mpi_f08
#define TYPE_MPI_COMM type(MPI_Comm)
#define TYPE_MPI_DATATYPE type(MPI_Datatype)
#define TYPE_MPI_REQUEST type(MPI_Request)
#define DTFFT_GET_MPI_VALUE( val ) val%MPI_VAL
#endif
