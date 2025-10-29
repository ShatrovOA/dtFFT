#if defined(DTFFT_USE_MPI)
use mpi
#define TYPE_MPI_COMM integer(int32)
#define TYPE_MPI_DATATYPE integer(int32)
#define TYPE_MPI_REQUEST integer(int32)
#define TYPE_MPI_GROUP integer(int32)
#define TYPE_MPI_WIN integer(int32)
#define GET_MPI_VALUE(val) val
#define ALL_REDUCE(buffer, dtype, op, comm, ierr) call all_reduce_inplace(buffer, op, comm, ierr)
#else
use mpi_f08
#define TYPE_MPI_COMM type(MPI_Comm)
#define TYPE_MPI_DATATYPE type(MPI_Datatype)
#define TYPE_MPI_REQUEST type(MPI_Request)
#define TYPE_MPI_GROUP type(MPI_Group)
#define TYPE_MPI_WIN type(MPI_Win)
#define GET_MPI_VALUE(val) val % MPI_VAL
#define ALL_REDUCE(buffer, dtype, op, comm, ierr) call MPI_Allreduce(MPI_IN_PLACE, buffer, 1, dtype, op, comm, ierr)
#endif
