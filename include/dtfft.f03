#define DTFFT_CHECK(ierr)                             \
  if( ierr /= DTFFT_SUCCESS ) then;                   \
    block;                                            \
      use iso_fortran_env, only: error_unit;          \
      character(len=:), allocatable :: string;        \
      integer :: mpi_err;                             \
      call dtfft_get_error_string(ierr, string);      \
      write(error_unit, '(a)') string;                \
      call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_err);  \
    endblock;                                         \
  endif
