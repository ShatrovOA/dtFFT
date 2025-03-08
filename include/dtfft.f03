#define DTFFT_CHECK(ierr)                                       \
  if( ierr /= DTFFT_SUCCESS ) then;                             \
    block;                                                      \
      use iso_fortran_env, only: error_unit;                    \
      integer :: mpi_err;                                       \
      write(error_unit, '(a)') dtfft_get_error_string(ierr);    \
      call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_err);            \
    endblock;                                                   \
  endif
