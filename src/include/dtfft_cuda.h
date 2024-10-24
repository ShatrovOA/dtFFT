#ifdef DTFFT_WITH_CUDA
# ifdef _DEVELOPMENT
#  define device contiguous
# endif
# define LOC_FUN c_devloc
# define C_ADDR c_devptr
#else
# define LOC_FUN c_loc
# define C_ADDR c_ptr
#endif

#define CUFFT_CALL(name, func)                                                                                                  \
  ierr = func;                                                                                                                  \
  if( ierr /= CUFFT_SUCCESS ) then;                                                                                             \
    block;                                                                                                                      \
      integer(IP) :: mpi_err;                                                                                                   \
      write(error_unit, '(a)') "Error occured during call to cuFFT function '"//name//"': "//cufftGetErrorString(ierr);         \
      call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_err);                                                                            \
    endblock;                                                                                                                   \
  endif
