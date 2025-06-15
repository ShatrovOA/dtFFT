program main
use dtfft
#ifdef HAVE_F08
use mpi_f08
#else
use mpi
#endif
implicit none
type(dtfft_plan_r2r_t) :: plan

#if defined(USE_MPI) && defined(HAVE_F08)
  call plan%create([2,2,2], comm=MPI_COMM_WORLD%MPI_VAL)
#else
  call plan%create([2,2,2], comm=MPI_COMM_WORLD)
#endif
end program main