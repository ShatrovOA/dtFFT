program main
use dtfft
use mpi_f08
implicit none
type(dtfft_plan_r2r) :: plan

#ifdef USE_MPI
  call plan%create([2,2,2], comm=MPI_COMM_WORLD%MPI_VAL)
#else
  call plan%create([2,2,2], comm=MPI_COMM_WORLD)
#endif
end program main