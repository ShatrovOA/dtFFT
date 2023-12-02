module test_utils
use iso_fortran_env
use mpi_f08
implicit none
private


contains

  subroutine report_single(test_name, tf, tb, local_error)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    real(real32),       intent(in)  :: local_error


  end subroutine report_single


  subroutine report_double(test_name, tf, tb, local_error)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    real(real64),       intent(in)  :: local_error

    
  end subroutine report_double

  subroutine report_internal(test_name, tf, tb, local_error, mpi_type, error_threshold)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    type(*),            intent(in)  :: local_error
    type(MPI_Datatype), intent(in)  :: mpi_type
    real(real64),       intent(in)  :: error_threshold
    integer(int32)                  :: comm_rank, comm_size

    call MPI_Comm_size(MPI_COMM_WORLD, comm_size)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank)

  end subroutine report_internal
end module