#include "dtfft_config.h"
module test_utils
use iso_fortran_env
use dtfft_utils, only: int_to_str, double_to_str
#include "dtfft_mpi.h"
#ifdef DTFFT_WITH_CUDA
use cudafor
#include "dtfft_cuda.h"
#endif
implicit none
private
public :: report
public :: attach_gpu_to_process

  interface report
    module procedure :: report_single
    module procedure :: report_double
  end interface report

contains

  subroutine report_single(test_name, tf, tb, local_error)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    real(real32),       intent(in)  :: local_error

    call report_internal(test_name, tf, tb, local_error, real(1e-5, real64))
  end subroutine report_single


  subroutine report_double(test_name, tf, tb, local_error)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    real(real64),       intent(in)  :: local_error

    call report_internal(test_name, tf, tb, local_error, real(1e-12, real64))
  end subroutine report_double

  subroutine report_internal(test_name, tf, tb, local_error, error_threshold)
    character(len=*),   intent(in)  :: test_name
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    class(*),           intent(in)  :: local_error
    real(real64),       intent(in)  :: error_threshold
    integer(int32)                  :: comm_rank, comm_size, ierr
    real(real64) :: local_error_, global_error


    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

    call write_timers("Forward", tf, comm_rank, comm_size)
    call write_timers("Backward", tb, comm_rank, comm_size)

    select type ( local_error )
    type is ( real(real32) )
      local_error_ = real(local_error, real64)
    type is ( real(real64) )
      local_error_ = local_error
    class default
      error stop
    endselect

    call MPI_Allreduce(local_error_, global_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
    if(comm_rank == 0) then
      if(global_error < error_threshold) then
        write(output_unit, '(a)') "Test '"//test_name//"' PASSED!"
      else
        write(error_unit, '(a, d16.5)') "Test '"//test_name//"' FAILED... error = ", global_error
        call MPI_Abort(MPI_COMM_WORLD, -1, ierr)
      endif
    endif
  end subroutine report_internal

  subroutine write_timers(direction, execution_time, comm_rank, comm_size)
    character(len=*), intent(in)  :: direction
    real(real64),     intent(in)  :: execution_time
    integer(int32),   intent(in)  :: comm_rank
    integer(int32),   intent(in)  :: comm_size
    real(real64) :: max_time, min_time, avg_time
    integer(int32) :: ierr

    call MPI_Allreduce(execution_time, max_time, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
    call MPI_Allreduce(execution_time, min_time, 1, MPI_REAL8, MPI_MIN, MPI_COMM_WORLD, ierr)
    call MPI_Allreduce(execution_time, avg_time, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
    avg_time = avg_time / real(comm_size, real64)

    if ( comm_rank == 0 ) then
      write(output_unit, '(a)') direction//" execution time"
      write(output_unit, '(a)') "  max: "//double_to_str(max_time)//" [s]"
      write(output_unit, '(a)') "  min: "//double_to_str(min_time)//" [s]"
      write(output_unit, '(a)') "  avg: "//double_to_str(avg_time)//" [s]"
    endif
  end subroutine write_timers

  subroutine attach_gpu_to_process
#ifdef DTFFT_WITH_CUDA
    integer(int32) :: comm_rank, ierr, host_rank, host_size, num_devices
    TYPE_MPI_COMM  :: host_comm

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, host_comm, ierr)
    call MPI_Comm_rank(host_comm, host_rank, ierr)
    call MPI_Comm_size(host_comm, host_size, ierr)
    call MPI_Comm_free(host_comm, ierr)

    CUDA_CALL( "cudaGetDeviceCount", cudaGetDeviceCount(num_devices) )
    if ( num_devices == 0 ) error stop "GPUs not found on host"
    if ( num_devices < host_size ) error stop "Number of GPU devices < Number of MPI processes"

    CUDA_CALL( "cudaSetDevice", cudaSetDevice(host_rank) )
#endif
  end subroutine attach_gpu_to_process
end module