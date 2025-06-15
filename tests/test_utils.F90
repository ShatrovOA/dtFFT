#include "dtfft_config.h"
module test_utils
use iso_fortran_env
use iso_c_binding
use dtfft
use dtfft_utils, only: int_to_str, double_to_str
#include "dtfft_mpi.h"
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#include "dtfft_cuda.h"
#endif
implicit none (type, external)
private
public :: report
public :: attach_gpu_to_process
public :: scaleFloatHost, scaleDoubleHost
public :: scaleComplexFloatHost, scaleComplexDoubleHost
public :: checkFloat, checkDouble
public :: checkComplexFloat, checkComplexDouble

#if defined(DTFFT_WITH_CUDA)
public :: scaleComplexFloat, scaleComplexDouble
public :: scaleFloat, scaleDouble

  interface
    subroutine scaleComplexFloat(buffer, count, scale, stream) bind(C, name="scaleComplexFloat")
    import
      type(c_ptr),          value :: buffer
      integer(c_size_t),    value :: count
      integer(c_size_t),    value :: scale
      type(dtfft_stream_t), value :: stream
    end subroutine scaleComplexFloat

    subroutine scaleComplexDouble(buffer, count, scale, stream) bind(C, name="scaleComplexDouble")
    import
      type(c_ptr),          value :: buffer
      integer(c_size_t),    value :: count
      integer(c_size_t),    value :: scale
      type(dtfft_stream_t), value :: stream
    end subroutine scaleComplexDouble

    subroutine scaleFloat(buffer, count, scale, stream) bind(C, name="scaleFloat")
    import
      type(c_ptr),          value :: buffer
      integer(c_size_t),    value :: count
      integer(c_size_t),    value :: scale
      type(dtfft_stream_t), value :: stream
    end subroutine scaleFloat

    subroutine scaleDouble(buffer, count, scale, stream) bind(C, name="scaleDouble")
    import
      type(c_ptr),          value :: buffer
      integer(c_size_t),    value :: count
      integer(c_size_t),    value :: scale
      type(dtfft_stream_t), value :: stream
    end subroutine scaleDouble
  end interface
#endif

  interface report
    module procedure :: reportSingle
    module procedure :: reportDouble
  end interface report

contains

  subroutine reportSingle(tf, tb, local_error, nx, ny, nz) bind(C, name="reportSingle")
    real(c_double),       intent(in)            :: tf
    real(c_double),       intent(in)            :: tb
    real(c_float),        intent(in)            :: local_error
    integer(c_int32_t),   intent(in)            :: nx
    integer(c_int32_t),   intent(in)            :: ny
    integer(c_int32_t),   intent(in), optional  :: nz
    integer(int32) :: temp
    real(real32)   :: errthr

    temp = nx * ny; if( present(nz) ) temp = temp * nz
    errthr = 5.0_real32 * log( real(temp, real32) ) / log(real(2.0, real32)) * (nearest(1._real32, 1._real32) - nearest(1._real32,-1._real32))
    call report_internal(tf, tb, real(local_error, real64), real(errthr, real64))
  end subroutine reportSingle

  subroutine reportDouble(tf, tb, local_error, nx, ny, nz)  bind(C, name="reportDouble")
    real(c_double),       intent(in)            :: tf
    real(c_double),       intent(in)            :: tb
    real(c_double),       intent(in)            :: local_error
    integer(c_int32_t),   intent(in)            :: nx
    integer(c_int32_t),   intent(in)            :: ny
    integer(c_int32_t),   intent(in), optional  :: nz
    integer(int32) :: temp
    real(real64)   :: errthr

    temp = nx * ny; if( present(nz) ) temp = temp * nz
    errthr = 5.0_real64 * log( real(temp, real64) ) / log(real(2.0, real64)) * (nearest(1._real64, 1._real64) - nearest(1._real64,-1._real64))
    call report_internal(tf, tb, local_error, errthr)
  end subroutine reportDouble

  subroutine report_internal(tf, tb, local_error, error_threshold)
    real(real64),       intent(in)  :: tf
    real(real64),       intent(in)  :: tb
    real(real64),       intent(in)  :: local_error
    real(real64),       intent(in)  :: error_threshold
    integer(int32)                  :: comm_rank, comm_size, ierr
    real(real64) :: global_error

    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

    call MPI_Allreduce(local_error, global_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
    if(comm_rank == 0) then
      write(output_unit, '(a)') repeat("*", 40)
      if(global_error < error_threshold .and. global_error >= 0._real64) then
        write(output_unit, '(a)') "            Test PASSED!"
      else
        write(output_unit, '(a, d16.5, a, d16.5)') "Test FAILED... error = ", global_error, ' threshold = ', error_threshold
        call MPI_Abort(MPI_COMM_WORLD, -1, ierr)
      endif
      write(output_unit, '(a)') repeat("*", 40)
    endif
    call write_timers("Forward", tf, comm_rank, comm_size)
    if(comm_rank == 0) write(output_unit, '(a)') repeat("-", 40)
    call write_timers("Backward", tb, comm_rank, comm_size)
    if(comm_rank == 0) write(output_unit, '(a)') repeat("*", 40)
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

  subroutine attach_gpu_to_process() bind(C)
#if defined(DTFFT_WITH_CUDA)
    integer(int32) :: comm_rank, ierr, host_rank, host_size, num_devices
    TYPE_MPI_COMM  :: host_comm
    character(len=5) :: platform_env

    call get_environment_variable("DTFFT_PLATFORM", platform_env)
    ! Default execution on device
    if ( trim(adjustl(platform_env)) == "host" ) return

    CUDA_CALL( "cudaGetDeviceCount", cudaGetDeviceCount(num_devices) )
    if ( num_devices == 0 ) return

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, host_comm, ierr)
    call MPI_Comm_rank(host_comm, host_rank, ierr)
    call MPI_Comm_size(host_comm, host_size, ierr)
    call MPI_Comm_free(host_comm, ierr)

    if ( host_size > num_devices ) return
    CUDA_CALL( "cudaSetDevice", cudaSetDevice(host_rank) )
#endif
  end subroutine attach_gpu_to_process

  subroutine scaleFloatHost(buffer, buf_size, scale) bind(C, name="scaleFloatHost")
    type(c_ptr),        value :: buffer
    integer(c_size_t),  value :: buf_size
    integer(c_size_t),  value :: scale
    real(real32), pointer :: buf(:)

    call c_f_pointer(buffer, buf, [buf_size])
    buf(:) = buf(:) / real(scale, real32)
  end subroutine scaleFloatHost

  subroutine scaleDoubleHost(buffer, buf_size, scale) bind(C, name="scaleDoubleHost")
    type(c_ptr),        value :: buffer
    integer(c_size_t),  value :: buf_size
    integer(c_size_t),  value :: scale
    real(real64), pointer :: buf(:)

    call c_f_pointer(buffer, buf, [buf_size])
    buf(:) = buf(:) / real(scale, real64)
  end subroutine scaleDoubleHost

  subroutine scaleComplexFloatHost(buffer, buf_size, scale) bind(C, name="scaleComplexFloatHost")
    type(c_ptr),        value :: buffer
    integer(c_size_t),  value :: buf_size
    integer(c_size_t),  value :: scale
    call scaleFloatHost(buffer, 2 * buf_size, scale)
  end subroutine scaleComplexFloatHost

  subroutine scaleComplexDoubleHost(buffer, buf_size, scale) bind(C, name="scaleComplexDoubleHost")
    type(c_ptr),        value :: buffer
    integer(c_size_t),  value :: buf_size
    integer(c_size_t),  value :: scale
    call scaleDoubleHost(buffer, 2 * buf_size, scale)
  end subroutine scaleComplexDoubleHost

  function checkFloat(check, buf, buf_size) result(err) bind(C, name="checkFloat")
    type(c_ptr),        value :: check
    type(c_ptr),        value :: buf
    integer(c_size_t),  value :: buf_size
    real(c_float)             :: err
    real(real32),  pointer :: check_(:)
    real(real32),  pointer :: buf_(:)

    call c_f_pointer(check, check_, [buf_size])
    call c_f_pointer(buf, buf_, [buf_size])
    err = maxval(abs(check_ - buf_))
  end function checkFloat

  function checkDouble(check, buf, buf_size) result(err) bind(C, name="checkDouble")
    type(c_ptr),        value :: check
    type(c_ptr),        value :: buf
    integer(c_size_t),  value :: buf_size
    real(c_double)            :: err
    real(real64),  pointer :: check_(:)
    real(real64),  pointer :: buf_(:)

    call c_f_pointer(check, check_, [buf_size])
    call c_f_pointer(buf, buf_, [buf_size])
    err = maxval(abs(check_ - buf_))
  end function checkDouble

  function checkComplexFloat(check, buf, buf_size) result(err) bind(C, name="checkComplexFloat")
    type(c_ptr),        value :: check
    type(c_ptr),        value :: buf
    integer(c_size_t),  value :: buf_size
    real(c_float)             :: err

    err = checkFloat(check, buf, 2 * buf_size)
  end function checkComplexFloat

  function checkComplexDouble(check, buf, buf_size) result(err) bind(C, name="checkComplexDouble")
    type(c_ptr),        value :: check
    type(c_ptr),        value :: buf
    integer(c_size_t),  value :: buf_size
    real(c_double)            :: err

    err = checkDouble(check, buf, 2 * buf_size)
  end function checkComplexDouble
end module