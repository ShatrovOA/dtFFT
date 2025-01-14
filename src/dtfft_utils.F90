#include "dtfft_config.h"
module dtfft_utils
use iso_c_binding,    only: c_char, c_null_char, c_int32_t, c_f_pointer, c_size_t
use iso_fortran_env,  only: int8, int32, int64, real64, output_unit, error_unit
use dtfft_parameters
#ifdef DTFFT_WITH_CUDA
use cudafor
# ifdef DTFFT_WITH_PROFILER
use nvtx
# endif
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
implicit none
private
public :: string_f2c, astring_f2c
public :: int_to_str, double_to_str
public :: write_message, dtfft_init, get_log_enabled
public :: get_inverse_kind
public :: dtfft_enable_z_slab, dtfft_disable_z_slab, get_z_slab_enabled
#ifdef DTFFT_WITH_CUDA
public :: count_unique
public :: get_user_stream, dtfft_set_stream
public :: destroy_stream
public :: dtfft_set_gpu_backend, get_user_gpu_backend
public :: get_mpi_enabled, dtfft_enable_mpi_backends, dtfft_disable_mpi_backends
public :: get_pipelined_enabled, dtfft_enable_pipelined_backends, dtfft_disable_pipelined_backends
# if defined(DTFFT_WITH_PROFILER)
public :: push_nvtx_domain_range, pop_nvtx_domain_range
#endif
#endif

  logical,                    save  :: is_log_enabled = .false.
  !< Should we log messages to stdout or not
  logical,                    save  :: is_z_slab_enabled = .true.
  !< Should we use z-slab decomposition or not
#ifdef DTFFT_WITH_CUDA
  integer(cuda_stream_kind),  save  :: dtfft_stream
  !< Default dtFFT CUDA stream
  integer(cuda_stream_kind),  save  :: dtfft_custom_stream
  !< CUDA stream set by the user
  logical,                    save  :: is_stream_created = .false.
  !< Is the default stream created?
  logical,                    save  :: is_custom_stream = .false.
  !< Is the custom stream provided by the user?
  logical,                    save  :: is_pipelined_enabled = .true.
  !< Should we use pipelined backends or not
  logical,                    save  :: is_mpi_enabled = .false.
  !< Should we use MPI backends or not
  integer(int8),              save  :: dtfft_gpu_backend = DTFFT_GPU_BACKEND_NCCL
  !< Default GPU backend
# if defined(DTFFT_WITH_PROFILER)
  type(nvtxDomainHandle),     save  :: dtfft_domain
  !< NVTX domain handle
  logical,                    save  :: domain_created = .false.
  !< Is the NVTX domain created?
# endif
#endif

  interface int_to_str
  !! Convert integer to string
    module procedure int_to_str_int8
    module procedure int_to_str_int32
  end interface int_to_str

contains

  integer(int32) function dtfft_init()
  !! Checks if MPI is initialized and reads the environment variable to enable logging
    integer(int32)    :: ierr             !< Error code
    logical           :: is_mpi_init      !< Is MPI initialized?
    character(len=1)  :: log_enabled_str  !< Log enabled string
    integer(int32)    :: value_len        !< Length of the environment variable
    integer(int32)    :: val              !< Value of the environment variable

    dtfft_init = DTFFT_SUCCESS

    call MPI_Initialized(is_mpi_init, ierr)
    if( .not. is_mpi_init ) then
      dtfft_init = DTFFT_ERROR_MPI_FINALIZED
      return
    endif

    call get_environment_variable("DTFFT_ENABLE_LOG", log_enabled_str, length=value_len)
    if ( value_len > 0 ) then
      read(log_enabled_str, *) val
      is_log_enabled = abs(val) > 0
    endif
  end function dtfft_init

  pure function get_log_enabled() result(log)
  !! Returns the value of the log_enabled variable
    logical :: log  !< Value of the log_enabled variable
    log = is_log_enabled
  end function get_log_enabled

  subroutine dtfft_enable_z_slab() bind(C)
  !! Enables previously disabled Z-slab optimization
  !!
  !! In order to take effect should be called before plan creation.
    is_z_slab_enabled = .true.
  end subroutine dtfft_enable_z_slab

  subroutine dtfft_disable_z_slab() bind(C)
  !! Disables Z-slab optimization
  !!
  !! One should consider disabling Z-slab optimization in order to resolve ``DTFFT_ERROR_VKFFT_R2R_2D_PLAN`` error 
  !! OR when underlying FFT implementation of 2D plan is too slow.
  !! In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
  !!
  !! In order to take effect should be called before plan creation.
  !!
  !! This option is only valid for 3d plans.
    is_z_slab_enabled = .false.
  end subroutine dtfft_disable_z_slab

  logical function get_z_slab_enabled()
  !! Whether Z-slab optimization is enabled or not
    get_z_slab_enabled = is_z_slab_enabled
  end function get_z_slab_enabled

  subroutine string_f2c(fstring, cstring, string_size)
  !! Convert Fortran string to C string
    character(len=*),           intent(in)    :: fstring        !< Fortran string
    character(kind=c_char),     intent(inout) :: cstring(*)     !< C string
    integer(int64),  optional,  intent(out)   :: string_size    !< Size of the C string
    integer                                   :: i, j           !< Loop indices
    logical                                   :: met_non_blank  !< Have we met a non-blank character?

    j = 1
    met_non_blank = .false.
    do i = 1, len_trim(fstring)
      if (met_non_blank) then
        cstring(j) = fstring(i:i)
        j = j + 1
      else if (fstring(i:i) /= ' ') then
        met_non_blank = .true.
        cstring(j) = fstring(i:i)
        j = j + 1
      end if
    end do

    cstring(j) = c_null_char
    if(present( string_size )) string_size = j
  end subroutine string_f2c

  subroutine astring_f2c(fstring, cstring, string_size)
  !! Convert Fortran string to C allocatable string
    character(len=*),                     intent(in)  :: fstring      !< Fortran string
    character(kind=c_char), allocatable,  intent(out) :: cstring(:)   !< C string
    integer(int64),         optional,     intent(out) :: string_size  !< Size of the C string

    allocate(cstring( len_trim(fstring) + 1 ))
    call string_f2c(fstring, cstring, string_size)
  end subroutine astring_f2c

  function int_to_str_int32(n) result(string)
  !! Convert 32-bit integer to string
    integer(int32),   intent(in)  :: n            !< Integer to convert
    character(len=:), allocatable :: string       !< Resulting string
    character(len=11)             :: temp         !< Temporary string

    write(temp, '(I11)') n
    allocate( string, source= trim(adjustl(temp)))
  end function int_to_str_int32

  function int_to_str_int8(n) result(string)
  !! Convert 8-bit integer to string
    integer(int8),    intent(in)  :: n            !< Integer to convert
    character(len=:), allocatable :: string       !< Resulting string
    character(len=3)              :: temp         !< Temporary string

    write(temp, '(I3)') n
    allocate( string, source= trim(adjustl(temp)))
  end function int_to_str_int8

  function double_to_str(n) result(string)
  !! Convert double to string
    real(real64),     intent(in)  :: n            !< Double to convert
    character(len=:), allocatable :: string       !< Resulting string
    character(len=23)             :: temp         !< Temporary string

    write(temp, '(F15.5)') n
    allocate( string, source= trim(adjustl(temp)))
  end function double_to_str

  subroutine write_message(unit, message, prefix)
  !! Write message to the specified unit
    integer(int32),   intent(in)            :: unit       !< Unit number
    character(len=*), intent(in)            :: message    !< Message to write
    character(len=*), intent(in), optional  :: prefix     !< Prefix to the message
    character(len=:), allocatable           :: prefix_    !< Dummy prefix
    integer(int32)                          :: comm_rank  !< Size of world communicator
    integer(int32)                          :: ierr       !< Error code

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    if ( comm_rank /= 0 ) return

    if ( present( prefix ) ) then
      allocate( prefix_, source=prefix )
    else
      allocate( prefix_, source="" )
    endif

    write(unit, '(a)') prefix_//trim(message)
    flush(unit)

    deallocate( prefix_ )
  end subroutine write_message

  elemental integer(int8) function get_inverse_kind(r2r_kind)
  !! Get the inverse R2R kind of transform for the given R2R kind
    integer(int8), intent(in)  :: r2r_kind        !< R2R kind

    get_inverse_kind = -1
    select case (r2r_kind)
    case ( DTFFT_DCT_1 )
      get_inverse_kind = DTFFT_DCT_1
    case ( DTFFT_DCT_2 )
      get_inverse_kind = DTFFT_DCT_3
    case ( DTFFT_DCT_3 )
      get_inverse_kind = DTFFT_DCT_2
    case ( DTFFT_DCT_4 )
      get_inverse_kind = DTFFT_DCT_4
    case ( DTFFT_DST_1 )
      get_inverse_kind = DTFFT_DST_1
    case ( DTFFT_DST_2 )
      get_inverse_kind = DTFFT_DST_3
    case ( DTFFT_DST_3 )
      get_inverse_kind = DTFFT_DST_2
    case ( DTFFT_DST_4 )
      get_inverse_kind = DTFFT_DST_4
    endselect
  end function get_inverse_kind

#ifdef DTFFT_WITH_CUDA
  integer(int32) function count_unique(x) result(n)
  !! Count the number of unique elements in the array
    integer(int32), intent(in)  :: x(:)   !< Array of integers
    integer(int32), allocatable :: y(:)   !< Array of unique integers

    allocate(y, source=x)
    n = 0
    do while (size(y) > 0)
        n = n + 1
        y = pack(y,mask=(y(:) /= y(1))) ! drops all elements that are 
                                        ! equals to the 1st one (included)
    end do
    deallocate(y)
  end function count_unique

  integer(cuda_stream_kind) function get_user_stream() result(stream)
  !! Returns either the custom provided by user or creates a new one
    if ( is_custom_stream ) then
      stream = dtfft_custom_stream
      return
    endif
    if (.not.is_stream_created) then
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(dtfft_stream) )
      is_stream_created = .true.
    endif
    stream = dtfft_stream
  end function get_user_stream

  subroutine dtfft_set_stream(stream, error_code)
  !! Set the custom CUDA stream for the dtFFT
  !!
  !! User is responsible in destroying this stream.
  !!
  !! This stream must not be destroyed before ``dtfft plan``.
  !!
  !! In order to take effect should be called before plan creation.
    integer(cuda_stream_kind),  intent(in)  :: stream       !< Custom stream
    integer(int32),  optional,  intent(out) :: error_code   !< Optional error code returned to user
    integer(int32)                          :: ierr         !< CUDA error code

    ierr = cudaStreamQuery(stream)
    if ( .not.any(ierr == [cudaSuccess, cudaErrorNotReady]) ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_GPU_INVALID_STREAM
      return
    endif
    dtfft_custom_stream = stream
    is_custom_stream = .true.
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine dtfft_set_stream

  subroutine destroy_stream
  !! Destroy the default stream if it was created
    if ( is_stream_created ) then
      CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(dtfft_stream) )
      is_stream_created = .false.
    endif
  end subroutine destroy_stream

  subroutine dtfft_set_gpu_backend(backend, error_code)
  !! Sets backend that will be used by dtfft when ``effort_flag`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  !!
  !! This call is optional. Default backend is ``DTFFT_GPU_BACKEND_NCCL``.
  !!
  !! In order to take effect should be called before plan creation.
    integer(int8),                 intent(in)  :: backend
    integer(int32),  optional,     intent(out) :: error_code

    if (.not.any(backend == VALID_GPU_BACKENDS)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_GPU_INVALID_BACKEND
      return
    endif
    dtfft_gpu_backend = backend
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine dtfft_set_gpu_backend

  integer(int8) function get_user_gpu_backend()
  !! Returns GPU backend set by the user or default one
    get_user_gpu_backend = dtfft_gpu_backend
  end function get_user_gpu_backend

  logical function get_pipelined_enabled()
  !! Whether pipelined backends are enabled or not
    get_pipelined_enabled = is_pipelined_enabled
  end function get_pipelined_enabled

  subroutine dtfft_enable_pipelined_backends() bind(C)
  !! Enables previously disabled pipelined GPU backends during plan autotuning.
  !!
  !! In order to take effect should be called before plan creation.
    is_pipelined_enabled = .true.
  end subroutine dtfft_enable_pipelined_backends

  subroutine dtfft_disable_pipelined_backends() bind(C)
  !! Disables pipelined GPU backends during plan autotuning.
  !!
  !! In order to take effect should be called before plan creation.
    is_pipelined_enabled = .false.
  end subroutine dtfft_disable_pipelined_backends

  logical function get_mpi_enabled()
  !! Whether MPI backends are enabled or not
    get_mpi_enabled = is_mpi_enabled
  end function get_mpi_enabled

  subroutine dtfft_enable_mpi_backends() bind(C)
  !! Enables MPI GPU Backends for autotuning.
  !! MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
  !!
  !! It was noticed that during plan autotuning GPU memory not being freed completely.
  !! For example:
  !! 1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, 
  !! plan autotuning will leak 8Gb GPU memory.
  !! Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.
  !!
  !! One of the workarounds is to disable MPI Backends by default, which is done here.
  !!
  !! Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
  !! but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms
  !!
  !! In order to take effect should be called before plan creation.
    is_mpi_enabled = .true.
  end subroutine dtfft_enable_mpi_backends

  subroutine dtfft_disable_mpi_backends() bind(C)
  !! Disables previously enabled MPI GPU Backends for during plan autotuning.
  !!
  !! In order to take effect should be called before plan creation.
    is_mpi_enabled = .false.
  end subroutine dtfft_disable_mpi_backends

# if defined(DTFFT_WITH_PROFILER)
  subroutine create_nvtx_domain
  !! Creates a new NVTX domain
    dtfft_domain = nvtxDomainCreate("dtFFT")
    domain_created = .true.
  end subroutine create_nvtx_domain

  subroutine push_nvtx_domain_range(message, color)
  !! Pushes a range to the NVTX domain
    character(len=*), intent(in)  :: message    !< Message to push
    integer(c_int),   intent(in)  :: color      !< Color of the range
    integer(c_int)                :: status     !< Status of the push
    type(nvtxEventAttributes)     :: range      !< Range to push

    if ( .not. domain_created ) call create_nvtx_domain()
    range = nvtxEventAttributes(message, color)
    status = nvtxDomainRangePushEx(dtfft_domain, range)
  end subroutine push_nvtx_domain_range

  subroutine pop_nvtx_domain_range()
  !! Pops a range from the NVTX domain
    integer(c_int)                :: status     !< Status of the pop

    status = nvtxDomainRangePop(dtfft_domain)
  end subroutine pop_nvtx_domain_range
# endif
#endif
end module dtfft_utils