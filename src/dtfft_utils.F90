#include "dtfft_config.h"
module dtfft_utils
use iso_c_binding
use iso_fortran_env,  only: int8, int32, int64, real64, output_unit, error_unit
use dtfft_parameters
#ifdef DTFFT_WITH_NVSHMEM
use dtfft_interface_nvshmem
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: string_f2c, astring_f2c
public :: int_to_str, double_to_str
public :: write_message, init_internal, get_log_enabled
public :: get_env, get_iters_from_env, get_datatype_from_env
public :: get_inverse_kind
public :: get_platform_from_env

public :: is_same_ptr, is_null_ptr
public :: mem_alloc_host, mem_free_host
#ifdef DTFFT_WITH_CUDA
public :: count_unique
public :: Comm_f2c
public :: is_device_ptr
public :: get_gpu_backend_from_env
#endif
#ifdef DTFFT_WITH_NVSHMEM
public :: is_nvshmem_ptr
#endif

  logical,                    save  :: is_log_enabled = .false.
  !! Should we log messages to stdout or not
  type(dtfft_platform_t),     save  :: platform_from_env = PLATFORM_UNDEFINED
  !! Platform obtained from environ
#ifdef DTFFT_WITH_CUDA
  type(dtfft_gpu_backend_t),  save  :: gpu_backend_from_env = BACKEND_NOT_SET
  !! Backend obtained from environ
#endif
  character(len=26), parameter :: UPPER_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  !! Upper case alphabet.
  character(len=26), parameter :: LOWER_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
  !! Lower case alphabet.

  interface int_to_str
    module procedure int_to_str_int8
    module procedure int_to_str_int32
  end interface int_to_str

  interface get_env
    module procedure :: get_env_base
    module procedure :: get_env_string
    module procedure :: get_env_int32
    module procedure :: get_env_int8
    module procedure :: get_env_logical
  end interface get_env

public :: string
  type :: string
  !! Class used to create array of strings
    character(len=:), allocatable :: raw                      !! String
  end type string

  interface string
    module procedure :: string_constructor
  end interface string

  interface
    subroutine mem_alloc_host(alloc_size, ptr) bind(C)
    import
      integer(c_size_t),  value :: alloc_size
      type(c_ptr)               :: ptr
    end subroutine mem_alloc_host

    subroutine mem_free_host(ptr) bind(C)
    import
      type(c_ptr),        value :: ptr
    end subroutine mem_free_host

#ifdef DTFFT_WITH_CUDA
    type(c_ptr) function Comm_f2c(fcomm) bind(C, name="Comm_f2c")
      import
      integer(c_int), value :: fcomm
    end function Comm_f2c

    function is_device_ptr(ptr) result(bool) bind(C)
    !! Checks if pointer can be accessed from device
    import
      type(c_ptr),    value :: ptr    !! Device pointer
      logical(c_bool)       :: bool   !! Result
    end function is_device_ptr
#endif
  end interface

contains

  type(string) function string_constructor(str)
    character(len=*), intent(in)  :: str
    allocate( string_constructor%raw, source=str )
  end function string_constructor

  integer(int32) function init_internal()
  !! Checks if MPI is initialized and reads the environment variable to enable logging
    integer(int32)    :: ierr             !! Error code
    logical           :: is_mpi_init      !! Is MPI initialized?

    init_internal = DTFFT_SUCCESS

    call MPI_Initialized(is_mpi_init, ierr)
    if( .not. is_mpi_init ) then
      init_internal = DTFFT_ERROR_MPI_FINALIZED
      return
    endif
    is_log_enabled = get_env("ENABLE_LOG", .false.)

#ifdef DTFFT_WITH_CUDA
    block
      type(string) :: platforms(2)
      character(len=:), allocatable :: pltfrm_env

      platforms(1) = string("host")
      platforms(2) = string("cuda")

      allocate( pltfrm_env, source=get_env("PLATFORM", "undefined", platforms) )
      if ( pltfrm_env == "undefined") then
        platform_from_env = PLATFORM_UNDEFINED
      else if ( pltfrm_env == "host" ) then
        platform_from_env = DTFFT_PLATFORM_HOST
      else if ( pltfrm_env == "cuda") then
        platform_from_env = DTFFT_PLATFORM_CUDA
      endif

      deallocate( platforms(1)%raw, platforms(2)%raw, pltfrm_env )
    endblock

    block
      type(string) :: backends(7)
      character(len=:), allocatable :: bcknd_env
      integer(int32) :: i

      backends(1) = string("mpi_dt")
      backends(2) = string("mpi_p2p")
      backends(3) = string("mpi_a2a")
      backends(4) = string("mpi_p2p_pipe")
      backends(5) = string("nccl")
      backends(6) = string("nccl_pipe")
      backends(7) = string("cufftmp")

      allocate( bcknd_env, source=get_env("GPU_BACKEND", "undefined", backends) )
      select case ( bcknd_env )
      case ( "undefined" )
        gpu_backend_from_env = BACKEND_NOT_SET
      case ( "mpi_dt" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_MPI_DATATYPE
      case ( "mpi_p2p" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_MPI_P2P
      case ( "mpi_a2a" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_MPI_A2A
      case ( "mpi_p2p_pipe" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED
      case ( "nccl" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_NCCL
      case ( "nccl_pipe" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_NCCL_PIPELINED
      case ( "cufftmp" )
        gpu_backend_from_env = DTFFT_GPU_BACKEND_CUFFTMP
      endselect

      deallocate( bcknd_env )
      do i = 1, size(backends)
        deallocate( backends(i)%raw )
      enddo
    endblock
#endif
  end function init_internal

  type(dtfft_platform_t) function get_platform_from_env()
    get_platform_from_env = platform_from_env
  end function get_platform_from_env

#ifdef DTFFT_WITH_CUDA
  type(dtfft_gpu_backend_t) function get_gpu_backend_from_env()
    get_gpu_backend_from_env = gpu_backend_from_env
  end function get_gpu_backend_from_env
#endif

  function get_env_base(name, full_name, is_external) result(env)
  !! Base function of obtaining dtFFT environment variable
    character(len=*), intent(in)                :: name               !! Name of environment variable without prefix
    character(len=:), intent(out),  allocatable, optional :: full_name          !! Prefixed environment variable name
    logical,          intent(in),   optional    :: is_external
    character(len=:), allocatable               :: env
    integer(int32)                              :: env_val_len        !! Length of the environment variable
    character(len=:), allocatable :: full_name_
    logical :: is_external_

    is_external_ = .false.
    if ( present(is_external) ) is_external_ = is_external

    if ( is_external_ ) then
      allocate( full_name_, source=name )
    else
      allocate( full_name_, source="DTFFT_"//name )
    endif
    if ( present(full_name) ) allocate(full_name, source=full_name_)

    call get_environment_variable(full_name_, length=env_val_len)
    allocate(character(env_val_len) :: env)
    if ( env_val_len == 0 ) then
      deallocate(full_name_)
      return
    endif
    call get_environment_variable(full_name_, env)
    deallocate(full_name_)
  end function get_env_base

  function get_env_string(name, default, valid_values, is_lower) result(env)
    character(len=*), intent(in)            :: name                 !! Name of environment variable without prefix
    character(len=*), intent(in)            :: default              !! Name of environment variable without prefix
    type(string),     intent(in)            :: valid_values(:)
    logical,          intent(in), optional  :: is_lower
    character(len=:), allocatable           :: env
    character(len=:), allocatable           :: full_name          !! Prefixed environment variable name
    character(len=:), allocatable           :: env_val_str        !! String value of the environment variable
    logical                                 :: is_correct         !! Is env value is correct
    integer(int32) :: i, j
    logical :: is_lower_

    ! env_val_str = get_env(name, full_name=full_name)
    allocate( env_val_str, source=get_env(name, full_name=full_name) )
    if ( len(env_val_str) == 0 ) then
      deallocate(env_val_str, full_name)
      allocate(env, source=default)
      return
    endif
    is_lower_ = .true.
    if(present(is_lower)) is_lower_ = is_lower

    if( is_lower_ ) then
      do i=1, len(env_val_str)
        j = index(UPPER_ALPHABET, env_val_str(i:i))
        if (j>0) env_val_str(i:i) = LOWER_ALPHABET(j:j)
      enddo
    endif

    is_correct = any([(env_val_str == valid_values(i)%raw, i=1,size(valid_values))])

    if ( is_correct ) then
      allocate( env, source=env_val_str )
      deallocate(env_val_str, full_name)
      return
    endif
    WRITE_ERROR("Invalid environment variable: "//full_name//", it has been ignored")
    allocate(env, source=default)
    deallocate(env_val_str, full_name)
  end function get_env_string

  integer(int32) function get_env_int32(name, default, valid_values, min_valid_value, is_external) result(env)
  !! Base Integer function of obtaining dtFFT environment variable
    character(len=*), intent(in)            :: name               !! Name of environment variable without prefix
    integer(int32),   intent(in)            :: default            !! Default value in case env is not set or it has wrong value
    integer(int32),   intent(in), optional  :: valid_values(:)    !! List of valid values
    integer(int32),   intent(in), optional  :: min_valid_value    !! Mininum valid value. Usually 0 or 1
    logical,          intent(in), optional  :: is_external
    character(len=:), allocatable           :: full_name          !! Prefixed environment variable name
    character(len=:), allocatable           :: env_val_str        !! String value of the environment variable
    logical                                 :: is_correct         !! Is env value is correct
    integer(int32)                          :: env_val_passed     !! Value of the environment variable

    if ( ( present(valid_values).and.present(min_valid_value) )           &
      .or.(.not.present(valid_values).and..not.present(min_valid_value))  &
    ) then
      error stop "dtFFT Internal error `get_env_int32`"
    endif

    allocate( env_val_str, source=get_env(name, full_name=full_name, is_external=is_external) )
    ! env_val_str = get_env(name, full_name=full_name, is_external=is_external)

    if ( len(env_val_str) == 0 ) then
      deallocate(env_val_str, full_name)
      env = default
      return
    endif
    read(env_val_str, *) env_val_passed
    is_correct = .false.
    if ( present( valid_values ) ) then
      is_correct = any(env_val_passed == valid_values)
    endif
    if ( present( min_valid_value ) ) then
      is_correct = env_val_passed >= min_valid_value
    endif
    if ( is_correct ) then
      env = env_val_passed
      deallocate(env_val_str, full_name)
      return
    endif
    WRITE_ERROR("Invalid environment variable: "//full_name//", it has been ignored")
    env = default
    deallocate(env_val_str, full_name)
  end function get_env_int32

  integer(int8) function get_env_int8(name, default, valid_values) result(env)
  !! Obtains int8 environment variable
    character(len=*), intent(in)  :: name               !! Name of environment variable without prefix
    integer(int8),    intent(in)  :: default            !! Default value in case env is not set or it has wrong value
    integer(int32),   intent(in)  :: valid_values(:)    !! List of valid values
    integer(int32)                :: val                !! Value of the environment variable

    val = get_env(name, int(default, int32), valid_values)
    env = int(val, int8)
  end function get_env_int8

  logical function get_env_logical(name, default, is_external) result(env)
  !! Obtains logical environment variable
    character(len=*), intent(in) :: name                !! Name of environment variable without prefix
    logical,          intent(in) :: default             !! Default value in case env is not set or it has wrong value
    logical,          intent(in), optional  :: is_external
    integer(int32) :: def, val

    if ( default ) then
      def = 1
    else
      def = 0
    endif

    val = get_env(name, def, [0, 1], is_external=is_external)
    env = val == 1
  end function get_env_logical

  integer(int32) function get_iters_from_env(is_warmup) result(n_iters)
  !! Obtains number of iterations from environment variable
    logical,  intent(in) :: is_warmup                   !! Warmup variable flag

    if ( is_warmup ) then
      n_iters = get_env("MEASURE_WARMUP_ITERS", 2, min_valid_value=0)
    else
      n_iters = get_env("MEASURE_ITERS", 5, min_valid_value=1)
    endif
  end function get_iters_from_env

  integer(int8) function get_datatype_from_env(name) result(env)
  !! Obtains datatype id from environment variable
    character(len=*), intent(in)  :: name               !! Name of environment variable without prefix
    env = get_env(name, 2_int8, [1, 2])
  end function get_datatype_from_env

  pure function get_log_enabled() result(log)
  !! Returns the value of the log_enabled variable
    logical :: log  !! Value of the log_enabled variable
    log = is_log_enabled
  end function get_log_enabled

  subroutine string_f2c(fstring, cstring, string_size)
  !! Convert Fortran string to C string
    character(len=*),           intent(in)    :: fstring        !! Fortran string
    character(kind=c_char),     intent(inout) :: cstring(*)     !! C string
    integer(int64),  optional,  intent(out)   :: string_size    !! Size of the C string
    integer                                   :: i, j           !! Loop indices
    logical                                   :: met_non_blank  !! Have we met a non-blank character?

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
    character(len=*),                     intent(in)  :: fstring      !! Fortran string
    character(kind=c_char), allocatable,  intent(out) :: cstring(:)   !! C string
    integer(int64),         optional,     intent(out) :: string_size  !! Size of the C string

    allocate(cstring( len_trim(fstring) + 1 ))
    call string_f2c(fstring, cstring, string_size)
  end subroutine astring_f2c

  function int_to_str_int32(n) result(string)
  !! Convert 32-bit integer to string
    integer(int32),   intent(in)  :: n            !! Integer to convert
    character(len=:), allocatable :: string       !! Resulting string
    character(len=11)             :: temp         !! Temporary string

    write(temp, '(I11)') n
    allocate( string, source= trim(adjustl(temp)))
  end function int_to_str_int32

  function int_to_str_int8(n) result(string)
  !! Convert 8-bit integer to string
    integer(int8),    intent(in)  :: n            !! Integer to convert
    character(len=:), allocatable :: string       !! Resulting string
    character(len=3)              :: temp         !! Temporary string

    write(temp, '(I3)') n
    allocate( string, source= trim(adjustl(temp)))
  end function int_to_str_int8

  function double_to_str(n) result(string)
  !! Convert double to string
    real(real64),     intent(in)  :: n            !! Double to convert
    character(len=:), allocatable :: string       !! Resulting string
    character(len=23)             :: temp         !! Temporary string

    write(temp, '(F15.5)') n
    allocate( string, source= trim(adjustl(temp)))
  end function double_to_str

  subroutine write_message(unit, message, prefix)
  !! Write message to the specified unit
    integer(int32),   intent(in)            :: unit         !! Unit number
    character(len=*), intent(in)            :: message      !! Message to write
    character(len=*), intent(in), optional  :: prefix       !! Prefix to the message
    character(len=:), allocatable           :: prefix_      !! Dummy prefix
    integer(int32)                          :: comm_rank    !! Size of world communicator
    integer(int32)                          :: ierr         !! Error code
    logical                                 :: is_finalized !! Is MPI Already finalized?

    call MPI_Finalized(is_finalized, ierr)
    if ( is_finalized ) then
      comm_rank = 0
    else
      call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    endif
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

  elemental function get_inverse_kind(r2r_kind) result(result_kind)
  !! Get the inverse R2R kind of transform for the given R2R kind
    type(dtfft_r2r_kind_t), intent(in)  :: r2r_kind        !! R2R kind
    type(dtfft_r2r_kind_t)              :: result_kind

    result_kind = dtfft_r2r_kind_t(-1)
    select case (r2r_kind%val)
    case ( DTFFT_DCT_1%val )
      result_kind = DTFFT_DCT_1
    case ( DTFFT_DCT_2%val )
      result_kind = DTFFT_DCT_3
    case ( DTFFT_DCT_3%val )
      result_kind = DTFFT_DCT_2
    case ( DTFFT_DCT_4%val )
      result_kind = DTFFT_DCT_4
    case ( DTFFT_DST_1%val )
      result_kind = DTFFT_DST_1
    case ( DTFFT_DST_2%val )
      result_kind = DTFFT_DST_3
    case ( DTFFT_DST_3%val )
      result_kind = DTFFT_DST_2
    case ( DTFFT_DST_4%val )
      result_kind = DTFFT_DST_4
    endselect
  end function get_inverse_kind

  logical function is_null_ptr(ptr)
    type(c_ptr),  intent(in) :: ptr

    is_null_ptr = is_same_ptr(ptr, c_null_ptr)
  end function is_null_ptr

  logical function is_same_ptr(ptr1, ptr2)
    type(c_ptr),  intent(in):: ptr1   !! First pointer
    type(c_ptr),  intent(in):: ptr2   !! Second pointer

    is_same_ptr = c_associated(ptr1, ptr2)
  end function is_same_ptr

#ifdef DTFFT_WITH_CUDA
  integer(int32) function count_unique(x) result(n)
  !! Count the number of unique elements in the array
    integer(int32), intent(in)  :: x(:)   !! Array of integers
    integer(int32), allocatable :: y(:)   !! Array of unique integers

    allocate(y, source=x)
    n = 0
    do while (size(y) > 0)
        n = n + 1
        y = pack(y,mask=(y(:) /= y(1))) ! drops all elements that are 
                                        ! equals to the 1st one (included)
    end do
    deallocate(y)
  end function count_unique
#endif
#ifdef DTFFT_WITH_NVSHMEM
  function is_nvshmem_ptr(ptr) result(bool)
  !! Checks if pointer is a symmetric nvshmem allocated pointer
    type(c_ptr)   :: ptr    !! Device pointer
    logical       :: bool   !! Result

    bool = is_null_ptr( nvshmem_ptr(ptr, nvshmem_my_pe()) )
  end function is_nvshmem_ptr
#endif
end module dtfft_utils