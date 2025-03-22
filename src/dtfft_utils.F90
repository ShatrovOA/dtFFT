!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
#include "dtfft_config.h"
module dtfft_utils
!! All Utilities functions are located here
use iso_c_binding
use iso_fortran_env,  only: int8, int32, int64, real64, output_unit, error_unit
use dtfft_parameters
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: string_f2c, string_c2f
public :: int_to_str, double_to_str
public :: write_message, init_internal, get_log_enabled
public :: get_env, get_iters_from_env, get_datatype_from_env
public :: get_inverse_kind
public :: get_platform_from_env, get_z_slab_from_env

public :: is_same_ptr, is_null_ptr
public :: mem_alloc_host, mem_free_host
#ifdef DTFFT_WITH_CUDA
public :: destroy_strings
public :: astring_f2c
public :: count_unique
public :: Comm_f2c
public :: is_device_ptr
public :: get_gpu_backend_from_env
public :: get_mpi_enabled_from_env, get_nccl_enabled_from_env, get_nvshmem_enabled_from_env, get_pipe_enabled_from_env
public :: load_library, load_symbol, unload_library, dynamic_load
#endif

  logical,                    save  :: is_init_called = .false.
    !! Has [[init_internal]] already been called or not
  logical,                    save  :: is_log_enabled
    !! Should we log messages to stdout or not
  type(dtfft_platform_t),     save  :: platform_from_env = PLATFORM_NOT_SET
    !! Platform obtained from environ
  integer(int32),             save  :: z_slab_from_env
    !! Should Z-slab be used if possible
#ifdef DTFFT_WITH_CUDA
  type(dtfft_gpu_backend_t),  save  :: gpu_backend_from_env
    !! Backend obtained from environ
  integer(int32),             save  :: mpi_enabled_from_env
    !! Should we use MPI backends during autotune or not
  integer(int32),             save  :: nccl_enabled_from_env
    !! Should we use NCCL backends during autotune or not
  integer(int32),             save  :: nvshmem_enabled_from_env
    !! Should we use NVSHMEM backends during autotune or not
  integer(int32),             save  :: pipe_enabled_from_env
    !! Should we use pipelined backends during autotune or not
#endif
  character(len=26), parameter :: UPPER_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    !! Upper case alphabet.
  character(len=26), parameter :: LOWER_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
    !! Lower case alphabet.

  interface int_to_str
  !! Converts integer to string
    module procedure int_to_str_int8
    module procedure int_to_str_int32
  end interface int_to_str

  interface is_null_ptr
  !! Checks if pointer is NULL
    module procedure is_null_ptr
#ifdef DTFFT_WITH_CUDA
    module procedure is_null_funptr
#endif
  end interface is_null_ptr

  interface get_env
  !! Obtains environment variable
    module procedure :: get_env_base
#ifdef DTFFT_WITH_CUDA
    module procedure :: get_env_string
#endif
    module procedure :: get_env_int32
    module procedure :: get_env_int8
    module procedure :: get_env_logical
  end interface get_env

#ifdef DTFFT_WITH_CUDA
public :: string
  type :: string
  !! Class used to create array of strings
    character(len=:), allocatable :: raw  !! String
  end type string

  interface string
  !! Creates [[string]] object
    module procedure :: string_constructor
  end interface string

  integer(c_int), parameter :: RTLD_LAZY = 1_c_int
    !! Options to dlopen

public :: dlopen, dlsym, dlclose
  interface dlopen
  !! Load and link a dynamic library or bundle
    function dlopen(filename, mode) bind(C)
    import
      type(c_ptr)           :: dlopen       !! Handle to the library
      character(c_char)     :: filename(*)  !! Name of the library
      integer(c_int), value :: mode         !! options to dlopen
    end function dlopen
  end interface

  interface dlsym
  !! Get address of a symbol
    function dlsym(handle, name) bind(C)
    import
      type(c_funptr)      :: dlsym          !! Address of the symbol
      type(c_ptr),  value :: handle         !! Handle to the library
      character(c_char)   :: name(*)        !! Name of the symbol
    end function dlsym
  end interface

  interface dlclose
  !! Close a dynamic library or bundle
    function dlclose(handle) bind(C)
    import
      integer(c_int)      :: dlclose        !! Result of the operation
      type(c_ptr), value  :: handle         !! Handle to the library
    end function dlclose
  end interface

  interface dlerror
  !! Get diagnostic information
    function dlerror() bind(C)
    import
      type(c_ptr)  :: dlerror !! Error message
    end function dlerror
  end interface
#endif

  interface mem_alloc_host
  !! Allocates memory using C11 Standard alloc_align with 16 bytes alignment
    subroutine mem_alloc_host(alloc_size, ptr) bind(C)
    import
      integer(c_size_t),  value :: alloc_size   !! Number of bytes to allocate
      type(c_ptr)               :: ptr          !! Pointer to allocate
    end subroutine mem_alloc_host
  end interface

  interface mem_free_host
  !! Frees memory allocated with [[mem_alloc_host]]
    subroutine mem_free_host(ptr) bind(C)
    import
      type(c_ptr),        value :: ptr          !! Pointer to free
    end subroutine mem_free_host
  end interface

#ifdef DTFFT_WITH_CUDA
  interface Comm_f2c
  !! Converts Fortran communicator to C
    type(c_ptr) function Comm_f2c(fcomm) bind(C, name="Comm_f2c")
      import
      integer(c_int), value :: fcomm            !! Fortran communicator
    end function Comm_f2c
  end interface

  interface is_device_ptr
  !! Checks if pointer can be accessed from device
    function is_device_ptr(ptr) result(bool) bind(C)
    import
      type(c_ptr),    value :: ptr    !! Device pointer
      logical(c_bool)       :: bool   !! Result
    end function is_device_ptr
  end interface
#endif

contains

#ifdef DTFFT_WITH_CUDA
  type(string) function string_constructor(str)
  !! Creates [[string]] object
    character(len=*), intent(in)  :: str  !! String
    allocate( string_constructor%raw, source=str )
  end function string_constructor

  subroutine destroy_strings(strings)
  !! Destroys array of [[string]] objects
    type(string), intent(inout), allocatable :: strings(:)  !! Array of strings
    integer(int32) :: i

    if ( .not. allocated(strings) ) return
    do i = 1, size(strings)
      if ( allocated(strings(i)%raw) ) deallocate( strings(i)%raw )
    end do
    deallocate( strings )
  end subroutine destroy_strings
#endif

  integer(int32) function init_internal()
  !! Checks if MPI is initialized and loads environment variables
    integer(int32)    :: ierr             !! Error code
    logical           :: is_mpi_init      !! Is MPI initialized?

    init_internal = DTFFT_SUCCESS

    call MPI_Initialized(is_mpi_init, ierr)
    if( .not. is_mpi_init ) then
      init_internal = DTFFT_ERROR_MPI_FINALIZED
      return
    endif
    ! Processing environment variables once
    if ( is_init_called ) return

    is_log_enabled = get_env("ENABLE_LOG", .false.)
    z_slab_from_env = get_env("ENABLE_Z_SLAB", VARIABLE_NOT_SET, valid_values=[0, 1])

#ifdef DTFFT_WITH_CUDA
    block
      type(string), allocatable :: platforms(:)
      character(len=:), allocatable :: pltfrm_env

      allocate( platforms(2) )
      platforms(1) = string("host")
      platforms(2) = string("cuda")

      allocate( pltfrm_env, source=get_env("PLATFORM", "undefined", platforms) )
      if ( pltfrm_env == "undefined") then
        platform_from_env = PLATFORM_NOT_SET
      else if ( pltfrm_env == "host" ) then
        platform_from_env = DTFFT_PLATFORM_HOST
      else if ( pltfrm_env == "cuda") then
        platform_from_env = DTFFT_PLATFORM_CUDA
      endif

      deallocate( platforms(1)%raw, platforms(2)%raw, pltfrm_env )
      deallocate( platforms )
    endblock

    block
      type(string), allocatable :: backends(:)
      character(len=:), allocatable :: bcknd_env
      integer(int32) :: i

      allocate( backends(7) )
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
      deallocate( backends )
    endblock

    mpi_enabled_from_env = get_env("ENABLE_MPI", VARIABLE_NOT_SET, valid_values=[0, 1])
    nccl_enabled_from_env = get_env("ENABLE_NCCL", VARIABLE_NOT_SET, valid_values=[0, 1])
    nvshmem_enabled_from_env = get_env("ENABLE_NVSHMEM", VARIABLE_NOT_SET, valid_values=[0, 1])
    pipe_enabled_from_env = get_env("ENABLE_PIPE", VARIABLE_NOT_SET, valid_values=[0, 1])
#endif
    is_init_called = .true.
  end function init_internal

  pure type(dtfft_platform_t) function get_platform_from_env()
  !! Returns execution platform set by environment variable
    get_platform_from_env = platform_from_env
  end function get_platform_from_env

  pure integer(int32) function get_z_slab_from_env()
  !! Returns Z-slab to be used set by environment variable
    get_z_slab_from_env = z_slab_from_env
  end function get_z_slab_from_env

#ifdef DTFFT_WITH_CUDA
  pure type(dtfft_gpu_backend_t) function get_gpu_backend_from_env()
  !! Returns GPU backend to use set by environment variable
    get_gpu_backend_from_env = gpu_backend_from_env
  end function get_gpu_backend_from_env

  pure integer(int32) function get_mpi_enabled_from_env()
  !! Returns usage of MPI Backends during autotune set by environment variable
    get_mpi_enabled_from_env = mpi_enabled_from_env
  end function get_mpi_enabled_from_env

  pure integer(int32) function get_nccl_enabled_from_env()
  !! Returns usage of NCCL Backends during autotune set by environment variable
    get_nccl_enabled_from_env = nccl_enabled_from_env
  end function get_nccl_enabled_from_env

  pure integer(int32) function get_nvshmem_enabled_from_env()
  !! Returns usage of NVSHMEM Backends during autotune set by environment variable
    get_nvshmem_enabled_from_env = nvshmem_enabled_from_env
  end function get_nvshmem_enabled_from_env

  pure integer(int32) function get_pipe_enabled_from_env()
  !! Returns usage of Pipelined Backends during autotune set by environment variable
    get_pipe_enabled_from_env = pipe_enabled_from_env
  end function get_pipe_enabled_from_env
#endif

  function get_env_base(name) result(env)
  !! Base function of obtaining dtFFT environment variable
    character(len=*), intent(in)    :: name         !! Name of environment variable without prefix
    character(len=:), allocatable   :: full_name    !! Prefixed environment variable name
    character(len=:), allocatable   :: env          !! Environment variable value
    integer(int32)                  :: env_val_len  !! Length of the environment variable

    allocate( full_name, source="DTFFT_"//name )

    call get_environment_variable(full_name, length=env_val_len)
    allocate(character(env_val_len) :: env)
    if ( env_val_len == 0 ) then
      deallocate(full_name)
      return
    endif
    call get_environment_variable(full_name, env)
    deallocate(full_name)
  end function get_env_base

#ifdef DTFFT_WITH_CUDA
  function get_env_string(name, default, valid_values) result(env)
  !! Obtains string environment variable
    character(len=*), intent(in)            :: name                 !! Name of environment variable without prefix
    character(len=*), intent(in)            :: default              !! Name of environment variable without prefix
    type(string),     intent(in)            :: valid_values(:)      !! List of valid variable values
    character(len=:), allocatable           :: env                  !! Environment variable value
    character(len=:), allocatable           :: env_val_str          !! String value of the environment variable
    logical                                 :: is_correct           !! Is env value is correct
    integer(int32) :: i, j

    allocate( env_val_str, source=get_env(name) )
    if ( len(env_val_str) == 0 ) then
      deallocate(env_val_str)
      allocate(env, source=default)
      return
    endif

    ! Converting to lowercase
    do i=1, len(env_val_str)
      j = index(UPPER_ALPHABET, env_val_str(i:i))
      if (j>0) env_val_str(i:i) = LOWER_ALPHABET(j:j)
    enddo

    is_correct = any([(env_val_str == valid_values(i)%raw, i=1,size(valid_values))])

    if ( is_correct ) then
      allocate( env, source=env_val_str )
      deallocate(env_val_str)
      return
    endif
    WRITE_ERROR("Invalid environment variable: `DTFFT_"//name//"`, it has been ignored")
    allocate(env, source=default)
    deallocate(env_val_str)
  end function get_env_string
#endif

  integer(int32) function get_env_int32(name, default, valid_values, min_valid_value) result(env)
  !! Base Integer function of obtaining dtFFT environment variable
    character(len=*), intent(in)            :: name               !! Name of environment variable without prefix
    integer(int32),   intent(in)            :: default            !! Default value in case env is not set or it has wrong value
    integer(int32),   intent(in), optional  :: valid_values(:)    !! List of valid values
    integer(int32),   intent(in), optional  :: min_valid_value    !! Mininum valid value. Usually 0 or 1
    character(len=:), allocatable           :: env_val_str        !! String value of the environment variable
    logical                                 :: is_correct         !! Is env value is correct
    integer(int32)                          :: env_val_passed     !! Value of the environment variable

    if ( ( present(valid_values).and.present(min_valid_value) )           &
      .or.(.not.present(valid_values).and..not.present(min_valid_value))  &
    ) then
      error stop "dtFFT Internal error `get_env_int32`"
    endif

    allocate( env_val_str, source=get_env(name) )

    if ( len(env_val_str) == 0 ) then
      deallocate(env_val_str)
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
      deallocate(env_val_str)
      return
    endif
    WRITE_ERROR("Invalid environment variable: `DTFFT_"//name//"`, it has been ignored")
    env = default
    deallocate(env_val_str)
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

  logical function get_env_logical(name, default) result(env)
  !! Obtains logical environment variable
    character(len=*), intent(in) :: name                !! Name of environment variable without prefix
    logical,          intent(in) :: default             !! Default value in case env is not set or it has wrong value
    integer(int32) :: def, val

    if ( default ) then
      def = 1
    else
      def = 0
    endif

    val = get_env(name, def, [0, 1])
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

  subroutine string_c2f(cstring, string)
  !! Convert C string to Fortran string
    type(c_ptr)                     :: cstring  !! C string
    character(len=:),   allocatable :: string   !! Fortran string
    character(len=256), pointer     :: fstring  !! Temporary Fortran string

    call c_f_pointer(cstring, fstring)
    allocate( string, source=fstring(1:index(fstring, c_null_char) - 1) )
  end subroutine string_c2f

#ifdef DTFFT_WITH_CUDA
  subroutine astring_f2c(fstring, cstring, string_size)
  !! Convert Fortran string to C allocatable string
    character(len=*),                     intent(in)  :: fstring      !! Fortran string
    character(kind=c_char), allocatable,  intent(out) :: cstring(:)   !! C string
    integer(int64),         optional,     intent(out) :: string_size  !! Size of the C string

    allocate(cstring( len_trim(fstring) + 1 ))
    call string_f2c(fstring, cstring, string_size)
  end subroutine astring_f2c

  subroutine dl_error(message)
  !! Writes error message to the error unit
    character(len=*), intent(in)  :: message      !! Message to write
    character(len=:), allocatable :: err_msg      !! Error string

    call string_c2f(dlerror(), err_msg)
    WRITE_ERROR(message//": "//err_msg)
    deallocate( err_msg )
  end subroutine dl_error

  function load_library(name) result(lib_handle)
  !! Dynamically loads library
    character(len=*), intent(in)  :: name         !! Name of library to load
    type(c_ptr)                   :: lib_handle   !! Loaded handle
    character(c_char),  allocatable :: cname(:)   !! Temporary string

    WRITE_DEBUG("Loading library: "//name)
    call astring_f2c(name//c_null_char, cname)
    lib_handle = dlopen(cname, RTLD_LAZY)
    deallocate( cname )
    if (is_null_ptr(lib_handle)) then
      call dl_error("Failed to load library '"//name//"'")
    endif
  end function load_library

  function load_symbol(handle, name) result(symbol_handle)
  !! Dynamically loads symbol from library
    type(c_ptr),      intent(in)  :: handle         !! Loaded handle
    character(len=*), intent(in)  :: name           !! Name of function to load
    type(c_funptr)                :: symbol_handle  !! Function pointer
    character(c_char),  allocatable :: cname(:)     !! Temporary string

    if ( is_null_ptr(handle) ) error stop "dtFFT Internal error: is_null_ptr(handle)"
    WRITE_DEBUG("Loading symbol: "//name)
    call astring_f2c(name//c_null_char, cname)
    symbol_handle = dlsym(handle, cname)
    deallocate(cname)
    if (is_null_ptr(symbol_handle)) then
      call dl_error("Failed to load symbol '"//name//"'")
    endif
  end function load_symbol

  subroutine unload_library(handle)
  !! Unloads library
    type(c_ptr),      intent(in)  :: handle         !! Loaded handle
    integer(int32)  :: ierr                         !! Error code

    ierr = dlclose(handle)
    if ( ierr /= 0 ) then
      call dl_error("Failed to unload library")
    endif
  end subroutine unload_library

  function dynamic_load(name, symbol_names, handle, symbols) result(error_code)
    character(len=*), intent(in)  :: name             !! Name of library to load
    type(string),     intent(in)  :: symbol_names(:)  !! Names of functions to load
    type(c_ptr),      intent(out) :: handle           !! Loaded handle
    type(c_funptr),   intent(out) :: symbols(:)       !! Function pointers
    integer(int32)                :: error_code       !! Error code
    integer(int32)                :: i                !! Loop index

    error_code = DTFFT_SUCCESS

    handle = load_library(name)
    if ( is_null_ptr(handle) ) then
      error_code = DTFFT_ERROR_DLOPEN_FAILED
      return
    endif

    do i = 1, size(symbol_names)
      symbols(i) = load_symbol(handle, symbol_names(i)%raw)
      if ( is_null_ptr(symbols(i)) ) then
        error_code = DTFFT_ERROR_DLSYM_FAILED
        return
      endif
    end do
  end function dynamic_load
#endif

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

  elemental logical function is_null_ptr(ptr)
  !! Checks if pointer is NULL
    type(c_ptr),  intent(in) :: ptr   !! Pointer to check
    is_null_ptr = .not.c_associated(ptr)
  end function is_null_ptr

#ifdef DTFFT_WITH_CUDA
  elemental logical function is_null_funptr(ptr)
  !! Checks if pointer is NULL
    type(c_funptr),  intent(in) :: ptr   !! Pointer to check
    is_null_funptr = .not.c_associated(ptr)
  end function is_null_funptr
#endif

  elemental logical function is_same_ptr(ptr1, ptr2)
  !! Checks if two pointer are the same
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
end module dtfft_utils