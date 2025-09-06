!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
use iso_fortran_env
use dtfft_errors
use dtfft_parameters
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: string_f2c, string_c2f
public :: to_str
public :: write_message

public :: get_inverse_kind
public :: is_same_ptr, is_null_ptr
public :: mem_alloc_host, mem_free_host
public :: string

#ifdef DTFFT_WITH_CUDA
public :: destroy_strings
public :: astring_f2c
public :: count_unique
public :: Comm_f2c
public :: is_device_ptr
public :: dynamic_load
#endif

  interface to_str
  !! Convert various types to string
    module procedure int8_to_string
    module procedure int32_to_string
    module procedure int64_to_string
    module procedure double_to_string
    module procedure float_to_string
  end interface to_str

  interface is_null_ptr
  !! Checks if pointer is NULL
    module procedure is_null_ptr
#ifdef DTFFT_WITH_CUDA
    module procedure is_null_funptr
#endif
  end interface is_null_ptr

#if defined(DTFFT_USE_MPI)
public :: all_reduce_inplace
  interface all_reduce_inplace
  !! MPI Allreduce inplace workaround
    module procedure :: all_reduce_inplace_i32
    module procedure :: all_reduce_inplace_i64
  end interface all_reduce_inplace
#endif

#ifdef DTFFT_WITH_CUDA
  integer(c_int), parameter :: RTLD_LAZY = 1_c_int
    !! Each external function reference is bound the first time the function is called.
  integer(c_int), parameter :: RTLD_NOW  = 2_c_int
    !! All external function references are bound when the library is loaded.

  interface
    function dlopen(filename, mode) bind(C)
    !! Load and link a dynamic library
    import
      type(c_ptr)           :: dlopen       !! Handle to the library
      character(c_char)     :: filename(*)  !! Name of the library
      integer(c_int), value :: mode         !! Options to dlopen
    end function dlopen
  end interface

  interface
    function dlsym(handle, name) bind(C)
    !! Get address of a symbol from a dynamic library
    import
      type(c_funptr)      :: dlsym          !! Address of the symbol
      type(c_ptr),  value :: handle         !! Handle to the library
      character(c_char)   :: name(*)        !! Name of the symbol
    end function dlsym
  end interface

  interface
    function dlclose(handle) bind(C)
    !! Close a dynamic library
    import
      integer(c_int)      :: dlclose        !! Result of the operation
      type(c_ptr), value  :: handle         !! Handle to the library
    end function dlclose
  end interface

  interface
    function dlerror() bind(C)
    !! Get diagnostic information
    import
      type(c_ptr)  :: dlerror !! Error message
    end function dlerror
  end interface
#endif

  interface
    function aligned_alloc(alignment, alloc_size) result(ptr) bind(C)
    !! Allocates memory using C11 Standard alloc_align with 16 bytes alignment
    import
      integer(c_size_t),  value :: alignment    !! Alignment in bytes (16 bytes by default)
      integer(c_size_t),  value :: alloc_size   !! Number of bytes to allocate
      type(c_ptr)               :: ptr          !! Pointer to allocate
    end function aligned_alloc
  end interface

  interface
    subroutine mem_free_host(ptr) bind(C, name="free")
    !! Frees memory allocated with [[aligned_alloc]]
    import
      type(c_ptr),        value :: ptr          !! Pointer to free
    end subroutine mem_free_host
  end interface

#ifdef DTFFT_WITH_CUDA
  interface
    type(c_ptr) function Comm_f2c(fcomm) bind(C, name="Comm_f2c")
    !! Converts Fortran communicator to C
      import
      integer(c_int), value :: fcomm            !! Fortran communicator
    end function Comm_f2c
  end interface

  interface
    function is_device_ptr(ptr) result(bool) bind(C)
    !! Checks if pointer can be accessed from device
    import
      type(c_ptr),    value :: ptr    !! Device pointer
      logical(c_bool)       :: bool   !! Result
    end function is_device_ptr
  end interface
#endif

  type :: string
  !! Class used to create array of strings
    character(len=:), allocatable :: raw  !! String
  contains
    procedure, pass(self) :: destroy => destroy_string
  end type string

  interface string
  !! Creates [[string]] object
    module procedure :: string_constructor
  end interface string

contains

  type(string) function string_constructor(str)
  !! Creates [[string]] object
    character(len=*), intent(in)  :: str  !! String
    allocate( string_constructor%raw, source=str )
  end function string_constructor

  subroutine destroy_string(self)
    class(string),  intent(inout) :: self
    if ( allocated(self%raw) ) deallocate( self%raw )
  end subroutine destroy_string

#ifdef DTFFT_WITH_CUDA
  subroutine destroy_strings(strings)
  !! Destroys array of [[string]] objects
    type(string), intent(inout), allocatable :: strings(:)  !! Array of strings
    integer(int32) :: i

    if ( .not. allocated(strings) ) return
    do i = 1, size(strings)
      call strings(i)%destroy()
    end do
    deallocate( strings )
  end subroutine destroy_strings
#endif

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

  subroutine string_c2f(cstring, fstring)
  !! Convert C string to Fortran string
    type(c_ptr)                     :: cstring    !! C string
    character(len=:),   allocatable :: fstring    !! Fortran string
    character(len=1024), pointer     :: fstring_  !! Temporary Fortran string

    call c_f_pointer(cstring, fstring_)
    allocate( fstring, source=fstring_(1:max(len(fstring_), index(fstring_, c_null_char)) - 1) )
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

    if ( is_null_ptr(handle) ) INTERNAL_ERROR("is_null_ptr(handle)")

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
  !! Dynamically loads library and its symbols
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
        call unload_library(handle)
        symbols(1:i) = c_null_funptr
        error_code = DTFFT_ERROR_DLSYM_FAILED
        return
      endif
    end do
  end function dynamic_load
#endif

  function int8_to_string(n) result(str)
  !! Convert 8-bit integer to string
    integer(int8),    intent(in)  :: n            !! Integer to convert
    character(len=:), allocatable :: str          !! Resulting string
    character(len=3)              :: temp         !! Temporary string

    write(temp, '(I3)') n
    allocate( str, source=trim(adjustl(temp)) )
  end function int8_to_string

  function int32_to_string(n) result(str)
  !! Convert 32-bit integer to string
    integer(int32),   intent(in)  :: n            !! Integer to convert
    character(len=:), allocatable :: str          !! Resulting string
    character(len=11)             :: temp         !! Temporary string

    write(temp, '(I11)') n
    allocate( str, source=trim(adjustl(temp)) )
  end function int32_to_string

  function int64_to_string(n) result(str)
  !! Convert 64-bit integer to string
    integer(int64),   intent(in)  :: n            !! Integer to convert
    character(len=:), allocatable :: str          !! Resulting string
    character(len=20)             :: temp         !! Temporary string

    write(temp, '(I20)') n
    allocate( str, source=trim(adjustl(temp)) )
  end function int64_to_string

  function double_to_string(n) result(str)
  !! Convert double to string
    real(real64),     intent(in)  :: n            !! Double to convert
    character(len=:), allocatable :: str          !! Resulting string
    character(len=23)             :: temp         !! Temporary string

    write(temp, '(F15.5)') n
    allocate( str, source=trim(adjustl(temp)) )
  end function double_to_string

  function float_to_string(n) result(str)
  !! Convert double to string
    real(real32),     intent(in)  :: n            !! Double to convert
    character(len=:), allocatable :: str          !! Resulting string
    character(len=15)             :: temp         !! Temporary string

    write(temp, '(F10.2)') n
    allocate( str, source=trim(adjustl(temp)) )
  end function float_to_string

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

  function mem_alloc_host(alloc_size) result(ptr)
  !! Allocates memory using C11 Standard alloc_align with 16 bytes alignment
    integer(int64), intent(in)  :: alloc_size   !! Number of bytes to allocate
    type(c_ptr)                 :: ptr          !! Pointer to allocate
    integer(int64) :: displ, alloc_size_

    displ = mod(alloc_size, int(ALLOC_ALIGNMENT, int64))
    alloc_size_ = alloc_size
    if ( displ > 0 ) then
      alloc_size_ = alloc_size_ + (ALLOC_ALIGNMENT - displ)
    endif
    ptr = aligned_alloc(int(ALLOC_ALIGNMENT, c_size_t), alloc_size_)
  end function mem_alloc_host

#if defined(DTFFT_USE_MPI)
! Some bug was noticed in mpich for macos
! For some reason MPI_IN_PLACE has not been recognized.
! This is some stupid workaround
  subroutine all_reduce_inplace_i64(buffer, op, comm)
    integer(int64), intent(inout) :: buffer
    integer(int32), intent(in)    :: op, comm
    integer(int64) :: tmp
    integer(int32) :: ierr

    call MPI_Allreduce(buffer, tmp, 1, MPI_INTEGER8, op, comm, ierr)
    buffer = tmp
  end subroutine all_reduce_inplace_i64

  subroutine all_reduce_inplace_i32(buffer, op, comm)
    integer(int32), intent(inout) :: buffer
    integer(int32), intent(in)    :: op, comm
    integer(int32) :: tmp
    integer(int32) :: ierr

    call MPI_Allreduce(buffer, tmp, 1, MPI_INTEGER4, op, comm, ierr)
    buffer = tmp
  end subroutine all_reduce_inplace_i32
#endif
end module dtfft_utils