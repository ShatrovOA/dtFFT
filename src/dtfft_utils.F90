module dtfft_utils
use iso_c_binding,    only: c_char, c_null_char
use iso_fortran_env,  only: output_unit, error_unit
use dtfft_parameters
use dtfft_precisions
#ifdef DTFFT_WITH_CUDA
use cufft
use cudafor, only: cuda_stream_kind, cudaStreamCreate
#endif
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
implicit none
private
public :: dtfft_string_f2c, dtfft_astring_f2c
public :: int_to_str, double_to_str
public :: write_debug, dtfft_init
public :: suppress_unused
public :: get_inverse_kind
#ifdef DTFFT_WITH_CUDA
public :: cufftGetErrorString
public :: CUFFT_SUCCESS
public :: dtfft_get_stream, dtfft_set_stream
#endif

  logical, save :: is_init_called = .false.


#ifdef DTFFT_WITH_CUDA
  integer(cuda_stream_kind),  save :: dtfft_stream
  logical,                    save :: is_stream_created = .false.
#endif

contains

  integer(IP) function dtfft_init()
    integer(IP) :: ierr
    logical :: is_mpi_init

    dtfft_init = DTFFT_SUCCESS

    call MPI_Initialized(is_mpi_init, ierr)
    if( .not. is_mpi_init ) then
      dtfft_init = DTFFT_ERROR_MPI_FINALIZED
      return
    endif
    is_init_called = .true.
  end function dtfft_init

  subroutine dtfft_string_f2c(fstring, cstring)
    character(len=*),       intent(in)  :: fstring
    character(kind=c_char), intent(inout) :: cstring(*)
    integer :: i, j
    logical :: met_non_blank

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
  end subroutine dtfft_string_f2c

  subroutine dtfft_astring_f2c(fstring, cstring)
    character(len=*),                     intent(in)  :: fstring
    character(kind=c_char), allocatable,  intent(out) :: cstring(:)

    allocate(cstring( len_trim(fstring) + 1 ))
    call dtfft_string_f2c(fstring, cstring)
  end subroutine dtfft_astring_f2c

  function int_to_str(n) result(string)
    integer(IP), intent(in)  :: n
    character(len=:), allocatable :: string
    character(len=11) :: temp

    write(temp, '(I11)') n
    allocate( string, source= trim(adjustl(temp)))
  end function int_to_str

  function double_to_str(n) result(string)
    real(R8P), intent(in)  :: n
    character(len=:), allocatable :: string
    character(len=23) :: temp

    write(temp, '(E23.15E3)') n
    allocate( string, source= trim(adjustl(temp)))
  end function double_to_str

  ! Suppress warnings from linter
  subroutine suppress_unused(x)
    type(*)   :: x(..)
    integer   :: i_size(1)
    i_size = shape(x)
  end subroutine suppress_unused

  subroutine write_debug(msg)
    character(len=*), intent(in)  :: msg
    integer(IP) :: comm_rank, ierr

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    if ( comm_rank == 0 ) write(output_unit, '(a)') 'DTFFT Debug: '//trim(msg)
  end subroutine write_debug

  elemental integer(IP) function get_inverse_kind(r2r_kind)
    integer(IP), intent(in)  :: r2r_kind

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
  integer(cuda_stream_kind) function dtfft_get_stream()
    integer :: ierr
    if (.not.is_stream_created) then
      CUFFT_CALL( "cudaStreamCreate", cudaStreamCreate(dtfft_stream) )
      is_stream_created = .true.
    endif
    dtfft_get_stream = dtfft_stream
  end function dtfft_get_stream

  subroutine dtfft_set_stream(stream)
    integer(cuda_stream_kind),  intent(in) :: stream

    dtfft_stream = stream
    is_stream_created = .true.
  end subroutine dtfft_set_stream

  function cufftGetErrorString(error_code) result(string)
    integer(IP),      intent(in) :: error_code
    character(len=:),allocatable :: string

    select case (error_code)
    case ( CUFFT_SUCCESS )
      allocate( string, source="CUFFT_SUCCESS" )
    case ( CUFFT_INVALID_PLAN )
      allocate( string, source="CUFFT_INVALID_PLAN" )
    case ( CUFFT_ALLOC_FAILED )
      allocate( string, source="CUFFT_ALLOC_FAILED" )
    case ( CUFFT_INVALID_TYPE )
      allocate( string, source="CUFFT_INVALID_TYPE" )
    case ( CUFFT_INVALID_VALUE )
      allocate( string, source="CUFFT_INVALID_VALUE" )
    case ( CUFFT_INTERNAL_ERROR )
      allocate( string, source="CUFFT_INTERNAL_ERROR" )
    case ( CUFFT_EXEC_FAILED )
      allocate( string, source="CUFFT_EXEC_FAILED" )
    case ( CUFFT_SETUP_FAILED )
      allocate( string, source="CUFFT_SETUP_FAILED" )
    case ( CUFFT_INVALID_SIZE )
      allocate( string, source="CUFFT_INVALID_SIZE" )
    case ( CUFFT_UNALIGNED_DATA )
      allocate( string, source="CUFFT_UNALIGNED_DATA" )
    case default
      allocate( string, source="CUFFT_UNKNOWN_ERROR" )
    endselect
  end function cufftGetErrorString
#endif
end module dtfft_utils