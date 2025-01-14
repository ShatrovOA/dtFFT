module dtfft_interface_cufft_native_m
use iso_c_binding, only: c_int32_t
use cufft
implicit none
private
public :: CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
public :: CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z
public :: CUFFT_SUCCESS
public :: cufftGetErrorString

contains

  function cufftGetErrorString(error_code) result(string)
  !! Returns a string representation of the cuFFT error code.
    integer(c_int32_t), intent(in)  :: error_code   !< cuFFT error code
    character(len=:),   allocatable :: string       !< String representation of the cuFFT error code

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
end module dtfft_interface_cufft_native_m