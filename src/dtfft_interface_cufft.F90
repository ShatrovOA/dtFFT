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
module dtfft_interface_cufft
use iso_c_binding,  only: c_int32_t, c_int, c_ptr, c_long_long, c_size_t
use dtfft_parameters, only: dtfft_stream_t
implicit none
private
public :: CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
public :: CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z
public :: cufftPlanMany, cufftXtExec, cufftDestroy, cufftSetStream
public :: cufftGetErrorString
public :: cufftReshapeHandle
public :: cufftMpCreateReshape, cufftMpAttachReshapeComm, cufftMpGetReshapeSize
public :: cufftMpMakeReshape, cufftMpExecReshapeAsync, cufftMpDestroyReshape


  integer(c_int), parameter,  public :: CUFFT_COMM_MPI = 0

  enum, bind(C)
    enumerator :: CUFFT_R2C = 42
    enumerator :: CUFFT_C2R = 44
    enumerator :: CUFFT_C2C = 41
    enumerator :: CUFFT_D2Z = 106
    enumerator :: CUFFT_Z2D = 108
    enumerator :: CUFFT_Z2Z = 105
  end enum

  enum, bind(C)
    enumerator :: CUFFT_SUCCESS = 0
    enumerator :: CUFFT_INVALID_PLAN = 1
    enumerator :: CUFFT_ALLOC_FAILED = 2
    enumerator :: CUFFT_INVALID_TYPE = 3
    enumerator :: CUFFT_INVALID_VALUE = 4
    enumerator :: CUFFT_INTERNAL_ERROR = 5
    enumerator :: CUFFT_EXEC_FAILED = 6
    enumerator :: CUFFT_SETUP_FAILED = 7
    enumerator :: CUFFT_INVALID_SIZE = 8
    enumerator :: CUFFT_UNALIGNED_DATA = 9
    enumerator :: CUFFT_INCOMPLETE_PARAMETER_LIST = 10
    enumerator :: CUFFT_INVALID_DEVICE = 11
    enumerator :: CUFFT_PARSE_ERROR = 12
    enumerator :: CUFFT_NO_WORKSPACE = 13
    enumerator :: CUFFT_NOT_IMPLEMENTED = 14
    enumerator :: CUFFT_LICENSE_ERROR = 15
    enumerator :: CUFFT_NOT_SUPPORTED = 16
  end enum

  type, bind(C) :: cufftReshapeHandle
  !! An opaque handle to a reshape operation.
    type(c_ptr) :: cptr
  end type cufftReshapeHandle

  interface
    function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, ffttype, batch)   &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftPlanMany")
    !! Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
    import
      type(c_ptr)                       :: plan         !! Pointer to an uninitialized cufftHandle object.
      integer(c_int),             value :: rank         !! Dimensionality of the transform (1, 2, or 3).
      integer(c_int)                    :: n(*)         !! Array of size rank, describing the size of each dimension,
                                                        !! n[0] being the size of the outermost 
                                                        !! and n[rank-1] innermost (contiguous) dimension of a transform.
      integer(c_int)                    :: inembed(*)   !! Pointer of size rank that indicates the storage dimensions of the input data in memory. 
                                                        !! If set to NULL all other advanced data layout parameters are ignored.
      integer(c_int),             value :: istride      !! Indicates the distance between two successive input elements in the least 
                                                        !! significant (i.e., innermost) dimension.
      integer(c_int),             value :: idist        !! Indicates the distance between the first element of two consecutive signals 
                                                        !! in a batch of the input data.
      integer(c_int)                    :: onembed(*)   !! Pointer of size rank that indicates the storage dimensions of the output data in memory. 
                                                        !! If set to NULL all other advanced data layout parameters are ignored.
      integer(c_int),             value :: ostride      !! Indicates the distance between two successive output elements in the output array 
                                                        !! in the least significant (i.e., innermost) dimension.
      integer(c_int),             value :: odist        !! Indicates the distance between the first element of two consecutive signals 
                                                        !! in a batch of the output data.
      integer(c_int),             value :: ffttype      !! The transform data type (e.g., CUFFT_R2C for single precision real to complex).
      integer(c_int),             value :: batch        !! Batch size for this transform.
      integer(c_int)                    :: cufftResult  !! The enumerated type cufftResult defines API call result codes.
    end function cufftPlanMany

    function cufftXtExec(plan, input, output, direction)                                                      &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftXtExec")
    !! Executes any cuFFT transform regardless of precision and type.
    !! In case of complex-to-real and real-to-complex transforms direction parameter is ignored.
    import
      type(c_ptr),                value :: plan         !! cufftHandle returned by cufftCreate
      type(c_ptr),                value :: input        !! Pointer to the input data (in GPU memory) to transform.
      type(c_ptr),                value :: output       !! Pointer to the output data (in GPU memory).
      integer(c_int),             value :: direction    !! The transform direction: CUFFT_FORWARD or CUFFT_INVERSE. 
                                                        !! Ignored for complex-to-real and real-to-complex transforms.
      integer(c_int)                    :: cufftResult  !! The enumerated type cufftResult defines API call result codes.
    end function cufftXtExec

    function cufftDestroy(plan)                                                                               &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftDestroy")
    !! Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
    import
      type(c_ptr),                value :: plan         !! Object of the plan to be destroyed.
      integer(c_int)                    :: cufftResult  !! The enumerated type cufftResult defines API call result codes.
    end function cufftDestroy

    function cufftSetStream(plan, stream)                                                                     &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftSetStream")
    !! Associates a CUDA stream with a cuFFT plan.
    import
      type(c_ptr),                value :: plan         !! Object to associate with the stream.
      type(dtfft_stream_t),       value :: stream       !! A valid CUDA stream
      integer(c_int)                    :: cufftResult  !! The enumerated type cufftResult defines API call result codes.
    end function cufftSetStream

    function cufftMpCreateReshape(reshapeHandle)                                                              &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftMpCreateReshape")
    !! This function initializes a reshape handle for future use. This function is not collective.
    import
      type(cufftReshapeHandle)          :: reshapeHandle      !! The reshape handle.
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpCreateReshape

    function cufftMpAttachReshapeComm(reshapeHandle, commType, fcomm)                                         &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftMpAttachReshapeComm")
    !! This function attaches a communication handle to a reshape. This function is not collective.
    import
      type(cufftReshapeHandle), value   :: reshapeHandle      !! The reshape handle.
      integer(c_int),           value   :: commType           !! An enum describing the communication type of the handle.
      type(c_ptr)                       :: fcomm              !! If comm_type is CUFFT_COMM_MPI, this should be a pointer to an MPI communicator. 
                                                              !! The pointer should remain valid until destruction of the handle. 
                                                              !! Otherwise, this should be NULL.
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpAttachReshapeComm

    function cufftMpGetReshapeSize(reshapeHandle, workSize)                                                   &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftMpGetReshapeSize")
    !! Returns the amount (in bytes) of workspace required to execute the handle. 
    !! There is no guarantee that the workspace_size will or will not change between versions of cuFFTMp.
    import
      type(cufftReshapeHandle), value   :: reshapeHandle      !! The reshape handle.
      integer(c_size_t)                 :: workSize           !! The size, in bytes, of the workspace required during reshape execution
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpGetReshapeSize

    function cufftMpMakeReshape(reshapeHandle, elementSize, rank, lower_input, upper_input, lower_output, upper_output, strides_input, strides_output)  &
      result(cufftResult)                                                                                                                               &
      bind(C, name="cufftMpMakeReshape")
    !! This function creates a reshape intended to re-distribute a global array of 3D data. 
    import
      type(cufftReshapeHandle), value   :: reshapeHandle      !! The reshape handle.
      integer(c_long_long),     value   :: elementSize        !! The size of the individual elements, in bytes. Allowed values are 4, 8 and 16.
      integer(c_int),           value   :: rank               !! The length of the lower_input, upper_input, lower_output, upper_output, strides_input and strides_output arrays. rank should be 3.
      integer(c_long_long)              :: lower_input(*)     !! An array of length rank, respresenting the lower-corner of the portion of the global nx * ny * nz array owned by the current process in the input descriptor.
      integer(c_long_long)              :: upper_input(*)     !! An array of length rank, respresenting the upper-corner of the portion of the global nx * ny * nz array owned by the current process in the input descriptor.
      integer(c_long_long)              :: lower_output(*)    !! An array of length rank, respresenting the lower-corner of the portion of the global nx * ny * nz array owned by the current process in the output descriptor.
      integer(c_long_long)              :: upper_output(*)    !! An array of length rank, respresenting the upper-corner of the portion of the global nx * ny * nz array owned by the current process in the output descriptor.
      integer(c_long_long)              :: strides_input(*)   !! An array of length rank, respresenting the local data layout of the input descriptor in memory. All entries much be decreasing and positive.
      integer(c_long_long)              :: strides_output(*)  !! An array of length rank, respresenting the local data layout of the output descriptor in memory. All entries much be decreasing and positive.
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpMakeReshape

    function cufftMpExecReshapeAsync(reshapeHandle, dataOut, dataIn, workSpace, stream)                       &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftMpExecReshapeAsync")
    !! Executes the reshape, redistributing data_in into data_out using the workspace in workspace.
    import
      type(cufftReshapeHandle), value   :: reshapeHandle      !! The reshape handle.
      type(c_ptr),              value   :: dataOut            !! A symmetric-heap pointer to the output data. This memory should be NVSHMEM allocated and identical on all processes.
      type(c_ptr),              value   :: dataIn             !! A symmetric-heap pointer to the input data. This memory should be NVSHMEM allocated and identical on all processes.
      type(c_ptr),              value   :: workSpace          !! A symmetric-heap pointer to the workspace data. This memory should be NVSHMEM allocated and identical on all processes.
      type(dtfft_stream_t),     value   :: stream             !! The CUDA stream in which to run the reshape operation.
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpExecReshapeAsync

    function cufftMpDestroyReshape(reshapeHandle)                                                             &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftMpDestroyReshape")
    !! Destroys a reshape and all its associated data.
    import
      type(cufftReshapeHandle), value   :: reshapeHandle      !! The reshape handle.
      integer(c_int)                    :: cufftResult        !! The enumerated type cufftResult defines API call result codes.
    end function cufftMpDestroyReshape
  end interface

contains

  function cufftGetErrorString(error_code) result(string)
  !! Returns a string representation of the cuFFT error code.
    integer(c_int32_t), intent(in)  :: error_code   !! cuFFT error code
    character(len=:),   allocatable :: string       !! String representation of the cuFFT error code

    select case (error_code)
    case (CUFFT_SUCCESS)
      allocate(string, source="CUFFT_SUCCESS")
    case (CUFFT_INVALID_PLAN)
      allocate(string, source="CUFFT_INVALID_PLAN")
    case (CUFFT_ALLOC_FAILED)
      allocate(string, source="CUFFT_ALLOC_FAILED")
    case (CUFFT_INVALID_TYPE)
      allocate(string, source="CUFFT_INVALID_TYPE")
    case (CUFFT_INVALID_VALUE)
      allocate(string, source="CUFFT_INVALID_VALUE")
    case (CUFFT_INTERNAL_ERROR)
      allocate(string, source="CUFFT_INTERNAL_ERROR")
    case (CUFFT_EXEC_FAILED)
      allocate(string, source="CUFFT_EXEC_FAILED")
    case (CUFFT_SETUP_FAILED)
      allocate(string, source="CUFFT_SETUP_FAILED")
    case (CUFFT_INVALID_SIZE)
      allocate(string, source="CUFFT_INVALID_SIZE")
    case (CUFFT_UNALIGNED_DATA)
      allocate(string, source="CUFFT_UNALIGNED_DATA")
    case (CUFFT_INCOMPLETE_PARAMETER_LIST)
      allocate(string, source="CUFFT_INCOMPLETE_PARAMETER_LIST")
    case (CUFFT_INVALID_DEVICE)
      allocate(string, source="CUFFT_INVALID_DEVICE")
    case (CUFFT_PARSE_ERROR)
      allocate(string, source="CUFFT_PARSE_ERROR")
    case (CUFFT_NO_WORKSPACE)
      allocate(string, source="CUFFT_NO_WORKSPACE")
    case (CUFFT_NOT_IMPLEMENTED)
      allocate(string, source="CUFFT_NOT_IMPLEMENTED")
    case (CUFFT_LICENSE_ERROR)
      allocate(string, source="CUFFT_LICENSE_ERROR")
    case (CUFFT_NOT_SUPPORTED)
      allocate(string, source="CUFFT_NOT_SUPPORTED")
    case default
      allocate(string, source="Unknown CUFFT error code")
  end select
  end function cufftGetErrorString
end module dtfft_interface_cufft