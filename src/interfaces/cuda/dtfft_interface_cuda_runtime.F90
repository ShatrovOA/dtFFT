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
module dtfft_interface_cuda_runtime
!! CUDA Runtime Interfaces
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
use dtfft_utils,      only: string_c2f
implicit none
private
public :: cudaGetErrorString

public :: cudaSuccess, cudaErrorNotReady
  enum, bind(c)
    enumerator :: cudaSuccess = 0
    enumerator :: cudaErrorInvalidValue = 1
    enumerator :: cudaErrorMemoryAllocation = 2
    enumerator :: cudaErrorInitializationError = 3
    enumerator :: cudaErrorCudartUnloading = 4
    enumerator :: cudaErrorProfilerDisabled = 5
    enumerator :: cudaErrorProfilerNotInitialized = 6
    enumerator :: cudaErrorProfilerAlreadyStarted = 7
    enumerator :: cudaErrorProfilerAlreadyStopped = 8
    enumerator :: cudaErrorInvalidConfiguration = 9
    enumerator :: cudaErrorInvalidPitchValue = 12
    enumerator :: cudaErrorInvalidSymbol = 13
    enumerator :: cudaErrorInvalidHostPointer = 16
    enumerator :: cudaErrorInvalidDevicePointer = 17
    enumerator :: cudaErrorInvalidTexture = 18
    enumerator :: cudaErrorInvalidTextureBinding = 19
    enumerator :: cudaErrorInvalidChannelDescriptor = 20
    enumerator :: cudaErrorInvalidMemcpyDirection = 21
    enumerator :: cudaErrorAddressOfConstant = 22
    enumerator :: cudaErrorTextureFetchFailed = 23
    enumerator :: cudaErrorTextureNotBound = 24
    enumerator :: cudaErrorSynchronizationError = 25
    enumerator :: cudaErrorInvalidFilterSetting = 26
    enumerator :: cudaErrorInvalidNormSetting = 27
    enumerator :: cudaErrorMixedDeviceExecution = 28
    enumerator :: cudaErrorNotYetImplemented = 31
    enumerator :: cudaErrorMemoryValueTooLarge = 32
    enumerator :: cudaErrorInsufficientDriver = 35
    enumerator :: cudaErrorInvalidSurface = 37
    enumerator :: cudaErrorDuplicateVariableName = 43
    enumerator :: cudaErrorDuplicateTextureName = 44
    enumerator :: cudaErrorDuplicateSurfaceName = 45
    enumerator :: cudaErrorDevicesUnavailable = 46
    enumerator :: cudaErrorIncompatibleDriverContext = 49
    enumerator :: cudaErrorMissingConfiguration = 52
    enumerator :: cudaErrorPriorLaunchFailure = 53
    enumerator :: cudaErrorLaunchMaxDepthExceeded = 65
    enumerator :: cudaErrorLaunchFileScopedTex = 66
    enumerator :: cudaErrorLaunchFileScopedSurf = 67
    enumerator :: cudaErrorSyncDepthExceeded = 68
    enumerator :: cudaErrorLaunchPendingCountExceeded = 69
    enumerator :: cudaErrorInvalidDeviceFunction = 98
    enumerator :: cudaErrorNoDevice = 100
    enumerator :: cudaErrorInvalidDevice = 101
    enumerator :: cudaErrorStartupFailure = 127
    enumerator :: cudaErrorInvalidKernelImage = 200
    enumerator :: cudaErrorDeviceUninitialized = 201
    enumerator :: cudaErrorMapBufferObjectFailed = 205
    enumerator :: cudaErrorUnmapBufferObjectFailed = 206
    enumerator :: cudaErrorArrayIsMapped = 207
    enumerator :: cudaErrorAlreadyMapped = 208
    enumerator :: cudaErrorNoKernelImageForDevice = 209
    enumerator :: cudaErrorAlreadyAcquired = 210
    enumerator :: cudaErrorNotMapped = 211
    enumerator :: cudaErrorNotMappedAsArray = 212
    enumerator :: cudaErrorNotMappedAsPointer = 213
    enumerator :: cudaErrorECCUncorrectable = 214
    enumerator :: cudaErrorUnsupportedLimit = 215
    enumerator :: cudaErrorDeviceAlreadyInUse = 216
    enumerator :: cudaErrorPeerAccessUnsupported = 217
    enumerator :: cudaErrorInvalidPtx = 218
    enumerator :: cudaErrorInvalidGraphicsContext = 219
    enumerator :: cudaErrorNvlinkUncorrectable = 220
    enumerator :: cudaErrorJitCompilerNotFound = 221
    enumerator :: cudaErrorInvalidSource = 300
    enumerator :: cudaErrorFileNotFound = 301
    enumerator :: cudaErrorSharedObjectSymbolNotFound = 302
    enumerator :: cudaErrorSharedObjectInitFailed = 303
    enumerator :: cudaErrorOperatingSystem = 304
    enumerator :: cudaErrorInvalidResourceHandle = 400
    enumerator :: cudaErrorIllegalState = 401
    enumerator :: cudaErrorSymbolNotFound = 500
    enumerator :: cudaErrorNotReady = 600
    enumerator :: cudaErrorIllegalAddress = 700
    enumerator :: cudaErrorLaunchOutOfResources = 701
    enumerator :: cudaErrorLaunchTimeout = 702
    enumerator :: cudaErrorLaunchIncompatibleTexturing = 703
    enumerator :: cudaErrorPeerAccessAlreadyEnabled = 704
    enumerator :: cudaErrorPeerAccessNotEnabled = 705
    enumerator :: cudaErrorSetOnActiveProcess = 708
    enumerator :: cudaErrorContextIsDestroyed = 709
    enumerator :: cudaErrorAssert = 710
    enumerator :: cudaErrorTooManyPeers = 711
    enumerator :: cudaErrorHostMemoryAlreadyRegistered = 712
    enumerator :: cudaErrorHostMemoryNotRegistered = 713
    enumerator :: cudaErrorHardwareStackError = 714
    enumerator :: cudaErrorIllegalInstruction = 715
    enumerator :: cudaErrorMisalignedAddress = 716
    enumerator :: cudaErrorInvalidAddressSpace = 717
    enumerator :: cudaErrorInvalidPc = 718
    enumerator :: cudaErrorLaunchFailure = 719
    enumerator :: cudaErrorCooperativeLaunchTooLarge = 720
    enumerator :: cudaErrorNotPermitted = 800
    enumerator :: cudaErrorNotSupported = 801
    enumerator :: cudaErrorSystemNotReady = 802
    enumerator :: cudaErrorSystemDriverMismatch = 803
    enumerator :: cudaErrorCompatNotSupportedOnDevice = 804
    enumerator :: cudaErrorStreamCaptureUnsupported = 900
    enumerator :: cudaErrorStreamCaptureInvalidated = 901
    enumerator :: cudaErrorStreamCaptureMerge = 902
    enumerator :: cudaErrorStreamCaptureUnmatched = 903
    enumerator :: cudaErrorStreamCaptureUnjoined = 904
    enumerator :: cudaErrorStreamCaptureIsolation = 905
    enumerator :: cudaErrorStreamCaptureImplicit = 906
    enumerator :: cudaErrorCapturedEvent = 907
    enumerator :: cudaErrorStreamCaptureWrongThread = 908
    enumerator :: cudaErrorTimeout = 909
    enumerator :: cudaErrorGraphExecUpdateFailure = 910
    enumerator :: cudaErrorUnknown = 999
    enumerator :: cudaErrorApiFailureBase = 10000
  end enum

public :: cudaMemcpyHostToHost,     &
          cudaMemcpyHostToDevice,   &
          cudaMemcpyDeviceToHost,   &
          cudaMemcpyDeviceToDevice, &
          cudaMemcpyDefault
  enum, bind(C)
    enumerator :: cudaMemcpyHostToHost = 0
    enumerator :: cudaMemcpyHostToDevice = 1
    enumerator :: cudaMemcpyDeviceToHost = 2
    enumerator :: cudaMemcpyDeviceToDevice = 3
    enumerator :: cudaMemcpyDefault = 4
  end enum



public :: cudaEvent
  type, bind(C) :: cudaEvent
  !! CUDA event types
    type(c_ptr) :: event  !! Handle
  end type cudaEvent

  integer(c_int), parameter, public :: cudaEventDisableTiming = 2


public :: cudaStreamQuery
  interface
  !! Queries an asynchronous stream for completion status.
    function cudaStreamQuery(stream)                              &
      result(cudaError_t)                                         &
      bind(C, name="cudaStreamQuery")
    import
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if all operations in stream have completed, 
                                                  !! or `cudaErrorNotReady` if not.
    end function cudaStreamQuery
  endinterface

public :: cudaStreamCreate
  interface
  !! Creates an asynchronous stream.
    function cudaStreamCreate(stream)                              &
      result(cudaError_t)                                          &
      bind(C, name="cudaStreamCreate")
    import
      type(dtfft_stream_t)        :: stream       !! Pointer to the created stream
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream was created successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaStreamCreate
  end interface

public :: cudaStreamDestroy
  interface
    !! Destroys an asynchronous stream.
    function cudaStreamDestroy(stream)                             &
      result(cudaError_t)                                          &
      bind(C, name="cudaStreamDestroy")
    import
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream was destroyed successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaStreamDestroy
  end interface

public :: cudaStreamSynchronize
  interface
    !! Waits for stream tasks to complete.
    function cudaStreamSynchronize(stream)                         &
      result(cudaError_t)                                          &
      bind(C, name="cudaStreamSynchronize")
    import
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream tasks completed successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaStreamSynchronize
  end interface

  interface
  !! Returns the string representation of an error code.
    function cudaGetErrorString_c(errcode)                         &
      result(string)                                               &
      bind(C, name="cudaGetErrorString")
    import
      integer(c_int), value :: errcode  !! Error code
      type(c_ptr)           :: string   !! Pointer to the error string
    end function cudaGetErrorString_c
  end interface

public :: cudaMalloc
  interface
    !! Allocates memory on the device.
    function cudaMalloc(ptr, count)                                &
      result(cudaError_t)                                          &
      bind(C, name="cudaMalloc")
    import
      type(c_ptr)               :: ptr          !! Pointer to allocated device memory
      integer(c_size_t), value  :: count        !! Requested allocation size in bytes
      integer(c_int)            :: cudaError_t  !! Returns `cudaSuccess` if memory was allocated successfully, 
                                                !! or `cudaErrorMemoryAllocation` if the memory could not be allocated.
    end function cudaMalloc
  end interface

public :: cudaFree
  interface
  !! Frees memory on the device.
    function cudaFree(ptr)                                         &
      result(cudaError_t)                                          &
      bind(C, name="cudaFree")
    import
      type(c_ptr), value :: ptr         !! Pointer to memory to free
      integer(c_int)     :: cudaError_t !! Returns `cudaSuccess` if memory was freed successfully, 
                                        !! or an error code if there was an issue.
    end function cudaFree
  end interface

public :: cudaMemset
  interface
  !! Initializes or sets device memory to a value.
    function cudaMemset(ptr, val, count)                           &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemset")
    import
      type(c_ptr),        value :: ptr          !! Pointer to device memory
      integer(c_int),     value :: val          !! Value to set
      integer(c_size_t),  value :: count        !! Size in bytes to set
      integer(c_int)            :: cudaError_t  !! Returns `cudaSuccess` if the memory was set successfully, 
                                                !! or an error code if there was an issue.
    end function cudaMemset
  end interface

public :: cudaEventCreateWithFlags
  interface
  !! Creates an event with the specified flags.
    function cudaEventCreateWithFlags(event, flags)                &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventCreateWithFlags")
    import
      type(cudaEvent)             :: event        !! Event identifier
      integer(c_int),   value     :: flags        !! Flags for event creation
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the event was created successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaEventCreateWithFlags
  end interface

public :: cudaEventRecord
  interface
  !! Records an event in a stream.
    function cudaEventRecord(event, stream)                        &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventRecord")
    import
      type(cudaEvent),      value :: event        !! Event identifier
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the event was recorded successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaEventRecord
  end interface

public :: cudaStreamWaitEvent
  interface
  !! Makes a stream wait on an event.
    function cudaStreamWaitEvent(stream, event, flags)             &
      result(cudaError_t)                                          &
      bind(C, name="cudaStreamWaitEvent")
    import
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      type(cudaEvent),      value :: event        !! Event identifier
      integer(c_int),       value :: flags        !! Flags for the wait operation
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream is waiting successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaStreamWaitEvent
  end interface

public :: cudaEventDestroy
  interface
  !! Destroys an event.
    function cudaEventDestroy(event)                               &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventDestroy")
    import
      type(cudaEvent), value :: event       !! Event identifier
      integer(c_int)         :: cudaError_t !! Returns `cudaSuccess` if the event was destroyed successfully, 
                                            !! or an error code if there was an issue.
    end function cudaEventDestroy
  end interface

public :: cudaEventCreate
  interface
  !! Creates an event.
    function cudaEventCreate(event)                                &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventCreate")
    import
      type(cudaEvent) :: event        !! Event identifier
      integer(c_int)  :: cudaError_t  !! Returns `cudaSuccess` if the event was created successfully, 
                                      !! or an error code if there was an issue.
    end function cudaEventCreate
  end interface

public :: cudaEventSynchronize
  interface
  !! Waits for an event to complete.
    function cudaEventSynchronize(event)                           &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventSynchronize")
    import
      type(cudaEvent), value :: event       !! Event identifier
      integer(c_int)         :: cudaError_t !! Returns `cudaSuccess` if the event completed successfully, 
                                            !! or an error code if there was an issue.
    end function cudaEventSynchronize
  end interface

public :: cudaEventElapsedTime
  interface
  !! Computes the elapsed time between two events.
    function cudaEventElapsedTime(time, start, end)                &
      result(cudaError_t)                                          &
      bind(C, name="cudaEventElapsedTime")
    import
      real(c_float)          :: time        !! Elapsed time in milliseconds
      type(cudaEvent), value :: start       !! Starting event
      type(cudaEvent), value :: end         !! Ending event
      integer(c_int)         :: cudaError_t !! Returns `cudaSuccess` if the elapsed time was computed successfully, 
                                            !! or an error code if there was an issue.
    end function cudaEventElapsedTime
  end interface

public :: cudaMemcpyAsync
  interface cudaMemcpyAsync
    !! Copies data asynchronously between host and device.
    function cudaMemcpyAsync_ptr(dst, src, count, kdir, stream)    &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemcpyAsync")
    import
      type(c_ptr),          value :: dst          !! Destination pointer
      type(c_ptr),          value :: src          !! Source pointer
      integer(c_size_t),    value :: count        !! Size in bytes to copy
      integer(c_int),       value :: kdir         !! Direction of copy (host-to-device, device-to-host, etc.)
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was initiated successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaMemcpyAsync_ptr

    function cudaMemcpyAsync_r32(dst, src, count, kdir, stream)    &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemcpyAsync")
    import
      real(c_float)               :: dst          !! Destination array (32-bit float)
      real(c_float)               :: src          !! Source array (32-bit float)
      integer(c_size_t),    value :: count        !! Number of elements to copy
      integer(c_int),       value :: kdir         !! Direction of copy
      type(dtfft_stream_t), value :: stream       !! Stream identifier
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was initiated successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaMemcpyAsync_r32
  end interface

public :: cudaMemcpy
  interface cudaMemcpy
    !! Copies data synchronously between host and device.
    function cudaMemcpy_ptr(dst, src, count, kdir)                 &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemcpy")
    import
      type(c_ptr),          value :: dst          !! Destination pointer
      type(c_ptr),          value :: src          !! Source pointer
      integer(c_size_t),    value :: count        !! Size in bytes to copy
      integer(c_int),       value :: kdir         !! Direction of copy
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was completed successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaMemcpy_ptr

    function cudaMemcpy_r64(dst, src, count, kdir)                 &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemcpy")
    import
      real(c_double)              :: dst(*)       !! Destination array (64-bit float)
      real(c_double)              :: src(*)       !! Source array (64-bit float)
      integer(c_size_t),    value :: count        !! Number of bytes to copy
      integer(c_int),       value :: kdir         !! Direction of copy
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was completed successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaMemcpy_r64

    function cudaMemcpy_r32(dst, src, count, kdir)                 &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemcpy")
    import
      real(c_float)               :: dst(*)       !! Destination array (32-bit float)
      real(c_float)               :: src(*)       !! Source array (32-bit float)
      integer(c_size_t),    value :: count        !! Number of bytes to copy
      integer(c_int),       value :: kdir         !! Direction of copy
      integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was completed successfully, 
                                                  !! or an error code if there was an issue.
    end function cudaMemcpy_r32
  end interface

public :: cudaGetDevice
  interface
    !! Returns the current device.
    function cudaGetDevice(num)                                    &
      result(cudaError_t)                                          &
      bind(C, name="cudaGetDevice")
    import
      integer(c_int) :: num         !! Device number
      integer(c_int) :: cudaError_t !! Returns `cudaSuccess` if the device was retrieved successfully, 
                                    !! or an error code if there was an issue.
    end function cudaGetDevice
  end interface

public :: cudaGetDeviceCount
  interface
    !! Returns the number of available devices.
    function cudaGetDeviceCount(num)                               &
      result(cudaError_t)                                          &
      bind(C, name="cudaGetDeviceCount")
    import
      integer(c_int) :: num         !! Number of devices
      integer(c_int) :: cudaError_t !! Returns `cudaSuccess` if the device count was retrieved successfully, 
                                    !! or an error code if there was an issue.
    end function cudaGetDeviceCount
  end interface

public :: cudaSetDevice
  interface
    !! Sets the current device.
    function cudaSetDevice(num)                                    &
      result(cudaError_t)                                          &
      bind(C, name="cudaSetDevice")
    import
      integer(c_int), value :: num          !! Device number
      integer(c_int)        :: cudaError_t  !! Returns `cudaSuccess` if the device was set successfully, 
                                            !! or an error code if there was an issue.
    end function cudaSetDevice
  end interface

public :: cudaMemGetInfo
  interface
    !! Returns the amount of free and total memory on the device.
    function cudaMemGetInfo(free, total)                           &
      result(cudaError_t)                                          &
      bind(C, name="cudaMemGetInfo")
    import
      integer(c_size_t)   :: free         !! Free memory in bytes
      integer(c_size_t)   :: total        !! Total memory in bytes
      integer(c_int)      :: cudaError_t  !! Returns `cudaSuccess` if the memory information was retrieved successfully, 
                                          !! or an error code if there was an issue.
    end function cudaMemGetInfo
  end interface

public :: cudaDeviceSynchronize
    interface
    !! Synchronizes the device, blocking until all preceding tasks in all streams have completed.
      function cudaDeviceSynchronize()                               &
        result(cudaError_t)                                          &
        bind(C, name="cudaDeviceSynchronize")
      import
        integer(c_int)    :: cudaError_t  !! Returns `cudaSuccess` if the device was set successfully, 
                                          !! or an error code if there was an issue.
      end function cudaDeviceSynchronize
    end interface

public :: device_props
  type, bind(C) :: device_props
  !! GPU device properties obtained from cudaDeviceProp
    integer(c_int)    :: sm_count                   !! Number of multiprocessors on device (cudaDeviceProp.multiProcessorCount)
    integer(c_int)    :: max_threads_per_sm         !! Maximum resident threads per multiprocessor (cudaDeviceProp.maxThreadsPerMultiProcessor)
    integer(c_int)    :: max_blocks_per_sm          !! Maximum number of resident blocks per multiprocessor (cudaDeviceProp.maxBlocksPerMultiProcessor)
    integer(c_size_t) :: shared_mem_per_sm          !! Shared memory per multiprocessor (cudaDeviceProp.sharedMemPerMultiprocessor)
    integer(c_int)    :: max_threads_per_block      !! Maximum number of threads per block (cudaDeviceProp.maxThreadsPerBlock)
    integer(c_size_t) :: shared_mem_per_block       !! Shared memory available per block in bytes (cudaDeviceProp.sharedMemPerBlock)
    integer(c_int)    :: l2_cache_size              !! Size of L2 cache in bytes (cudaDeviceProp.l2CacheSize)
    integer(c_int)    :: compute_capability_major   !! Major compute capability (cudaDeviceProp.major)
    integer(c_int)    :: compute_capability_minor   !! Minor compute capability (cudaDeviceProp.minor)
  end type device_props

public :: get_device_props
  interface
    subroutine get_device_props(device, props)         &
      bind(C, name="get_device_props_cuda")
    !! Returns the CUDA device properties for a given device number.
    import
      integer(c_int), value   :: device   !! Device number
      type(device_props)      :: props    !! GPU Properties
    end subroutine get_device_props
  end interface


contains

  function cudaGetErrorString(errcode) result(string)
  !! Helper function that returns a string describing the given nvrtcResult code
  !! If the error code is not recognized, "unrecognized error code" is returned.
    integer(c_int),   intent(in)  :: errcode        !! CUDA Runtime Compilation API result code.
    character(len=:), allocatable :: string         !! Result string

    call string_c2f(cudaGetErrorString_c(errcode), string)
  end function cudaGetErrorString

end module dtfft_interface_cuda_runtime