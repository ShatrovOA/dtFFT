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
module dtfft_interface_cuda_runtime
!! CUDA Runtime Interfaces
use iso_c_binding
use iso_fortran_env
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_MOCK_ENABLED
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#endif
implicit none
private
public :: cudaGetErrorString
public :: cudaStreamQuery, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize
public :: cudaMalloc, cudaFree, cudaMemset
public :: cudaEventCreateWithFlags, cudaEventRecord, cudaStreamWaitEvent
public :: cudaEventDestroy, cudaEventCreate, cudaEventSynchronize, cudaEventElapsedTime
public :: cudaMemcpyAsync, cudaMemcpy
public :: cudaGetDevice, cudaGetDeviceCount, cudaSetDevice
public :: cudaMemGetInfo, cudaDeviceSynchronize, cudaGetLastError
public :: get_device_props

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
#ifdef DTFFT_WITH_MOCK_ENABLED
    type :: cudaEvent
        real(real64) :: t
    end type cudaEvent
#else
    type, bind(C) :: cudaEvent
    !! CUDA event types
        type(c_ptr) :: event  !! Handle
    end type cudaEvent
#endif

    integer(c_int), parameter, public :: cudaEventDisableTiming = 2

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

#ifndef DTFFT_WITH_MOCK_ENABLED
! Real CUDA Runtime interfaces

    interface
        function cudaStreamQuery(stream)                                    &
            result(cudaError_t)                                             &
            bind(C, name="cudaStreamQuery")
        !! Queries an asynchronous stream for completion status.
        import
            type(dtfft_stream_t), value :: stream       !! Stream identifier
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if all operations in stream have completed,
                                                        !! or `cudaErrorNotReady` if not.
        end function cudaStreamQuery
    endinterface

    interface
        function cudaStreamCreate(stream)                                   &
            result(cudaError_t)                                             &
            bind(C, name="cudaStreamCreate")
        !! Creates an asynchronous stream.
        import
            type(dtfft_stream_t)        :: stream       !! Pointer to the created stream
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream was created successfully,
                                                        !! or an error code if there was an issue.
        end function cudaStreamCreate
    end interface

    interface
        function cudaStreamDestroy(stream)                                  &
            result(cudaError_t)                                             &
            bind(C, name="cudaStreamDestroy")
        !! Destroys an asynchronous stream.
        import
            type(dtfft_stream_t), value :: stream       !! Stream identifier
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream was destroyed successfully,
                                                        !! or an error code if there was an issue.
        end function cudaStreamDestroy
    end interface

    interface
        function cudaStreamSynchronize(stream)                              &
            result(cudaError_t)                                             &
            bind(C, name="cudaStreamSynchronize")
        !! Waits for stream tasks to complete.
        import
            type(dtfft_stream_t), value :: stream       !! Stream identifier
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream tasks completed successfully,
                                                        !! or an error code if there was an issue.
        end function cudaStreamSynchronize
    end interface

    interface
        function cudaGetErrorString_c(errcode)                              &
            result(string)                                                  &
            bind(C, name="cudaGetErrorString")
        !! Returns the string representation of an error code.
        import
            integer(c_int), value :: errcode  !! Error code
            type(c_ptr)           :: string   !! Pointer to the error string
        end function cudaGetErrorString_c
    end interface

    interface
        function cudaMalloc(ptr, count)                                     &
            result(cudaError_t)                                             &
            bind(C, name="cudaMalloc")
        !! Allocates memory on the device.
        import
            type(c_ptr)               :: ptr          !! Pointer to allocated device memory
            integer(c_size_t), value  :: count        !! Requested allocation size in bytes
            integer(c_int)            :: cudaError_t  !! Returns `cudaSuccess` if memory was allocated successfully,
                                                        !! or `cudaErrorMemoryAllocation` if the memory could not be allocated.
        end function cudaMalloc
    end interface

    interface
        function cudaFree(ptr)                                              &
            result(cudaError_t)                                             &
            bind(C, name="cudaFree")
        !! Frees memory on the device.
        import
            type(c_ptr), value :: ptr         !! Pointer to memory to free
            integer(c_int)     :: cudaError_t !! Returns `cudaSuccess` if memory was freed successfully,
                                                !! or an error code if there was an issue.
        end function cudaFree
    end interface

    interface
        function cudaMemset(ptr, val, count)                                &
            result(cudaError_t)                                             &
            bind(C, name="cudaMemset")
        !! Initializes or sets device memory to a value.
        import
            type(c_ptr),        value :: ptr          !! Pointer to device memory
            integer(c_int),     value :: val          !! Value to set
            integer(c_size_t),  value :: count        !! Size in bytes to set
            integer(c_int)            :: cudaError_t  !! Returns `cudaSuccess` if the memory was set successfully,
                                                        !! or an error code if there was an issue.
        end function cudaMemset
    end interface

    interface
        function cudaEventCreateWithFlags(event, flags)                     &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventCreateWithFlags")
        !! Creates an event with the specified flags.
        import
            type(cudaEvent)             :: event        !! Event identifier
            integer(c_int),   value     :: flags        !! Flags for event creation
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the event was created successfully,
                                                        !! or an error code if there was an issue.
        end function cudaEventCreateWithFlags
    end interface

    interface
        function cudaEventRecord(event, stream)                             &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventRecord")
        !! Records an event in a stream.
        import
            type(cudaEvent),      value :: event        !! Event identifier
            type(dtfft_stream_t), value :: stream       !! Stream identifier
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the event was recorded successfully,
                                                        !! or an error code if there was an issue.
        end function cudaEventRecord
    end interface

    interface
        function cudaStreamWaitEvent(stream, event, flags)                  &
            result(cudaError_t)                                             &
            bind(C, name="cudaStreamWaitEvent")
        !! Makes a stream wait on an event.
        import
            type(dtfft_stream_t), value :: stream       !! Stream identifier
            type(cudaEvent),      value :: event        !! Event identifier
            integer(c_int),       value :: flags        !! Flags for the wait operation
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the stream is waiting successfully,
                                                        !! or an error code if there was an issue.
        end function cudaStreamWaitEvent
    end interface

    interface
        function cudaEventDestroy(event)                                    &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventDestroy")
        !! Destroys an event.
        import
            type(cudaEvent), value :: event       !! Event identifier
            integer(c_int)         :: cudaError_t !! Returns `cudaSuccess` if the event was destroyed successfully,
                                                    !! or an error code if there was an issue.
        end function cudaEventDestroy
    end interface

    interface
        function cudaEventCreate(event)                                     &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventCreate")
        !! Creates an event.
        import
            type(cudaEvent) :: event        !! Event identifier
            integer(c_int)  :: cudaError_t  !! Returns `cudaSuccess` if the event was created successfully,
                                            !! or an error code if there was an issue.
        end function cudaEventCreate
    end interface

    interface
        function cudaEventSynchronize(event)                                &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventSynchronize")
        !! Waits for an event to complete.
        import
            type(cudaEvent), value :: event         !! Event identifier
            integer(c_int)         :: cudaError_t   !! Returns `cudaSuccess` if the event completed successfully,
                                                    !! or an error code if there was an issue.
        end function cudaEventSynchronize
    end interface

    interface
        function cudaEventElapsedTime(time, start, end)                     &
            result(cudaError_t)                                             &
            bind(C, name="cudaEventElapsedTime")
        !! Computes the elapsed time between two events.
        import
            real(c_float)          :: time        !! Elapsed time in milliseconds
            type(cudaEvent), value :: start       !! Starting event
            type(cudaEvent), value :: end         !! Ending event
            integer(c_int)         :: cudaError_t !! Returns `cudaSuccess` if the elapsed time was computed successfully,
                                                    !! or an error code if there was an issue.
        end function cudaEventElapsedTime
    end interface

    interface cudaMemcpyAsync
        !! Copies data asynchronously between host and device.
        function cudaMemcpyAsync_ptr(dst, src, count, kdir, stream)         &
            result(cudaError_t)                                             &
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

        function cudaMemcpyAsync_r32(dst, src, count, kdir, stream)         &
            result(cudaError_t)                                             &
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

    interface cudaMemcpy
        !! Copies data synchronously between host and device.
        function cudaMemcpy_ptr(dst, src, count, kdir)                      &
            result(cudaError_t)                                             &
            bind(C, name="cudaMemcpy")
        import
            type(c_ptr),          value :: dst          !! Destination pointer
            type(c_ptr),          value :: src          !! Source pointer
            integer(c_size_t),    value :: count        !! Size in bytes to copy
            integer(c_int),       value :: kdir         !! Direction of copy
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was completed successfully,
                                                        !! or an error code if there was an issue.
        end function cudaMemcpy_ptr

        function cudaMemcpy_r64(dst, src, count, kdir)                      &
            result(cudaError_t)                                             &
            bind(C, name="cudaMemcpy")
        import
            real(c_double)              :: dst(*)       !! Destination array (64-bit float)
            real(c_double)              :: src(*)       !! Source array (64-bit float)
            integer(c_size_t),    value :: count        !! Number of bytes to copy
            integer(c_int),       value :: kdir         !! Direction of copy
            integer(c_int)              :: cudaError_t  !! Returns `cudaSuccess` if the copy was completed successfully,
                                                        !! or an error code if there was an issue.
        end function cudaMemcpy_r64

        function cudaMemcpy_r32(dst, src, count, kdir)                      &
            result(cudaError_t)                                             &
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

    interface
        function cudaGetDevice(num)                                         &
            result(cudaError_t)                                             &
            bind(C, name="cudaGetDevice")
        !! Returns the current device.
        import
            integer(c_int) :: num         !! Device number
            integer(c_int) :: cudaError_t !! Returns `cudaSuccess` if the device was retrieved successfully,
                                            !! or an error code if there was an issue.
        end function cudaGetDevice
    end interface

    interface
        function cudaGetDeviceCount(num)                                    &
            result(cudaError_t)                                             &
            bind(C, name="cudaGetDeviceCount")
        !! Returns the number of available devices.
        import
            integer(c_int) :: num         !! Number of devices
            integer(c_int) :: cudaError_t !! Returns `cudaSuccess` if the device count was retrieved successfully,
                                            !! or an error code if there was an issue.
        end function cudaGetDeviceCount
    end interface

    interface
        function cudaSetDevice(num)                                         &
            result(cudaError_t)                                             &
            bind(C, name="cudaSetDevice")
        !! Sets the current device.
        import
            integer(c_int), value :: num          !! Device number
            integer(c_int)        :: cudaError_t  !! Returns `cudaSuccess` if the device was set successfully,
                                                    !! or an error code if there was an issue.
        end function cudaSetDevice
    end interface

    interface
        function cudaMemGetInfo(free, total)                                &
            result(cudaError_t)                                             &
            bind(C, name="cudaMemGetInfo")
        !! Returns the amount of free and total memory on the device.
        import
            integer(c_size_t)   :: free         !! Free memory in bytes
            integer(c_size_t)   :: total        !! Total memory in bytes
            integer(c_int)      :: cudaError_t  !! Returns `cudaSuccess` if the memory information was retrieved successfully,
                                                !! or an error code if there was an issue.
        end function cudaMemGetInfo
    end interface

    interface
        function cudaDeviceSynchronize()                                    &
            result(cudaError_t)                                             &
            bind(C, name="cudaDeviceSynchronize")
        !! Synchronizes the device, blocking until all preceding tasks in all streams have completed.
        import
            integer(c_int)    :: cudaError_t  !! Returns `cudaSuccess` if syncronization was
                                                !! or an error code if there was an issue.
        end function cudaDeviceSynchronize
    end interface

    interface
        function cudaGetLastError()                                         &
            result(cudaError_t)                                             &
            bind(C, name="cudaGetLastError")
        !! Returns the last error from a runtime call.
        import
            integer(c_int)    :: cudaError_t  !! Returns `cudaSuccess` if no error was detected
                                                !! or an error code if there was an issue.
        end function cudaGetLastError
    end interface

    interface
        subroutine get_device_props(device, props)                          &
            bind(C, name="get_device_props_cuda")
        !! Returns the CUDA device properties for a given device number.
        import
            integer(c_int), value   :: device   !! Device number
            type(device_props)      :: props    !! GPU Properties
        end subroutine get_device_props
    end interface

#else
! Mock CUDA Runtime interfaces for CPU testing

    interface cudaMemcpyAsync
        module procedure cudaMemcpyAsync_ptr
        module procedure cudaMemcpyAsync_r32
    end interface cudaMemcpyAsync

    interface cudaMemcpy
        module procedure cudaMemcpy_ptr
        module procedure cudaMemcpy_r64
        module procedure cudaMemcpy_r32
    end interface cudaMemcpy

    ! C-wrappers for mock functions (public for C interoperability)
    public :: cudaDeviceSynchronize_c
    public :: cudaStreamCreate_c
    public :: cudaStreamSynchronize_c
    public :: cudaStreamDestroy_c
    public :: cudaMallocManaged_c
    public :: cudaFree_c
    public :: cudaMemset_c
    public :: cudaGetErrorString_c_wrapper

#endif


contains

#ifdef DTFFT_WITH_MOCK_ENABLED
  ! Mock implementations for CPU testing
  
    function cudaStreamQuery(stream) result(cudaError_t)
    !! Mock: Always returns cudaSuccess
        type(dtfft_stream_t), value :: stream
        integer(c_int)              :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaStreamQuery

    function cudaStreamCreate(stream) result(cudaError_t)
    !! Mock: Creates a dummy stream
        type(dtfft_stream_t)        :: stream
        integer(c_int)              :: cudaError_t
        stream%stream = c_null_ptr
        cudaError_t = cudaSuccess
    end function cudaStreamCreate

    function cudaStreamDestroy(stream) result(cudaError_t)
    !! Mock: Does nothing
        type(dtfft_stream_t), intent(in) :: stream
        integer(c_int)                   :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaStreamDestroy

    function cudaStreamSynchronize(stream) result(cudaError_t)
    !! Mock: Does nothing, just returns success
        type(dtfft_stream_t), intent(in) :: stream
        integer(c_int)                   :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaStreamSynchronize

    function cudaGetErrorString_c(errcode) result(string)
    !! Mock: Returns error string
        integer(c_int), intent(in) :: errcode
        type(c_ptr)                :: string
        character(len=20), target, save :: error_msg = "Mock CUDA Error"
        string = c_loc(error_msg)
    end function cudaGetErrorString_c

    function cudaMalloc(ptr, count) result(cudaError_t)
    !! Mock: Allocates memory on CPU using allocate
        use, intrinsic :: iso_c_binding
        type(c_ptr)                  :: ptr
        integer(c_size_t), intent(in) :: count
        integer(c_int)               :: cudaError_t

        ptr = mem_alloc_host(count)
        if( is_null_ptr(ptr) ) then
        cudaError_t = cudaErrorMemoryAllocation
        else
        cudaError_t = cudaSuccess
        end if
    end function cudaMalloc

    function cudaFree(ptr) result(cudaError_t)
    !! Mock: Frees memory allocated on CPU
        type(c_ptr), intent(in) :: ptr
        integer(c_int)          :: cudaError_t

        if ( is_null_ptr(ptr) ) then
        cudaError_t = cudaErrorInvalidValue
        return
        end if
        call mem_free_host(ptr)
        cudaError_t = cudaSuccess
    end function cudaFree

    function cudaMemset(ptr, val, count) result(cudaError_t)
    !! Mock: Sets memory on CPU
        type(c_ptr),        intent(in) :: ptr
        integer(c_int),     intent(in) :: val
        integer(c_size_t),  intent(in) :: count
        integer(c_int)                 :: cudaError_t
        integer(c_int8_t), pointer :: temp_array(:)
        integer(c_size_t) :: i

        if ( is_null_ptr(ptr) ) then
            cudaError_t = cudaErrorInvalidValue
            return
        end if

        call c_f_pointer(ptr, temp_array, [count])
        do i = 1, count
            temp_array(i) = int(val, c_int8_t)
        end do
        cudaError_t = cudaSuccess
    end function cudaMemset

    function cudaEventCreateWithFlags(event, flags) result(cudaError_t)
    !! Mock: Creates dummy event
        type(cudaEvent)            :: event
        integer(c_int), intent(in) :: flags
        integer(c_int)             :: cudaError_t
        event%t = 0.0_real64
        cudaError_t = cudaSuccess
    end function cudaEventCreateWithFlags

    function cudaEventRecord(event, stream) result(cudaError_t)
    !! Mock: Does nothing
        type(cudaEvent),      intent(inout) :: event
        type(dtfft_stream_t), intent(in)    :: stream
        integer(c_int)                      :: cudaError_t

        event%t = MPI_Wtime()
        cudaError_t = cudaSuccess
    end function cudaEventRecord

    function cudaStreamWaitEvent(stream, event, flags) result(cudaError_t)
    !! Mock: Does nothing
        type(dtfft_stream_t), intent(in) :: stream
        type(cudaEvent),      intent(in) :: event
        integer(c_int),       intent(in) :: flags
        integer(c_int)                   :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaStreamWaitEvent

    function cudaEventDestroy(event) result(cudaError_t)
    !! Mock: Does nothing
        type(cudaEvent), intent(in) :: event
        integer(c_int)              :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaEventDestroy

    function cudaEventCreate(event) result(cudaError_t)
    !! Mock: Creates dummy event
        type(cudaEvent) :: event
        integer(c_int)  :: cudaError_t
        event%t = 0.0_real64
        cudaError_t = cudaSuccess
    end function cudaEventCreate

    function cudaEventSynchronize(event) result(cudaError_t)
    !! Mock: Does nothing
        type(cudaEvent), intent(in) :: event
        integer(c_int)              :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaEventSynchronize

    function cudaEventElapsedTime(time, start, end) result(cudaError_t)
    !! Mock: Returns zero time
        real(c_float),   intent(out) :: time
        type(cudaEvent), intent(in)  :: start
        type(cudaEvent), intent(in)  :: end
        integer(c_int)               :: cudaError_t
        time = 1000.0_c_float * real(end%t - start%t, real32)
        cudaError_t = cudaSuccess
    end function cudaEventElapsedTime

    function cudaMemcpyAsync_ptr(dst, src, count, kdir, stream) result(cudaError_t)
    !! Mock: Synchronous copy on CPU
        type(c_ptr),          intent(in) :: dst
        type(c_ptr),          intent(in) :: src
        integer(c_size_t),    intent(in) :: count
        integer(c_int),       intent(in) :: kdir
        type(dtfft_stream_t), intent(in) :: stream
        integer(c_int)                   :: cudaError_t
        integer(c_int8_t), pointer :: src_array(:), dst_array(:)

        if ( is_null_ptr(src) .or. is_null_ptr(dst)) then
            cudaError_t = cudaErrorInvalidValue
            return
        endif

        call c_f_pointer(src, src_array, [count])
        call c_f_pointer(dst, dst_array, [count])
        dst_array(:) = src_array(:)
        cudaError_t = cudaSuccess
    end function cudaMemcpyAsync_ptr

    function cudaMemcpyAsync_r32(dst, src, count, kdir, stream) result(cudaError_t)
    !! Mock: Synchronous copy on CPU for r32
        real(c_float), target,  intent(out) :: dst
        real(c_float), target,  intent(in)  :: src
        integer(c_size_t),      intent(in)  :: count
        integer(c_int),         intent(in)  :: kdir
        type(dtfft_stream_t),   intent(in)  :: stream
        integer(c_int)                    :: cudaError_t

        cudaError_t = cudaMemcpyAsync(c_loc(dst), c_loc(src), count, kdir, stream)
    end function cudaMemcpyAsync_r32

    function cudaMemcpy_ptr(dst, src, count, kdir) result(cudaError_t)
    !! Mock: Synchronous copy on CPU
        type(c_ptr),       intent(in) :: dst
        type(c_ptr),       intent(in) :: src
        integer(c_size_t), intent(in) :: count
        integer(c_int),    intent(in) :: kdir
        integer(c_int)                :: cudaError_t

        cudaError_t = cudaMemcpyAsync(dst, src, count, kdir, NULL_STREAM)
    end function cudaMemcpy_ptr

    function cudaMemcpy_r64(dst, src, count, kdir) result(cudaError_t)
    !! Mock: Synchronous copy on CPU for r64
        real(c_double),    target,  intent(out) :: dst(*)
        real(c_double),    target,  intent(in)  :: src(*)
        integer(c_size_t),          intent(in)  :: count
        integer(c_int),             intent(in)  :: kdir
        integer(c_int)                          :: cudaError_t

        cudaError_t = cudaMemcpy(c_loc(dst), c_loc(src), count, kdir)
    end function cudaMemcpy_r64

    function cudaMemcpy_r32(dst, src, count, kdir) result(cudaError_t)
    !! Mock: Synchronous copy on CPU for r32
        real(c_float),    target,   intent(out) :: dst(*)
        real(c_float),    target,   intent(in)  :: src(*)
        integer(c_size_t),          intent(in)  :: count
        integer(c_int),             intent(in)  :: kdir
        integer(c_int)                          :: cudaError_t

        cudaError_t = cudaMemcpy(c_loc(dst), c_loc(src), count, kdir)
    end function cudaMemcpy_r32

    function cudaGetDevice(num) result(cudaError_t)
    !! Mock: Returns rank of MPI_COMM_WORLD
        integer(c_int), intent(out) :: num
        integer(c_int)              :: cudaError_t
        integer(int32) :: mpi_err

        call MPI_Comm_rank(MPI_COMM_WORLD, num, mpi_err)
        cudaError_t = cudaSuccess
    end function cudaGetDevice

    function cudaGetDeviceCount(num) result(cudaError_t)
    !! Mock: Returns size of MPI_COMM_TYPE_SHARED
        integer(c_int), intent(out) :: num
        integer(c_int)              :: cudaError_t
        TYPE_MPI_COMM  :: local_comm
        integer(int32) :: ierr

        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
        call MPI_Comm_size(local_comm, num, ierr)
        call MPI_Comm_free(local_comm, ierr)
        cudaError_t = cudaSuccess
    end function cudaGetDeviceCount

    function cudaSetDevice(num) result(cudaError_t)
    !! Mock: Does nothing
        integer(c_int), intent(in) :: num
        integer(c_int)             :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaSetDevice

    function cudaMemGetInfo(free, total) result(cudaError_t)
    !! Mock: Returns dummy memory info
        integer(c_size_t)   :: free, total
        integer(c_int)      :: cudaError_t
        total = 8_c_size_t * 1024_c_size_t * 1024_c_size_t * 1024_c_size_t  ! 8 GB
        free = 4_c_size_t * 1024_c_size_t * 1024_c_size_t * 1024_c_size_t   ! 4 GB
        cudaError_t = cudaSuccess
    end function cudaMemGetInfo

    function cudaDeviceSynchronize() result(cudaError_t)
    !! Mock: Does nothing
        integer(c_int)    :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaDeviceSynchronize

    function cudaGetLastError() result(cudaError_t)
    !! Mock: Returns success
        integer(c_int)    :: cudaError_t
        cudaError_t = cudaSuccess
    end function cudaGetLastError

    subroutine get_device_props(device, props)
    !! Mock: Returns dummy device properties
        integer(c_int),    intent(in)  :: device
        type(device_props), intent(out) :: props

        props%sm_count                   = 108
        props%max_threads_per_sm         = 2048
        props%max_blocks_per_sm          = 32
        props%shared_mem_per_sm          = 164 * 1024
        props%max_threads_per_block      = 1024
        props%shared_mem_per_block       = 48 * 1024
        props%l2_cache_size              = 40960 * 1024
        props%compute_capability_major   = 8
        props%compute_capability_minor   = 0
    end subroutine get_device_props

#endif

    function cudaGetErrorString(errcode) result(string)
    !! Helper function that returns a string describing the given nvrtcResult code
    !! If the error code is not recognized, "unrecognized error code" is returned.
        integer(c_int),   intent(in)  :: errcode        !! CUDA Runtime Compilation API result code.
        character(len=:), allocatable :: string         !! Result string

#ifndef DTFFT_WITH_MOCK_ENABLED
        call string_c2f(cudaGetErrorString_c(errcode), string)
#else
        if (errcode == cudaSuccess) then
            allocate(string, source="cudaSuccess (mock)")
        else
            allocate(string, source="cudaError (mock)")
        end if
#endif
    end function cudaGetErrorString

#ifdef DTFFT_WITH_MOCK_ENABLED
    ! C-wrapper functions for mock CUDA functions
    function cudaDeviceSynchronize_c() result(cudaError_t) bind(C, name="cudaDeviceSynchronize")
        integer(c_int) :: cudaError_t
        cudaError_t = cudaDeviceSynchronize()
    end function cudaDeviceSynchronize_c

    function cudaStreamCreate_c(stream) result(cudaError_t) bind(C, name="cudaStreamCreate")
        type(dtfft_stream_t) :: stream
        integer(c_int)       :: cudaError_t
        cudaError_t = cudaStreamCreate(stream)
    end function cudaStreamCreate_c

    function cudaStreamSynchronize_c(stream) result(cudaError_t) bind(C, name="cudaStreamSynchronize")
        type(dtfft_stream_t), value :: stream
        integer(c_int)              :: cudaError_t
        cudaError_t = cudaStreamSynchronize(stream)
    end function cudaStreamSynchronize_c

    function cudaStreamDestroy_c(stream) result(cudaError_t) bind(C, name="cudaStreamDestroy")
        type(dtfft_stream_t), value :: stream
        integer(c_int)              :: cudaError_t
        cudaError_t = cudaStreamDestroy(stream)
    end function cudaStreamDestroy_c

    function cudaMallocManaged_c(ptr, count, flags) result(cudaError_t) bind(C, name="cudaMallocManaged")
        type(c_ptr)              :: ptr
        integer(c_size_t), value :: count
        integer(c_int), value    :: flags
        integer(c_int)           :: cudaError_t
        ! For mock version, just call regular cudaMalloc (ignore flags)
        cudaError_t = cudaMalloc(ptr, count)
    end function cudaMallocManaged_c

    function cudaFree_c(ptr) result(cudaError_t) bind(C, name="cudaFree")
        type(c_ptr), value :: ptr
        integer(c_int)     :: cudaError_t
        cudaError_t = cudaFree(ptr)
    end function cudaFree_c

    function cudaMemset_c(ptr, val, count) result(cudaError_t) bind(C, name="cudaMemset")
        type(c_ptr), value       :: ptr
        integer(c_int), value    :: val
        integer(c_size_t), value :: count
        integer(c_int)           :: cudaError_t
        cudaError_t = cudaMemset(ptr, val, count)
    end function cudaMemset_c

    function cudaGetErrorString_c_wrapper(errcode) result(string) bind(C, name="cudaGetErrorString")
        integer(c_int), value :: errcode
        type(c_ptr)           :: string
        character(len=50), target, save :: error_msg

        if (errcode == cudaSuccess) then
            error_msg = "cudaSuccess (mock)" // c_null_char
        else
            error_msg = "cudaError (mock)" // c_null_char
        end if
        string = c_loc(error_msg)
    end function cudaGetErrorString_c_wrapper
#endif

end module dtfft_interface_cuda_runtime