module dtfft_interface_cuda
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
implicit none
private
public :: cudaGetErrorString
public :: cudaStreamQuery, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize
public :: cudaSuccess, cudaErrorNotReady
public :: cudaEvent


  type, bind(C) :: dim3
    integer(c_int) :: x,y,z
  end type

public :: dim3


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

  enum, bind(C)
    enumerator :: cudaMemcpyHostToHost = 0
    enumerator :: cudaMemcpyHostToDevice = 1
    enumerator :: cudaMemcpyDeviceToHost = 2
    enumerator :: cudaMemcpyDeviceToDevice = 3
    enumerator :: cudaMemcpyDefault = 4
  end enum

  public :: cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault

  type, bind(C) :: cudaEvent
    type(c_ptr) :: event
  end type cudaEvent

  integer(c_int), parameter, public :: cudaEventDisableTiming = 2
! public :: cudaStream_t

!   type, bind(C) :: cudaStream_t
!     type(c_ptr) :: ptr
!   end type cudaStream_t

  interface
    integer(c_int) function cudaStreamQuery(stream) bind(C, name="cudaStreamQuery")
    import
      type(dtfft_stream_t), value :: stream
    end function cudaStreamQuery

    integer(c_int) function cudaStreamCreate(stream) bind(C, name="cudaStreamCreate")
    import
      type(dtfft_stream_t)        :: stream
    end function cudaStreamCreate

    integer(c_int) function cudaStreamDestroy(stream) bind(C, name="cudaStreamDestroy")
    import
      type(dtfft_stream_t), value :: stream
    end function cudaStreamDestroy

    integer(c_int) function cudaStreamSynchronize(stream) bind(C, name="cudaStreamSynchronize")
    import
      type(dtfft_stream_t), value :: stream
    end function cudaStreamSynchronize
  end interface


  interface
    function cudaGetErrorString_c(errcode) result(string) bind(C, name="cudaGetErrorString")
      import
      integer(c_int),   value :: errcode
      type(c_ptr)             :: string
    end function cudaGetErrorString_c
  end interface

public :: cudaMalloc, cudaFree, cudaMemset
  interface
    integer(c_int) function cudaMalloc(ptr, count) bind(C, name="cudaMalloc")
    import
    type(c_ptr)               :: ptr
    integer(c_size_t), value  :: count
    end function cudaMalloc

    integer(c_int) function cudaFree(ptr) bind(C, name="cudaFree")
    import
      type(c_ptr), value :: ptr
    end function cudaFree

    integer(c_int) function cudaMemset(ptr, val, count) bind(C, name="cudaMemset")
    import
      type(c_ptr),        value :: ptr
      integer(c_int),     value :: val
      integer(c_size_t),  value :: count
    end function cudaMemset
  end interface

public :: cudaEventCreateWithFlags, cudaEventRecord, cudaStreamWaitEvent
public :: cudaEventDestroy, cudaEventCreate, cudaEventSynchronize
public :: cudaEventElapsedTime
  interface
    integer(c_int) function cudaEventCreateWithFlags(event, flags) bind(C, name="cudaEventCreateWithFlags")
    import
      type(cudaEvent)             :: event
      integer(c_int),   value     :: flags
    end function cudaEventCreateWithFlags

    integer(c_int) function cudaEventRecord(event, stream) bind(C, name="cudaEventRecord")
    import
      type(cudaEvent),      value :: event
      type(dtfft_stream_t), value :: stream
    end function cudaEventRecord

    integer(c_int) function cudaStreamWaitEvent(stream, event, flags) bind(C, name="cudaStreamWaitEvent")
    import
      type(dtfft_stream_t), value :: stream
      type(cudaEvent),      value :: event
      integer(c_int),       value :: flags
    end function cudaStreamWaitEvent

    integer(c_int) function cudaEventDestroy(event) bind(C, name="cudaEventDestroy")
    import
      type(cudaEvent),      value :: event
    end function cudaEventDestroy

    integer(c_int) function cudaEventCreate( event ) bind(C, name="cudaEventCreate")
    import
      type(cudaEvent)             :: event
    end function cudaEventCreate

    integer(c_int) function cudaEventSynchronize( event ) bind(C, name="cudaEventSynchronize")
    import
      type(cudaEvent),      value :: event
    end function cudaEventSynchronize

    integer(c_int) function cudaEventElapsedTime(time, start, end) bind(C, name="cudaEventElapsedTime")
    import
      real(c_float)               :: time
      type(cudaEvent),      value :: start
      type(cudaEvent),      value :: end
    end function cudaEventElapsedTime
  end interface

public :: cudaMemcpyAsync
  interface cudaMemcpyAsync
    integer(c_int) function cudaMemcpyAsync_ptr(dst, src, count, kdir, stream) bind(C, name="cudaMemcpyAsync")
    import
    type(c_ptr),          value :: dst
    type(c_ptr),          value :: src
    integer(c_size_t),    value :: count
    integer(c_int),       value :: kdir
    type(dtfft_stream_t), value :: stream
    end function cudaMemcpyAsync_ptr

    integer(c_int) function cudaMemcpyAsync_r32(dst, src, count, kdir, stream) bind(C, name="cudaMemcpyAsync")
    import
      real(c_float)               :: dst
      real(c_float)               :: src
      integer(c_size_t),    value :: count
      integer(c_int),       value :: kdir
      type(dtfft_stream_t), value :: stream
    end function cudaMemcpyAsync_r32
  end interface cudaMemcpyAsync

public :: cudaMemcpy
  interface cudaMemcpy
    integer(c_int) function cudaMemcpy_ptr(dst, src, count, kdir) bind(C, name="cudaMemcpy")
    import
      type(c_ptr),          value :: dst
      type(c_ptr),          value :: src
      integer(c_size_t),    value :: count
      integer(c_int),       value :: kdir
    end function cudaMemcpy_ptr

    integer(c_int) function cudaMemcpy_r64(dst, src, count, kdir) bind(C, name="cudaMemcpy")
    import
      real(c_double)              :: dst(*)
      real(c_double)              :: src(*)
      integer(c_size_t),    value :: count
      integer(c_int),       value :: kdir
    end function cudaMemcpy_r64
  end interface cudaMemcpy

public :: cudaGetDevice, cudaGetDeviceCount, cudaSetDevice
  interface
    integer(c_int) function cudaGetDevice(num) bind(C, name="cudaGetDevice")
    import
      integer(c_int)              :: num
    end function cudaGetDevice

    integer(c_int) function cudaGetDeviceCount(num) bind(C, name="cudaGetDeviceCount")
    import
      integer(c_int)              :: num
    end function cudaGetDeviceCount

    integer(c_int) function cudaSetDevice(num) bind(C, name="cudaSetDevice")
    import
      integer(c_int),   value     :: num
    end function cudaSetDevice
  end interface

public :: get_cuda_architecture
  interface
    subroutine get_cuda_architecture(device, major, minor) bind(C)
    import
      integer(c_int), value :: device
      integer(c_int)        :: major
      integer(c_int)        :: minor
    end subroutine get_cuda_architecture
  end interface


contains

  function cudaGetErrorString(errcode) result(string)
  !! Helper function that returns a string describing the given nvrtcResult code
  !! For unrecognized enumeration values, it returns "NVRTC_ERROR unknown"
    integer(c_int),   intent(in)  :: errcode     !< CUDA Runtime Compilation API result code.
    character(len=:), allocatable :: string         !< Result string
    type(c_ptr)                   :: c_string       !< Pointer to C string
    character(len=256), pointer   :: f_string       !< Pointer to Fortran string

    c_string = cudaGetErrorString_c(errcode)
    call c_f_pointer(c_string, f_string)
    allocate( string, source=f_string(1:index(f_string, c_null_char) - 1) )
  end function cudaGetErrorString

end module dtfft_interface_cuda