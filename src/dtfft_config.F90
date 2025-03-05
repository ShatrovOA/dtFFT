#include "dtfft_config.h"
module dtfft_config
use iso_c_binding
use iso_fortran_env
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda
#endif
#include "dtfft_cuda.h"
#include "dtfft_mpi.h"
implicit none
private
public :: dtfft_config_t
public :: dtfft_create_config, dtfft_set_config
public :: get_z_slab_flag
public :: get_platform
#ifdef DTFFT_WITH_CUDA
public :: get_user_stream
public :: destroy_stream
public :: get_user_gpu_backend
public :: get_mpi_enabled, get_nvshmem_enabled, get_nccl_enabled, get_pipelined_enabled
#endif


  logical,                    save  :: is_z_slab_enabled = .true.
  !< Should we use z-slab decomposition or not
  type(dtfft_platform_t),     save  :: platform = DTFFT_PLATFORM_HOST
  !< Default platform

#ifdef DTFFT_WITH_CUDA
# ifdef DTFFT_WITH_NCCL
  type(dtfft_gpu_backend_t),  parameter :: DEFAULT_GPU_BACKEND = DTFFT_GPU_BACKEND_NCCL
# else
  type(dtfft_gpu_backend_t),  parameter :: DEFAULT_GPU_BACKEND = DTFFT_GPU_BACKEND_MPI_P2P
# endif

  type(dtfft_stream_t),       save  :: main_stream
  !< Default dtFFT CUDA stream
  type(dtfft_stream_t),       save  :: custom_stream
  !< CUDA stream set by the user
  logical,                    save  :: is_stream_created = .false.
  !< Is the default stream created?
  logical,                    save  :: is_custom_stream = .false.
  !< Is the custom stream provided by the user?
  logical,                    save  :: is_pipelined_enabled = .true.
  !< Should we use pipelined backends or not
  logical,                    save  :: is_mpi_enabled = .false.
  !< Should we use MPI backends or not
  logical,                    save  :: is_nccl_enabled = .true.
  !< Should we use NCCL backends or not
  logical,                    save  :: is_nvshmem_enabled = .true.
  !< Should we use NCCL backends or not
  type(dtfft_gpu_backend_t),  save  :: gpu_backend = DEFAULT_GPU_BACKEND
  !< Default GPU backend
#endif


  type, bind(C) :: dtfft_config_t
    logical(c_bool)           :: enable_z_slab
    !! Should dtFFT use Z-slab optimization or not.
    !!
    !! Default is true.
    !!
    !! One should consider disabling Z-slab optimization in order to resolve `DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error 
    !! OR when underlying FFT implementation of 2D plan is too slow.
    !! In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
#ifdef DTFFT_WITH_CUDA
    type(dtfft_platform_t)    :: platform
    !! Selects platform to execute plan.
    !!
    !! Default is DTFFT_PLATFORM_HOST
    !!
    !! This option is only defined with device support build.
    !! Even when dtFFT is build with device support it does not nessasary means that all plans must be related to device.
    !! This enables single library installation to be compiled with both host, CUDA and HIP plans.
    type(dtfft_stream_t) :: stream
    !! Main CUDA stream that will be used in dtFFT.
    !!
    !! This parameter is a placeholder for user to set custom stream.
    !!
    !! Stream that is actually used by dtFFT plan is returned by `plan%get_stream` function.
    !!
    !! When user sets stream he is responsible of destroying it.
    !!
    !! Stream must not be destroyed before call to `plan%destroy`.
    type(dtfft_gpu_backend_t) :: gpu_backend
    !! Backend that will be used by dtFFT when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
    !!
    !! Default is `DTFFT_GPU_BACKEND_NCCL`
    logical(c_bool)           :: enable_mpi_backends
    !! Should MPI GPU Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    !!
    !! Default is false.
    !!
    !! MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
    !! It was noticed that during plan autotuning GPU memory not being freed completely.
    !! For example:
    !! 1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
    !! Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.
    !!
    !! One of the workarounds is to disable MPI Backends by default, which is done here.
    !!
    !! Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
    !! but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms
    logical(c_bool)           :: enable_pipelined_backends
    !! Should pipelined GPU backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    !!
    !! Default is true.
    !!
    !! Pipelined backends require additional buffer that user has no control over.
    logical(c_bool)           :: enable_nccl_backends
    !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    !!
    !! Default is true.
    logical(c_bool)           :: enable_nvshmem_backends
    !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    !!
    !! Default is true.
#endif
  end type dtfft_config_t

  interface dtfft_config_t
    module procedure config_constructor
  end interface dtfft_config_t

contains

  subroutine dtfft_create_config(config) bind(C, name="dtfft_create_config_c")
    type(dtfft_config_t), intent(out) :: config

    config%enable_z_slab = .true.
#ifdef DTFFT_WITH_CUDA
    config%platform = DTFFT_PLATFORM_HOST
    config%stream = NULL_STREAM
    config%gpu_backend = DEFAULT_GPU_BACKEND
    config%enable_mpi_backends = .false.
    config%enable_pipelined_backends = .true.
    config%enable_nccl_backends = .true.
    config%enable_nvshmem_backends = .true.
#endif
  end subroutine dtfft_create_config

  function config_constructor() result(config)
    type(dtfft_config_t) :: config

    call dtfft_create_config(config)
  end function config_constructor

  subroutine dtfft_set_config(config, error_code)
    type(dtfft_config_t),     intent(in)  :: config
    integer(int32), optional, intent(out) :: error_code

    is_z_slab_enabled = config%enable_z_slab

#ifdef DTFFT_WITH_CUDA
    if (.not.is_valid_gpu_backend(config%gpu_backend)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_GPU_INVALID_BACKEND
      return
    endif
    gpu_backend = config%gpu_backend

    if ( .not.is_null_ptr(config%stream%stream) ) then
      block
        integer(int32) :: ierr

        ierr = cudaStreamQuery(config%stream)
        if ( .not.any(ierr == [cudaSuccess, cudaErrorNotReady]) ) then
          if ( present( error_code ) ) error_code = DTFFT_ERROR_GPU_INVALID_STREAM
          return
        endif
        custom_stream = config%stream
        is_custom_stream = .true.
      endblock
    endif

    if (  .not.is_valid_platform(config%platform) ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_PLATFORM
      return
    endif
    platform = config%platform

    is_mpi_enabled = config%enable_mpi_backends
    is_pipelined_enabled = config%enable_pipelined_backends
    is_nccl_enabled = config%enable_nccl_backends
    is_nvshmem_enabled = config%enable_nvshmem_backends
#endif
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine dtfft_set_config

  logical function get_z_slab_flag()
  !! Whether Z-slab optimization is enabled or not
    get_z_slab_flag = is_z_slab_enabled
  end function get_z_slab_flag

  type(dtfft_platform_t) function get_platform()
  !! Returns platform set by the user or default one
    get_platform = platform
    if ( get_platform_from_env() /= DTFFT_PLATFORM_UNDEFINED ) get_platform = get_platform_from_env()
  end function get_platform

#ifdef DTFFT_WITH_CUDA
  type(dtfft_stream_t) function get_user_stream() result(stream)
  !! Returns either the custom provided by user or creates a new one
    if ( is_custom_stream ) then
      stream = custom_stream
      return
    endif
    if (.not.is_stream_created) then
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(main_stream) )
      is_stream_created = .true.
    endif
    stream = main_stream
  end function get_user_stream

  subroutine destroy_stream
  !! Destroy the default stream if it was created
    if ( is_stream_created ) then
      CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(main_stream) )
      is_stream_created = .false.
    endif
  end subroutine destroy_stream

  type(dtfft_gpu_backend_t) function get_user_gpu_backend()
  !! Returns GPU backend set by the user or default one
    get_user_gpu_backend = gpu_backend
  end function get_user_gpu_backend

  logical function get_pipelined_enabled()
  !! Whether pipelined backends are enabled or not
    get_pipelined_enabled = is_pipelined_enabled
  end function get_pipelined_enabled

  logical function get_mpi_enabled()
  !! Whether MPI backends are enabled or not
#if !defined(DTFFT_WITH_NCCL) && !defined(DTFFT_WITH_NVSHMEM)
    get_mpi_enabled = .true.
#else
    get_mpi_enabled = is_mpi_enabled
#endif
  end function get_mpi_enabled

  logical function get_nccl_enabled()
  !! Whether NCCL backends are enabled or not
#ifdef DTFFT_WITH_NCCL
    get_nccl_enabled = is_nccl_enabled
#else
    get_nccl_enabled = .false.
#endif
  end function get_nccl_enabled

  logical function get_nvshmem_enabled()
  !! Whether nvshmem backends are enabled or not
#ifdef DTFFT_WITH_NVSHMEM
    get_nvshmem_enabled = is_nvshmem_enabled
#else
    get_nvshmem_enabled = .false.
#endif
  end function get_nvshmem_enabled
#endif
end module dtfft_config