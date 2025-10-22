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
module dtfft_config
!! Configuration module for dtFFT.
!! It handles both runtime (environment variables) and compile-time ([[dtfft_config_t]]) configurations.
use iso_c_binding, only: c_bool, c_int32_t
use iso_fortran_env
use dtfft_errors
use dtfft_parameters
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: init_internal
public :: dtfft_config_t
public :: string
public :: dtfft_create_config, dtfft_set_config
public :: get_conf_log_enabled, get_conf_z_slab_enabled, get_conf_y_slab_enabled
public :: get_conf_platform
public :: get_conf_measure_warmup_iters, get_conf_measure_iters
public :: get_env, get_datatype_from_env
public :: get_conf_backend, get_conf_datatype_enabled
public :: get_conf_mpi_enabled, get_conf_pipelined_enabled
#ifdef DTFFT_WITH_CUDA
public :: destroy_stream
public :: get_conf_stream
public :: get_conf_nvshmem_enabled, get_conf_nccl_enabled
public :: get_conf_kernel_optimization_enabled, get_conf_configs_to_test
public :: get_conf_forced_kernel_optimization
#endif

  logical,                    save  :: is_init_called = .false.
    !! Has [[init_internal]] already been called or not
  integer(int32),             save  :: log_enabled_from_env = VARIABLE_NOT_SET
    !! Should we log messages to stdout or not
  type(dtfft_platform_t),     save  :: platform_from_env = PLATFORM_NOT_SET
    !! Platform obtained from environ
  integer(int32),             save  :: z_slab_from_env = VARIABLE_NOT_SET
    !! Should Z-slab be used if possible
  integer(int32),             save  :: y_slab_from_env = VARIABLE_NOT_SET
    !! Should Y-slab be used if possible
  integer(int32),             save  :: n_measure_warmup_iters_from_env = VARIABLE_NOT_SET
    !! Number of warmup iterations for measurements
  integer(int32),             save  :: n_measure_iters_from_env = VARIABLE_NOT_SET
    !! Number of measurement iterations
  logical,                    save  :: is_log_enabled = .false.
    !! Should we print additional information during plan creation
  logical,                    save  :: is_z_slab_enabled = .true.
    !! Should we use Z-slab optimization or not
  logical,                    save  :: is_y_slab_enabled = .false.
    !! Should we use Y-slab optimization or not
  type(dtfft_platform_t),     save  :: platform = DTFFT_PLATFORM_HOST
    !! Default platform
  integer(int32),             save  :: n_measure_warmup_iters = CONF_DTFFT_MEASURE_WARMUP_ITERS
    !! Number of warmup iterations for measurements
  integer(int32),             save  :: n_measure_iters = CONF_DTFFT_MEASURE_ITERS
    !! Number of measurement iterations

  type(dtfft_backend_t),      save  :: backend_from_env = BACKEND_NOT_SET
    !! Backend obtained from environ
  integer(int32),             save  :: datatype_enabled_from_env = VARIABLE_NOT_SET
    !! Should we use MPI Datatype backend during autotune or not
  integer(int32),             save  :: mpi_enabled_from_env = VARIABLE_NOT_SET
    !! Should we use MPI backends during autotune or not
  integer(int32),             save  :: pipelined_enabled_from_env = VARIABLE_NOT_SET
    !! Should we use pipelined backends during autotune or not
#ifdef DTFFT_WITH_CUDA
  integer(int32),             save  :: nccl_enabled_from_env = VARIABLE_NOT_SET
    !! Should we use NCCL backends during autotune or not
  integer(int32),             save  :: nvshmem_enabled_from_env = VARIABLE_NOT_SET
    !! Should we use NVSHMEM backends during autotune or not
  integer(int32),             save  :: kernel_optimization_enabled_from_env = VARIABLE_NOT_SET
    !! Should we enable kernel block optimization during autotune or not
  integer(int32),             save  :: n_configs_to_test_from_env = VARIABLE_NOT_SET
    !! Number of blocks to test during nvrtc kernel autotune
  integer(int32),             save  :: forced_kernel_optimization_from_env = VARIABLE_NOT_SET
    !! Should we force kernel optimization even when effort is not DTFFT_PATIENT
  type(dtfft_backend_t),  parameter :: DEFAULT_BACKEND = BACKEND_NOT_SET
    !! Default backend when cuda is enabled
  type(dtfft_stream_t),       save  :: main_stream = NULL_STREAM
    !! Default dtFFT CUDA stream
  type(dtfft_stream_t),       save  :: custom_stream = NULL_STREAM
    !! CUDA stream set by the user
  logical,                    save  :: is_stream_created = .false.
    !! Is the default stream created?
  logical,                    save  :: is_custom_stream = .false.
    !! Is the custom stream provided by the user?
#else
  type(dtfft_backend_t),  parameter :: DEFAULT_BACKEND = DTFFT_BACKEND_MPI_DATATYPE
    !! Default host backend
#endif
  logical,                    save  :: is_datatype_enabled = .true.
    !! Should we use MPI Datatype backend or not
  logical,                    save  :: is_pipelined_enabled = .true.
    !! Should we use pipelined backends or not
  logical,                    save  :: is_mpi_enabled = .false.
    !! Should we use MPI backends or not
#ifdef DTFFT_WITH_CUDA
  logical,                    save  :: is_nccl_enabled = .true.
    !! Should we use NCCL backends or not
  logical,                    save  :: is_nvshmem_enabled = .true.
    !! Should we use NCCL backends or not
  logical,                    save  :: is_kernel_optimization_enabled = .true.
    !! Should we use kernel optimization or not
  integer(int32),             save  :: n_configs_to_test = CONF_DTFFT_CONFIGS_TO_TEST
    !! Number of different NVRTC kernel configurations to try during autotune
  logical,                    save  :: is_forced_kernel_optimization = .false.
    !! Should we use forced kernel optimization or not
#endif
  type(dtfft_backend_t),      save  :: backend = DEFAULT_BACKEND
    !! Default backend

  character(len=26), parameter :: UPPER_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    !! Upper case alphabet.
  character(len=26), parameter :: LOWER_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
    !! Lower case alphabet.

  type, bind(C) :: dtfft_config_t
  !! Type that can be used to set additional configuration parameters to ``dtFFT``
    logical(c_bool)           :: enable_log
      !! Should dtFFT print additional information during plan creation or not.
      !!
      !! Default is false.
    logical(c_bool)           :: enable_z_slab
      !! Should dtFFT use Z-slab optimization or not.
      !!
      !! Default is true.
      !!
      !! One should consider disabling Z-slab optimization in order to resolve `DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error
      !! OR when underlying FFT implementation of 2D plan is too slow.
      !! In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
    logical(c_bool)           :: enable_y_slab
      !! Should dtFFT use Y-slab optimization or not.
      !!
      !! Default is false.
      !!
      !! One should consider disabling Y-slab optimization in order to resolve `DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error
      !! OR when underlying FFT implementation of 2D plan is too slow.
      !! In all other cases it is considered that Y-slab is always faster, since it reduces number of data transpositions.
    integer(c_int32_t)        :: n_measure_warmup_iters
      !! Number of warmup iterations to execute when effort level is higher or equal to `DTFFT_MEASURE`
      !!
      !! Default is 2.
    integer(c_int32_t)        :: n_measure_iters
      !! Number of iterations to execute when effort level is higher or equal to `DTFFT_MEASURE`
      !!
      !! Default is 5.
      !! When `dtFFT` is built with CUDA support, this value also used to determine number
      !! of iterations when selecting block of threads for NVRTC transpose kernel
#ifdef DTFFT_WITH_CUDA
    type(dtfft_platform_t)    :: platform
      !! Selects platform to execute plan.
      !!
      !! Default is DTFFT_PLATFORM_HOST
      !!
      !! This option is only defined with device support build.
      !! Even when dtFFT is build with device support it does not nessasary means that all plans must be related to device.
      !! This enables single library installation to be compiled with both host, CUDA and HIP plans.

    type(dtfft_stream_t)      :: stream
      !! Main CUDA stream that will be used in dtFFT.
      !!
      !! This parameter is a placeholder for user to set custom stream.
      !!
      !! Stream that is actually used by dtFFT plan is returned by `plan%get_stream` function.
      !!
      !! When user sets stream he is responsible of destroying it.
      !!
      !! Stream must not be destroyed before call to `plan%destroy`.
#endif
    type(dtfft_backend_t)     :: backend
      !! Backend that will be used by dtFFT when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
      !!
      !! Default is `DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise `DTFFT_BACKEND_MPI_P2P`.

    logical(c_bool)           :: enable_datatype_backend
      !! Should `DTFFT_BACKEND_MPI_DATATYPE` be enabled when `effort` is `DTFFT_PATIENT` or not.
      !!
      !! Default is true.
      !!
      !! This option works when `platform` is `DTFFT_PLATFORM_HOST`.

    logical(c_bool)           :: enable_mpi_backends
      !! Should MPI Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
      !!
      !! Default is false.
      !!
      !! The following applies only to CUDA builds.
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
#ifdef DTFFT_WITH_CUDA
    logical(c_bool)           :: enable_nccl_backends
      !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
      !!
      !! Default is true.

    logical(c_bool)           :: enable_nvshmem_backends
      !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
      !!
      !! Default is true.
    logical(c_bool)           :: enable_kernel_optimization
      !! Should dtFFT try to optimize NVRTC kernel block size when `effort` is `DTFFT_PATIENT` or not.
      !!
      !! Default is true.
      !!
      !! This option is only defined when dtFFT is built with CUDA support.
      !!
      !! Enabling this option will make autotuning process longer, but may result in better performance for some problem sizes.
      !! It is recommended to keep this option enabled.

    integer(c_int32_t)        :: n_configs_to_test
      !! Number of top theoretical best performing blocks of threads to test for transposition kernels
      !! when `effort` is `DTFFT_PATIENT`  or `force_kernel_optimization` set to `true`.
      !!
      !! Default is 5.
      !!
      !! This option is only defined when dtFFT is built with CUDA support.
      !!
      !! It is recommended to keep this value between 3 and 10.
      !! Maximum possible value is 25.
      !! Setting this value to zero or one will disable kernel optimization.
    logical(c_bool)            :: force_kernel_optimization
      !! Whether to force kernel optimization when `effort` is not `DTFFT_PATIENT`.
      !!
      !! Default is false.
      !!
      !! This option is only defined when dtFFT is built with CUDA support.
      !!
      !! Enabling this option will make plan creation process longer, but may result in better performance for a long run.
      !! Since kernel optimization is performed without data transfers, the overall autotuning time increase should not be significant.
#endif
  end type dtfft_config_t

  interface dtfft_config_t
  !! Interface to create a new configuration
    module procedure config_constructor !! Default constructor
  end interface dtfft_config_t

  interface get_conf_internal
  !! Returns value from configuration unless environment variable is set
    module procedure get_conf_internal_logical  !! For logical values
    module procedure get_conf_internal_int32    !! For integer(int32) values
  end interface get_conf_internal

  interface get_env
  !! Obtains environment variable
    module procedure :: get_env_base    !! Base procedure
    module procedure :: get_env_string  !! For string values
    module procedure :: get_env_int32   !! For integer(int32) values
    module procedure :: get_env_int8    !! For integer(int8) values
    module procedure :: get_env_logical !! For logical values
  end interface get_env

contains

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

    call init_environment()
    is_init_called = .true.
  end function init_internal

  subroutine init_environment()

    log_enabled_from_env = get_env("ENABLE_LOG", VARIABLE_NOT_SET, valid_values=[0, 1])
    z_slab_from_env = get_env("ENABLE_Z_SLAB", VARIABLE_NOT_SET, valid_values=[0, 1])
    y_slab_from_env = get_env("ENABLE_Y_SLAB", VARIABLE_NOT_SET, valid_values=[0, 1])

    n_measure_warmup_iters_from_env = get_env("MEASURE_WARMUP_ITERS", VARIABLE_NOT_SET, min_valid_value=0)
    n_measure_iters_from_env = get_env("MEASURE_ITERS", VARIABLE_NOT_SET, min_valid_value=1)

#ifdef DTFFT_WITH_CUDA
    block
      type(string), allocatable :: platforms(:)
      type(string) :: pltfrm_env

      allocate( platforms(2) )
      platforms(1) = string("host")
      platforms(2) = string("cuda")

      pltfrm_env = get_env("PLATFORM", "undefined", platforms)
      if ( pltfrm_env%raw == "undefined") then
        platform_from_env = PLATFORM_NOT_SET
      else if ( pltfrm_env%raw == "host" ) then
        platform_from_env = DTFFT_PLATFORM_HOST
      else if ( pltfrm_env%raw == "cuda") then
        platform_from_env = DTFFT_PLATFORM_CUDA
      endif

      call pltfrm_env%destroy()
      call destroy_strings(platforms)
    endblock
#endif

    block
      type(string), allocatable :: backends(:)
      type(string) :: bcknd_env

      allocate( backends(10) )
      backends(1) = string("mpi_dt")
      backends(2) = string("mpi_p2p")
      backends(3) = string("mpi_a2a")
      backends(4) = string("mpi_p2p_pipe")
      backends(5) = string("nccl")
      backends(6) = string("nccl_pipe")
      backends(7) = string("cufftmp")
      backends(8) = string("cufftmp_pipe")
      backends(9) = string("mpi_rma")
      backends(10) = string("mpi_rma_pipe")

      bcknd_env = get_env("BACKEND", "undefined", backends)

      select case ( bcknd_env%raw )
      case ( "undefined" )
        backend_from_env = BACKEND_NOT_SET
      case ( "mpi_dt" )
        backend_from_env = DTFFT_BACKEND_MPI_DATATYPE
      case ( "mpi_p2p" )
        backend_from_env = DTFFT_BACKEND_MPI_P2P
      case ( "mpi_a2a" )
        backend_from_env = DTFFT_BACKEND_MPI_A2A
      case ( "mpi_p2p_pipe" )
        backend_from_env = DTFFT_BACKEND_MPI_P2P_PIPELINED
      case ( "nccl" )
        backend_from_env = DTFFT_BACKEND_NCCL
      case ( "nccl_pipe" )
        backend_from_env = DTFFT_BACKEND_NCCL_PIPELINED
      case ( "cufftmp" )
        backend_from_env = DTFFT_BACKEND_CUFFTMP
      case ( "cufftmp_pipe")
        backend_from_env = DTFFT_BACKEND_CUFFTMP_PIPELINED
      case ( "mpi_rma" )
        backend_from_env = DTFFT_BACKEND_MPI_RMA
      case ( "mpi_rma_pipe" )
        backend_from_env = DTFFT_BACKEND_MPI_RMA_PIPELINED
      endselect

      if ( backend_from_env /= BACKEND_NOT_SET .and. .not.is_valid_backend(backend_from_env) ) then
        WRITE_ERROR("Backend '"//bcknd_env%raw//"' is not available in this build.")
        WRITE_ERROR("Environment variable 'DTFFT_BACKEND' has been ignored")
        backend_from_env = BACKEND_NOT_SET
      endif

      call bcknd_env%destroy()
      call destroy_strings(backends)
    endblock

    datatype_enabled_from_env = get_env("ENABLE_MPI_DT", VARIABLE_NOT_SET, valid_values=[0, 1])
    mpi_enabled_from_env = get_env("ENABLE_MPI", VARIABLE_NOT_SET, valid_values=[0, 1])
    pipelined_enabled_from_env = get_env("ENABLE_PIPE", VARIABLE_NOT_SET, valid_values=[0, 1])
#ifdef DTFFT_WITH_CUDA
    nccl_enabled_from_env = get_env("ENABLE_NCCL", VARIABLE_NOT_SET, valid_values=[0, 1])
    nvshmem_enabled_from_env = get_env("ENABLE_NVSHMEM", VARIABLE_NOT_SET, valid_values=[0, 1])
    kernel_optimization_enabled_from_env = get_env("ENABLE_KERNEL_OPTIMIZATION", VARIABLE_NOT_SET, valid_values=[0, 1])
    n_configs_to_test_from_env = get_env("CONFIGS_TO_TEST", VARIABLE_NOT_SET, min_valid_value=0)
    forced_kernel_optimization_from_env = get_env("FORCE_KERNEL_OPTIMIZATION", VARIABLE_NOT_SET, valid_values=[0, 1])
#endif
  end subroutine init_environment

  pure subroutine dtfft_create_config(config) bind(C, name="dtfft_create_config_c")
  !! Creates a new configuration and sets default values.
  !!
  !! C interface
    type(dtfft_config_t), intent(out) :: config !! Configuration to create
    config = dtfft_config_t()
  end subroutine dtfft_create_config

#ifdef DTFFT_WITH_CUDA
  pure function config_constructor(                                       &
    enable_log, enable_z_slab, enable_y_slab,                             &
    n_measure_warmup_iters, n_measure_iters,                              &
    platform, stream, backend, enable_datatype_backend,                   &
    enable_mpi_backends, enable_pipelined_backends,                       &
    enable_nccl_backends, enable_nvshmem_backends,                        &
    enable_kernel_optimization, n_configs_to_test,                        &
    force_kernel_optimization) result(config)
#else
  pure function config_constructor(                                       &
    enable_log, enable_z_slab, enable_y_slab,                             &
    n_measure_warmup_iters, n_measure_iters,                              &
    backend, enable_datatype_backend,                                     &
    enable_mpi_backends, enable_pipelined_backends) result(config)
#endif
  !! Creates a new configuration
    logical,                optional, intent(in)  :: enable_log
      !! Should dtFFT use Z-slab optimization or not.
    logical,                optional, intent(in)  :: enable_z_slab
      !! Should dtFFT use Z-slab optimization or not.
    logical,                optional, intent(in)  :: enable_y_slab
      !! Should dtFFT use Y-slab optimization or not.
    integer(int32),         optional, intent(in)  :: n_measure_warmup_iters
      !! Number of warmup iterations for measurements
    integer(int32),         optional, intent(in)  :: n_measure_iters
      !! Number of measurement iterations
#ifdef DTFFT_WITH_CUDA
    type(dtfft_platform_t), optional, intent(in)  :: platform
      !! Selects platform to execute plan.
    type(dtfft_stream_t),   optional, intent(in)  :: stream
      !! Main CUDA stream that will be used in dtFFT.
#endif
    type(dtfft_backend_t),  optional, intent(in)  :: backend
      !! Backend that will be used by dtFFT when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
    logical,                optional, intent(in)  :: enable_datatype_backend
      !! Should `DTFFT_BACKEND_MPI_DATATYPE` be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_mpi_backends
      !! Should MPI GPU Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_pipelined_backends
      !! Should pipelined GPU backends be enabled when `effort` is `DTFFT_PATIENT` or not.
#ifdef DTFFT_WITH_CUDA
    logical,                optional, intent(in)  :: enable_nccl_backends
      !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_nvshmem_backends
      !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_kernel_optimization
      !! Should dtFFT try to optimize NVRTC kernel block size during autotune or not.
    integer(int32),         optional, intent(in)  :: n_configs_to_test
      !! Number of top theoretical best performing blocks of threads to test for transposition kernels when `effort` is `DTFFT_PATIENT`.
    logical,                optional, intent(in)  :: force_kernel_optimization
      !! Whether to force kernel optimization when `effort` is not `DTFFT_PATIENT`.
#endif

    type(dtfft_config_t) :: config
      !! Constructed `dtFFT` config ready to be set by call to [[dtfft_set_config]]
    config%enable_log = .false.;                if ( present(enable_log) ) config%enable_log = enable_log
    config%enable_z_slab = .true.;              if ( present(enable_z_slab) ) config%enable_z_slab = enable_z_slab
    config%enable_y_slab = .false.;             if ( present(enable_y_slab) ) config%enable_y_slab = enable_y_slab
    config%n_measure_warmup_iters = CONF_DTFFT_MEASURE_WARMUP_ITERS
      if ( present(n_measure_warmup_iters) ) config%n_measure_warmup_iters = n_measure_warmup_iters
    config%n_measure_iters = CONF_DTFFT_MEASURE_ITERS
      if ( present(n_measure_iters) ) config%n_measure_iters = n_measure_iters
#ifdef DTFFT_WITH_CUDA
    config%platform = DTFFT_PLATFORM_HOST;      if ( present(platform) ) config%platform = platform
    config%stream = NULL_STREAM;                if ( present(stream) ) config%stream = stream
#endif
    config%backend = DEFAULT_BACKEND;           if ( present(backend) ) config%backend = backend
    config%enable_datatype_backend = .true.;    if ( present(enable_datatype_backend) ) config%enable_datatype_backend = enable_datatype_backend
    config%enable_mpi_backends = .false.;       if ( present(enable_mpi_backends) ) config%enable_mpi_backends = enable_mpi_backends
    config%enable_pipelined_backends = .true.;  if ( present(enable_pipelined_backends) ) config%enable_pipelined_backends = enable_pipelined_backends
#ifdef DTFFT_WITH_CUDA
    config%enable_nccl_backends = .true.;       if ( present(enable_nccl_backends) ) config%enable_nccl_backends = enable_nccl_backends
    config%enable_nvshmem_backends = .true.;    if ( present(enable_nvshmem_backends) ) config%enable_nvshmem_backends = enable_nvshmem_backends
    config%enable_kernel_optimization = .true.; if ( present(enable_kernel_optimization) ) config%enable_kernel_optimization = enable_kernel_optimization
    config%n_configs_to_test = CONF_DTFFT_CONFIGS_TO_TEST
      if ( present(n_configs_to_test) ) config%n_configs_to_test = n_configs_to_test
    config%force_kernel_optimization = .false.; if ( present(force_kernel_optimization) ) config%force_kernel_optimization = force_kernel_optimization
#endif
  end function config_constructor

  subroutine dtfft_set_config(config, error_code)
  !! Sets configuration parameters
    type(dtfft_config_t),     intent(in)  :: config     !! Configuration to set
    integer(int32), optional, intent(out) :: error_code !! Error code
    integer(int32) :: ierr

    ierr = init_internal()
    if ( ierr /=DTFFT_SUCCESS ) then
      if ( present( error_code ) ) error_code = ierr
      return
    endif
    is_log_enabled = config%enable_log
    is_z_slab_enabled = config%enable_z_slab
    is_y_slab_enabled = config%enable_y_slab

    if ( config%n_measure_warmup_iters < 0 ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS
      return
    endif
    n_measure_warmup_iters = config%n_measure_warmup_iters
    if ( config%n_measure_iters < 1 ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_MEASURE_ITERS
      return
    endif
    n_measure_iters = config%n_measure_iters

    is_datatype_enabled = config%enable_datatype_backend
    is_mpi_enabled = config%enable_mpi_backends
    is_pipelined_enabled = config%enable_pipelined_backends

#ifdef DTFFT_WITH_CUDA
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

    if ( .not.is_valid_platform(config%platform) ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_PLATFORM
      return
    endif
    platform = config%platform

    is_nccl_enabled = config%enable_nccl_backends
    is_nvshmem_enabled = config%enable_nvshmem_backends
    is_forced_kernel_optimization = config%force_kernel_optimization
    is_kernel_optimization_enabled = config%enable_kernel_optimization
    n_configs_to_test = config%n_configs_to_test
    if ( n_configs_to_test <= 1 ) then
      is_kernel_optimization_enabled = .false.
    endif
#endif

    if ( config%backend /= BACKEND_NOT_SET .and. .not.is_valid_backend(config%backend)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_BACKEND
      return
    endif
    backend = get_correct_backend(config%backend)

    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine dtfft_set_config

  elemental type(dtfft_backend_t) function get_correct_backend(back)
    type(dtfft_backend_t), intent(in) :: back

    if ( back == BACKEND_NOT_SET ) then
#ifdef DTFFT_WITH_CUDA
      if ( get_conf_platform() == DTFFT_PLATFORM_CUDA ) then
# ifdef DTFFT_WITH_NCCL
        get_correct_backend = DTFFT_BACKEND_NCCL
# else
        get_correct_backend = DTFFT_BACKEND_MPI_P2P
# endif
      else
        get_correct_backend = DTFFT_BACKEND_MPI_DATATYPE
      endif
#else
      get_correct_backend = DTFFT_BACKEND_MPI_DATATYPE
#endif
    else
      get_correct_backend = back
    endif
  end function get_correct_backend

  elemental logical function get_conf_internal_logical(from_conf, from_env)
  !! Returns value from configuration unless environment variable is set
    logical,        intent(in) :: from_conf   !! Value from configuration
    integer(int32), intent(in) :: from_env    !! Value from environment variable
    get_conf_internal_logical = from_conf
    if ( from_env /= VARIABLE_NOT_SET ) get_conf_internal_logical = from_env == 1
  end function get_conf_internal_logical

  elemental integer(int32) function get_conf_internal_int32(from_conf, from_env)
  !! Returns value from configuration unless environment variable is set
    integer(int32), intent(in) :: from_conf   !! Value from configuration
    integer(int32), intent(in) :: from_env    !! Value from environment variable
    get_conf_internal_int32 = from_conf
    if ( from_env /= VARIABLE_NOT_SET ) get_conf_internal_int32 = from_env
  end function get_conf_internal_int32

  elemental function get_conf_log_enabled() result(bool)
  !! Whether logging is enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_log_enabled, log_enabled_from_env)
  end function get_conf_log_enabled

  elemental function get_conf_z_slab_enabled() result(bool)
  !! Whether Z-slab optimization is enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_z_slab_enabled, z_slab_from_env)
  end function get_conf_z_slab_enabled

  elemental function get_conf_y_slab_enabled() result(bool)
  !! Whether Y-slab optimization is enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_y_slab_enabled, y_slab_from_env)
  end function get_conf_y_slab_enabled

  elemental function get_conf_measure_warmup_iters() result(iters)
  !! Returns the number of warmup iterations
    integer(int32) :: iters  !! Result
    iters = get_conf_internal(n_measure_warmup_iters, n_measure_warmup_iters_from_env)
  end function get_conf_measure_warmup_iters

  elemental function get_conf_measure_iters() result(iters)
  !! Returns the number of measurement iterations
    integer(int32) :: iters  !! Result
    iters = get_conf_internal(n_measure_iters, n_measure_iters_from_env)
  end function get_conf_measure_iters

  elemental type(dtfft_platform_t) function get_conf_platform()
  !! Returns platform set by the user or default one
    get_conf_platform = platform
    if ( platform_from_env /= PLATFORM_NOT_SET ) get_conf_platform = platform_from_env
  end function get_conf_platform

#ifdef DTFFT_WITH_CUDA
  type(dtfft_stream_t) function get_conf_stream() result(stream)
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
  end function get_conf_stream

  subroutine destroy_stream
  !! Destroy the default stream if it was created
    if ( is_stream_created ) then
      CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(main_stream) )
      is_stream_created = .false.
    endif
  end subroutine destroy_stream
#endif

  elemental type(dtfft_backend_t) function get_conf_backend()
  !! Returns backend set by the user or default one
    get_conf_backend = get_correct_backend(backend)
    if ( backend_from_env /= BACKEND_NOT_SET) get_conf_backend = backend_from_env
  end function get_conf_backend

  elemental function get_conf_datatype_enabled() result(bool)
  !! Whether MPI Datatype backend is enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_datatype_enabled, datatype_enabled_from_env)
  end function get_conf_datatype_enabled

  elemental function get_conf_pipelined_enabled() result(bool)
  !! Whether pipelined backends are enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_pipelined_enabled, pipelined_enabled_from_env)
  end function get_conf_pipelined_enabled

  elemental function get_conf_mpi_enabled() result(bool)
  !! Whether MPI backends are enabled or not
    logical :: bool  !! Result flag
#if !defined(DTFFT_WITH_NCCL) && !defined(DTFFT_WITH_NVSHMEM)
    if ( get_conf_platform() == DTFFT_PLATFORM_HOST) then
      bool = get_conf_internal(is_mpi_enabled, mpi_enabled_from_env)
    else
      bool = .true.
    endif
    ! Should not be .false. if only MPI backends are possible
#else
    bool = get_conf_internal(is_mpi_enabled, mpi_enabled_from_env)
#endif
  end function get_conf_mpi_enabled

#ifdef DTFFT_WITH_CUDA
  elemental function get_conf_nccl_enabled() result(bool)
  !! Whether NCCL backends are enabled or not
    logical :: bool  !! Result flag
# ifdef DTFFT_WITH_NCCL
    bool = get_conf_internal(is_nccl_enabled, nccl_enabled_from_env)
# else
    bool = .false.
# endif
  end function get_conf_nccl_enabled

  elemental function get_conf_nvshmem_enabled() result(bool)
  !! Whether nvshmem backends are enabled or not
    logical :: bool  !! Result flag
# ifdef DTFFT_WITH_NVSHMEM
    bool = get_conf_internal(is_nvshmem_enabled, nvshmem_enabled_from_env)
# else
    bool = .false.
# endif
  end function get_conf_nvshmem_enabled

  elemental function get_conf_kernel_optimization_enabled() result(bool)
  !! Whether kernel optimization is enabled or not
    logical :: bool  !! Result flag
    bool = get_conf_internal(is_kernel_optimization_enabled, kernel_optimization_enabled_from_env)
  end function get_conf_kernel_optimization_enabled

  pure function get_conf_configs_to_test() result(n)
  !! Returns the number of configurations to test
    integer(int32) :: n  !! Result
    n = get_conf_internal(n_configs_to_test, n_configs_to_test_from_env)
  end function get_conf_configs_to_test

  elemental function get_conf_forced_kernel_optimization() result(bool)
  !! Whether forced kernel optimization is enabled or not
    logical :: bool  !! Result flag
    bool = get_conf_internal(is_forced_kernel_optimization, forced_kernel_optimization_from_env)
  end function get_conf_forced_kernel_optimization
#endif

  type(string) function get_env_base(name) result(env)
  !! Base function of obtaining dtFFT environment variable
    character(len=*), intent(in)    :: name         !! Name of environment variable without prefix
    type(string)                    :: full_name    !! Prefixed environment variable name
    integer(int32)                  :: env_val_len  !! Length of the environment variable

    full_name = string("DTFFT_"//name)

    call get_environment_variable(full_name%raw, length=env_val_len)
    allocate(character(env_val_len) :: env%raw)
    if ( env_val_len == 0 ) then
      call full_name%destroy()
      return
    endif
    call get_environment_variable(full_name%raw, env%raw)
    call full_name%destroy()
  end function get_env_base

  type(string) function get_env_string(name, default, valid_values) result(env)
  !! Obtains string environment variable
    character(len=*), intent(in)            :: name                 !! Name of environment variable without prefix
    character(len=*), intent(in)            :: default              !! Name of environment variable without prefix
    type(string),     intent(in)            :: valid_values(:)      !! List of valid variable values
    logical                                 :: is_correct           !! Is env value is correct
    integer(int32)    :: i            !! Index in string
    integer(int32)    :: j            !! Index in alphabet
    type(string)      :: env_val_str  !! String value of the environment variable

    env_val_str = get_env(name)
    if ( len(env_val_str%raw) == 0 ) then
      call env_val_str%destroy()
      env = string(default)
      return
    endif

    ! Converting to lowercase
    do i=1, len(env_val_str%raw)
      j = index(UPPER_ALPHABET, env_val_str%raw(i:i))
      if (j>0) env_val_str%raw(i:i) = LOWER_ALPHABET(j:j)
    enddo

    is_correct = any([(env_val_str%raw == valid_values(i)%raw, i=1,size(valid_values))])

    if ( is_correct ) then
      env = string(env_val_str%raw)
      call env_val_str%destroy()
      return
    endif
    WRITE_ERROR("Invalid environment variable: `DTFFT_"//name//"`, it has been ignored")
    call env_val_str%destroy()
    env = string(default)
  end function get_env_string

  integer(int32) function get_env_int32(name, default, valid_values, min_valid_value) result(env)
  !! Base Integer function of obtaining dtFFT environment variable
    character(len=*), intent(in)            :: name               !! Name of environment variable without prefix
    integer(int32),   intent(in)            :: default            !! Default value in case env is not set or it has wrong value
    integer(int32),   intent(in), optional  :: valid_values(:)    !! List of valid values
    integer(int32),   intent(in), optional  :: min_valid_value    !! Mininum valid value. Usually 0 or 1
    type(string)                            :: env_val_str        !! String value of the environment variable
    logical                                 :: is_correct         !! Is env value is correct
    integer(int32)                          :: env_val_passed     !! Value of the environment variable
    integer(int32)                          :: io_status          !! IO status of reading env variable

#ifdef DTFFT_DEBUG
    if ( ( present(valid_values).and.present(min_valid_value) )           &
      .or.(.not.present(valid_values).and..not.present(min_valid_value))  &
    ) then
      INTERNAL_ERROR("`get_env_int32`")
    endif
#endif

    env_val_str = get_env(name)
    if ( len(env_val_str%raw) == 0 ) then
      deallocate(env_val_str%raw)
      env = default
      return
    endif
    read(env_val_str%raw, *, iostat=io_status) env_val_passed
    if (io_status /= 0) then
      WRITE_ERROR("Invalid integer value for environment variable: `DTFFT_"//name//"`=<"//env_val_str%raw//">, it has been ignored")
      env = default
      deallocate(env_val_str%raw)
      return
    endif
    is_correct = .false.
    if ( present( valid_values ) ) then
      is_correct = any(env_val_passed == valid_values)
    endif
    if ( present( min_valid_value ) ) then
      is_correct = env_val_passed >= min_valid_value
    endif
    if ( is_correct ) then
      env = env_val_passed
      deallocate(env_val_str%raw)
      return
    endif
    WRITE_ERROR("Invalid integer value for environment variable: `DTFFT_"//name//"`=<"//env_val_str%raw//">, it has been ignored")
    env = default
    deallocate(env_val_str%raw)
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

  integer(int8) function get_datatype_from_env(name) result(env)
  !! Obtains datatype id from environment variable
    character(len=*), intent(in)  :: name               !! Name of environment variable without prefix
    env = get_env(name, 2_int8, [1, 2])
  end function get_datatype_from_env

end module dtfft_config