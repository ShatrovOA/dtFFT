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
#ifdef DTFFT_WITH_COMPRESSION
use dtfft_abstract_compressor
#endif
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
public :: get_conf_backend, get_conf_reshape_backend
public :: get_conf_datatype_enabled, get_conf_mpi_enabled, get_conf_pipelined_enabled
#ifdef DTFFT_WITH_CUDA
public :: destroy_stream
public :: get_conf_stream
public :: get_conf_nvshmem_enabled, get_conf_nccl_enabled
#endif
public :: get_conf_kernel_autotune_enabled
public :: get_conf_fourier_reshape_enabled
public :: get_conf_rma_enabled, get_conf_fused_enabled
public :: get_conf_transpose_mode, get_conf_access_mode
public :: get_conf_compression_enabled
public :: get_correct_backend
#ifdef DTFFT_WITH_COMPRESSION
public :: get_conf_transpose, get_conf_reshape
#endif

    logical,                    protected  :: is_init_called = .false.
        !! Has [[init_internal]] already been called or not
    integer(int32),             protected  :: log_enabled_from_env = VARIABLE_NOT_SET
        !! Should we log messages to stdout or not
    type(dtfft_platform_t),     protected  :: platform_from_env = PLATFORM_NOT_SET
        !! Platform obtained from environ
    integer(int32),             protected  :: z_slab_from_env = VARIABLE_NOT_SET
        !! Should Z-slab be used if possible
    integer(int32),             protected  :: y_slab_from_env = VARIABLE_NOT_SET
        !! Should Y-slab be used if possible
    integer(int32),             protected  :: n_measure_warmup_iters_from_env = VARIABLE_NOT_SET
        !! Number of warmup iterations for measurements
    integer(int32),             protected  :: n_measure_iters_from_env = VARIABLE_NOT_SET
        !! Number of measurement iterations
    logical,                    protected  :: is_log_enabled = .false.
        !! Should we print additional information during plan creation
    logical,                    protected  :: is_z_slab_enabled = .true.
        !! Should we use Z-slab optimization or not
    logical,                    protected  :: is_y_slab_enabled = .false.
        !! Should we use Y-slab optimization or not
    type(dtfft_platform_t),     protected  :: platform = DTFFT_PLATFORM_HOST
        !! Default platform
    integer(int32),             protected  :: n_measure_warmup_iters = CONF_DTFFT_MEASURE_WARMUP_ITERS
        !! Number of warmup iterations for measurements
    integer(int32),             protected  :: n_measure_iters = CONF_DTFFT_MEASURE_ITERS
        !! Number of measurement iterations

    type(dtfft_backend_t),      protected  :: backend_from_env = BACKEND_NOT_SET
        !! Backend obtained from environ
    type(dtfft_backend_t),      protected  :: reshape_backend_from_env = BACKEND_NOT_SET
        !! Reshape backend obtained from environ
    integer(int32),             protected  :: datatype_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use MPI Datatype backend during autotune or not
    integer(int32),             protected  :: mpi_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use MPI backends during autotune or not
    integer(int32),             protected  :: pipelined_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use pipelined backends during autotune or not
    integer(int32),             protected  :: kernel_autotune_enabled_from_env = VARIABLE_NOT_SET
        !! Should we enable kernel autotune or not
    integer(int32),             protected  :: fourier_reshape_enabled_from_env = VARIABLE_NOT_SET
        !! Should we enable fourier space reshape or not
#ifdef DTFFT_WITH_RMA
    integer(int32),             protected  :: rma_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use RMA backends during autotune or not
#endif
    integer(int32),             protected  :: fused_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use fused backends during autotune or not
    type(dtfft_transpose_mode_t), protected :: transpose_mode_from_env = TRANSPOSE_MODE_NOT_SET
        !! Transpose mode obtained from environ
    type(dtfft_access_mode_t),    protected :: access_mode_from_env = ACCESS_MODE_NOT_SET
        !! Access mode obtained from environ

#ifdef DTFFT_WITH_COMPRESSION
    integer(int32),               protected :: compression_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use compressed backends during autotune or not
#endif

#ifdef DTFFT_WITH_CUDA
    integer(int32),             protected  :: nccl_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use NCCL backends during autotune or not
    integer(int32),             protected  :: nvshmem_enabled_from_env = VARIABLE_NOT_SET
        !! Should we use NVSHMEM backends during autotune or not
    type(dtfft_backend_t),  parameter :: DEFAULT_BACKEND = BACKEND_NOT_SET
        !! Default backend when cuda is enabled
    type(dtfft_stream_t),       protected  :: main_stream = NULL_STREAM
        !! Default dtFFT CUDA stream
    type(dtfft_stream_t),       protected  :: custom_stream = NULL_STREAM
        !! CUDA stream set by the user
    logical,                    protected  :: is_stream_created = .false.
        !! Is the default stream created?
    logical,                    protected  :: is_custom_stream = .false.
        !! Is the custom stream provided by the user?
#else
    type(dtfft_backend_t),  parameter :: DEFAULT_BACKEND = DTFFT_BACKEND_MPI_DATATYPE
        !! Default host backend
#endif
    logical,                    protected  :: is_datatype_enabled = .true.
        !! Should we use MPI Datatype backend or not
    logical,                    protected  :: is_pipelined_enabled = .true.
        !! Should we use pipelined backends or not
    logical,                    protected  :: is_mpi_enabled = .false.
        !! Should we use MPI backends or not
#ifdef DTFFT_WITH_CUDA
    logical,                    protected  :: is_nccl_enabled = .true.
        !! Should we use NCCL backends or not
    logical,                    protected  :: is_nvshmem_enabled = .true.
        !! Should we use NCCL backends or not
#endif
    logical,                    protected  :: is_kernel_autotune_enabled = .false.
        !! Should we use kernel autotune or not
    logical,                    protected  :: is_fourier_reshape_enabled = .false.
        !! Should we use reshape in fourier space or not
    logical,                    protected  :: is_rma_enabled = .true.
        !! Should we use RMA backends or not
    logical,                    protected  :: is_fused_enabled = .true.
        !! Should we use fused backends or not
    type(dtfft_backend_t),      protected  :: backend = DEFAULT_BACKEND
        !! Default backend
    type(dtfft_backend_t),      protected  :: reshape_backend = DEFAULT_BACKEND
        !! Default reshape backend
    type(dtfft_transpose_mode_t), protected :: transpose_mode = DTFFT_TRANSPOSE_MODE_PACK
        !! Default transpose mode
    type(dtfft_access_mode_t),    protected :: access_mode = DTFFT_ACCESS_MODE_WRITE
        !! Default access mode

#ifdef DTFFT_WITH_COMPRESSION
    logical,                          protected :: is_compression_enabled = .false.
        !! Should we use compressed backends or not during autotuning
    type(dtfft_compression_config_t), protected :: config_transpose = DEFAULT_COMPRESSION_CONFIG
        !! Configuration for compression during transpositions
    type(dtfft_compression_config_t), protected :: config_reshape = DEFAULT_COMPRESSION_CONFIG
        !! Configuration for compression during reshape operations
#endif

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
        !! Number of warmup iterations to execute during backend and kernel autotuning when effort level is `DTFFT_MEASURE` or higher.
        !!
        !! Default is 2.
        integer(c_int32_t)        :: n_measure_iters
        !! Number of iterations to execute during backend and kernel autotuning when effort level is `DTFFT_MEASURE` or higher.
        !!
        !! Default is 5.
#ifdef DTFFT_WITH_CUDA
        type(dtfft_platform_t)    :: platform
        !! Selects platform to execute plan.
        !!
        !! Default is `DTFFT_PLATFORM_HOST`.
        !!
        !! This option is only available when dtFFT is built with device support.
        !! Even when dtFFT is built with device support, it does not necessarily mean that all plans must be device-related.
        !! This enables a single library installation to support both host and CUDA plans.

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
        !! Default for HOST platform is `DTFFT_BACKEND_MPI_DATATYPE`.
        !!
        !! Default for CUDA platform is `DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise `DTFFT_BACKEND_MPI_P2P`.

        type(dtfft_backend_t)     :: reshape_backend
        !! Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
        !!
        !! Default for HOST platform is `DTFFT_BACKEND_MPI_DATATYPE`.
        !!
        !! Default for CUDA platform is `DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise `DTFFT_BACKEND_MPI_P2P`.

        logical(c_bool)           :: enable_datatype_backend
        !! Should `DTFFT_BACKEND_MPI_DATATYPE` be considered for autotuning when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
        !!
        !! This option only works when `platform` is `DTFFT_PLATFORM_HOST`.
        !! When `platform` is `DTFFT_PLATFORM_CUDA`, `DTFFT_BACKEND_MPI_DATATYPE` is always disabled during autotuning.

        logical(c_bool)           :: enable_mpi_backends
        !! Should MPI Backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
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
        !! Should pipelined backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
        logical(c_bool)           :: enable_rma_backends
        !! Should RMA backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
        logical(c_bool)           :: enable_fused_backends
        !! Should fused backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
#ifdef DTFFT_WITH_CUDA
        logical(c_bool)           :: enable_nccl_backends
        !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
        !!
        !! This option is only defined when dtFFT is built with CUDA support.

        logical(c_bool)           :: enable_nvshmem_backends
        !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is true.
        !!
        !! This option is only defined when dtFFT is built with CUDA support.
#endif
        logical(c_bool)            :: enable_kernel_autotune
        !! Should dtFFT try to optimize kernel launch parameters during plan creation when `effort` is below `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is false.
        !!
        !! Kernel optimization is always enabled for `DTFFT_EXHAUSTIVE` effort level.
        !! Setting this option to true enables kernel optimization for lower effort levels (`DTFFT_ESTIMATE`, `DTFFT_MEASURE`, `DTFFT_PATIENT`).
        !! This may increase plan creation time but can improve runtime performance.
        !! Since kernel optimization is performed without data transfers, the time increase is usually minimal.

        logical(c_bool)             :: enable_fourier_reshape
        !! Should dtFFT execute reshapes from pencils to bricks and vice versa in Fourier space during calls to `execute`.
        !!
        !! Default is false.
        !!
        !! When enabled, data will be in brick layout in Fourier space, which may be useful for certain operations
        !! between forward and backward transforms. However, this requires additional data transpositions
        !! and will reduce overall FFT performance.

        type(dtfft_transpose_mode_t) :: transpose_mode
        !! Specifies at which stage the local transposition is performed during global exchange.
        !!
        !! Default is `DTFFT_TRANSPOSE_MODE_PACK`.
        !!
        !! It affects only Generic backends that perform explicit packing/unpacking.
        !! This option only takes effect when `effort` is less than `DTFFT_EXHAUSTIVE`.
        !!
        !! For `DTFFT_EXHAUSTIVE` effort level, dtFFT will always choose the best transpose mode based on internal benchmarking.

        type(dtfft_access_mode_t) :: access_mode
        !! Specifies the memory access pattern (optimization target) for local transposition.
        !!
        !! Default is `DTFFT_ACCESS_MODE_WRITE`.
        !!
        !! This setting applies only to Host (CPU) Generic backends.
        !!
        !! This option allows user to force specific access mode (`DTFFT_ACCESS_MODE_WRITE` or `DTFFT_ACCESS_MODE_READ`) when autotuning is disabled.
        !! When autotuning is enabled (e.g. `effort` is `DTFFT_EXHAUSTIVE`), this option is ignored and best access mode is selected automatically.
#ifdef DTFFT_WITH_COMPRESSION
        logical(c_bool)           :: enable_compressed_backends
        !! Should compressed backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
        !!
        !! Default is false.
        !!
        !! Only fixed-rate compression can be used during autotuning, since it provides predictable performance characteristics and does not require data-dependent decisions at runtime.
        !! To enable compressed backends during autotuning, set this option to true, set compression type to `DTFFT_COMPRESSION_MODE_FIXED_RATE` and provide desired compression rate.

        type(dtfft_compression_config_t)  :: compression_config_transpose
        !! Options for compression approach during transpositions

        type(dtfft_compression_config_t)  :: compression_config_reshape
        !! Options for compression approach during reshape operations
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

#if defined(DTFFT_WITH_CUDA) && defined(DTFFT_WITH_MOCK_ENABLED)
    if ( get_conf_platform() == DTFFT_PLATFORM_CUDA ) then
      WRITE_WARN("This is mock version of CUDA platform")
    endif
#endif
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

    backend_from_env = get_backend_from_env("BACKEND")
    reshape_backend_from_env = get_backend_from_env("RESHAPE_BACKEND")

    datatype_enabled_from_env = get_env("ENABLE_MPI_DT", VARIABLE_NOT_SET, valid_values=[0, 1])
    mpi_enabled_from_env = get_env("ENABLE_MPI", VARIABLE_NOT_SET, valid_values=[0, 1])
    pipelined_enabled_from_env = get_env("ENABLE_PIPE", VARIABLE_NOT_SET, valid_values=[0, 1])
#ifdef DTFFT_WITH_RMA
    rma_enabled_from_env = get_env("ENABLE_RMA", VARIABLE_NOT_SET, valid_values=[0, 1])
#endif
    fused_enabled_from_env = get_env("ENABLE_FUSED", VARIABLE_NOT_SET, valid_values=[0, 1])
#ifdef DTFFT_WITH_CUDA
    nccl_enabled_from_env = get_env("ENABLE_NCCL", VARIABLE_NOT_SET, valid_values=[0, 1])
    nvshmem_enabled_from_env = get_env("ENABLE_NVSHMEM", VARIABLE_NOT_SET, valid_values=[0, 1])
#endif
    kernel_autotune_enabled_from_env = get_env("ENABLE_KERNEL_AUTOTUNE", VARIABLE_NOT_SET, valid_values=[0, 1])
    fourier_reshape_enabled_from_env = get_env("ENABLE_FOURIER_RESHAPE", VARIABLE_NOT_SET, valid_values=[0, 1])
#ifdef DTFFT_WITH_COMPRESSION
    compression_enabled_from_env = get_env("ENABLE_COMPRESSED", VARIABLE_NOT_SET, valid_values=[0, 1])
#endif

    block
      type(string), allocatable :: transpose_modes(:)
      type(string) :: t_mode_env

      allocate( transpose_modes(2) )
      transpose_modes(1) = string("pack")
      transpose_modes(2) = string("unpack")

      t_mode_env = get_env("TRANSPOSE_MODE", "undefined", transpose_modes)
      select case ( t_mode_env%raw )
      case ( "undefined" )
        transpose_mode_from_env = TRANSPOSE_MODE_NOT_SET
      case ( "pack" )
        transpose_mode_from_env = DTFFT_TRANSPOSE_MODE_PACK
      case ( "unpack" )
        transpose_mode_from_env = DTFFT_TRANSPOSE_MODE_UNPACK
      endselect

      call t_mode_env%destroy()
      call destroy_strings(transpose_modes)
    endblock

    block
      type(string), allocatable :: access_modes(:)
      type(string) :: access_mode_env

      allocate( access_modes(2) )
      access_modes(1) = string("write")
      access_modes(2) = string("read")

      access_mode_env = get_env("ACCESS_MODE", "undefined", access_modes)
      select case ( access_mode_env%raw )
      case ( "undefined" )
        access_mode_from_env = ACCESS_MODE_NOT_SET
      case ( "write" )
        access_mode_from_env = DTFFT_ACCESS_MODE_WRITE
      case ( "read" )
        access_mode_from_env = DTFFT_ACCESS_MODE_READ
      endselect

      call access_mode_env%destroy()
      call destroy_strings(access_modes)
    endblock
  end subroutine init_environment

  type(dtfft_backend_t) function get_backend_from_env(name)
  !! Returns backend or reshape backend obtained from environment variable
    character(len=*), intent(in) :: name !! Name of the environment variable
    type(string), allocatable :: backends(:)
    type(string) :: bcknd_env

      allocate( backends(17) )
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
      backends(11) = string("mpi_p2p_sched")
      backends(12) = string("mpi_p2p_fused")
      backends(13) = string("mpi_p2p_compressed")
      backends(14) = string("mpi_rma_fused")
      backends(15) = string("mpi_rma_compressed")
      backends(16) = string("nccl_compressed")
      backends(17) = string("adaptive")

      bcknd_env = get_env(name, "undefined", backends)

      select case ( bcknd_env%raw )
      case ( "undefined" )
        get_backend_from_env = BACKEND_NOT_SET
      case ( "mpi_dt" )
        get_backend_from_env = DTFFT_BACKEND_MPI_DATATYPE
      case ( "mpi_p2p" )
        get_backend_from_env = DTFFT_BACKEND_MPI_P2P
      case ( "mpi_a2a" )
        get_backend_from_env = DTFFT_BACKEND_MPI_A2A
      case ( "mpi_p2p_pipe" )
        get_backend_from_env = DTFFT_BACKEND_MPI_P2P_PIPELINED
      case ( "nccl" )
        get_backend_from_env = DTFFT_BACKEND_NCCL
      case ( "nccl_pipe" )
        get_backend_from_env = DTFFT_BACKEND_NCCL_PIPELINED
      case ( "cufftmp" )
        get_backend_from_env = DTFFT_BACKEND_CUFFTMP
      case ( "cufftmp_pipe")
        get_backend_from_env = DTFFT_BACKEND_CUFFTMP_PIPELINED
      case ( "mpi_rma" )
        get_backend_from_env = DTFFT_BACKEND_MPI_RMA
      case ( "mpi_rma_pipe" )
        get_backend_from_env = DTFFT_BACKEND_MPI_RMA_PIPELINED
      case ( "mpi_p2p_sched" )
        get_backend_from_env = DTFFT_BACKEND_MPI_P2P_SCHEDULED
      case ( "mpi_p2p_fused" )
        get_backend_from_env = DTFFT_BACKEND_MPI_P2P_FUSED
      case ( "mpi_p2p_compressed" )
        get_backend_from_env = DTFFT_BACKEND_MPI_P2P_COMPRESSED
      case ( "mpi_rma_fused" )
        get_backend_from_env = DTFFT_BACKEND_MPI_RMA_FUSED
      case ( "mpi_rma_compressed" )
        get_backend_from_env = DTFFT_BACKEND_MPI_RMA_COMPRESSED
      case ( "nccl_compressed" )
        get_backend_from_env = DTFFT_BACKEND_NCCL_COMPRESSED
      case ( "adaptive" )
        get_backend_from_env = DTFFT_BACKEND_ADAPTIVE
      endselect

      if ( get_backend_from_env /= BACKEND_NOT_SET .and. .not.is_valid_backend(get_backend_from_env) ) then
        WRITE_ERROR("Backend '"//bcknd_env%raw//"' is not available in this build.")
        WRITE_ERROR("Environment variable 'DTFFT_"//name//"' has been ignored")
        get_backend_from_env = BACKEND_NOT_SET
      endif

      call bcknd_env%destroy()
      call destroy_strings(backends)
  end function get_backend_from_env

  pure subroutine dtfft_create_config(config) bind(C, name="dtfft_create_config_c")
  !! Creates a new configuration and sets default values.
  !!
  !! C interface
    type(dtfft_config_t), intent(out) :: config !! Configuration to create
    config = dtfft_config_t()
  end subroutine dtfft_create_config

  pure function config_constructor(                                       &
    enable_log, enable_z_slab, enable_y_slab,                             &
    n_measure_warmup_iters, n_measure_iters,                              &
#ifdef DTFFT_WITH_CUDA
    platform, stream,                                                     &
#endif
    backend, reshape_backend, enable_datatype_backend,                    &
    enable_mpi_backends, enable_pipelined_backends,                       &
    enable_rma_backends, enable_fused_backends,                           &
#ifdef DTFFT_WITH_CUDA
    enable_nccl_backends, enable_nvshmem_backends,                        &
#endif
    enable_kernel_autotune, enable_fourier_reshape,                       &
    transpose_mode, access_mode                                           &
#ifdef DTFFT_WITH_COMPRESSION
    ,enable_compressed_backends                                           &
    ,compression_config_transpose, compression_config_reshape             &
#endif
  ) result(config)
  !! Creates a new configuration
    logical,                optional, intent(in)  :: enable_log
      !! Should dtFFT print additional information during plan creation or not.
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
    type(dtfft_backend_t),  optional, intent(in)  :: reshape_backend
      !! Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
    logical,                optional, intent(in)  :: enable_datatype_backend
      !! Should `DTFFT_BACKEND_MPI_DATATYPE` be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_mpi_backends
      !! Should MPI Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_pipelined_backends
      !! Should pipelined backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_rma_backends
      !! Should RMA backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_fused_backends
      !! Should fused backends be enabled when `effort` is `DTFFT_PATIENT` or not.
#ifdef DTFFT_WITH_CUDA
    logical,                optional, intent(in)  :: enable_nccl_backends
      !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_nvshmem_backends
      !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
#endif
    logical,                optional, intent(in)  :: enable_kernel_autotune
      !! Should dtFFT try to autotune transpose/packing/unpacking kernels size during autotune process or not.
    logical,                optional, intent(in)  :: enable_fourier_reshape
      !! Should dtFFT execute reshapes from pencils to bricks and vice versa in Fourier space during calls to `execute` or not.
    type(dtfft_transpose_mode_t), optional, intent(in) :: transpose_mode
      !! Specifies at which stage the local transposition is performed during global exchange.
    type(dtfft_access_mode_t),    optional, intent(in) :: access_mode
      !! Specifies the memory access pattern (optimization target) for local transposition in Generic backends.
#ifdef DTFFT_WITH_COMPRESSION
    logical,                          optional, intent(in)  :: enable_compressed_backends
      !! Should compressed backends be enabled when `effort` is `DTFFT_PATIENT` or `DTFFT_EXHAUSTIVE`.
    type(dtfft_compression_config_t), optional, intent(in)  :: compression_config_transpose
      !! Options for compression approach during transpositions
    type(dtfft_compression_config_t), optional, intent(in)  :: compression_config_reshape
      !! Options for compression approach during transpositions
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
    config%reshape_backend = DEFAULT_BACKEND;   if ( present(reshape_backend) ) config%reshape_backend = reshape_backend
    config%enable_datatype_backend = .true.;    if ( present(enable_datatype_backend) ) config%enable_datatype_backend = enable_datatype_backend
    config%enable_mpi_backends = .false.;       if ( present(enable_mpi_backends) ) config%enable_mpi_backends = enable_mpi_backends
    config%enable_pipelined_backends = .true.;  if ( present(enable_pipelined_backends) ) config%enable_pipelined_backends = enable_pipelined_backends
    config%enable_rma_backends = .true.;        if ( present(enable_rma_backends) ) config%enable_rma_backends = enable_rma_backends
    config%enable_fused_backends = .true.;      if ( present(enable_fused_backends) ) config%enable_fused_backends = enable_fused_backends
#ifdef DTFFT_WITH_CUDA
    config%enable_nccl_backends = .true.;       if ( present(enable_nccl_backends) ) config%enable_nccl_backends = enable_nccl_backends
    config%enable_nvshmem_backends = .true.;    if ( present(enable_nvshmem_backends) ) config%enable_nvshmem_backends = enable_nvshmem_backends
#endif
    config%enable_kernel_autotune = .false.;    if ( present(enable_kernel_autotune) ) config%enable_kernel_autotune = enable_kernel_autotune
    config%enable_fourier_reshape = .false.;    if ( present(enable_fourier_reshape) ) config%enable_fourier_reshape = enable_fourier_reshape
    config%transpose_mode = DTFFT_TRANSPOSE_MODE_PACK; if ( present(transpose_mode) ) config%transpose_mode = transpose_mode
    config%access_mode = DTFFT_ACCESS_MODE_WRITE; if ( present(access_mode) ) config%access_mode = access_mode
#ifdef DTFFT_WITH_COMPRESSION
    config%enable_compressed_backends = .false.;                      if ( present(enable_compressed_backends) ) config%enable_compressed_backends = enable_compressed_backends
    config%compression_config_transpose = DEFAULT_COMPRESSION_CONFIG; if ( present(compression_config_transpose) ) config%compression_config_transpose = compression_config_transpose
    config%compression_config_reshape = DEFAULT_COMPRESSION_CONFIG;   if ( present(compression_config_reshape) ) config%compression_config_reshape = compression_config_reshape
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
    is_rma_enabled = config%enable_rma_backends
    is_fused_enabled = config%enable_fused_backends

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
#endif

    is_kernel_autotune_enabled = config%enable_kernel_autotune
    is_fourier_reshape_enabled = config%enable_fourier_reshape
    if ( .not. is_valid_transpose_mode(config%transpose_mode) ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_TRANSPOSE_MODE
      return
    endif
    transpose_mode = config%transpose_mode

    if ( .not. is_valid_access_mode(config%access_mode) ) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_ACCESS_MODE
      return
    endif
    access_mode = config%access_mode

    if ( config%backend /= BACKEND_NOT_SET .and. .not.is_valid_backend(config%backend)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_BACKEND
      return
    endif
    if ( config%reshape_backend /= BACKEND_NOT_SET .and. .not.is_valid_backend(config%reshape_backend)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_INVALID_BACKEND
      return
    endif
    backend = get_correct_backend(config%backend)
    reshape_backend = get_correct_backend(config%reshape_backend)
#ifdef DTFFT_WITH_COMPRESSION
    is_compression_enabled = config%enable_compressed_backends
    config_transpose = config%compression_config_transpose
    config_reshape = config%compression_config_reshape
#endif
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
      CUDA_CALL( cudaStreamCreate(main_stream) )
      is_stream_created = .true.
    endif
    stream = main_stream
  end function get_conf_stream

  subroutine destroy_stream
  !! Destroy the default stream if it was created
    if ( is_stream_created ) then
      CUDA_CALL( cudaStreamDestroy(main_stream) )
      is_stream_created = .false.
    endif
  end subroutine destroy_stream
#endif

  elemental type(dtfft_backend_t) function get_conf_backend()
  !! Returns backend set by the user or default one
    get_conf_backend = get_correct_backend(backend)
    if ( backend_from_env /= BACKEND_NOT_SET) get_conf_backend = backend_from_env
  end function get_conf_backend

  elemental type(dtfft_backend_t) function get_conf_reshape_backend()
  !! Returns reshape backend set by the user or default one
    get_conf_reshape_backend = get_correct_backend(reshape_backend)
    if ( reshape_backend_from_env /= BACKEND_NOT_SET) get_conf_reshape_backend = reshape_backend_from_env
  end function get_conf_reshape_backend

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

  elemental function get_conf_rma_enabled() result(bool)
  !! Whether RMA backends are enabled or not
    logical :: bool   !! Result flag
#ifdef DTFFT_WITH_RMA
    bool = get_conf_internal(is_rma_enabled, rma_enabled_from_env)
#else
    bool = .false.
#endif
  end function get_conf_rma_enabled

  elemental function get_conf_fused_enabled() result(bool)
  !! Whether fused backends are enabled or not
    logical :: bool   !! Result flag
    bool = get_conf_internal(is_fused_enabled, fused_enabled_from_env)
  end function get_conf_fused_enabled

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
#endif

  elemental function get_conf_kernel_autotune_enabled() result(bool)
  !! Whether kernel optimization is enabled or not
    logical :: bool  !! Result flag
    bool = get_conf_internal(is_kernel_autotune_enabled, kernel_autotune_enabled_from_env)
  end function get_conf_kernel_autotune_enabled

  elemental function get_conf_fourier_reshape_enabled() result(bool)
  !! Whether reshape in Fourier space is enabled or not
    logical :: bool  !! Result flag
    bool = get_conf_internal(is_fourier_reshape_enabled, fourier_reshape_enabled_from_env)
  end function get_conf_fourier_reshape_enabled

  elemental function get_conf_transpose_mode() result(mode)
  !! Returns transpose mode set by the user or default one
    type(dtfft_transpose_mode_t) :: mode  !! Result
    mode = transpose_mode
    if ( transpose_mode_from_env /= TRANSPOSE_MODE_NOT_SET ) mode = transpose_mode_from_env
  end function get_conf_transpose_mode

  elemental function get_conf_access_mode() result(mode)
  !! Returns access mode set by the user or default one
    type(dtfft_access_mode_t) :: mode  !! Result
    mode = access_mode
    if ( access_mode_from_env /= ACCESS_MODE_NOT_SET ) mode = access_mode_from_env
  end function get_conf_access_mode

  elemental function get_conf_compression_enabled() result(bool)
  !! Whether compression is enabled or not
    logical :: bool  !! Result flag
#ifdef DTFFT_WITH_COMPRESSION
    bool = get_conf_internal(is_compression_enabled, compression_enabled_from_env)
#else
    bool = .false.
#endif
  end function get_conf_compression_enabled

#ifdef DTFFT_WITH_COMPRESSION
  elemental function get_conf_transpose() result(config)
  !! Returns compression config for transposes
    type(dtfft_compression_config_t) :: config  !! Result
    config = config_transpose
  end function get_conf_transpose


  elemental function get_conf_reshape() result(config)
  !! Returns compression config for reshapes
    type(dtfft_compression_config_t) :: config  !! Result

    config = config_reshape
  end function get_conf_reshape
#endif

end module dtfft_config