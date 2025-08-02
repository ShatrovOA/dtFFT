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
#include "dtfft_config.h"
module dtfft_config
!! This module handles additional configuration ``dtFFT``, provided by [[dtfft_config_t]]
!! or environment variables
use iso_c_binding
use iso_fortran_env
use dtfft_parameters
use dtfft_errors
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
#include "dtfft_cuda.h"
#include "dtfft_mpi.h"
implicit none
private
public :: dtfft_config_t
public :: dtfft_create_config, dtfft_set_config
public :: get_z_slab
public :: get_user_platform
#ifdef DTFFT_WITH_CUDA
public :: get_user_stream
public :: destroy_stream
public :: get_user_gpu_backend
public :: get_mpi_enabled, get_nvshmem_enabled, get_nccl_enabled, get_pipelined_enabled
#endif


  logical,                    save  :: is_z_slab_enabled = .true.
    !! Should we use z-slab decomposition or not
  type(dtfft_platform_t),     save  :: platform = DTFFT_PLATFORM_HOST
    !! Default platform

#ifdef DTFFT_WITH_CUDA
# ifdef DTFFT_WITH_NCCL
  type(dtfft_backend_t),  parameter :: DEFAULT_GPU_BACKEND = DTFFT_BACKEND_NCCL
# else
  type(dtfft_backend_t),  parameter :: DEFAULT_GPU_BACKEND = DTFFT_BACKEND_MPI_P2P
# endif
    !! Default GPU backend 

  type(dtfft_stream_t),       save  :: main_stream
    !! Default dtFFT CUDA stream
  type(dtfft_stream_t),       save  :: custom_stream
    !! CUDA stream set by the user
  logical,                    save  :: is_stream_created = .false.
    !! Is the default stream created?
  logical,                    save  :: is_custom_stream = .false.
    !! Is the custom stream provided by the user?
  logical,                    save  :: is_pipelined_enabled = .true.
    !! Should we use pipelined backends or not
  logical,                    save  :: is_mpi_enabled = .false.
    !! Should we use MPI backends or not
  logical,                    save  :: is_nccl_enabled = .true.
    !! Should we use NCCL backends or not
  logical,                    save  :: is_nvshmem_enabled = .true.
    !! Should we use NCCL backends or not
  type(dtfft_backend_t),      save  :: backend = DEFAULT_GPU_BACKEND
    !! Default GPU backend
#endif


  type, bind(C) :: dtfft_config_t
  !! Type that can be used to set additional configuration parameters to ``dtFFT``
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

    type(dtfft_backend_t)     :: backend
      !! Backend that will be used by dtFFT when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
      !!
      !! Default is `DTFFT_GPU_BACKEND_NCCL` if NCCL is enabled, otherwise `DTFFT_BACKEND_MPI_P2P`.

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
  !! Interface to create a new configuration
    module procedure config_constructor
  end interface dtfft_config_t

contains

  pure subroutine dtfft_create_config(config) bind(C, name="dtfft_create_config_c")
  !! Creates a new configuration with default values. 
  !!
  !! C interface
    type(dtfft_config_t), intent(out) :: config !! Configuration to create
    config = dtfft_config_t()
  end subroutine dtfft_create_config

#ifdef DTFFT_WITH_CUDA
  pure function config_constructor(                                       &
    enable_z_slab, platform, stream, backend,                             &
    enable_mpi_backends, enable_pipelined_backends,                       &
    enable_nccl_backends, enable_nvshmem_backends) result(config)
#else
  pure function config_constructor(enable_z_slab) result(config)
#endif
  !! Creates a new configuration
    logical,                optional, intent(in)  :: enable_z_slab
      !! Should dtFFT use Z-slab optimization or not.
#ifdef DTFFT_WITH_CUDA
    type(dtfft_platform_t), optional, intent(in)  :: platform
      !! Selects platform to execute plan.
    type(dtfft_stream_t),   optional, intent(in)  :: stream
      !! Main CUDA stream that will be used in dtFFT.
    type(dtfft_backend_t),  optional, intent(in)  :: backend
      !! Backend that will be used by dtFFT when `effort` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
    logical,                optional, intent(in)  :: enable_mpi_backends
      !! Should MPI GPU Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_pipelined_backends
      !! Should pipelined GPU backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_nccl_backends
      !! Should NCCL Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
    logical,                optional, intent(in)  :: enable_nvshmem_backends
      !! Should NVSHMEM Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
#endif
    type(dtfft_config_t) :: config
      !! Constructed `dtFFT` config ready to be set by call to [[dtfft_set_config]]

    config%enable_z_slab = .true.;              if ( present(enable_z_slab) ) config%enable_z_slab = enable_z_slab
#ifdef DTFFT_WITH_CUDA
    config%platform = DTFFT_PLATFORM_HOST;      if ( present(platform) ) config%platform = platform
    config%stream = NULL_STREAM;                if ( present(stream) ) config%stream = stream
    config%backend = DEFAULT_GPU_BACKEND;       if ( present(backend) ) config%backend = backend
    config%enable_mpi_backends = .false.;       if ( present(enable_mpi_backends) ) config%enable_mpi_backends = enable_mpi_backends
    config%enable_pipelined_backends = .true.;  if ( present(enable_pipelined_backends) ) config%enable_pipelined_backends = enable_pipelined_backends
    config%enable_nccl_backends = .true.;       if ( present(enable_nccl_backends) ) config%enable_nccl_backends = enable_nccl_backends
    config%enable_nvshmem_backends = .true.;    if ( present(enable_nvshmem_backends) ) config%enable_nvshmem_backends = enable_nvshmem_backends
#endif
  end function config_constructor

  subroutine dtfft_set_config(config, error_code)
  !! Sets configuration parameters
    type(dtfft_config_t),     intent(in)  :: config     !! Configuration to set
    integer(int32), optional, intent(out) :: error_code !! Error code

    is_z_slab_enabled = config%enable_z_slab

#ifdef DTFFT_WITH_CUDA
    if (.not.is_valid_gpu_backend(config%backend)) then
      if ( present( error_code ) ) error_code = DTFFT_ERROR_GPU_INVALID_BACKEND
      return
    endif
    backend = config%backend

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

    is_mpi_enabled = config%enable_mpi_backends
    is_pipelined_enabled = config%enable_pipelined_backends
    is_nccl_enabled = config%enable_nccl_backends
    is_nvshmem_enabled = config%enable_nvshmem_backends
#endif
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine dtfft_set_config

  pure logical function get_z_slab()
  !! Whether Z-slab optimization is enabled or not
    get_z_slab = is_z_slab_enabled
    if ( get_z_slab_from_env() /= VARIABLE_NOT_SET ) get_z_slab = get_z_slab_from_env() == 1
  end function get_z_slab

  pure type(dtfft_platform_t) function get_user_platform()
  !! Returns platform set by the user or default one
    get_user_platform = platform
    if ( get_platform_from_env() /= PLATFORM_NOT_SET ) get_user_platform = get_platform_from_env()
  end function get_user_platform

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

  pure type(dtfft_backend_t) function get_user_gpu_backend()
  !! Returns GPU backend set by the user or default one
    get_user_gpu_backend = backend
    if ( get_backend_from_env() /= BACKEND_NOT_SET) get_user_gpu_backend = get_backend_from_env()
  end function get_user_gpu_backend

  pure logical function get_pipelined_enabled()
  !! Whether pipelined backends are enabled or not
    get_pipelined_enabled = is_pipelined_enabled
    if ( get_pipe_enabled_from_env() /= VARIABLE_NOT_SET ) get_pipelined_enabled = get_pipe_enabled_from_env() == 1
  end function get_pipelined_enabled

  pure logical function get_mpi_enabled()
  !! Whether MPI backends are enabled or not
#if !defined(DTFFT_WITH_NCCL) && !defined(DTFFT_WITH_NVSHMEM)
    get_mpi_enabled = .true.
    ! Should not be .false. if only MPI backends are possible
#else
    get_mpi_enabled = is_mpi_enabled
    if ( get_mpi_enabled_from_env() /= VARIABLE_NOT_SET ) get_mpi_enabled = get_mpi_enabled_from_env() == 1
#endif
  end function get_mpi_enabled

  pure logical function get_nccl_enabled()
  !! Whether NCCL backends are enabled or not
#ifdef DTFFT_WITH_NCCL
    get_nccl_enabled = is_nccl_enabled
    if ( get_nccl_enabled_from_env() /= VARIABLE_NOT_SET ) get_nccl_enabled = get_nccl_enabled_from_env() == 1
#else
    get_nccl_enabled = .false.
#endif
  end function get_nccl_enabled

  pure logical function get_nvshmem_enabled()
  !! Whether nvshmem backends are enabled or not
#ifdef DTFFT_WITH_NVSHMEM
    get_nvshmem_enabled = is_nvshmem_enabled
    if ( get_nvshmem_enabled_from_env() /= VARIABLE_NOT_SET ) get_nvshmem_enabled = get_nvshmem_enabled_from_env() == 1
#else
    get_nvshmem_enabled = .false.
#endif
  end function get_nvshmem_enabled
#endif
end module dtfft_config