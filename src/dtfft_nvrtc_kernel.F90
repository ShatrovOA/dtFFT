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
module dtfft_nvrtc_kernel
!! This module describes NVRTC Kernel class [[nvrtc_kernel]]
use iso_c_binding
use iso_fortran_env
use dtfft_config
use dtfft_interface_cuda
use dtfft_interface_cuda_runtime
use dtfft_interface_nvrtc
use dtfft_nvrtc_block_optimizer,  only: N_CANDIDATES,                       &
                                        kernel_config,                      &
                                        generate_candidates,                &
                                        evaluate_analytical_performance,    &
                                        sort_candidates_by_score
use dtfft_nvrtc_kernel_cache,     only: cache
use dtfft_nvrtc_kernel_generator
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_profile.h"
#include "_dtfft_private.h"
implicit none
private
public :: nvrtc_kernel
public :: get_kernel, get_kernel_args

  integer(int32),   parameter, public :: DEF_TILE_SIZE = 32
    !! Default tile size
  character(len=*), parameter :: DEFAULT_KERNEL_NAME = "dtfft_kernel"
    !! Basic kernel name
  integer(int32),   parameter :: TARGET_THREADS_PER_BLOCK = 256
    !! Target number of threads per block for unpacked kernels

  type :: nvrtc_kernel
  !! nvRTC Compiled kernel class
  private
    logical                       :: is_created = .false.     !! Kernel is created flag.
    logical                       :: is_dummy = .false.       !! If kernel should do anything or not.
    type(CUfunction)              :: cuda_kernel              !! Pointer to CUDA kernel.
    type(dim3)                    :: blocks                   !! Grid of blocks.
    type(dim3)                    :: threads                  !! Thread block.
    type(kernel_type_t)           :: kernel_type              !! Type of kernel to execute.
    type(kernelArgs)              :: kernelParams             !! Kernel arguments.
    integer(int32),   allocatable :: pointers(:,:)            !! Optional pointers that hold info about counts and displacements
                                                              !! in ``KERNEL_UNPACK_PIPELINED`` kernel.
    type(c_ptr)                   :: device_pointers(3)       !! Device pointers for kernel arguments.
    logical                       :: has_device_pointers      !! Flag indicating if device pointers are present
    integer(int64)                :: copy_bytes               !! Number of bytes to copy for `KERNEL_UNPACK_SIMPLE_COPY` kernel
  contains
  private
    procedure,  pass(self), public  :: create                 !! Creates kernel
    procedure,  pass(self), public  :: execute                !! Executes kernel
    procedure,  pass(self), public  :: destroy                !! Destroys kernel
  end type nvrtc_kernel

contains

  subroutine create(self, comm, dims, effort, base_storage, transpose_type, kernel_type, pointers, force_effort)
  !! Creates kernel
    class(nvrtc_kernel),      intent(inout) :: self               !! nvRTC Compiled kernel class
    TYPE_MPI_COMM,            intent(in)    :: comm               !! MPI Communicator
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    type(dtfft_effort_t),     intent(in)    :: effort             !! Effort level for generating transpose kernels
    integer(int64),           intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(dtfft_transpose_t),  intent(in)    :: transpose_type     !! Type of transposition to perform
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel to build
    integer(int32), optional, intent(in)    :: pointers(:,:)      !! Optional pointers to unpack kernels
    logical,        optional, intent(in)    :: force_effort       !! Should effort be forced or not
    type(device_props)                      :: props              !! GPU architecture properties
    integer(int32)                          :: device_id          !! Device ID
    ! integer(int32) :: mpi_ierr

    call self%destroy()

    if ( any(dims == 0) ) then
      self%is_created = .true.
      self%is_dummy = .true.
      return
    endif
    self%is_dummy = .false.
    self%kernel_type = kernel_type

    if ( kernel_type == KERNEL_DUMMY ) then
      self%is_created = .true.
      self%is_dummy = .true.
      return
    endif

    if ( kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) then
      self%is_created = .true.
      self%copy_bytes = base_storage * product(dims)
      return
    endif

    self%has_device_pointers = .false.
    if ( any([kernel_type == [KERNEL_TRANSPOSE_PACKED, KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED, KERNEL_UNPACK_PARTIAL]]) ) then
      if ( .not. present(pointers) ) INTERNAL_ERROR("Pointer required")

      if( kernel_type == KERNEL_UNPACK_PIPELINED ) then
        allocate( self%pointers, source=pointers )
      else
        block
          integer(int32) :: i
          do i = 1, size(self%device_pointers)
            call create_device_pointer(self%device_pointers(i), pointers(i, :))
          enddo
          self%has_device_pointers = .true.
        endblock
      endif
    endif

    CUDA_CALL( "cudaGetDevice", cudaGetDevice(device_id) )
    call get_device_props(device_id, props)
    call get_kernel(comm, dims, transpose_type, kernel_type, effort, base_storage, props, self%device_pointers, self%blocks, self%threads, self%cuda_kernel, force_effort=force_effort)
    call get_kernel_args(comm, dims, transpose_type, kernel_type, self%threads%y, self%device_pointers, self%kernelParams)

    self%is_created = .true.
  end subroutine create

  subroutine execute(self, in, out, stream, source)
  !! Executes kernel on stream
    class(nvrtc_kernel),          intent(inout) :: self               !! nvRTC Compiled kernel class
    real(real32),    target,      intent(in)    :: in(:)              !! Source pointer
    real(real32),    target,      intent(in)    :: out(:)             !! Target pointer
    type(dtfft_stream_t),         intent(in)    :: stream             !! CUDA Stream
    integer(int32),   optional,   intent(in)    :: source             !! Source rank for pipelined unpacking

    if ( self%is_dummy ) return
    if ( .not. self%is_created ) INTERNAL_ERROR("`execute` called while plan not created")

    if ( self%kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) then
      CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(c_loc(out), c_loc(in), self%copy_bytes, cudaMemcpyDeviceToDevice, stream) )
#ifdef DTFFT_DEBUG
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
      return
    endif

    if ( self%kernel_type == KERNEL_UNPACK_PIPELINED ) then
      if ( .not. present(source) ) INTERNAL_ERROR("Source is not passed")
      self%kernelParams%ints(1:5) = self%pointers(1:5, source)
      call get_contiguous_execution_blocks(self%pointers(4, source), self%blocks, self%threads)
    endif
    CUDA_CALL( "cuLaunchKernel", cuLaunchKernel(self%cuda_kernel, c_loc(in), c_loc(out), self%blocks, self%threads, stream, self%kernelParams) )
#ifdef DTFFT_DEBUG
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys kernel
    class(nvrtc_kernel),          intent(inout) :: self               !! nvRTC Compiled kernel class
    integer(int32)  :: i  !! Counter

    if ( .not. self%is_created ) return
    if ( self%is_dummy .or. self%kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) return

    call cache%remove(self%cuda_kernel)

    if ( self%has_device_pointers ) then
      do i = 1, size(self%device_pointers)
        CUDA_CALL( "cudaFree", cudaFree(self%device_pointers(i)) )
      enddo
    endif
    if ( allocated(self%pointers) ) deallocate(self%pointers)
    self%has_device_pointers = .false.
    self%is_created = .false.
  end subroutine destroy

  subroutine get_contiguous_execution_blocks(size, blocks, threads)
  !! Gets the number of blocks and threads for a contiguous execution
    integer(int32), intent(in)  :: size         !! Total amount of iterations required
    type(dim3),     intent(out) :: blocks       !! Grid of blocks.
    type(dim3),     intent(out) :: threads      !! Thread block.
    integer(int32)              :: block_size   !! Number of threads in a block

    if ( size < TARGET_THREADS_PER_BLOCK ) then
      block_size = size
    else
      block_size = TARGET_THREADS_PER_BLOCK
    endif
    threads%x = block_size
    threads%y = 1
    threads%z = 1

    blocks%x = (size + block_size - 1) / block_size
    blocks%y = 1
    blocks%z = 1
  end subroutine get_contiguous_execution_blocks

  subroutine create_device_pointer(ptr, values)
  !! Allocates memory on a device and copies ``values`` to it.
    type(c_ptr),        intent(inout)       :: ptr            !! Device pointer
    integer(c_int),     intent(in), target  :: values(:)      !! Values to copy
    integer(c_size_t)                       :: n_bytes        !! Number of bytes to copy

    n_bytes = size(values) * c_sizeof(c_int)
    CUDA_CALL( "cudaMalloc", cudaMalloc(ptr, n_bytes) )
    CUDA_CALL( "cudaMemcpy", cudaMemcpy(ptr, c_loc(values), n_bytes, cudaMemcpyHostToDevice) )
  end subroutine create_device_pointer

  subroutine get_kernel_args(comm, dims, transpose_type, kernel_type, block_rows, ptrs, params)
  !! Populates kernel arguments based on kernel type
    TYPE_MPI_COMM,            intent(in)    :: comm
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    type(dtfft_transpose_t),  intent(in)    :: transpose_type     !! Type of transposition to perform
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel to build
    integer(int32),           intent(in)    :: block_rows         !! Number of rows in each block
    type(c_ptr),              intent(in)    :: ptrs(3)
    type(kernelArgs),         intent(out)   :: params            !! Kernel arguments
    integer(int32)  :: n_args             !! Number of arguments set by this subroutine
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: mpi_ierr

    call MPI_Comm_rank(comm, comm_rank, mpi_ierr)
    call MPI_Comm_size(comm, comm_size, mpi_ierr)

    params%n_ptrs = 0
    ! params%n_longs = 0
    if ( kernel_type == KERNEL_UNPACK .or. kernel_type == KERNEL_UNPACK_PARTIAL) then
      params%n_ints = 3
      params%ints(1) = product(dims)
      if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        params%ints(2) = dims(1) * dims(2)
      else
        params%ints(2) = dims(1)
      endif
      params%ints(3) = comm_size
      n_args = 5
      if( kernel_type == KERNEL_UNPACK_PARTIAL ) then
        params%n_ints = 4
        params%ints(4) = comm_rank
      endif
      params%n_ptrs = 3
      params%ptrs(:) = ptrs(:)
    else if ( kernel_type == KERNEL_UNPACK_PIPELINED ) then
      params%n_ints = 5
      ! All 5 ints are populated during kernel execution based on sender pointers
    else
      params%n_ints = size(dims) + 1
      params%ints(1) = block_rows
      params%ints(2) = dims(1)
      params%ints(3) = dims(2)
      if ( size(dims) == 3 ) then
        params%ints(4) = dims(3)

        if ( kernel_type == KERNEL_TRANSPOSE_PACKED ) then
          params%n_ints = params%n_ints + 1
          params%ints(params%n_ints) = comm_size
          params%n_ptrs = 3
          params%ptrs(:) = ptrs(:)
        endif
      endif
    endif
  end subroutine get_kernel_args

  subroutine get_transpose_kernel(comm, kernel_name, dims, transpose_type, kernel_type, base_storage, props, config, blocks, threads, kernel)
    TYPE_MPI_COMM,            intent(in)  :: comm               !! MPI Communicator
    character(len=*),         intent(in)  :: kernel_name
    integer(int32),           intent(in)  :: dims(:)            !! Local dimensions to process
    type(dtfft_transpose_t),  intent(in)  :: transpose_type     !! Type of transposition to perform
    type(kernel_type_t),      intent(in)  :: kernel_type        !! Type of kernel to build
    integer(int64),           intent(in)  :: base_storage       !! Number of bytes needed to store single element
    type(device_props),       intent(in)  :: props
    type(kernel_config),      intent(in)  :: config
    type(dim3),               intent(out) :: blocks             !! 
    type(dim3),               intent(out) :: threads            !! 
    type(CUfunction),         intent(out) :: kernel             !! Compiled kernel to return
    type(kernel_codegen) :: code

    code = get_transpose_kernel_code(kernel_name, size(dims, kind=int8), base_storage, transpose_type, kernel_type == KERNEL_TRANSPOSE_PACKED, config%padding)
    kernel = compile_and_cache(comm, kernel_name, kernel_type, transpose_type, code, props, base_storage, config%threads%x, config%padding)
    blocks = config%blocks
    threads = config%threads
    call code%destroy()
  end subroutine get_transpose_kernel

  subroutine get_kernel(comm, dims, transpose_type, kernel_type, effort, base_storage, props, ptrs, blocks, threads, kernel, force_effort)
  !! Compiles kernel and caches it. Returns compiled kernel.
    TYPE_MPI_COMM,            intent(in)    :: comm               !! MPI Communicator
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    type(dtfft_transpose_t),  intent(in)    :: transpose_type     !! Type of transposition to perform
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel to build
    type(dtfft_effort_t),     intent(in)    :: effort             !! How thoroughly `dtFFT` searches for the optimal transpose kernel
    integer(int64),           intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(device_props),       intent(in)    :: props              !! GPU architecture properties
    type(c_ptr),              intent(in)    :: ptrs(3)            !! Array of device pointers required by certain kernels
    type(dim3),               intent(out)   :: blocks             !! Selected grid of blocks
    type(dim3),               intent(out)   :: threads            !! Selected thread configuration
    type(CUfunction),         intent(out)   :: kernel             !! Compiled kernel to return
    logical,        optional, intent(in)    :: force_effort       !! Should effort be forced or not
    character(len=:), allocatable :: kernel_name              !! Name of the kernel
    type(kernel_codegen)          :: code                     !! Kernel code
    type(kernel_config)           :: candidates(N_CANDIDATES) !! Candidate kernel configurations
    type(kernel_config)           :: config                   !! Current candidate
    integer(int32)                :: num_candidates           !! Number of candidate configurations generated
    integer(int32)                :: i                        !! Loop index
    real(real32),     allocatable :: scores(:)                !! Scores for each candidate configuration
    integer(int32),   allocatable :: sorted(:)                !! Sorted indices of candidate configurations
    integer(int32)                :: tile_dim                 !! Tile dimension
    integer(int32)                :: other_dim                !! Dimension that is not part of shared memory
    integer(int32)                :: fixed_dims(3)            !! Dimensions fixed to the shared memory
    integer(int32)                :: ndims                    !! Number of dimensions
    integer(int32)                :: test_size                !! Number of test configurations to run
    integer(int32)                :: test_id                  !! Current test configuration ID
    integer(int32)                :: iter                     !! Loop index
    integer(int32)                :: best_kernel_id           !! Best kernel configuration ID
    type(c_ptr)                   :: in                       !! Input buffer
    type(c_ptr)                   :: out                      !! Output buffer
    type(cudaEvent)               :: timer_start              !! Timer start event
    type(cudaEvent)               :: timer_stop               !! Timer stop event
    real(real32)                  :: execution_time           !! Execution time
    real(real32)                  :: best_time                !! Best execution time
    type(dtfft_stream_t)          :: stream                   !! CUDA stream for kernel execution
    real(real32)                  :: bandwidth                !! Bandwidth for kernel execution
    type(kernelArgs)              :: params                   !! Kernel arguments
    integer(int32)                :: n_iters                  !! Number of iterations to perform when testing kernel
    integer(int32)                :: n_warmup_iters           !! Number of warmup iterations to perform before testing kernel
    logical                       :: force_effort_            !! Should effort be forced or not
    character(len=:), allocatable :: global_phase             !! Global phase name for profiling
    character(len=:), allocatable :: local_phase              !! Local phase name for profiling

    kernel_name = DEFAULT_KERNEL_NAME // "_"
    if ( kernel_type == KERNEL_TRANSPOSE .or. kernel_type == KERNEL_TRANSPOSE_PACKED ) then
      select case ( transpose_type%val )
      case ( DTFFT_TRANSPOSE_X_TO_Y%val )
        kernel_name = kernel_name // "xy"
      case ( DTFFT_TRANSPOSE_Y_TO_X%val )
        kernel_name = kernel_name // "yx"
      case ( DTFFT_TRANSPOSE_X_TO_Z%val )
        kernel_name = kernel_name // "xz"
      case ( DTFFT_TRANSPOSE_Z_TO_X%val )
        kernel_name = kernel_name // "zx"
      case ( DTFFT_TRANSPOSE_Y_TO_Z%val )
        kernel_name = kernel_name // "yz"
      case ( DTFFT_TRANSPOSE_Z_TO_Y%val )
        kernel_name = kernel_name // "zy"
      endselect
    else if ( kernel_type == KERNEL_UNPACK ) then
      kernel_name = kernel_name // "unpack"
    else if ( kernel_type == KERNEL_UNPACK_PIPELINED ) then
      kernel_name = kernel_name // "unpack_pipelined"
    else if ( kernel_type == KERNEL_UNPACK_PARTIAL ) then
      kernel_name = kernel_name // "unpack_partial"
    else
      INTERNAL_ERROR("Unknown kernel type")
    endif

    if ( is_unpack_kernel(kernel_type) ) then
      if ( kernel_type == KERNEL_UNPACK .or. kernel_type == KERNEL_UNPACK_PARTIAL) then
        call get_contiguous_execution_blocks(product(dims), blocks, threads)
      endif

      if ( kernel_type == KERNEL_UNPACK_PIPELINED ) then
        code = get_unpack_pipelined_kernel_code(kernel_name, base_storage)
      else
        code = get_unpack_kernel_code(kernel_name, base_storage, kernel_type == KERNEL_UNPACK_PARTIAL)
      endif

      kernel = compile_and_cache(comm, kernel_name, kernel_type, transpose_type, code, props, base_storage, VARIABLE_NOT_SET, VARIABLE_NOT_SET)
      call code%destroy()
      deallocate(kernel_name)
      return
    endif
    ! Tranpose kernels only
    if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
      tile_dim = 2
      other_dim = 3
    else
      tile_dim = 3
      other_dim = 2
    endif

    ndims = size(dims)
    fixed_dims(:) = 1
    fixed_dims(1:ndims) = dims(1:ndims)

    call generate_candidates(fixed_dims, tile_dim, other_dim, base_storage, props, candidates, num_candidates)
    allocate(scores(num_candidates), sorted(num_candidates))
    do i = 1, num_candidates
      scores(i) = evaluate_analytical_performance(fixed_dims, transpose_type, candidates(i), props, base_storage)
    enddo
    call sort_candidates_by_score(scores, num_candidates, sorted)

    force_effort_ = .false.; if( present(force_effort) ) force_effort_ = force_effort

    if ( (effort == DTFFT_ESTIMATE .and. force_effort_) .or.                                                                                &
          .not. ( (effort == DTFFT_PATIENT .and. get_conf_kernel_optimization_enabled()) .or. (get_conf_forced_kernel_optimization())) ) then
      call get_transpose_kernel(comm, kernel_name, dims, transpose_type, kernel_type, base_storage, props, candidates(sorted(1)), blocks, threads, kernel)
      deallocate(scores, sorted)
      deallocate(kernel_name)
      return
    endif

    CUDA_CALL( "cudaMalloc", cudaMalloc(in, base_storage * product(dims)) )
    CUDA_CALL( "cudaMalloc", cudaMalloc(out, base_storage * product(dims)) )
    CUDA_CALL( "cudaEventCreate", cudaEventCreate(timer_start) )
    CUDA_CALL( "cudaEventCreate", cudaEventCreate(timer_stop) )
    stream = get_conf_stream()

    global_phase = "Testing nvRTC "//TRANSPOSE_NAMES(transpose_type%val)//" kernel perfomances..."
    PHASE_BEGIN(global_phase, COLOR_AUTOTUNE)
    WRITE_INFO(global_phase)

    n_warmup_iters = get_conf_measure_warmup_iters()
    n_iters = get_conf_measure_iters()

    best_time = MAX_REAL32
    test_size = get_conf_configs_to_test()
    if ( test_size > num_candidates ) test_size = num_candidates

    do test_id = 1, test_size
      config = candidates(sorted(test_id))

      call get_transpose_kernel(comm, kernel_name, dims, transpose_type, kernel_type, base_storage, props, config, blocks, threads, kernel)
      call get_kernel_args(comm, dims, transpose_type, kernel_type, config%threads%y, ptrs, params)
      local_phase = "Testing block: "//to_str(config%threads%x)//"x"//to_str(config%threads%y)
      PHASE_BEGIN(local_phase, COLOR_AUTOTUNE2)
      WRITE_INFO("    "//local_phase)

      PHASE_BEGIN("Warmup", COLOR_TRANSPOSE)
      do iter = 1, n_warmup_iters
        CUDA_CALL( "cuLaunchKernel", cuLaunchKernel(kernel, in, out, blocks, threads, stream, params) )
      enddo
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
      PHASE_END("Warmup")

      PHASE_BEGIN("Measure", COLOR_EXECUTE)
      CUDA_CALL( "cudaEventRecord", cudaEventRecord(timer_start, stream) )
      do iter = 1, n_iters
        CUDA_CALL( "cuLaunchKernel", cuLaunchKernel(kernel, in, out, blocks, threads, stream, params) )
      enddo

      CUDA_CALL( "cudaEventRecord", cudaEventRecord(timer_stop, stream) )
      CUDA_CALL( "cudaEventSynchronize", cudaEventSynchronize(timer_stop) )
      PHASE_END("Measure")
      CUDA_CALL( "cudaEventElapsedTime", cudaEventElapsedTime(execution_time, timer_start, timer_stop) )
      execution_time = execution_time / real(n_iters, real32)
      bandwidth = 2.0 * 1000.0 * real(base_storage * product(dims), real32) / real(1024 * 1024 * 1024, real32) / execution_time
      WRITE_INFO("        Average execution time = "//to_str(real(execution_time, real64))//" [ms]")
      WRITE_INFO("        Bandwidth = "//to_str(bandwidth)//" [GB/s]")

      if ( execution_time < best_time ) then
        best_time = execution_time
        best_kernel_id = test_id
      endif
      call cache%remove(kernel)
      PHASE_END(local_phase)
    enddo
    config = candidates(sorted(best_kernel_id))
    PHASE_END(global_phase)
    WRITE_INFO("  Best configuration is: "//to_str(config%threads%x)//"x"//to_str(config%threads%y))
    call get_transpose_kernel(comm, kernel_name, dims, transpose_type, kernel_type, base_storage, props, config, blocks, threads, kernel)

    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(timer_start) )
    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(timer_stop) )
    CUDA_CALL( "cudaFree", cudaFree(in) )
    CUDA_CALL( "cudaFree", cudaFree(out) )
    deallocate(scores, sorted)
    deallocate(kernel_name)
    deallocate(global_phase, local_phase)
  end subroutine get_kernel

  function compile_and_cache(comm, kernel_name, kernel_type, transpose_type, code, props, base_storage, tile_size, padding) result(kernel)
  !! Compiles kernel stored in `code` and caches pointer to CUfunction
    TYPE_MPI_COMM,            intent(in)  :: comm               !! MPI Communicator
    character(len=*),         intent(in)  :: kernel_name        !! Kernel name
    type(kernel_type_t),      intent(in)  :: kernel_type        !! Type of kernel to build
    type(dtfft_transpose_t),  intent(in)  :: transpose_type     !! Type of transposition to perform
    type(kernel_codegen),     intent(in)  :: code               !! Kernel code to compile
    type(device_props),       intent(in)  :: props              !! GPU architecture properties
    integer(int64),           intent(in)  :: base_storage       !! Number of bytes needed to store single element
    integer(int32),           intent(in)  :: tile_size          !! Tile size to use in shared memory
    integer(int32),           intent(in)  :: padding            !! Padding to use in shared memory
    type(CUfunction)                      :: kernel             !! Compiled kernel to return
    type(nvrtcProgram)                :: prog               !! nvRTC Program
    integer(c_size_t)                 :: cubinSizeRet       !! Size of cubin
    character(c_char),    allocatable :: cubin(:)           !! Compiled binary
    character(c_char),    allocatable :: cstr(:)            !! Temporary string
    character(c_char),    allocatable :: c_code(:)          !! CUDA C Code to compile
    type(string), target, allocatable :: options(:)         !! Compilation options
    type(c_ptr),          allocatable :: c_options(:)       !! C style, null-string terminated options
    integer(int32)                    :: num_options        !! Number of compilation options
    integer(int32)                    :: i                  !! Loop index
    integer(int32)                    :: ierr               !! Error code
    integer(int32)                    :: mpi_ierr           !! MPI error code
    type(CUmodule)                    :: CUmod              !! CUDA module
    character(len=:),     allocatable :: phase_name         !! Phase name for profiling

    ! Checking for existing kernel and early exit
    kernel = cache%get(transpose_type, kernel_type, base_storage, tile_size, padding)
    if ( .not.is_null_ptr(kernel%ptr) ) return
    call code%to_cstr(c_code)
    ! WRITE_INFO(code%raw)
    phase_name = "Compiling nvRTC kernel: "//kernel_name
    PHASE_BEGIN(phase_name, COLOR_FFT)
    WRITE_DEBUG(phase_name)

#ifdef DTFFT_DEBUG
    num_options = 4
#else
    num_options = 2
#endif

    allocate( c_options(num_options), options(num_options) )
    options(1) = string("--gpu-architecture=sm_"//to_str(props%compute_capability_major)//to_str(props%compute_capability_minor) // c_null_char)
    options(2) = string("-DTILE_DIM="//to_str(tile_size) // c_null_char)
#ifdef DTFFT_DEBUG
    options(3) = string("--device-debug" // c_null_char)
    options(4) = string("--generate-line-info" // c_null_char)
#endif
    do i = 1, num_options
      c_options(i) = c_loc(options(i)%raw)
    enddo

    NVRTC_CALL( "nvrtcCreateProgram", nvrtcCreateProgram(prog, c_code, kernel_name//c_null_char, 0, c_null_ptr, c_null_ptr) )
    ierr = nvrtcCompileProgram(prog, num_options, c_options)
    ! It is assumed here that ierr can only be positive
    ! ALL_REDUCE(ierr, MPI_INTEGER4, MPI_MAX, comm, mpi_ierr)
    if ( ierr /= 0 ) then
      block
        type(c_ptr) :: c_log
        character(len=:), allocatable :: f_log
        ! integer(int32) :: global_rank

        NVRTC_CALL( "nvrtcGetProgramLog", nvrtcGetProgramLog(prog, c_log))
        call string_c2f(c_log, f_log)

        ! call MPI_Comm_rank(comm, global_rank, mpi_ierr)
        ! if ( global_rank == 0 ) then
        write(error_unit, "(a)") "dtFFT Internal Error: failed to compile kernel"
        write(error_unit, "(a)") "CUDA Code:"
        write(error_unit, "(a)") code%raw
        write(error_unit, "(a)") "Compilation log:"
        write(error_unit, "(a)") f_log
        ! endif
        call MPI_Abort(comm, ierr, mpi_ierr)
      endblock
    endif

    NVRTC_CALL( "nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, cubinSizeRet) )
    allocate( cubin( cubinSizeRet ) )
    NVRTC_CALL( "nvrtcGetCUBIN", nvrtcGetCUBIN(prog, cubin) )
    NVRTC_CALL( "nvrtcDestroyProgram", nvrtcDestroyProgram(prog) )

    CUDA_CALL( "cuModuleLoadData", cuModuleLoadData(CUmod, cubin) )
    call astring_f2c(kernel_name//c_null_char, cstr)
    CUDA_CALL( "cuModuleGetFunction", cuModuleGetFunction(kernel, CUmod, cstr) )
    deallocate(cstr)

    PHASE_END(phase_name)

    deallocate( phase_name )
    deallocate( c_code )
    call destroy_strings(options)
    deallocate( c_options )
    deallocate( cubin )

    call cache%add(CUmod, kernel, kernel_type, transpose_type, tile_size, padding, base_storage)
  end function compile_and_cache
end module dtfft_nvrtc_kernel