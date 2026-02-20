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
module dtfft_kernel_device
!! This module defines [[kernel_device]] type and its type bound procedures.
!! It extends [[abstract_kernel]] type and implements its type bound procedures.
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_config
use dtfft_interface_cuda,         only: MAX_KERNEL_ARGS, dim3, CUfunction, cuLaunchKernel
use dtfft_interface_cuda_runtime
use dtfft_nvrtc_block_optimizer,  only: N_CANDIDATES,                       &
                                        kernel_config,                      &
                                        generate_candidates,                &
                                        evaluate_analytical_performance,    &
                                        sort_candidates_by_score
use dtfft_nvrtc_module_cache
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_profile.h"
#include "_dtfft_private.h"
implicit none
private
public :: kernel_device

    type, extends(abstract_kernel) :: kernel_device
    !! Device kernel class
    private
        type(kernel_type_t)   :: internal_kernel_type     !! Actual kernel type used for execution, can be different from `kernel_type`
        type(CUfunction)      :: cuda_kernel              !! Pointer to CUDA kernel.
        integer(int32)        :: tile_size                !! Tile size used for this kernel
        integer(int32)        :: block_rows               !! Number of rows in each block processed by each thread
        integer(int64)        :: copy_bytes               !! Number of bytes to copy for `KERNEL_COPY` kernel
    contains
        procedure :: create_private => create   !! Creates kernel
        procedure :: execute_private => execute !! Executes kernel
        procedure :: destroy_private => destroy !! Destroys kernel
    end type kernel_device

contains

    subroutine create(self, effort, base_storage, force_effort)
    !! Creates kernel
        class(kernel_device),     intent(inout) :: self             !! Device kernel class
        type(dtfft_effort_t),     intent(in)    :: effort           !! Effort level for generating transpose kernels
        integer(int64),           intent(in)    :: base_storage     !! Number of bytes needed to store single element
        logical,        optional, intent(in)    :: force_effort     !! Should effort be forced or not
        type(device_props)                      :: props              !! GPU architecture properties
        integer(int32)                          :: device_id          !! Device ID

        call self%destroy()
#ifdef DTFFT_DEBUG
        if ( any(self%kernel_type == [KERNEL_UNPACK_FORWARD, KERNEL_UNPACK_FORWARD_PIPELINED, KERNEL_UNPACK_BACKWARD, KERNEL_UNPACK_BACKWARD_PIPELINED]) ) then
            INTERNAL_ERROR("Invalid kernel type for CUDA: "//self%kernel_string%raw)
        endif
#endif
        self%base_storage = base_storage
        if ( self%kernel_type == KERNEL_COPY .or. self%kernel_type == KERNEL_COPY_PIPELINED) then
            self%is_created = .true.
            self%copy_bytes = base_storage * product(self%dims)
            return
        endif

        self%internal_kernel_type = self%kernel_type
        select case ( self%kernel_type%val )
            case ( KERNEL_PACK%val )
                self%internal_kernel_type = KERNEL_PACK_PIPELINED
            case ( KERNEL_UNPACK%val )
                self%internal_kernel_type = KERNEL_UNPACK_PIPELINED
            case ( KERNEL_PERMUTE_BACKWARD_END%val )
                self%internal_kernel_type = KERNEL_PERMUTE_BACKWARD_END_PIPELINED
        end select

        CUDA_CALL( cudaGetDevice(device_id) )
        call get_device_props(device_id, props)
        if ( allocated( self%neighbor_data ) ) then
            call get_kernel(self%dims, self%internal_kernel_type, self%kernel_string, effort, base_storage, props,    &
                            self%tile_size, self%block_rows, self%cuda_kernel, force_effort=force_effort, neighbor_data=self%neighbor_data(:, 1))
        else
            call get_kernel(self%dims, self%internal_kernel_type, self%kernel_string, effort, base_storage, props,    &
                            self%tile_size, self%block_rows, self%cuda_kernel, force_effort=force_effort)
        endif
    end subroutine create

    subroutine execute(self, in, out, stream, sync, neighbor)
    !! Executes kernel on stream
        class(kernel_device),       intent(inout)   :: self         !! Device kernel class
        type(c_ptr),                intent(in)      :: in           !! Source buffer, can be device or host pointer
        type(c_ptr),                intent(in)      :: out          !! Target buffer, can be device or host pointer
        type(dtfft_stream_t),       intent(in)      :: stream       !! Stream to execute on
        logical,                    intent(in)      :: sync         !! Sync stream after kernel execution, unused here
        integer(int32),   optional, intent(in)      :: neighbor     !! Source rank for pipelined unpacking
        integer(int32) :: nargs, n
        integer(int32) :: args(MAX_KERNEL_ARGS)
        type(dim3) :: blocks, threads
        type(c_ptr) :: in_ptr, out_ptr

        if ( self%kernel_type == KERNEL_COPY ) then
            CUDA_CALL( cudaMemcpyAsync(out, in, self%copy_bytes, cudaMemcpyDeviceToDevice, stream) )
#ifdef DTFFT_DEBUG
            CUDA_CALL( cudaStreamSynchronize(stream) )
#endif
            if ( sync ) then
                CUDA_CALL( cudaStreamSynchronize(stream) )
            endif
            return
        endif

        if ( self%kernel_type == KERNEL_COPY_PIPELINED ) then
            out_ptr = ptr_offset(out, self%base_storage * self%neighbor_data(5, neighbor))
            in_ptr = ptr_offset(in, self%base_storage * self%neighbor_data(4, neighbor))
            CUDA_CALL( cudaMemcpyAsync(out_ptr, in_ptr, self%base_storage * product(self%neighbor_data(1:3, neighbor)), cudaMemcpyDeviceToDevice, stream) )
#ifdef DTFFT_DEBUG
            CUDA_CALL( cudaStreamSynchronize(stream) )
#endif
            if ( sync ) then
                CUDA_CALL( cudaStreamSynchronize(stream) )
            endif
            return
        endif

        if( any(self%kernel_type == [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD, KERNEL_PERMUTE_BACKWARD_START]) ) then
            call get_kernel_launch_params(self%kernel_type, self%dims, self%tile_size, self%block_rows, blocks, threads)
            call get_kernel_args(self%kernel_type, self%dims, nargs, args)
            CUDA_CALL( cuLaunchKernel(self%cuda_kernel, in, out, blocks, threads, stream, nargs, args) )
#ifdef DTFFT_DEBUG
            CUDA_CALL( cudaStreamSynchronize(stream) )
#endif
            if ( sync ) then
                CUDA_CALL( cudaStreamSynchronize(stream) )
            endif
            return
        endif

        if ( any(self%kernel_type == [KERNEL_UNPACK_PIPELINED, KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_PACK_FORWARD, KERNEL_PACK_BACKWARD]) ) then
            call get_kernel_launch_params(self%kernel_type, self%neighbor_data(1:3, neighbor), self%tile_size, self%block_rows, blocks, threads )
            call get_kernel_args(self%kernel_type, self%dims, nargs, args, self%neighbor_data(:, neighbor))
            CUDA_CALL( cuLaunchKernel(self%cuda_kernel, in, out, blocks, threads, stream, nargs, args) )
#ifdef DTFFT_DEBUG
            CUDA_CALL( cudaStreamSynchronize(stream) )
#endif
            if ( sync ) then
                CUDA_CALL( cudaStreamSynchronize(stream) )
            endif
            return
        endif

        do n = 1, size(self%neighbor_data, dim=2)
            call get_kernel_launch_params(self%internal_kernel_type, self%neighbor_data(1:3, n), self%tile_size, self%block_rows, blocks, threads )
            call get_kernel_args(self%internal_kernel_type, self%dims, nargs, args, self%neighbor_data(:, n))
            CUDA_CALL( cuLaunchKernel(self%cuda_kernel, in, out, blocks, threads, stream, nargs, args) )
#ifdef DTFFT_DEBUG
            CUDA_CALL( cudaStreamSynchronize(stream) )
#endif
        enddo
        if ( sync ) then
            CUDA_CALL( cudaStreamSynchronize(stream) )
        endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys kernel
    class(kernel_device), intent(inout) :: self !! Device kernel class

    if ( .not. self%is_created ) return
    if ( self%is_dummy .or. self%kernel_type == KERNEL_COPY ) return
  end subroutine destroy

  subroutine get_kernel_args(kernel_type, dims, nargs, args, neighbor_data)
  !! Populates kernel arguments based on kernel type
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    integer(int32),           intent(out)   :: nargs              !! Number of arguments set by this subroutine
    integer(int32),           intent(out)   :: args(MAX_KERNEL_ARGS)  !! Kernel arguments
    integer(int32), optional, intent(in)    :: neighbor_data(:)   !! Neighbor data for pipelined kernels

    nargs = 0
    nargs = nargs + 1;  args(nargs) = dims(1)
    nargs = nargs + 1;  args(nargs) = dims(2)
    nargs = nargs + 1
    if ( size(dims) == 2 ) then
      args(nargs) = 1
    else
      args(nargs) = dims(3)
    endif

    if ( is_pack_kernel(kernel_type) .or. is_unpack_kernel(kernel_type) ) then
      nargs = nargs + 1;  args(nargs) = neighbor_data(1)
      nargs = nargs + 1;  args(nargs) = neighbor_data(2)
      nargs = nargs + 1;  args(nargs) = neighbor_data(3)
      nargs = nargs + 1;  args(nargs) = neighbor_data(4)
      nargs = nargs + 1;  args(nargs) = neighbor_data(5)
    endif
    ! if ( size(dims) == 2 .or. kernel_type == KERNEL_UNPACK_PIPELINED .or. kernel_type == KERNEL_PACK ) return

    ! if ( any(kernel_type == [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD, KERNEL_PERMUTE_BACKWARD_START]) ) then
    !   nargs = nargs + 1; args(nargs) = dims(3)
    !   return
    ! endif

    ! nargs = nargs + 1; args(nargs) = neighbor_data(1)
    ! nargs = nargs + 1; args(nargs) = neighbor_data(2)
    ! nargs = nargs + 1; args(nargs) = neighbor_data(3)
    ! nargs = nargs + 1; args(nargs) = neighbor_data(4)
    ! nargs = nargs + 1; args(nargs) = neighbor_data(5)
  end subroutine get_kernel_args

  subroutine get_kernel_launch_params(kernel_type, dims, tile_size, block_rows, blocks, threads)
  !! Computes kernel launch parameters based on kernel type and dimensions
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    integer(int32),           intent(in)    :: tile_size          !! Size of the tile in shared memory
    integer(int32),           intent(in)    :: block_rows         !! Number of rows in each block
    type(dim3),               intent(out)   :: blocks             !! Number of blocks to launch
    type(dim3),               intent(out)   :: threads            !! Number of threads per block
    integer(int32) :: tile_dim, other_dim

    threads%x = tile_size
    threads%y = block_rows
    threads%z = 1

    if ( any(kernel_type == [KERNEL_PERMUTE_FORWARD, KERNEL_PACK_FORWARD, KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_UNPACK_PIPELINED, KERNEL_PACK_PIPELINED]) ) then
      tile_dim = 2
      other_dim = 3
    else
      ! KERNEL_PERMUTE_BACKWARD_START or KERNEL_PERMUTE_BACKWARD
      tile_dim = 3
      other_dim = 2
    endif

    blocks%x = (dims(1) + tile_size - 1) / tile_size
    blocks%y = (dims(tile_dim) + tile_size - 1) / tile_size
    if ( size(dims) == 2 ) then
      blocks%z = 1
    else
      blocks%z = dims(other_dim)
    endif
  end subroutine get_kernel_launch_params

  subroutine get_kernel(dims, kernel_type, kernel_string, effort, base_storage, props, tile_size, block_rows, kernel, force_effort, neighbor_data)
  !! Compiles kernel and caches it. Returns compiled kernel.
    integer(int32),           intent(in)    :: dims(:)            !! Local dimensions to process
    type(kernel_type_t),      intent(in)    :: kernel_type        !! Type of kernel to build
    type(string),             intent(in)    :: kernel_string      !! Kernel string
    type(dtfft_effort_t),     intent(in)    :: effort             !! How thoroughly `dtFFT` searches for the optimal transpose kernel
    integer(int64),           intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(device_props),       intent(in)    :: props              !! GPU architecture properties
    integer(int32),           intent(out)   :: tile_size          !! Size of the tile in shared memory
    integer(int32),           intent(out)   :: block_rows         !! Number of rows in each block processed by each thread
    type(CUfunction),         intent(out)   :: kernel             !! Compiled kernel to return
    logical,        optional, intent(in)    :: force_effort       !! Should effort be forced or not
    integer(int32), optional, intent(in)    :: neighbor_data(:)   !! Neighbor data for pipelined kernels
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
    integer(int32)                :: test_id                  !! Current test configuration ID
    integer(int32)                :: iter                     !! Loop index
    integer(int32)                :: best_kernel_id           !! Best kernel configuration ID
    type(c_ptr)                   :: in                       !! Input buffer
    type(c_ptr)                   :: out                      !! Output buffer
    type(dim3)                    :: blocks                   !! Blocks configuration
    type(dim3)                    :: threads                  !! Threads configuration
    type(cudaEvent)               :: timer_start              !! Timer start event
    type(cudaEvent)               :: timer_stop               !! Timer stop event
    real(real32)                  :: execution_time           !! Execution time
    real(real32)                  :: best_time                !! Best execution time
    type(dtfft_stream_t)          :: stream                   !! CUDA stream for kernel execution
    real(real32)                  :: bandwidth                !! Bandwidth for kernel execution
    integer(int32)                :: n_iters                  !! Number of iterations to perform when testing kernel
    integer(int32)                :: n_warmup_iters           !! Number of warmup iterations to perform before testing kernel
    logical                       :: force_effort_            !! Should effort be forced or not
    character(len=:), allocatable :: global_phase             !! Global phase name for profiling
    character(len=:), allocatable :: local_phase              !! Local phase name for profiling
    integer(int32)                :: nargs                    !! Number of kernel arguments
    integer(int32)                :: args(MAX_KERNEL_ARGS)    !! Kernel arguments


    if ( any(kernel_type == [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_UNPACK_PIPELINED, KERNEL_PACK_PIPELINED]) ) then
      tile_dim = 2
      other_dim = 3
    else
      ! KERNEL_PERMUTE_BACKWARD_START or KERNEL_PERMUTE_BACKWARD
      tile_dim = 3
      other_dim = 2
    endif

    ndims = size(dims)
    fixed_dims(:) = 1
    fixed_dims(1:ndims) = dims(1:ndims)
    if ( is_unpack_kernel(kernel_type) .or. is_pack_kernel(kernel_type)) fixed_dims(1:ndims) = neighbor_data(1:ndims)
    call generate_candidates(fixed_dims, tile_dim, other_dim, base_storage, props, candidates, num_candidates)
    allocate(scores(num_candidates), sorted(num_candidates))
    do i = 1, num_candidates
      scores(i) = evaluate_analytical_performance(fixed_dims, tile_dim, other_dim, kernel_type, candidates(i), props, base_storage, neighbor_data)
    enddo
    call sort_candidates_by_score(scores, num_candidates, sorted)

    call create_nvrtc_module(ndims, kernel_type, base_storage, candidates(1:num_candidates), props)

    force_effort_ = .false.; if( present(force_effort) ) force_effort_ = force_effort

    if ((effort == DTFFT_ESTIMATE .and. force_effort_) .or. &
          .not. ( effort == DTFFT_EXHAUSTIVE .or. get_conf_kernel_autotune_enabled()) ) then
      config = candidates(sorted(1))
      tile_size = config%tile_size
      block_rows = config%block_rows
      kernel = get_kernel_instance(ndims, kernel_type, base_storage, tile_size, block_rows)
      deallocate(scores, sorted)
      return
    endif

    CUDA_CALL( cudaMalloc(in, base_storage * product(dims)) )
    CUDA_CALL( cudaMalloc(out, base_storage * product(dims)) )
    CUDA_CALL( cudaEventCreate(timer_start) )
    CUDA_CALL( cudaEventCreate(timer_stop) )
    stream = get_conf_stream()

    allocate( global_phase, source="Testing nvRTC kernel: '"//kernel_string%raw//"' perfomances..." )
    PHASE_BEGIN(global_phase, COLOR_STEEL_BLUE)
    WRITE_INFO(global_phase)

    n_warmup_iters = get_conf_measure_warmup_iters()
    n_iters = get_conf_measure_iters()

    best_time = MAX_REAL32

    do test_id = 1, num_candidates
      config = candidates(sorted(test_id))
      tile_size = config%tile_size
      block_rows = config%block_rows

      call get_kernel_launch_params(kernel_type, fixed_dims, tile_size, block_rows, blocks, threads)
      call get_kernel_args(kernel_type, dims, nargs, args, neighbor_data)

      kernel = get_kernel_instance(ndims, kernel_type, base_storage, tile_size, block_rows)

      allocate( local_phase, source="Testing block: "//to_str(tile_size)//"x"//to_str(block_rows) )
      REGION_BEGIN(local_phase, COLOR_ORCHID)
      WRITE_INFO("    "//local_phase)

      REGION_BEGIN("Warmup", COLOR_VIOLET)
      do iter = 1, n_warmup_iters
        CUDA_CALL( cuLaunchKernel(kernel, in, out, blocks, threads, stream, nargs, args) )
      enddo
      CUDA_CALL( cudaStreamSynchronize(stream) )
      REGION_END("Warmup")

      REGION_BEGIN("Measure", COLOR_DODGER_BLUE)
      CUDA_CALL( cudaEventRecord(timer_start, stream) )
      do iter = 1, n_iters
        CUDA_CALL( cuLaunchKernel(kernel, in, out, blocks, threads, stream, nargs, args) )
      enddo

      CUDA_CALL( cudaEventRecord(timer_stop, stream) )
      CUDA_CALL( cudaEventSynchronize(timer_stop) )
      REGION_END("Measure")
      CUDA_CALL( cudaEventElapsedTime(execution_time, timer_start, timer_stop) )
      execution_time = execution_time / real(n_iters, real32)
      WRITE_INFO("        Average execution time = "//to_str(real(execution_time, real64))//" [ms]")
      if ( execution_time > 0._real32 ) then
        bandwidth = 2.0 * 1000.0 * real(base_storage * product(fixed_dims), real32) / real(1024 * 1024 * 1024, real32) / execution_time
        WRITE_INFO("        Bandwidth = "//to_str(bandwidth)//" [GB/s]")
      endif

      if ( execution_time < best_time ) then
        best_time = execution_time
        best_kernel_id = test_id
      endif
      REGION_END(local_phase)
      deallocate(local_phase)
    enddo
    config = candidates(sorted(best_kernel_id))
    PHASE_END(global_phase)
    tile_size = config%tile_size
    block_rows = config%block_rows
    kernel = get_kernel_instance(ndims, kernel_type, base_storage, tile_size, block_rows)
    WRITE_INFO("  Best configuration is: "//to_str(tile_size)//"x"//to_str(block_rows))

    CUDA_CALL( cudaEventDestroy(timer_start) )
    CUDA_CALL( cudaEventDestroy(timer_stop) )
    CUDA_CALL( cudaFree(in) )
    CUDA_CALL( cudaFree(out) )
    deallocate(scores, sorted)
    deallocate(global_phase)
  end subroutine get_kernel
end module dtfft_kernel_device