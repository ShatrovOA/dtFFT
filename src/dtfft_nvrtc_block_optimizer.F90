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
module dtfft_nvrtc_block_optimizer
!! Module that provides functionality to analytically optimize CUDA kernel configurations
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_config,         only: get_conf_log_enabled
use dtfft_interface_cuda, only: dim3
use dtfft_interface_cuda_runtime
use dtfft_parameters
use dtfft_utils
#include "_dtfft_private.h"
implicit none
private
public :: generate_candidates
public :: evaluate_analytical_performance
public :: get_ampere_architecture, get_volta_architecture
public :: count_bank_conflicts
public :: sort_candidates_by_score

  integer(int32), parameter :: NUM_BANKS = 32
    !! Number of banks in shared memory
  integer(int32), parameter :: WARP_SIZE = 32
    !! Warp size in threads
  integer(int32), parameter :: BANK_WIDTH_BYTES = 4
    !! Bank width in bytes

  integer(int32), parameter, public :: N_TILES_CANDIDATES = 5
    !! Maximum number of tile candidates to generate
  integer(int32), parameter, public :: N_BLOCKS_CANDIDATES = 5
    !! Maximum number of block candidates to generate
  integer(int32), parameter, public :: N_CANDIDATES = N_TILES_CANDIDATES * N_BLOCKS_CANDIDATES
    !! Maximum number of candidates to generate

  type, public :: kernel_config
  !! Configuration for the potential kernel
    ! type(dim3)      :: blocks     !! Number of blocks in the grid
    ! type(dim3)      :: threads    !! Number of threads per block
    integer(int32)  :: tile_size  !! Tile size (number of columns)
    integer(int32)  :: block_rows !! Block rows
    integer(int32)  :: padding    !! Padding added to the tile
  end type kernel_config

contains

  function get_ampere_architecture() result(props)
  !! Ampere architecture (Compute Capability 8.0)
    type(device_props) :: props !! Ampere architecture properties

    props%sm_count                   = 108
    props%max_threads_per_sm         = 2048
    props%max_blocks_per_sm          = 32
    props%shared_mem_per_sm          = 164 * 1024
    props%max_threads_per_block      = 1024
    props%shared_mem_per_block       = 48 * 1024
    props%l2_cache_size              = 40960 * 1024
    props%compute_capability_major   = 8
    props%compute_capability_minor   = 0
  end function get_ampere_architecture

  function get_volta_architecture() result(props)
  !! Volta architecture (Compute Capability 7.0)
    type(device_props) :: props !! Volta architecture properties

    props%sm_count                   = 80
    props%max_threads_per_sm         = 2048
    props%max_blocks_per_sm          = 32
    props%shared_mem_per_sm          = 96 * 1024
    props%max_threads_per_block      = 1024
    props%shared_mem_per_block       = 48 * 1024
    props%l2_cache_size              = 6144 * 1024
    props%compute_capability_major   = 7
    props%compute_capability_minor   = 0
  end function get_volta_architecture

  pure function count_bank_conflicts(tile_size, block_rows, base_storage, padding) result(total_conflicts)
  !! Counts bank conflicts for a given tile size, padding, element size, and block rows.
    integer(int32), intent(in)  :: tile_size          !! Size of the tile
    integer(int32), intent(in)  :: block_rows         !! Number of rows in the block
    integer(int64), intent(in)  :: base_storage       !! Number of bytes needed to store single element
    integer(int32), intent(in)  :: padding            !! Padding added to the tile
    integer(int32)              :: total_conflicts    !! Total number of bank conflicts
    integer(int32) :: stride                          !! Stride of the tile
    integer(int32) :: thread_idx                      !! Thread index within the warp
    integer(int32) :: bank                            !! Bank index
    integer(int32) :: element_address                 !! Address of the element
    integer(int32) :: address_in_bytes                !! Address of the element in bytes
    integer(int32) :: bank_accesses(0:NUM_BANKS - 1)  !! Array to count bank accesses
    integer(int32) :: thread_x                        !! Thread index in the x dimension
    integer(int32) :: thread_y                        !! Thread index in the y dimension
    integer(int32) :: load_conflicts                  !! Number of load bank conflicts
    integer(int32) :: store_conflicts                 !! Number of store bank conflicts
    integer(int32) :: offset                          !! Offset for the tile

    ! Check for validity
    if (block_rows > tile_size) then
      total_conflicts = MAX_INT32
      return
    end if

    ! Shared memory: T tile[TILE_DIM][TILE_DIM + padding]
    ! stride = tile_size + padding
    stride = tile_size + padding
    load_conflicts = 0
    ! Computing write conflicts
    do offset = 0, tile_size - 1, block_rows
      bank_accesses(:) = 0

      do thread_idx = 0, WARP_SIZE - 1
        thread_x = mod(thread_idx, tile_size)
        thread_y = thread_idx / tile_size

        if (thread_y >= block_rows) cycle
        if (thread_y + offset >= tile_size) cycle

        ! Load address: tile[threadIdx.x][threadIdx.y + y_offset]
        element_address = thread_x * stride + (thread_y + offset)
        address_in_bytes = element_address * int(base_storage, int32)
        bank = mod(address_in_bytes / BANK_WIDTH_BYTES, NUM_BANKS)

        bank_accesses(bank) = bank_accesses(bank) + 1
      end do

      do bank = 0, NUM_BANKS - 1
        if (bank_accesses(bank) > 1) then
          load_conflicts = load_conflicts + (bank_accesses(bank) - 1)
        end if
      end do
    end do

    ! Computing read conflicts
    store_conflicts = 0
    do offset = 0, tile_size-1, block_rows
      bank_accesses(:) = 0

      do thread_idx = 0, WARP_SIZE-1
        thread_x = mod(thread_idx, tile_size)
        thread_y = thread_idx / tile_size

        if (thread_y >= block_rows) cycle
        if (thread_y + offset >= tile_size) cycle

        ! Read address: tile[threadIdx.y + offset][threadIdx.x]
        element_address = (thread_y + offset) * stride + thread_x
        address_in_bytes = element_address * int(base_storage, int32)
        bank = mod(address_in_bytes / BANK_WIDTH_BYTES, NUM_BANKS)

        bank_accesses(bank) = bank_accesses(bank) + 1
      end do

      do bank = 0, NUM_BANKS-1
        if (bank_accesses(bank) > 1) then
          store_conflicts = store_conflicts + (bank_accesses(bank) - 1)
        end if
      end do
    end do

    total_conflicts = load_conflicts + store_conflicts
  end function count_bank_conflicts

  pure function estimate_optimal_padding(tile_size, block_rows, base_storage) result(padding)
  !! Estimates the optimal padding for a given tile size and element size
    integer(int32), intent(in)  :: tile_size      !! Size of the tile
    integer(int32), intent(in)  :: block_rows     !! Number of rows in the block
    integer(int64), intent(in)  :: base_storage   !! Number of bytes needed to store single element
    integer(int32)              :: padding        !! Optimal padding to reduce bank conflicts
    integer(int32) :: best_padding    !! Best padding found
    integer(int32) :: min_conflicts   !! Minimum conflicts found
    integer(int32) :: n_conflicts     !! Number of bank conflicts for the current padding

    best_padding = 1  ! This is default, works in most cases
    min_conflicts = MAX_INT32

    ! Starting from 1. We do not consider zero padding
    do padding = 1, 8
      n_conflicts = count_bank_conflicts(tile_size, block_rows, base_storage, padding)

      if (n_conflicts < min_conflicts) then
        min_conflicts = n_conflicts
        best_padding = padding
        ! Early exit in case of best padding found
        if (n_conflicts == 0) exit
      end if
    end do

    padding = best_padding
  end function estimate_optimal_padding

  pure function estimate_bank_conflict_ratio(config, base_storage) result(ratio)
  !! Estimates the bank conflict ratio for a given kernel configuration
    type(kernel_config),  intent(in)  :: config         !! Kernel configuration
    integer(int64),       intent(in)  :: base_storage   !! Number of bytes needed to store single element
    real(real32)                      :: ratio          !! Bank conflict estimation
    integer(int32) :: actual_conflicts      !! Actual number of bank conflicts for given configuration
    integer(int32) :: worst_case_conflicts  !! Worst-case number of bank conflicts for given configuration

    actual_conflicts = count_bank_conflicts(config%tile_size, config%block_rows, base_storage, config%padding)
    if (actual_conflicts == MAX_INT32) then
      ! Invalid configuration, return immediately
      ratio = 1.0
      return
    end if
    ! Worst case is using same configuration but without padding
    worst_case_conflicts = count_bank_conflicts(config%tile_size, config%block_rows, base_storage, 0)

    ! Worst case can also have no conflicts
    if (worst_case_conflicts > 0) then
      ratio = real(actual_conflicts) / real(worst_case_conflicts)
      ratio = max(0.0, min(1.0, ratio))
    else
      ratio = 0.0
    end if
  end function estimate_bank_conflict_ratio

  pure function estimate_occupancy(config, props, base_storage) result(occupancy)
  !! Calculates theoretical occupancy for a given kernel configuration
    type(kernel_config),    intent(in) :: config        !! Kernel configuration
    type(device_props),     intent(in) :: props         !! GPU architecture properties
    integer(int64),         intent(in) :: base_storage  !! Number of bytes needed to store single element
    real(real32)                       :: occupancy     !! Estimated occupancy
    integer(int32) :: threads_per_block         !! Number of threads per block
    integer(int32) :: max_blocks_by_threads     !! Maximum number of blocks by threads
    integer(int32) :: max_blocks_by_shared_mem  !! Maximum number of blocks by shared memory
    integer(int32) :: max_active_blocks         !! Maximum number of active blocks
    integer(int64) :: shared_mem_per_block      !! Shared memory per block

    ! Limits by number of threads in multiprocessor
    threads_per_block = config%tile_size * config%block_rows
    if (threads_per_block > 0) then
      max_blocks_by_threads = props%max_threads_per_sm / threads_per_block
    else
      max_blocks_by_threads = 0
    end if

    ! Limits by shared memory in multiprocessor
    shared_mem_per_block = config%tile_size * (config%tile_size + config%padding) * base_storage
    if (shared_mem_per_block > 0) then
      max_blocks_by_shared_mem = int(props%shared_mem_per_sm / shared_mem_per_block, int32)
    else
      max_blocks_by_shared_mem = props%max_blocks_per_sm
    end if

    max_active_blocks = min(max_blocks_by_threads, max_blocks_by_shared_mem, props%max_blocks_per_sm)
    if (props%max_threads_per_sm > 0) then
      occupancy = real(max_active_blocks * threads_per_block, real32) / real(props%max_threads_per_sm, real32)
    else
      occupancy = 0.0
    end if
    occupancy = max(0.0, min(1.0, occupancy))
  end function estimate_occupancy

  function estimate_memory_pressure(dims, tile_dim, other_dim, base_storage, props) result(pressure)
  !! Analytical estimation of memory pressure based on GPU architecture
    integer(int32),         intent(in)  :: dims(:)        !! Size of the problem
    integer(int32),         intent(in)  :: tile_dim       !! Tile dimension
    integer(int32),         intent(in)  :: other_dim      !! Other dimension (not tiled)
    integer(int64),         intent(in)  :: base_storage   !! Number of bytes needed to store single element
    type(device_props),     intent(in)  :: props           !! GPU architecture properties
    real(real32)                        :: pressure       !! Pressure metric
    integer(int64)  :: matrix_size_bytes      !! Size of the matrix in bytes
    integer(int64)  :: shared_mem_total_bytes !! Total shared memory available for a GPU
    real(real32)    :: memory_utilization     !! Memory utilization metric
    real(real32)    :: cache_efficiency       !! Cache efficiency metric
    real(real32)    :: locality_factor        !! Locality factor metric
    real(real32)    :: aspect_ratio           !! Aspect ratio metric
    integer(int32)  :: nx                     !! Size of the matrix in the x dimension
    integer(int32)  :: ny                     !! Size of the matrix in the y dimension
    integer(int32)  :: nz                     !! Size of the matrix in the z dimension
    integer(int32)  :: min_dim                !! Minimum dimension size

    nx = dims(1)
    ny = dims(tile_dim)
    nz = dims(other_dim)
    min_dim = min(nx, ny)

    matrix_size_bytes = product(dims) * base_storage
    shared_mem_total_bytes = props%shared_mem_per_sm * int(props%sm_count, int64)

    ! 2D model of memory pressure: Shared Memory + L2 Cache vs Global Memory
    if (matrix_size_bytes > props%l2_cache_size) then
      ! Matrix does not fit in L2 - high pressure on global memory
      memory_utilization = 0.9
    else if (matrix_size_bytes > shared_mem_total_bytes) then
      ! Matrix fits in L2 but not in shared memory - medium pressure
      memory_utilization = 0.6
    else if (matrix_size_bytes < 1024_int64) then
      ! No pressure
      memory_utilization = 0.1
    else
      ! Matrix fits in shared memory - low pressure
      memory_utilization = 0.3
    end if

    ! Analytical estimation of cache efficiency based on access pattern
    aspect_ratio = real(max(nx, ny)) / real(min_dim)
    if (nz == 1) then
      ! 2D case
      if (nx == ny) then
        ! Square matrix - good spatial locality when transposed
        cache_efficiency = 0.8
      else if (min_dim < 32) then
        ! Narrow matrix - poor spatial locality
        cache_efficiency = 0.5
      else
        ! Rectangular matrix - average locality
        cache_efficiency = 0.7
      end if
    else
      ! 3D case - additional dimension affects cache efficiency
      if (nx == ny .and. nz <= 16) then
        ! Square layers with small depth - good locality
        cache_efficiency = 0.7
      else if (min_dim < 32 .or. nz > 64) then
        ! Narrow layers or large depth - poor locality
        cache_efficiency = 0.4
      else
        ! Average case for 3D
        cache_efficiency = 0.6
      end if
      aspect_ratio = aspect_ratio * (1.0 + real(nz) / 32.0)  ! Z-dimension affects locality
    end if
    if (aspect_ratio > 16.0) then
      locality_factor = 0.3  ! Very asymmetric - poor locality
    else if (aspect_ratio > 4.0) then
      locality_factor = 0.6  ! Moderately asymmetric
    else
      locality_factor = 0.9  ! Close to square - good locality
    end if

    ! Combined memory pressure estimation
    pressure = memory_utilization * (1.0 - cache_efficiency * locality_factor + 0.1)
    pressure = max(0.0, min(1.0, pressure))
  end function estimate_memory_pressure

  function estimate_coalescing(dims, tile_dim, other_dim, kernel_type, config, base_storage, neighbor_data) result(score)
  !! Estimate memory coalescing efficiency for a given kernel configuration and transpose type
    integer(int32),           intent(in)  :: dims(:)        !! Local dimensions of the input data
    integer(int32),           intent(in)  :: tile_dim       !! Tile dimension
    integer(int32),           intent(in)  :: other_dim      !! Other dimension (not tiled)
    type(kernel_type_t),      intent(in)  :: kernel_type    !! Type of kernel
    type(kernel_config),      intent(in)  :: config         !! Kernel configuration
    integer(int64),           intent(in)  :: base_storage   !! Number of bytes needed to store single element
    integer(int32), optional, intent(in)  :: neighbor_data(:) !! Neighboring data dimensions for pipelined kernels
    real(real32)                          :: score          !! Coalescing score
    real(real32)    :: read_efficiency     !! Read efficiency score
    real(real32)    :: write_efficiency    !! Write efficiency score
    real(real32)    :: thread_utilization  !! Thread utilization score
    real(real32)    :: dimension_penalty   !! Penalty based on dimension sizes
    integer(int32)  :: tile_size           !! Tile size (X dimension)
    integer(int32)  :: block_rows          !! Number of rows in the block (Y dimension)
    integer(int32)  :: threads_per_block   !! Number of threads per block
    integer(int32)  :: nx, ny, nz          !! Problem dimensions
    integer(int32)  :: read_stride         !! Stride for reading
    integer(int32)  :: write_stride        !! Stride for writing
    real(real32)    :: cache_efficiency    !! Cache line utilization efficiency

    tile_size = config%tile_size
    block_rows = config%block_rows
    threads_per_block = tile_size * block_rows
    nx = dims(1)
    ny = dims(tile_dim)
    nz = dims(other_dim)

    ! Analyze memory access patterns based on transpose type
    select case ( kernel_type%val )
    case ( KERNEL_PERMUTE_FORWARD%val )
      read_stride = nx
      write_stride = ny * nz
    case ( KERNEL_PERMUTE_BACKWARD%val )
      read_stride = nx * ny
      write_stride = nz
    case ( KERNEL_PERMUTE_BACKWARD_START%val )
      read_stride = nx * ny
      write_stride = ny * nz
    case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
      read_stride = neighbor_data(1) * neighbor_data(3)
      write_stride = nx
    case ( KERNEL_UNPACK_PIPELINED%val )
      read_stride = neighbor_data(1)
      write_stride = nx
    case ( KERNEL_PACK%val )
      read_stride = nx
      write_stride = neighbor_data(1)
    case default
      INTERNAL_ERROR("estimate_coalescing: unknown kernel_type")
    endselect

    ! Calculate read efficiency based on memory access pattern
    if (tile_size >= WARP_SIZE) then
      ! Full warp coalescing for reads
      read_efficiency = 1.0
    else
      read_efficiency = real(tile_size) / real(WARP_SIZE)
    end if

    ! Apply stride penalty for reads
    if (read_stride == 1) then
      ! Perfect coalescing
      read_efficiency = read_efficiency * 1.0
    else if (read_stride <= 4) then
      ! Good coalescing
      read_efficiency = read_efficiency * 0.95
    else if (read_stride <= 32) then
      ! Acceptable coalescing
      read_efficiency = read_efficiency * 0.85
    else if (read_stride <= 128) then
      ! Poor coalescing
      read_efficiency = read_efficiency * 0.70
    else
      ! Very poor coalescing
      read_efficiency = read_efficiency * 0.50
    end if

    ! Calculate write efficiency based on stride and dimension sizes
    if (write_stride == 1) then
      write_efficiency = 1.0
    else if (write_stride <= 4) then
      write_efficiency = 0.95
    else if (write_stride <= 16) then
      write_efficiency = 0.85
    else if (write_stride <= 64) then
      write_efficiency = 0.70
    else if (write_stride <= 256) then
      write_efficiency = 0.55
    else if (write_stride <= 1024) then
      write_efficiency = 0.40
    else
      write_efficiency = 0.25
    end if

    ! Apply dimension-specific penalties
    select case ( kernel_type%val )
    case ( KERNEL_PERMUTE_BACKWARD_START%val, KERNEL_PERMUTE_BACKWARD%val )
      if (min(nx, nz) < 16) then
        dimension_penalty = 0.75
      else if (min(nx, nz) < 32) then
        dimension_penalty = 0.85
      else
        dimension_penalty = 0.92
      end if
    case default
      if (min(ny, nz) < 16) then
        dimension_penalty = 0.80
      else if (min(ny, nz) < 32) then
        dimension_penalty = 0.88
      else
        dimension_penalty = 0.95
      end if
    endselect

    write_efficiency = write_efficiency * dimension_penalty

    ! Cache line utilization based on element size
    if (base_storage == DOUBLE_COMPLEX_STORAGE_SIZE) then
      cache_efficiency = 0.85
    else if (base_storage == DOUBLE_STORAGE_SIZE) then
      cache_efficiency = 0.92
    else
      cache_efficiency = 1.0
    end if

    read_efficiency = read_efficiency * cache_efficiency
    write_efficiency = write_efficiency * cache_efficiency

    ! Thread utilization based on block size
    if (threads_per_block >= 512) then
      thread_utilization = 1.0
    else if (threads_per_block >= 256) then
      thread_utilization = 0.95
    else if (threads_per_block >= 128) then
      thread_utilization = 0.90
    else if (threads_per_block >= 64) then
      thread_utilization = 0.80
    else
      thread_utilization = 0.60
    end if

    ! Combined score - writing is more critical for transpose operations
    score = 0.20 * read_efficiency + 0.65 * write_efficiency + 0.15 * thread_utilization
  end function estimate_coalescing

  subroutine generate_candidates(dims, tile_dim, other_dim, base_storage, props, candidates, num_candidates)
  !! Generate kernel configuration candidates for given problem
    integer(int32),           intent(in)  :: dims(:)        !! Local dimensions of the input data, always 3D
    integer(int32),           intent(in)  :: tile_dim       !! Tile dimension
    integer(int32),           intent(in)  :: other_dim      !! Other dimension (not tiled)
    integer(int64),           intent(in)  :: base_storage   !! Number of bytes needed to store single element
    type(device_props),       intent(in)  :: props          !! GPU architecture properties
    type(kernel_config),      intent(out) :: candidates(:)  !! Generated kernel configurations
    integer(int32),           intent(out) :: num_candidates !! Number of generated candidates
    integer(int32)  :: nx                 !! Local dimension X
    integer(int32)  :: ny                 !! Local dimension Y
    integer(int32)  :: tile_sizes(N_TILES_CANDIDATES)      !! Tile sizes to consider
    integer(int32)  :: block_heights(N_BLOCKS_CANDIDATES)   !! Block heights to consider
    integer(int32)  :: i                  !! Counter
    integer(int32)  :: j                  !! Counter
    integer(int32)  :: k                  !! Counter
    integer(int32)  :: tile_size          !! Current tile size
    integer(int32)  :: block_rows         !! Current number of block rows
    logical         :: is_asymmetric      !! Flag for asymmetric problem
    logical         :: is_square          !! Flag for square problem
    logical         :: is_large           !! Flag for large problem
    logical         :: is_found           !! Flag for found configuration
    real(real32)    :: aspect_ratio       !! Aspect ratio of the problem
    real(real32)    :: memory_bandwidth   !! Estimated memory bandwidth pressure
    integer(int32)  :: optimal_tile_size  !! Optimal tile size
    integer(int32)  :: optimal_block_rows !! Optimal number of block rows
    integer(int32)  :: threads_per_block  !! Number of threads per block
    integer(int32)  :: padding            !! Padding size
    integer(int64)  :: shared_mem_usage   !! Shared memory usage per block

    ! WRITE_DEBUG("Generating kernel configuration candidates..")
    nx = dims(1)
    ny = dims(tile_dim)

    is_square = abs(nx - ny) <= max(nx, ny) / 10
    is_large = 1_int64 * product(dims) > (int(props%l2_cache_size, int64) / base_storage)
    is_asymmetric = (nx > 4 * ny) .or. (ny > 4 * nx)
    aspect_ratio = real(max(nx, ny)) / real(min(nx, ny))

    memory_bandwidth = estimate_memory_pressure(dims, tile_dim, other_dim, base_storage, props)

    if (is_asymmetric) then
      if (nx > ny) then
        ! Prefer more rows
        optimal_tile_size = min(64, max(16, ny))
        optimal_block_rows = min(16, max(8, ny / 4))
      else
        ! Prefer fewer rows
        optimal_tile_size = min(128, max(32, nx))
        optimal_block_rows = min(8, max(2, nx / 16))
      end if
      call find_valid_combination(optimal_tile_size, optimal_block_rows)
    else if (is_large .and. memory_bandwidth > 0.7) then
      ! Large matrix with high memory pressure - prefer larger blocks
      optimal_tile_size = 64
      if (base_storage >= DOUBLE_STORAGE_SIZE) then
        optimal_block_rows = 8
      else
        optimal_block_rows = 16
      end if
    else if (.not. is_large ) then
      ! Small matrix with low pressure - prefer smaller blocks
      optimal_tile_size = 16
      optimal_block_rows = 4
    else
      ! Default case
      optimal_tile_size = 32
      if (base_storage >= DOUBLE_STORAGE_SIZE) then
        optimal_block_rows = 4
      else
        optimal_block_rows = 8
      end if
    end if

    ! Forming 5 variants of tile sizes around the optimal
    if ( mod(optimal_tile_size / 4, WARP_SIZE) == 0) then
      tile_sizes(1) = max(8, optimal_tile_size / 4)
    else
      tile_sizes(1) = 16
    endif
    if ( mod(optimal_tile_size / 2, WARP_SIZE) == 0) then
      tile_sizes(2) = max(16, optimal_tile_size / 2)
    else
      tile_sizes(2) = 32
    endif
    ! tile_sizes(2) = max(16, optimal_tile_size / 2)
    tile_sizes(3) = optimal_tile_size
    tile_sizes(4) = min(128, optimal_tile_size * 2)
    tile_sizes(5) = min(256, optimal_tile_size * 4)

    ! Forming 5 variants of block heights around the optimal
    ! Avoiding too small configurations (minimum 2 rows for reasonable occupancy)
    block_heights(1) = max(2, optimal_block_rows / 2)
    block_heights(2) = optimal_block_rows
    block_heights(3) = min(16, optimal_block_rows * 2)
    block_heights(4) = min(32, optimal_block_rows * 4)
    block_heights(5) = min(64, optimal_block_rows * 8)

    num_candidates = 0
    do i = 1, size(tile_sizes)
      do j = 1, size(block_heights)
        if (num_candidates > size(candidates)) exit
        tile_size = tile_sizes(i)
        block_rows = block_heights(j)
        threads_per_block = tile_size * block_rows

        if (tile_size * block_rows > props%max_threads_per_block) cycle
        if (threads_per_block < 2 * WARP_SIZE) cycle
        if (tile_size < block_rows) cycle
        if (mod(tile_size * block_rows, WARP_SIZE) /= 0) cycle

        padding = estimate_optimal_padding(tile_size, block_rows, base_storage)
        shared_mem_usage = tile_size * (tile_size + padding) * base_storage
        ! Reserving 10% of shared memory
        if (shared_mem_usage >= int(props%shared_mem_per_block * 0.9_real64, int64)) cycle

        ! Checking if candidate was added before
        is_found = .false.
        do k = 1, num_candidates
          if ( candidates(k)%tile_size == tile_size .and. candidates(k)%block_rows == block_rows ) is_found = .true.
        enddo
        if ( is_found ) cycle
        WRITE_DEBUG("Adding candidate for consideration: "//to_str(tile_size)//"x"//to_str(block_rows)//", padding = "//to_str(padding))
        num_candidates = num_candidates + 1
        candidates(num_candidates)%tile_size = tile_size
        candidates(num_candidates)%block_rows = block_rows
        ! candidates(num_candidates)%threads%x = tile_size
        ! candidates(num_candidates)%threads%y = block_rows
        ! candidates(num_candidates)%threads%z = 1
        ! candidates(num_candidates)%blocks%x = (nx + tile_size - 1) / tile_size
        ! candidates(num_candidates)%blocks%y = (ny + tile_size - 1) / tile_size
        ! candidates(num_candidates)%blocks%z = dims(other_dim)
        candidates(num_candidates)%padding = padding
      end do
      if (num_candidates > size(candidates)) exit
    end do
    ! WRITE_DEBUG("Generated "//to_str(num_candidates)//" candidates")
  end subroutine generate_candidates

  subroutine find_valid_combination(base_tile, base_rows)
  !! This subroutine optimizes the tile size and number of rows for narrow matrices
  !! by adjusting them to be compatible with the warp size.
    integer(int32), intent(inout) :: base_tile    !!< Tile size
    integer(int32), intent(inout) :: base_rows    !!< Number of rows
    integer(int32) :: best_tile !! Optimized tile size
    integer(int32) :: best_rows !! Optimized number of rows

    best_tile = (base_tile + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE
    best_tile = min(256, max(8, best_tile))

    best_rows = ((base_rows + best_tile - 1) / best_tile) * best_tile
    best_rows = min(4, max(best_tile, best_rows))

    if (mod(best_tile * best_rows, WARP_SIZE) /= 0) then
      best_rows = best_rows + best_tile
      best_rows = min(4, best_rows)
    end if

    base_tile = best_tile
    base_rows = best_rows
  end subroutine find_valid_combination

  function evaluate_analytical_performance(dims, tile_dim, other_dim, kernel_type, config, props, base_storage, neighbor_data) result(score)
  !! This function evaluates the performance of a kernel configuration
  !! based on various architectural and problem-specific parameters.
    integer(int32),           intent(in)  :: dims(:)        !! Problem dimensions
    integer(int32),           intent(in)  :: tile_dim       !! Tile dimension
    integer(int32),           intent(in)  :: other_dim      !! Other dimension (not tiled)
    type(kernel_type_t),      intent(in)  :: kernel_type    !! Type of kernel_type to evaluate
    type(kernel_config),      intent(in)  :: config         !! Kernel configuration
    type(device_props),       intent(in)  :: props          !! GPU architecture properties
    integer(int64),           intent(in)  :: base_storage   !! Number of bytes needed to store single element
    integer(int32), optional, intent(in)  :: neighbor_data(:) !! Neighboring data dimensions for pipelined kernels
    real(real32)                          :: score          !! Performance score
    integer(int32)  :: n_bank_conflicts             !! Number of bank conflicts
    integer(int32)  :: tile_size                    !! Tile size
    integer(int32)  :: block_rows                   !! Number of rows in a block
    real(real32)    :: occupancy_score              !! Occupancy score
    real(real32)    :: memory_access_score          !! Memory access score
    real(real32)    :: computation_efficiency_score !! Computation efficiency score
    real(real32)    :: occupancy                    !! Raw occupancy score
    real(real32)    :: coalescing_efficiency        !! Coalescing efficiency score
    real(real32)    :: bank_conflict_ratio          !! Bank conflict ratio
    ! real(real32)    :: x_waste                      !! X waste
    ! real(real32)    :: y_waste                      !! Y waste
    ! real(real32)    :: total_efficiency             !! Total efficiency

    tile_size = config%tile_size
    block_rows = config%block_rows

    occupancy = estimate_occupancy(config, props, base_storage)
    if (occupancy >= 0.5 .and. occupancy <= 0.75) then
      occupancy_score = 1.0
    else if (occupancy >= 0.3 .and. occupancy < 0.5) then
      occupancy_score = 0.7 + 0.3 * (occupancy - 0.3) / 0.2
    else if (occupancy > 0.75 .and. occupancy <= 1.0) then
      occupancy_score = 1.0 - 0.2 * (occupancy - 0.75) / 0.25
    else
      occupancy_score = 0.3
    end if

    coalescing_efficiency = estimate_coalescing(dims, tile_dim, other_dim, kernel_type, config, base_storage, neighbor_data)
    bank_conflict_ratio = estimate_bank_conflict_ratio(config, base_storage)
    memory_access_score = 0.4 * coalescing_efficiency + 0.6 * (1.0 - bank_conflict_ratio)

    if (mod(tile_size, WARP_SIZE) == 0) then
      computation_efficiency_score = 1.0
    else if (mod(tile_size, 16) == 0) then
      computation_efficiency_score = 0.9
    else if (mod(tile_size, 8) == 0) then
      computation_efficiency_score = 0.8
    else
      computation_efficiency_score = 0.7
    end if

    ! x_waste = real(config%blocks%x * tile_size - dims(1)) / real(config%blocks%x * tile_size)
    ! y_waste = real(config%blocks%y * tile_size - dims(2)) / real(config%blocks%y * tile_size)

    ! total_efficiency = (1.0 - x_waste) * (1.0 - y_waste)

    ! if (total_efficiency >= 0.95) then
    !   workload_balance_score = 1.0
    ! else if (total_efficiency >= 0.85) then
    !   workload_balance_score = 0.9 + 0.1 * (total_efficiency - 0.85) / 0.10
    ! else if (total_efficiency >= 0.75) then
    !   workload_balance_score = 0.8 + 0.1 * (total_efficiency - 0.75) / 0.10
    ! else if (total_efficiency >= 0.50) then
    !   workload_balance_score = 0.6 + 0.2 * (total_efficiency - 0.50) / 0.25
    ! else
    !   workload_balance_score = 0.3 + 0.3 * total_efficiency / 0.50
    ! end if

    score = 0.35 * occupancy_score                &
          + 0.55 * memory_access_score            &
          + 0.10 * computation_efficiency_score
          ! + 0.10 * workload_balance_score

    score = max(0.0, min(1.0, score))
    n_bank_conflicts = count_bank_conflicts(tile_size, block_rows, base_storage, config%padding)

    ! WRITE_DEBUG("=== Performance Analysis for "//to_str(tile_size)//"x"//to_str(block_rows)//", padding = "//to_str(config%padding)//" ===")
    ! WRITE_DEBUG("  Occupancy score:                "//to_str(occupancy_score)//" (occupancy: "//to_str(occupancy)//")")
    ! WRITE_DEBUG("  Memory access score:            "//to_str(memory_access_score))
    ! WRITE_DEBUG("    - Coalescing:                 "//to_str(coalescing_efficiency))
    ! WRITE_DEBUG("    - Bank conflict ratio:        "//to_str(bank_conflict_ratio))
    ! WRITE_DEBUG("    - Number of bank conflicts:   "//to_str(n_bank_conflicts))
    ! WRITE_DEBUG("  Computational efficiency score: "//to_str(computation_efficiency_score))
    ! WRITE_DEBUG("  Workload balance score:         "//to_str(workload_balance_score)//" (efficiency: "//to_str(total_efficiency)//")")
    ! WRITE_DEBUG("  Final score: "//to_str(score))
  end function evaluate_analytical_performance

  subroutine sort_candidates_by_score(scores, num_candidates, sorted_indices)
  !! Sorting candidates by their performance scores
    real(real32),   intent(in)  :: scores(:)          !! Performance scores of candidates generated by `evaluate_analytical_performance`
    integer(int32), intent(in)  :: num_candidates     !! Number of candidates
    integer(int32), intent(out) :: sorted_indices(:)  !! Sorted indices of candidates
    integer(int32)  :: i            !! Counter
    integer(int32)  :: j            !! Counter
    integer(int32)  :: max_idx      !! Index of the maximum element
    real(real32)    :: max_score    !! Maximum score found
    logical :: used(num_candidates) !! Array to track used candidates

    used = .false.

    do i = 1, num_candidates
      max_score = -1.0
      max_idx = 0
      do j = 1, num_candidates
        if (.not. used(j) .and. scores(j) > max_score) then
          max_score = scores(j)
          max_idx = j
        end if
      end do

      if (max_idx > 0) then
        sorted_indices(i) = max_idx
        used(max_idx) = .true.
      else
        sorted_indices(i) = i  ! Fallback
      end if
    end do
  end subroutine sort_candidates_by_score
end module dtfft_nvrtc_block_optimizer