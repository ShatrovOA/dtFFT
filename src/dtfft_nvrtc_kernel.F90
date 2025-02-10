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
module dtfft_nvrtc_kernel
!! This module describes NVRTC Kernel class.
!! It uses caching of compiled kernels to avoid recompilation similar kernels.
use iso_c_binding
use iso_fortran_env
use dtfft_utils
use dtfft_nvrtc_interfaces
use dtfft_parameters
use cudafor
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft_profile.h"
implicit none
private
public :: nvrtc_kernel
public :: DEF_TILE_SIZE
public :: clean_unused_cache

  integer(int32), parameter :: DEF_TILE_SIZE = 16
  !! Default tile size
  integer(int32), parameter :: MIN_TILE_SIZE = 8
  !! Minimum tile size. Will launch 2 warps
  integer(int32), parameter :: TARGET_THREADS_PER_BLOCK = DEF_TILE_SIZE * DEF_TILE_SIZE
  !! Maximum number of threads to run in a block (256)

  character(len=*), parameter :: DEFAULT_KERNEL_NAME = "dtfft_kernel"
  !! Basic kernel name

  integer(int8), parameter, public  :: KERNEL_TRANSPOSE           = 1
  !! Basic transpose kernel type.
  integer(int8), parameter, public  :: KERNEL_TRANSPOSE_PACKED    = 2
  !! Transposes data and packs it into contiguous buffer.
  !! Should be used only in X-Y 3D plans.
  integer(int8), parameter, public  :: KERNEL_UNPACK              = 3
  !! Unpacks contiguous buffer.
  ! integer(int8), parameter, public  :: KERNEL_UNPACK_SIMPLE_COPY  = 4
  !! Doesn't actually unpacks anything. Performs ``cudaMemcpyAsync`` call.
  !! Should be used only when backend is ``DTFFT_GPU_BACKEND_CUFFTMP``.
  integer(int8), parameter, public  :: KERNEL_UNPACK_PIPELINED    = 5
  !! Unpacks pack of contiguous buffer recieved from rank.

  type :: kernel_code
  !! Class to build CUDA kernel code
  private
    character(len=:), allocatable :: raw                      !< String that holds CUDA code
  contains
  private
    procedure, pass(self),  public :: to_cstr                 !< Converts Fortran CUDA code to C pointer
    procedure, pass(self),  public :: add_line                !< Adds new line to CUDA code
    procedure, pass(self),  public :: destroy => destroy_code !< Frees all memory
  end type kernel_code

  type :: string
  !! Class used to create array of strings
  private
    character(len=:), allocatable :: raw                      !< String
  end type string

  type :: nvrtc_kernel
  !! nvRTC Compiled kernel class
  private
    logical                       :: is_created = .false.     !< Kernel is created flag.
    logical                       :: is_dummy = .false.       !< If kernel should do anything or not.
    type(c_ptr)                   :: cuda_kernel              !< Pointer to CUDA kernel.
    type(dim3)                    :: num_blocks               !< Grid of blocks.
    type(dim3)                    :: block_size               !< Thread block.
    integer(int8)                 :: kernel_type              !< Type of kernel to execute.
    type(kernelArgs)              :: args                     !< Kernel arguments.
    integer(int32),   allocatable :: pointers(:,:)            !< Optional pointers that hold info about counts and displacements
                                                              !< in ``KERNEL_UNPACK_PIPELINED`` kernel.
  contains
  private
    procedure,  pass(self), public  :: create                 !< Creates kernel
    procedure,  pass(self), public  :: execute                !< Executes kernel
    procedure,  pass(self), public  :: destroy                !< Destroys kernel
  end type nvrtc_kernel

  type :: nvrtc_cache
  !! Class to cache compiled kernels
  private
    integer(int32)                :: ref_count = 0            !< Number of references to this kernel
    type(c_ptr)                   :: cuda_module = c_null_ptr !< Pointer to CUDA Module.
    type(c_ptr)                   :: cuda_kernel = c_null_ptr !< Pointer to CUDA kernel.
    integer(int8)                 :: kernel_type              !< Type of kernel to execute.
    type(dtfft_transpose_type_t)  :: transpose_type             !< Type of transpose
    integer(int32)                :: tile_size                !< Tile size of transpose kernel
    integer(int8)                 :: base_storage             !< Number of bytes needed to store single element
    logical                       :: has_inner_loop           !< If kernel has inner loop
  end type nvrtc_cache

  integer(int32),     parameter         :: CACHE_PREALLOC_SIZE = 10
  !! Number of preallocated cache entries
  type(nvrtc_cache),  allocatable, save :: cache(:)
  !! Cache of compiled kernels
  integer(int32),                  save :: cache_size = 0
  !! Number of entries in cache
contains

  subroutine to_cstr(self, c_code)
  !! Converts Fortran CUDA code to C pointer
    class(kernel_code),             intent(in)    :: self     !< Kernel code
    character(c_char), allocatable, intent(out)   :: c_code(:)!< C pointer to code

    call astring_f2c(self%raw, c_code)
  end subroutine to_cstr

  subroutine add_line(self, line)
  !! Adds new line to CUDA code
    class(kernel_code),             intent(inout) :: self     !< Kernel code
    character(len=*),               intent(in)    :: line     !< Line to add

    if ( .not. allocated( self%raw ) ) allocate(self%raw, source="")
    self%raw = self%raw // line // c_new_line
  end subroutine add_line

  subroutine destroy_code(self)
  !! Frees all memory
    class(kernel_code),             intent(inout) :: self     !< Kernel code

    if ( allocated( self%raw ) ) deallocate(self%raw)
  end subroutine destroy_code

  subroutine create_device_pointer(ptr, values)
  !! Allocates memory on a device and copies ``values`` to it.
    type(c_devptr),     intent(inout)       :: ptr            !< Device pointer
    integer(c_int),     intent(in), target  :: values(:)      !< Values to copy
    integer(c_size_t)                       :: n_bytes        !< Number of bytes to copy

    n_bytes = size(values) * c_sizeof(c_int)
    CUDA_CALL( "cudaMalloc", cudaMalloc(ptr, n_bytes) )
    CUDA_CALL( "cudaMemcpy", cudaMemcpy(ptr, c_loc(values), n_bytes, cudaMemcpyHostToDevice) )
  end subroutine create_device_pointer

  integer(int32) function get_tile_size(x, y)
  !! Returns tile size to use in a tranpose kernel
    integer(int32), intent(in)  :: x     !< Number of elements in x direction
    integer(int32), intent(in)  :: y     !< Number of elements in y direction

    if ( minval([x, y]) >= DEF_TILE_SIZE ) then
      get_tile_size = DEF_TILE_SIZE
    else
      get_tile_size = MIN_TILE_SIZE
    endif
  end function get_tile_size

  subroutine get_contiguous_execution_blocks(size, num_blocks, block_sizes)
    integer(int32), intent(in)  :: size         !< Total amount of iterations required
    type(dim3),     intent(out) :: num_blocks   !< Grid of blocks.
    type(dim3),     intent(out) :: block_sizes  !< Thread block.
    integer(int32)              :: block_size   !< Number of threads in a block

    if ( size < TARGET_THREADS_PER_BLOCK ) then
      block_size = size
    else
      block_size = TARGET_THREADS_PER_BLOCK
    endif
    block_sizes%x = block_size
    block_sizes%y = 1
    block_sizes%z = 1

    num_blocks%x = (size + block_size - 1) / block_size
    num_blocks%y = 1
    num_blocks%z = 1
  end subroutine get_contiguous_execution_blocks

  subroutine create(self, comm, dims, base_storage, transpose_type, kernel_type, pointers)
  !! Creates kernel
    class(nvrtc_kernel),          intent(inout) :: self               !< nvRTC Compiled kernel class
    TYPE_MPI_COMM,                intent(in)    :: comm               !< MPI Communicator
    integer(int32), target,       intent(in)    :: dims(0:)           !< Global dimensions to process
    integer(int8),                intent(in)    :: base_storage       !< Number of bytes needed to store single element
    type(dtfft_transpose_type_t), intent(in)    :: transpose_type     !< Type of transposition to perform
    integer(int8),                intent(in)    :: kernel_type        !< Type of kernel to build
    integer(int32), optional,     intent(in)    :: pointers(:,:)      !< Optional pointers to unpack kernels
    integer(int32)  :: comm_size          !< Number of processes in current MPI communicator
    integer(int32)  :: mpi_ierr           !< Error code
    integer(int32)  :: tile_dim           !< Dimension to tile
    integer(int32)  :: other_dim          !< Dimension not used to tile
    integer(int32)  :: tile_size          !< Tile size
    integer(int32)  :: scaler             !< Scaler to adjust number of blocks
    logical         :: has_inner_loop     !< If kernel has inner loop

    call self%destroy()

    if ( any(dims == 0) ) then
      self%is_created = .true.
      self%is_dummy = .true.
      return
    endif
    self%is_dummy = .false.
    self%kernel_type = kernel_type

    ! if ( kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) then
    !   self%is_created = .true.
    !   self%args%ints(1) = product(dims) * base_storage
    !   return
    ! endif

    call MPI_Comm_size(comm, comm_size, mpi_ierr)

    has_inner_loop = .false.
    tile_size = 0
    if ( kernel_type == KERNEL_UNPACK ) then
      call get_contiguous_execution_blocks(product(dims), self%num_blocks, self%block_size)
    else if ( (kernel_type == KERNEL_TRANSPOSE) .or. (kernel_type == KERNEL_TRANSPOSE_PACKED) ) then
      if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        tile_dim = 1
        other_dim = 2
      else
        tile_dim = 2
        other_dim = 1
      endif

      tile_size = get_tile_size(dims(0), dims(tile_dim))

      self%block_size%x = tile_size
      self%block_size%y = tile_size
      self%block_size%z = 1

      self%num_blocks%x = (dims(0) + tile_size - 1) / tile_size
      self%num_blocks%y = (dims(tile_dim) + tile_size - 1) / tile_size

      scaler = 1
      if ( size(dims) == 3 ) then
        if ( tile_size == MIN_TILE_SIZE .and. dims(other_dim) > TARGET_THREADS_PER_BLOCK ) then
          has_inner_loop = .true.
          select case (base_storage)
          case (FLOAT_STORAGE_SIZE)
            scaler = 8
          case (DOUBLE_STORAGE_SIZE)
            scaler = 4
          case (DOUBLE_COMPLEX_STORAGE_SIZE)
            scaler = 2
          endselect
        endif
        self%num_blocks%z = (dims(other_dim) + scaler - 1) / scaler
      else
        self%num_blocks%z = 1
      endif
    endif

    if ( kernel_type == KERNEL_UNPACK ) then
      self%args%n_ints = 3
      self%args%ints(1) = product(dims)
      if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        self%args%ints(2) = dims(0) * dims(1)
      else
        self%args%ints(2) = dims(0)
      endif
      self%args%ints(3) = comm_size
    else if ( kernel_type == KERNEL_UNPACK_PIPELINED ) then
      self%args%n_ints = 5
      self%args%ints(1) = product(dims)
      self%args%ints(2) = dims(0)
      ! Other 3 args are populated during kernel execution based on sender pointers
    else
      self%args%n_ints = size(dims)
      self%args%ints(1) = dims(0)
      self%args%ints(2) = dims(1)
      if ( size(dims) == 3 ) then
        self%args%ints(3) = dims(2)
        if ( kernel_type == KERNEL_TRANSPOSE_PACKED ) then
          self%args%n_ints = 4
          self%args%ints(4) = comm_size
        endif
      endif
    endif

    self%args%n_ptrs = 0
    if ( kernel_type == KERNEL_TRANSPOSE_PACKED .or. kernel_type == KERNEL_UNPACK .or. kernel_type == KERNEL_UNPACK_PIPELINED ) then
      if ( .not. present(pointers) ) error stop "Pointer required"

      if (kernel_type == KERNEL_TRANSPOSE_PACKED .or. kernel_type == KERNEL_UNPACK) then
        block
          integer(int32) :: i
          self%args%n_ptrs = size(self%args%ptrs)
          do i = 1, self%args%n_ptrs
            call create_device_pointer(self%args%ptrs(i), pointers(:, i))
          enddo
        endblock
      else
        allocate( self%pointers, source=pointers )
      endif
    endif

    self%cuda_kernel = compile_and_cache(comm, dims, transpose_type, kernel_type, base_storage, tile_size, has_inner_loop)
    self%is_created = .true.
  end subroutine create

  subroutine execute(self, in, out, stream, source)
  !! Executes kernel on stream
    class(nvrtc_kernel),          intent(inout) :: self               !< nvRTC Compiled kernel class
    real(real32),  DEVICE_PTR     intent(in)    :: in(:)              !< Source pointer
    real(real32),  DEVICE_PTR     intent(in)    :: out(:)             !< Target pointer
    integer(cuda_stream_kind),    intent(in)    :: stream             !< CUDA Stream
    integer(int32),   optional,   intent(in)    :: source             !< Source rank for pipelined unpacking
    integer(int32)    :: n_align_sent !< Number of aligned elements sent
    integer(int32)    :: displ_in     !< Displacement in source buffer
    integer(int32)    :: displ_out    !< Displacement in target buffer

    if ( self%is_dummy ) return
    if ( .not. self%is_created ) error stop "dtFFT Internal Error: `execute` called while plan not created"

    ! if ( self%kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) then
    !   CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(out, in, self%args%ints(1), cudaMemcpyDeviceToDevice, stream) )
    !   return
    ! endif

    if ( self%kernel_type == KERNEL_UNPACK_PIPELINED ) then
      if ( .not. present(source) ) error stop "Source is not passed"
      displ_in = self%pointers(source, 1)
      displ_out = self%pointers(source, 2)
      n_align_sent = self%pointers(source, 3)

      self%args%ints(3) = n_align_sent
      self%args%ints(4) = displ_in
      self%args%ints(5) = displ_out

      call get_contiguous_execution_blocks(self%pointers(source, 4), self%num_blocks, self%block_size)
    endif

    CUDA_CALL( "cuLaunchKernel", run_cuda_kernel(self%cuda_kernel, c_devloc(in), c_devloc(out), self%num_blocks, self%block_size, stream, self%args) )
    ! CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  end subroutine execute

  subroutine destroy(self)
  !! Destroys kernel
    class(nvrtc_kernel),          intent(inout) :: self               !< nvRTC Compiled kernel class
    integer(int32)  :: i  !< Counter

    if ( .not. self%is_created ) return
    if ( self%is_dummy ) return
    !  .or. self%kernel_type == KERNEL_UNPACK_SIMPLE_COPY ) return

    call mark_unused(self%cuda_kernel)

    do i = 1, self%args%n_ptrs
      CUDA_CALL( "cudaFree", cudaFree(self%args%ptrs(i)) )
    enddo
    if ( allocated(self%pointers) ) deallocate(self%pointers)

    self%args%n_ints = 0
    self%args%n_ptrs = 0
    self%cuda_kernel = c_null_ptr
    self%is_created = .false.
  end subroutine destroy

  function get_cached_kernel(transpose_type, kernel_type, base_storage, tile_size, has_inner_loop) result(kernel)
  !! Returns cached kernel if it exists.
  !! If not returns null pointer.
    type(dtfft_transpose_type_t), intent(in)    :: transpose_type       !< Type of transposition to perform
    integer(int8),                intent(in)    :: kernel_type        !< Type of kernel to build
    integer(int8),                intent(in)    :: base_storage       !< Number of bytes needed to store single element
    integer(int32),               intent(in)    :: tile_size          !< Tile size
    logical,                      intent(in)    :: has_inner_loop     !< If kernel has inner loop
    type(c_ptr)                   :: kernel             !< Cached kernel
    type(dtfft_transpose_type_t)  :: transpose_type_    !< Fixed id of transposition
    integer(int32)                :: i                !< Counter

    kernel = c_null_ptr
    transpose_type_ = get_true_transpose_type(transpose_type)
    if ( .not. allocated(cache) ) return
    do i = 1, cache_size
      if ( cache(i)%transpose_type == transpose_type_         &
     .and. cache(i)%kernel_type == kernel_type            &
     .and. cache(i)%base_storage == base_storage          &
     .and. cache(i)%tile_size == tile_size                &
     .and. ( (cache(i)%has_inner_loop .and. has_inner_loop) .or. (.not.cache(i)%has_inner_loop .and. .not.has_inner_loop)) &
     .or. ( cache(i)%kernel_type == kernel_type .and. (kernel_type == KERNEL_UNPACK .or. kernel_type == KERNEL_UNPACK_PIPELINED) )  &
     ) then
      kernel = cache(i)%cuda_kernel
      cache(i)%ref_count = cache(i)%ref_count + 1
      return
     endif
    end do
  end function get_cached_kernel

  function get_true_transpose_type(transpose_type) result(transpose_type_)
  !! Returns generic transpose id.
  !! Since X-Y and Y-Z transpositions are symmectric, it returns only one of them.
  !! X-Z and Z-X are not symmetric
    type(dtfft_transpose_type_t), intent(in)    :: transpose_type       !< Type of transposition to perform
    type(dtfft_transpose_type_t)                :: transpose_type_      !< Fixed id of transposition

    if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
      transpose_type_ = transpose_type
    else
      transpose_type_%val = abs(transpose_type%val)
    endif
  end function get_true_transpose_type

  function compile_and_cache(comm, dims, transpose_type, kernel_type, base_storage, tile_size, has_inner_loop) result(kernel)
  !! Compiles kernel and caches it. Returns compiled kernel.
    TYPE_MPI_COMM,                intent(in)    :: comm               !< MPI Communicator
    integer(int32), target,       intent(in)    :: dims(:)            !< Global dimensions to process
    type(dtfft_transpose_type_t), intent(in)    :: transpose_type     !< Type of transposition to perform
    integer(int8),                intent(in)    :: kernel_type        !< Type of kernel to build
    integer(int8),                intent(in)    :: base_storage       !< Number of bytes needed to store single element
    integer(int32),               intent(in)    :: tile_size          !< Tile size
    logical,                      intent(in)    :: has_inner_loop     !< If kernel has inner loop
    type(c_ptr)                             :: kernel             !< Compiled kernel to return
    type(nvrtc_cache),        allocatable   :: temp(:)            !< Temporary cache
    integer(int32)                          :: i                  !< Counter
    character(len=:),         allocatable   :: kernel_name        !< Name of kernel
    type(kernel_code)                       :: code               !< CUDA Code to compile
    character(c_char),        allocatable   :: c_code(:)          !< CUDA C Code to compile
    type(string),   target,   allocatable   :: options(:)         !< Compilation options
    type(c_ptr),              allocatable   :: c_options(:)       !< C style, null-string terminated options
    integer(int32)                          :: num_options        !< Number of compilation options
    type(dtfft_transpose_type_t)            :: transpose_type_      !< Fixed id of transposition
    integer(int32)                          :: device_id          !< Current device number
    type(cudaDeviceProp)                    :: prop               !< Current device properties
    integer(int32)                          :: ierr               !< Error code
    integer(int32)                          :: mpi_ierr           !< MPI Error code
    type(c_ptr)                             :: prog               !< nvRTC Program
    integer(c_size_t)                       :: cubinSizeRet       !< Size of cubin
    character(c_char),        allocatable   :: cubin(:)           !< Compiled binary

    ! Check if kernel already been compiled
    kernel = get_cached_kernel(transpose_type, kernel_type, base_storage, tile_size, has_inner_loop)
    if ( c_associated(kernel) ) return

    PHASE_BEGIN("Building nvRTC kernel", COLOR_EXECUTE)

    if ( .not. allocated(cache) ) allocate( cache( CACHE_PREALLOC_SIZE ) )
    ! Need more cache
    if ( cache_size == size(cache) ) then

      allocate( temp(cache_size) )
      do i = 1, cache_size
        temp(i) = cache(i)
      enddo
      deallocate( cache )

      allocate( cache(cache_size + CACHE_PREALLOC_SIZE) )
      do i = 1, cache_size
        cache(i) = temp(i)
      enddo
      deallocate( temp )
    endif

    transpose_type_ = get_true_transpose_type(transpose_type)

    kernel_name = DEFAULT_KERNEL_NAME // "_"
    if ( kernel_type == KERNEL_TRANSPOSE .or. kernel_type == KERNEL_TRANSPOSE_PACKED ) then
      select case ( transpose_type_%val )
      case ( DTFFT_TRANSPOSE_X_TO_Y%val )
        kernel_name = kernel_name // "xy"
      case ( DTFFT_TRANSPOSE_X_TO_Z%val )
        kernel_name = kernel_name // "xz"
      case ( DTFFT_TRANSPOSE_Z_TO_X%val )
        kernel_name = kernel_name // "zx"
      case ( DTFFT_TRANSPOSE_Y_TO_Z%val )
        kernel_name = kernel_name // "yz"
      endselect
    else
      kernel_name = kernel_name // "unpack"
    endif

    if ( kernel_type == KERNEL_UNPACK ) then
      code = get_unpack_kernel_code(kernel_name, base_storage)
    else if ( kernel_type == KERNEL_UNPACK_PIPELINED ) then
      code = get_unpack_pipelined_kernel_code(kernel_name, base_storage)
    else
      code = get_transpose_kernel_code(kernel_name, size(dims, kind=int8), base_storage, transpose_type, kernel_type == KERNEL_TRANSPOSE_PACKED, has_inner_loop)
    endif
    call code%to_cstr(c_code)

#ifdef __DEBUG
    num_options = 4
#else
    num_options = 2
#endif

    CUDA_CALL( "cudaGetDevice", cudaGetDevice(device_id) )
    CUDA_CALL( "cudaGetDeviceProperties", cudaGetDeviceProperties(prop, device_id) )

    allocate( c_options(num_options), options(num_options) )
    options(1)%raw = "--gpu-architecture=sm_"//int_to_str(prop%major)//int_to_str(prop%minor)
    options(2)%raw = "-DTILE_DIM="//int_to_str(tile_size)
#ifdef __DEBUG
    options(3)%raw = "--device-debug"
    options(4)%raw = "--generate-line-info"
#endif
    do i = 1, num_options
      options(i)%raw = options(i)%raw // c_null_char
      c_options(i) = c_loc(options(i)%raw)
    enddo

    NVRTC_CALL( "nvrtcCreateProgram", nvrtcCreateProgram(prog, c_code, "nvrtc_kernel.cu"//c_null_char, 0, c_null_ptr, c_null_ptr) )
    ierr = nvrtcCompileProgram(prog, num_options, c_options)
    ! It is assumed here that ierr can only be positive
    call MPI_Allreduce(MPI_IN_PLACE, ierr, 1, MPI_INTEGER4, MPI_MAX, comm, mpi_ierr)
    if ( ierr /= 0 ) then
      block
        type(c_ptr) :: c_log
        character(len=:), allocatable, target :: f_log
        integer(int32) :: global_rank

        f_log = repeat(" ", 5000)
        c_log = c_loc(f_log)

        NVRTC_CALL( "nvrtcGetProgramLog", nvrtcGetProgramLog(prog, c_log))

        call MPI_Comm_rank(comm, global_rank, mpi_ierr)
        if ( global_rank == 0 ) then
          write(error_unit, "(a)") "dtFFT Internal Error: failed to compile kernel"
          write(error_unit, "(a)") "CUDA Code:"
          write(error_unit, "(a)") code%raw
          write(error_unit, "(a)") "Compilation log:"
          write(error_unit, "(a)") trim(f_log)
        endif
        call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_ierr)
      endblock
    endif

    NVRTC_CALL( "nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, cubinSizeRet) )
    allocate( cubin( cubinSizeRet ) )
    NVRTC_CALL( "nvrtcGetCUBIN", nvrtcGetCUBIN(prog, cubin) )
    NVRTC_CALL( "nvrtcDestroyProgram", nvrtcDestroyProgram(prog) )

    cache_size = cache_size + 1
    cache(cache_size)%base_storage = base_storage
    cache(cache_size)%kernel_type = kernel_type
    cache(cache_size)%tile_size = tile_size
    cache(cache_size)%transpose_type = transpose_type_
    cache(cache_size)%has_inner_loop = has_inner_loop
    cache(cache_size)%ref_count = 1

    CUDA_CALL( "cuModuleLoadDataEx", cuModuleLoadDataEx(cache(cache_size)%cuda_module, cubin, 0, c_null_ptr, c_null_ptr) )
    CUDA_CALL( "cuModuleGetFunction", cuModuleGetFunction(cache(cache_size)%cuda_kernel, cache(cache_size)%cuda_module, kernel_name//c_null_char) )
    ! Result -- compiled function
    kernel = cache(cache_size)%cuda_kernel

    deallocate( c_code )
    do i = 1, num_options
      deallocate(options(i)%raw)
    enddo
    deallocate( options )
    deallocate( c_options )
    deallocate( cubin )
    deallocate( kernel_name )
    call code%destroy()

    PHASE_END("Building nvRTC kernel")
  end function compile_and_cache

  subroutine mark_unused(kernel)
  !! Takes CUDA kernel as an argument and searches for it in cache
  !! If kernel is found than reduces `ref_count` and return null pointer
    type(c_ptr),  intent(inout) :: kernel   !< CUDA kernel to search for
    integer(int32)              :: i        !< Counter

    if ( .not. allocated(cache) ) return
    do i = 1, cache_size
      if ( c_associated(cache(i)%cuda_kernel, kernel) ) then
        kernel = c_null_ptr
        cache(i)%ref_count = cache(i)%ref_count - 1
        return
      endif
    end do
  end subroutine mark_unused

  subroutine clean_unused_cache()
  !! Removes unused modules from cuda context
    integer(int32)  :: i  !< Counter

    do i = 1, cache_size
      if ( cache(i)%ref_count == 0 .and. c_associated(cache(i)%cuda_module) ) then
        CUDA_CALL( "cuModuleUnload", cuModuleUnload(cache(i)%cuda_module) )
        cache(i)%cuda_module = c_null_ptr
        cache(i)%cuda_kernel = c_null_ptr
        cache(i)%base_storage = 0
        cache(i)%kernel_type = 0
        cache(i)%tile_size = -1
        cache(i)%transpose_type%val = 0
      endif
    enddo
    if ( all( cache(:)%ref_count == 0 ) ) then
      deallocate( cache )
      cache_size = 0
    endif
  end subroutine clean_unused_cache

  subroutine get_neighbor_function_code(code)
  !! Generated device function that is used to determite id of process that to which data is being sent or from which data has been recieved
  !! based on local element coordinate
    type(kernel_code),  intent(inout) :: code   !< Resulting code

    call code%add_line("__device__")
    call code%add_line("int findNeighborIdx(const int *a, int n, int val) {")
    ! call code%add_line("  if ( a[0] == val ) return 0;")
    call code%add_line("  if ( a[n - 1] <= val) return n - 1;")
    call code%add_line("  if ( n == 2 ) {")
    call code%add_line("    // Since [n - 1] already been handled")
    call code%add_line("    return 0;")
    call code%add_line("  }")
    call code%add_line("  int low = 0, high = n - 1;")
    call code%add_line("  while (1) {")
    call code%add_line("    int mid = (low + high) / 2;")
    call code%add_line("    if (a[mid] <= val && a[mid + 1] > val) {")
    call code%add_line("      return mid;")
    call code%add_line("    } else if (a[mid] < val) {")
    call code%add_line("      low = mid;")
    call code%add_line("    } else {")
    call code%add_line("      high = mid;")
    call code%add_line("    }")
    call code%add_line("  }")
    call code%add_line("}")
  end subroutine get_neighbor_function_code

  subroutine get_code_init(kernel_name, base_storage, code, buffer_type)
  !! Generates basic code that is used in all other kernels
    character(len=*),                         intent(in)    :: kernel_name        !< Name of CUDA kernel
    integer(int8),                            intent(in)    :: base_storage       !< Number of bytes needed to store single element
    type(kernel_code),                        intent(inout) :: code               !< Resulting code
    character(len=:), optional, allocatable,  intent(out)   :: buffer_type        !< Type of buffer that should be used
    character(len=:),           allocatable                 :: buffer_type_       !< Type of buffer that should be used

    select case ( base_storage )
    case ( FLOAT_STORAGE_SIZE )
      allocate( buffer_type_, source="float" )
    case ( DOUBLE_STORAGE_SIZE )
      allocate( buffer_type_, source="double" )
    case ( DOUBLE_COMPLEX_STORAGE_SIZE )
      allocate( buffer_type_, source="double2" )
    case default
      error stop "dtFFT Internal Error: unknown `base_storage`"
    endselect

    call code%add_line('extern "C" __global__')
    call code%add_line("void")
    call code%add_line("__launch_bounds__("//int_to_str(TARGET_THREADS_PER_BLOCK)//")")
    call code%add_line(kernel_name)
    call code%add_line("(")
    call code%add_line("  "//buffer_type_//" * __restrict__ out")
    call code%add_line("   ,const "//buffer_type_//" * __restrict__ in")
    if ( present(buffer_type) ) allocate( buffer_type, source=buffer_type_ )
    deallocate(buffer_type_)
  end subroutine get_code_init

  function get_transpose_kernel_code(kernel_name, ndims, base_storage, transpose_type, enable_packing, enable_multiprocess) result(code)
  !! Generates code that will be used to locally tranpose data and prepares to send it to other processes
    character(len=*),               intent(in)  :: kernel_name              !< Name of CUDA kernel
    integer(int8),                  intent(in)  :: ndims                    !< Number of dimensions
    integer(int8),                  intent(in)  :: base_storage             !< Number of bytes needed to store single element
    type(dtfft_transpose_type_t),   intent(in)  :: transpose_type           !< Transpose id
    logical,                        intent(in)  :: enable_packing           !< If data should be manually packed or not
    logical,                        intent(in)  :: enable_multiprocess      !< If thread should process more then one element
    type(kernel_code)               :: code                     !< Resulting code
    character(len=:),   allocatable :: buffer_type              !< Type of buffer that should be used
    character(len=2) :: temp                                    !< Temporary string

    if ( ndims == 2 .and. (enable_packing .or. enable_multiprocess) ) error stop "dtFFT Internal error: ndims == 2 .and. (enable_packing .or. enable_multiprocess)"


    if ( enable_packing ) then
      call get_neighbor_function_code(code)
    endif

    call get_code_init(kernel_name, base_storage, code, buffer_type)
    call code%add_line("   ,const int nx")
    call code%add_line("   ,const int ny")
    if ( ndims == 3 ) call code%add_line("   ,const int nz")
    if ( enable_packing ) then
      call code%add_line("   ,const int n_neighbors")
      call code%add_line("   ,const int * __restrict__ local_starts")
      call code%add_line("   ,const int * __restrict__ local_counts")
      call code%add_line("   ,const int * __restrict__ pack_displs")
    endif
    call code%add_line(")")
    call code%add_line("{")
    call code%add_line("__shared__ "//buffer_type//" tile[TILE_DIM][TILE_DIM + 1];")

    call code%add_line("int x_in, x_out, ind_in, ind_out;")
    call code%add_line("int y_in, y_out;")
    call code%add_line("int lx = threadIdx.x;")
    call code%add_line("int ly = threadIdx.y;")
    call code%add_line("int bx = blockIdx.x;")
    call code%add_line("int by = blockIdx.y;")
    call code%add_line("x_in = lx + TILE_DIM * bx;")
    call code%add_line("y_in = ly + TILE_DIM * by;")
    if ( .not. enable_multiprocess ) then
      call code%add_line("int z = blockIdx.z;")
    endif
    call code%add_line("x_out = ly + TILE_DIM * bx;")
    call code%add_line("y_out = lx + TILE_DIM * by;")
    if ( ndims == 3 ) then
      if ( enable_packing ) then
        ! Only X-Y 3d transpose
        call code%add_line("int neighbor_idx = findNeighborIdx(local_starts, n_neighbors, x_out);")
        call code%add_line("int shift_out = pack_displs[neighbor_idx];")
        call code%add_line("int target_count = local_counts[neighbor_idx];")
      endif
      if ( enable_multiprocess ) then
        if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
          call code%add_line(" for (int z = blockIdx.z; z < nz; z += gridDim.z) { ")
        else
          call code%add_line(" for (int z = blockIdx.z; z < ny; z += gridDim.z) { ")
        endif
      endif
      if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        call code%add_line("ind_in = x_in + (y_in + z * ny) * nx;")
      else
        call code%add_line("ind_in = x_in + (z + y_in * ny) * nx;")
      endif
      if ( enable_packing ) then
        call code%add_line("ind_out = shift_out + (z * ny * target_count) + x_out * ny + y_out;")
      else
        select case ( transpose_type%val )
        case ( DTFFT_TRANSPOSE_X_TO_Y%val, DTFFT_TRANSPOSE_Y_TO_X%val )
          call code%add_line("ind_out = y_out + (x_out + z * nx) * ny;")
        case ( DTFFT_TRANSPOSE_X_TO_Z%val )
          call code%add_line("ind_out = y_out + (x_out + z * nx) * nz;")
        case ( DTFFT_TRANSPOSE_Z_TO_X%val )
          call code%add_line("ind_out = y_out + (z + x_out * nz) * ny;")
        case ( DTFFT_TRANSPOSE_Y_TO_Z%val, DTFFT_TRANSPOSE_Z_TO_Y%val )
          call code%add_line("ind_out = y_out + (z + x_out * ny) * nz;")
        endselect
      endif
    else !! ndims == 2
      call code%add_line("ind_in = x_in + y_in * nx;")
      call code%add_line("ind_out = y_out + x_out * ny;")
    endif
    if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
      temp = "ny"
    else
      temp = "nz"
    endif
    call code%add_line("if( x_in < nx && y_in < "//temp//")")
    call code%add_line("    tile[lx][ly] = in[ind_in];")

    call code%add_line("__syncthreads();")
    call code%add_line("if( x_out < nx && y_out < "//temp//")")
    call code%add_line("    out[ind_out] = tile[ly][lx];")

    if ( enable_multiprocess ) then
      call code%add_line("__syncthreads(); ")
      call code%add_line("}")
    endif

    call code%add_line("}")
    deallocate( buffer_type )
  end function get_transpose_kernel_code

  function get_unpack_kernel_code(kernel_name, base_storage) result(code)
  !! Generates code that will be used to unpack data when it is recieved
    character(len=*),   intent(in)  :: kernel_name        !< Name of CUDA kernel
    integer(int8),      intent(in)  :: base_storage       !< Number of bytes needed to store single element
    type(kernel_code)               :: code               !< Resulting code

    call get_neighbor_function_code(code)
    call get_code_init(kernel_name, base_storage, code)
    call code%add_line("  ,const int n_total")
    call code%add_line("  ,const int n_align")
    call code%add_line("  ,const int n_neighbors")
    call code%add_line("  ,const int* __restrict__ recv_displs")
    call code%add_line("  ,const int* __restrict__ recv_starts")
    call code%add_line("  ,const int* __restrict__ send_sizes")
    call code%add_line(") {")
    call code%add_line("  int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    call code%add_line("")
    call code%add_line("  if (idx < n_total) {")
    call code%add_line("    int neighbor = findNeighborIdx(recv_displs, n_neighbors, idx);")
    call code%add_line("    int start = recv_starts[neighbor];")
    call code%add_line("    int shift_out = idx - recv_displs[neighbor];")
    call code%add_line("    int sent_size = send_sizes[neighbor];")
    call code%add_line("    int mod_sent_size = shift_out - (shift_out / sent_size) * sent_size;")
    ! call code%add_line("    int mod_sent_size = shift_out % sent_size;")
    call code%add_line("    int div_sent_size = (shift_out - mod_sent_size) / sent_size;")
    call code%add_line("    int ind_out = start + div_sent_size * n_align + mod_sent_size;")
    call code%add_line("    out[ind_out] = in[idx];")
    call code%add_line("	}")
    call code%add_line("}")
  end function get_unpack_kernel_code

  function get_unpack_pipelined_kernel_code(kernel_name, base_storage) result(code)
  !! Generates code that will be used to partially unpack data when it is recieved from other process
    character(len=*),   intent(in)  :: kernel_name        !< Name of CUDA kernel
    integer(int8),      intent(in)  :: base_storage       !< Number of bytes needed to store single element
    type(kernel_code)               :: code               !< Resulting code

    call get_code_init(kernel_name, base_storage, code)
    call code%add_line("  ,const int n_total")
    call code%add_line("  ,const int n_align")
    call code%add_line("  ,const int n_align_sent")
    call code%add_line("  ,const int displ_in")
    call code%add_line("  ,const int displ_out")
    call code%add_line(") {")
    call code%add_line("  int idx = blockIdx.x * blockDim.x + threadIdx.x;")

    call code%add_line("  if (displ_in + idx < n_total) {")
    ! call code%add_line("    int ind_mod = (idx % n_align_sent);")
    call code%add_line("    int ind_mod = idx - (idx / n_align_sent) * n_align_sent;")
    call code%add_line("    int ind_out = (idx - ind_mod) / n_align_sent * n_align + ind_mod;")
    call code%add_line("    out[displ_out + ind_out] = in[displ_in + idx];")
    call code%add_line("  }")
    call code%add_line("}")
  end function get_unpack_pipelined_kernel_code
end module dtfft_nvrtc_kernel