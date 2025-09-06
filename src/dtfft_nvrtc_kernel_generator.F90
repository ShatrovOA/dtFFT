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
module dtfft_nvrtc_kernel_generator
use iso_c_binding
use iso_fortran_env
use dtfft_interface_cuda
use dtfft_interface_cuda_runtime
use dtfft_interface_nvrtc
use dtfft_nvrtc_block_optimizer
use dtfft_parameters
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "_dtfft_private.h"
implicit none
private
public :: kernel_codegen
public :: get_transpose_kernel_code
public :: get_unpack_kernel_code
public :: get_unpack_pipelined_kernel_code


  type :: kernel_codegen
  !! Class to build CUDA kernel code
    character(len=:), allocatable :: raw                      !! String that holds CUDA code
  contains
  private
    procedure, pass(self),  public :: to_cstr                 !! Converts Fortran CUDA code to C pointer
    procedure, pass(self),  public :: add_line                !! Adds new line to CUDA code
    procedure, pass(self),  public :: destroy => destroy_code !! Frees all memory
  end type kernel_codegen

contains

  subroutine to_cstr(self, c_code)
  !! Converts Fortran CUDA code to C pointer
    class(kernel_codegen),          intent(in)    :: self     !! Kernel code
    character(c_char), allocatable, intent(out)   :: c_code(:)!! C pointer to code

    call astring_f2c(self%raw, c_code)
  end subroutine to_cstr

  subroutine add_line(self, line)
  !! Adds new line to CUDA code
    class(kernel_codegen),          intent(inout) :: self     !! Kernel code
    character(len=*),               intent(in)    :: line     !! Line to add

    if ( .not. allocated( self%raw ) ) allocate(self%raw, source="")
    self%raw = self%raw // line // c_new_line
  end subroutine add_line

  subroutine destroy_code(self)
  !! Frees all memory
    class(kernel_codegen),          intent(inout) :: self     !! Kernel code

    if ( allocated( self%raw ) ) deallocate(self%raw)
  end subroutine destroy_code

  subroutine get_neighbor_function_code(code)
  !! Generated device function that is used to determite id of process that to which data is being sent or from which data has been recieved
  !! based on local element coordinate
    type(kernel_codegen),  intent(inout) :: code   !! Resulting code

    call code%add_line("__device__ __forceinline__")
    call code%add_line("int findNeighborIdx(const int *a, int n, int val) {")
    call code%add_line("  if ( a[n - 1] <= val) return n - 1;")
    ! Since [n - 1] already been handled
    call code%add_line("  if ( n == 2 ) return 0;")
    call code%add_line("  int low = 0, high = n - 1;")
    call code%add_line("  while (high - low > 1) {")
    call code%add_line("    int mid = (low + high) >> 1;")
    call code%add_line("    low = (a[mid] <= val) ? mid : low;")
    call code%add_line("    high = (a[mid] <= val) ? high : mid;")
    call code%add_line("  }")
    call code%add_line("  return low;")
    call code%add_line("}")
  end subroutine get_neighbor_function_code

  subroutine get_code_init(kernel_name, base_storage, code, buffer_type)
  !! Generates basic code that is used in all other kernels
    character(len=*),                         intent(in)    :: kernel_name        !! Name of CUDA kernel
    integer(int64),                           intent(in)    :: base_storage       !! Number of bytes needed to store single element
    type(kernel_codegen),                     intent(inout) :: code               !! Resulting code
    character(len=:), optional, allocatable,  intent(out)   :: buffer_type        !! Type of buffer that should be used
    character(len=:),           allocatable                 :: buffer_type_       !! Type of buffer that should be used

    select case ( base_storage )
    case ( FLOAT_STORAGE_SIZE )
      allocate( buffer_type_, source="float" )
    case ( DOUBLE_STORAGE_SIZE )
      allocate( buffer_type_, source="double" )
    case ( DOUBLE_COMPLEX_STORAGE_SIZE )
      allocate( buffer_type_, source="double2" )
    case default
      INTERNAL_ERROR("unknown `base_storage`")
    endselect

    call code%add_line('extern "C" __global__')
    call code%add_line("void")
    ! call code%add_line("__launch_bounds__("//to_str(TARGET_THREADS_PER_BLOCK)//")")
    call code%add_line(kernel_name)
    call code%add_line("(")
    call code%add_line("  "//buffer_type_//" * __restrict__ out")
    call code%add_line("   ,const "//buffer_type_//" * __restrict__ in")
    if ( present(buffer_type) ) allocate( buffer_type, source=buffer_type_ )
    deallocate(buffer_type_)
  end subroutine get_code_init

  function get_transpose_kernel_code(kernel_name, ndims, base_storage, transpose_type, enable_packing, padding) result(code)
  !! Generates code that will be used to locally tranpose data and prepares to send it to other processes
    character(len=*),         intent(in)  :: kernel_name              !! Name of CUDA kernel
    integer(int8),            intent(in)  :: ndims                    !! Number of dimensions
    integer(int64),           intent(in)  :: base_storage             !! Number of bytes needed to store single element
    type(dtfft_transpose_t),  intent(in)  :: transpose_type           !! Transpose id
    logical,                  intent(in)  :: enable_packing           !! If data should be manually packed or not
    integer(int32),           intent(in)  :: padding
    type(kernel_codegen)               :: code                     !! Resulting code
    character(len=:),   allocatable :: buffer_type              !! Type of buffer that should be used
    character(len=2) :: temp                                    !! Temporary string

    if ( ndims == 2 .and. enable_packing ) INTERNAL_ERROR(" ndims == 2 .and. enable_packing ")

    if ( enable_packing ) then
      call get_neighbor_function_code(code)
    endif

    call get_code_init(kernel_name, base_storage, code, buffer_type)
    call code%add_line("   ,const int block_rows")
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
    call code%add_line("  __shared__ "//buffer_type//" tile[TILE_DIM][TILE_DIM + "//to_str(padding)//"];")
    call code%add_line("  const int lx = threadIdx.x;")
    call code%add_line("  const int ly = threadIdx.y;")
    call code%add_line("  const int bx = blockIdx.x;")
    call code%add_line("  const int by = blockIdx.y;")
    call code%add_line("  const int z = blockIdx.z;")
    call code%add_line("  const int x_in = lx + TILE_DIM * bx;")
    call code%add_line("  const int y_in = ly + TILE_DIM * by;")
    call code%add_line("  const int x_out = ly + TILE_DIM * bx;")
    call code%add_line("  const int y_out = lx + TILE_DIM * by;")
    if ( ndims == 3 ) then
      ! if ( enable_packing ) then
      !   ! Only X-Y 3d transpose
      !   call code%add_line("  const int neighbor_idx = findNeighborIdx(local_starts, n_neighbors, x_out);")
      !   call code%add_line("  const int shift_out = pack_displs[neighbor_idx];")
      !   call code%add_line("  const int target_count = local_counts[neighbor_idx];")
      ! endif

      if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
        call code%add_line("  const int base_in = x_in + z * ny * nx;")
      else
        call code%add_line("  const int base_in = x_in + z * nx;")
      endif
      if ( enable_packing ) then
        ! call code%add_line("  const int base_out = shift_out + (z * ny * target_count) + y_out;")
      else
        select case ( transpose_type%val )
        case ( DTFFT_TRANSPOSE_X_TO_Y%val, DTFFT_TRANSPOSE_Y_TO_X%val )
          call code%add_line("  const int base_out = y_out + z * nx * ny;")
        case ( DTFFT_TRANSPOSE_X_TO_Z%val )
          call code%add_line("  const int base_out = y_out + z * nx * nz;")
        case ( DTFFT_TRANSPOSE_Z_TO_X%val )
          call code%add_line("  const int base_out = y_out + z * ny;")
        case ( DTFFT_TRANSPOSE_Y_TO_Z%val, DTFFT_TRANSPOSE_Z_TO_Y%val )
          call code%add_line("  const int base_out = y_out + z * nz;")
        endselect
      endif
    else !! ndims == 2
      call code%add_line("  const int base_in = x_in;")
      call code%add_line("  const int base_out = y_out;")
    endif

    if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
      temp = "ny"
    else
      temp = "nz"
    endif
    call code%add_line("  for(int offset = 0; offset < TILE_DIM; offset+=block_rows) {")
    call code%add_line("    int y = y_in + offset;")
    call code%add_line("    if( x_in < nx && y < "//temp//") {")
    if ( abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Y%val .or. transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
    call code%add_line("      int ind_in = base_in + y * nx;")
    else
    call code%add_line("      int ind_in = base_in + y * nx * ny;")
    endif
    call code%add_line("      tile[lx][ly + offset] = in[ind_in];")
    call code%add_line("    }")
    call code%add_line("  }")
    call code%add_line("  __syncthreads();")
    call code%add_line("  for(int offset = 0; offset < TILE_DIM; offset+=block_rows) {")
    call code%add_line("    int x = x_out + offset;")
    if ( enable_packing ) then
      ! Only X-Y 3d transpose
      call code%add_line("  const int neighbor_idx = findNeighborIdx(local_starts, n_neighbors, x);")
      call code%add_line("  const int shift_out = pack_displs[neighbor_idx];")
      call code%add_line("  const int target_count = local_counts[neighbor_idx];")
      call code%add_line("  const int base_out = shift_out + (z * ny * target_count) + y_out;")

    endif
    call code%add_line("    if( x < nx && y_out < "//temp//") {")
    select case ( transpose_type%val )
    case ( DTFFT_TRANSPOSE_X_TO_Y%val, DTFFT_TRANSPOSE_Y_TO_X%val )
    call code%add_line("      int ind_out = base_out + x * ny;")
    case ( DTFFT_TRANSPOSE_X_TO_Z%val )
    call code%add_line("      int ind_out = base_out + x * nz;")
    case ( DTFFT_TRANSPOSE_Z_TO_X%val, DTFFT_TRANSPOSE_Y_TO_Z%val, DTFFT_TRANSPOSE_Z_TO_Y%val )
    call code%add_line("      int ind_out = base_out + x * ny * nz;")
    endselect
    call code%add_line("      out[ind_out] = tile[ly + offset][lx];")
    call code%add_line("    }")
    call code%add_line("  }")
    call code%add_line("}")
    deallocate( buffer_type )
  end function get_transpose_kernel_code

  function get_unpack_kernel_code(kernel_name, base_storage, is_partial) result(code)
  !! Generates code that will be used to unpack data when it is recieved
    character(len=*),   intent(in)  :: kernel_name        !! Name of CUDA kernel
    integer(int64),     intent(in)  :: base_storage       !! Number of bytes needed to store single element
    logical,            intent(in)  :: is_partial
    type(kernel_codegen)            :: code               !! Resulting code

    call get_neighbor_function_code(code)
    call get_code_init(kernel_name, base_storage, code)
    call code%add_line("  ,const int n_total")
    call code%add_line("  ,const int n_align")
    call code%add_line("  ,const int n_neighbors")
    if ( is_partial ) then
      call code%add_line("  ,const int me")
    endif
    call code%add_line("  ,const int* __restrict__ recv_displs")
    call code%add_line("  ,const int* __restrict__ recv_starts")
    call code%add_line("  ,const int* __restrict__ send_sizes")
    call code%add_line(") {")
    call code%add_line("  int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    call code%add_line("")
    call code%add_line("  if (idx < n_total) {")
    call code%add_line("    int neighbor = findNeighborIdx(recv_displs, n_neighbors, idx);")
    if ( is_partial ) then
    call code%add_line("    if (neighbor == me) return;")
    endif
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
    character(len=*),   intent(in)  :: kernel_name        !! Name of CUDA kernel
    integer(int64),     intent(in)  :: base_storage       !! Number of bytes needed to store single element
    type(kernel_codegen)               :: code               !! Resulting code

    call get_code_init(kernel_name, base_storage, code)
    call code%add_line("  ,const int displ_in")
    call code%add_line("  ,const int displ_out")
    call code%add_line("  ,const int sent_size")
    call code%add_line("  ,const int n_total")
    call code%add_line("  ,const int n_align")
    call code%add_line(") {")
    call code%add_line("  int idx = blockIdx.x * blockDim.x + threadIdx.x;")

    call code%add_line("  if ( idx < n_total ) {")
    ! call code%add_line("    int ind_mod = (idx % sent_size);")
    call code%add_line("    int ind_mod = idx - (idx / sent_size) * sent_size;")
    call code%add_line("    int ind_out = (idx - ind_mod) / sent_size * n_align + ind_mod;")
    call code%add_line("    out[displ_out + ind_out] = in[displ_in + idx];")
    call code%add_line("  }")
    call code%add_line("}")
  end function get_unpack_pipelined_kernel_code

end module dtfft_nvrtc_kernel_generator