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
module dtfft_kernel_host
!! This module defines `kernel_host` type and its type bound procedures.
!! The host kernel is an implementation of the `abstract_kernel` type
!! that runs on the host CPU.
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_kernel
use dtfft_parameters
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: kernel_host

  type, extends(abstract_kernel) :: kernel_host
  !! Host kernel implementation
    integer(int64) :: base_storage
  contains
    procedure :: create_private => create_host    !! Creates kernel
    procedure :: execute_private => execute_host  !! Executes kernel
    procedure :: destroy_private => destroy_host  !! Destroys kernel
  end type kernel_host

contains

  subroutine create_host(self, effort, base_storage, force_effort)
  !! Creates kernel
    class(kernel_host),       intent(inout) :: self         !! Host kernel class
    type(dtfft_effort_t),     intent(in)    :: effort       !! Effort level for generating transpose kernels
    integer(int64),           intent(in)    :: base_storage !! Number of bytes needed to store single element
    logical,        optional, intent(in)    :: force_effort !! Should effort be forced or not
    self%base_storage = base_storage
  end subroutine create_host

  subroutine execute_host(self, in, out, stream, neighbor)
  !! Executes host kernel
    class(kernel_host),         intent(inout) :: self       !! Host kernel class
    real(real32),     target,   intent(in)    :: in(:)      !! Source host-allocated buffer
    real(real32),     target,   intent(inout) :: out(:)     !! Target host-allocated buffer
    type(dtfft_stream_t),       intent(in)    :: stream     !! Stream to execute on, unused here
    integer(int32),   optional, intent(in)    :: neighbor   !! Source rank for pipelined unpacking
    type(c_ptr) :: pin, pout
    integer(int32) :: scaler

    pin = c_loc(in)
    pout = c_loc(out)
    scaler = int(self%base_storage / FLOAT_STORAGE_SIZE, int32)

    select case( self%base_storage )
    case ( FLOAT_STORAGE_SIZE )
      select case( self%kernel_type%val )
      case( KERNEL_PERMUTE_FORWARD%val )
        call permute_forward_f32(in, out, self%dims)
      case( KERNEL_PERMUTE_BACKWARD%val )
        call permute_backward_f32(in, out, self%dims)
      case( KERNEL_PERMUTE_BACKWARD_START%val )
        call permute_backward_start_f32(in, out, self%dims)
      case( KERNEL_PERMUTE_BACKWARD_END%val )
        call permute_backward_end_f32(in, out, self%dims, self%neighbor_data)
      case( KERNEL_UNPACK%val )
        call unpack_f32(in, out, self%dims, self%neighbor_data)
      case( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
        call permute_backward_end_pipelined_f32(in, out, self%dims, self%neighbor_data(:, neighbor))
      case( KERNEL_UNPACK_PIPELINED%val )
        call unpack_pipelined_f32(in, out, self%dims, self%neighbor_data(:, neighbor))
      endselect
    case ( DOUBLE_STORAGE_SIZE )
      block
        real(real64), pointer :: inbuf(:), outbuf(:)

        call c_f_pointer(pin, inbuf, [size(in) / scaler])
        call c_f_pointer(pout, outbuf, [size(out) / scaler])

        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
          call permute_forward_f64(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
          call permute_backward_f64(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
          call permute_backward_start_f64(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_END%val )
          call permute_backward_end_f64(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_UNPACK%val )
          call unpack_f64(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
          call permute_backward_end_pipelined_f64(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case( KERNEL_UNPACK_PIPELINED%val )
          call unpack_pipelined_f64(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
      endblock
    case ( DOUBLE_COMPLEX_STORAGE_SIZE )
      block
        complex(real64), pointer :: inbuf(:), outbuf(:)

        call c_f_pointer(pin, inbuf, [size(in) / scaler])
        call c_f_pointer(pout, outbuf, [size(out) / scaler])

        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
          call permute_forward_f128(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
          call permute_backward_f128(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
          call permute_backward_start_f128(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_END%val )
          call permute_backward_end_f128(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_UNPACK%val )
          call unpack_f128(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
          call permute_backward_end_pipelined_f128(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case( KERNEL_UNPACK_PIPELINED%val )
          call unpack_pipelined_f128(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
      endblock
    endselect
  end subroutine execute_host

  subroutine destroy_host(self)
  !! Destroys host kernel
    class(kernel_host), intent(inout) :: self !! Host kernel class

    if ( self%is_created ) return
  end subroutine destroy_host

#define PREC _f128
#define BUFFER_TYPE complex(real64)
#include "_dtfft_kernel_host_routines.inc"

#define PREC _f64
#define BUFFER_TYPE real(real64)
#include "_dtfft_kernel_host_routines.inc"

#define PREC _f32
#define BUFFER_TYPE real(real32)
#include "_dtfft_kernel_host_routines.inc"
end module dtfft_kernel_host