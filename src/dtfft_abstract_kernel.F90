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
module dtfft_abstract_kernel
!! This module defines `abstract_kernel` type and its type bound procedures.
!!
!! The abstract kernel is used in `transpose_handle_generic` type and
!! is resposible for packing/unpacking/permute operations.
!! The actual implementation of the kernel is deferred to the
!! `create_private`, `execute_private` and `destroy_private` procedures.
use iso_fortran_env
use dtfft_parameters, only: dtfft_effort_t, dtfft_stream_t, COLOR_EXECUTE
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: abstract_kernel
public :: kernel_type_t, get_kernel_string
public :: is_unpack_kernel, is_transpose_kernel

public :: operator(==)
  interface operator(==)
    module procedure kernel_type_eq     !! Check if two `kernel_type_t` are equal
  end interface

public :: operator(/=)
  interface operator(/=)
    module procedure kernel_type_ne     !! Check if two `kernel_type_t` are not equal
  end interface


  type :: kernel_type_t
  !! nvRTC Kernel type
    integer(int32) :: val
  end type kernel_type_t

  type(kernel_type_t), parameter, public  :: KERNEL_DUMMY               = kernel_type_t(-1)
    !! Dummy kernel, does nothing
  ! type(kernel_type_t), parameter, public  :: KERNEL_TRANSPOSE           = kernel_type_t(1)
  !   !! Basic transpose kernel type.
  ! type(kernel_type_t), parameter, public  :: KERNEL_TRANSPOSE_PACKED    = kernel_type_t(2)
  !   !! Transposes data and packs it into contiguous buffer.
  !   !! Should be used only in X-Y 3D plans.
  type(kernel_type_t), parameter, public  :: KERNEL_UNPACK              = kernel_type_t(3)
    !! Unpacks contiguous buffer.
  type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_SIMPLE_COPY  = kernel_type_t(4)
    !! Doesn't actually unpacks anything. Performs ``cudaMemcpyAsync`` call.
    !! Should be used only when backend is ``DTFFT_BACKEND_CUFFTMP``.
  type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_PIPELINED    = kernel_type_t(5)
    !! Unpacks pack of contiguous buffer recieved from rank.
  ! type(kernel_type_t), parameter, public  :: KERNEL_UNPACK_PARTIAL      = kernel_type_t(6)
    !! Unpacks contiguous buffer recieved from everyone except myself.
    !! Should be used only when backend is ``DTFFT_BACKEND_NCCL_PIPELINED``.
  type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_FORWARD     = kernel_type_t(7)

  type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD    = kernel_type_t(8)

  type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_START    = kernel_type_t(9)

  type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_END    = kernel_type_t(10)

  type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_END_PIPELINED   = kernel_type_t(11)

  ! type(kernel_type_t), parameter, public  :: KERNEL_PERMUTE_BACKWARD_END_PARTIAL = kernel_type_t(12)
    !! Unpacks contiguous buffer recieved from everyone except myself.
    !! Should be used only when backend is ``DTFFT_BACKEND_NCCL_PIPELINED``.

  type(kernel_type_t), parameter          :: TRANSPOSE_KERNELS(*) = [KERNEL_PERMUTE_FORWARD, KERNEL_PERMUTE_BACKWARD, KERNEL_PERMUTE_BACKWARD_START]
    !! List of all transpose kernel types
  type(kernel_type_t), parameter          :: UNPACK_KERNELS(*) = [KERNEL_PERMUTE_BACKWARD_END_PIPELINED, KERNEL_UNPACK_PIPELINED]
    !! List of all unpack kernel types

  type, abstract :: abstract_kernel
  !! Abstract kernel type
  !!
  !! This kernel type is used in `transpose_handle_generic` type and
  !! is resposible for packing/unpacking/permute operations.
    logical                       :: is_created = .false.     !! Kernel is created flag.
    logical                       :: is_dummy = .false.       !! If kernel should do anything or not.
    type(kernel_type_t)           :: kernel_type              !! Type of the kernel
    character(len=:), allocatable :: kernel_string
    integer(int32),   allocatable :: neighbor_data(:,:)       !! Neighbor data for pipelined unpacking
    integer(int32),   allocatable :: dims(:)                  !! Local dimensions to process
  contains
    procedure,                    pass(self)  :: create           !! Creates kernel
    procedure,                    pass(self)  :: execute          !! Executes kernel
    procedure,                    pass(self)  :: destroy          !! Destroys kernel
    procedure(create_interface),  deferred    :: create_private   !! Creates underlying kernel
    procedure(execute_interface), deferred    :: execute_private  !! Executes underlying kernel
    procedure(destroy_interface), deferred    :: destroy_private  !! Destroys underlying kernel
  end type abstract_kernel

  abstract interface
    subroutine create_interface(self, effort, base_storage, force_effort)
    import
    !! Creates underlying kernel
      class(abstract_kernel),   intent(inout) :: self             !! Abstract kernel
      type(dtfft_effort_t),     intent(in)    :: effort           !! Effort level for generating transpose kernels
      integer(int64),           intent(in)    :: base_storage     !! Number of bytes needed to store single element
      logical,        optional, intent(in)    :: force_effort     !! Should effort be forced or not
    end subroutine create_interface

    subroutine execute_interface(self, in, out, stream, neighbor)
    import
    !! Executes underlying kernel
      class(abstract_kernel),     intent(inout) :: self           !! Abstract kernel
      real(real32),    target,    intent(in)    :: in(:)          !! Source buffer, can be device or host pointer
      real(real32),    target,    intent(inout) :: out(:)         !! Target buffer, can be device or host pointer
      type(dtfft_stream_t),       intent(in)    :: stream         !! Stream to execute on, used only for device pointers
      integer(int32),   optional, intent(in)    :: neighbor       !! Source rank for pipelined unpacking
    end subroutine execute_interface

    subroutine destroy_interface(self)
    import
    !! Destroys underlying kernel
      class(abstract_kernel), intent(inout) :: self               !! Abstract kernel
    end subroutine destroy_interface
  end interface

contains

  function get_kernel_string(kernel) result(string)
  !! Gets the string description of a kernel
    type(kernel_type_t), intent(in) :: kernel !! kernel type
    character(len=:), allocatable   :: string !! kernel string

    select case ( kernel%val )
    case ( KERNEL_UNPACK%val )
      allocate(string, source="dtfft_kernel_unpack")
    case ( KERNEL_UNPACK_PIPELINED%val )
      allocate(string, source="dtfft_kernel_unpack_pipe")
    ! case ( KERNEL_UNPACK_PARTIAL%val )
    !   allocate(string, source="Unpack part")
    case ( KERNEL_PERMUTE_FORWARD%val )
      allocate(string, source="dtfft_kernel_forward")
    case ( KERNEL_PERMUTE_BACKWARD%val )
      allocate(string, source="dtfft_kernel_backward")
    case ( KERNEL_PERMUTE_BACKWARD_START%val )
      allocate(string, source="dtfft_kernel_backward_start")
    case ( KERNEL_PERMUTE_BACKWARD_END%val )
      allocate(string, source="dtfft_kernel_backward_end")
    case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
      allocate(string, source="dtfft_kernel_backward_end_pipe")
    ! case ( KERNEL_PERMUTE_BACKWARD_END_PARTIAL%val )
    !   allocate(string, source="Backward end part")
    case default
      allocate(string, source="Unknown kernel")
    endselect
  end function get_kernel_string

  subroutine create(self, dims, effort, base_storage, kernel_type, neighbor_data, force_effort)
  !! Creates kernel
    class(abstract_kernel),   intent(inout) :: self             !! Abstract kernel
    integer(int32),           intent(in)    :: dims(:)          !! Local dimensions to process
    type(dtfft_effort_t),     intent(in)    :: effort           !! Effort level for generating transpose kernels
    integer(int64),           intent(in)    :: base_storage     !! Number of bytes needed to store single element
    type(kernel_type_t),      intent(in)    :: kernel_type      !! Type of kernel to build
    integer(int32), optional, intent(in)    :: neighbor_data(:,:) !! Optional pointers for unpack kernels
    logical,        optional, intent(in)    :: force_effort     !! Should effort be forced or not

    call self%destroy()

    if ( any(dims == 0) .or. kernel_type == KERNEL_DUMMY) then
      self%is_created = .true.
      self%is_dummy = .true.
      return
    endif
    self%is_dummy = .false.
    self%kernel_type = kernel_type

#ifdef DTFFT_DEBUG
    if ( any(kernel_type == [KERNEL_PERMUTE_BACKWARD_START, KERNEL_PERMUTE_BACKWARD_END,              &
                            KERNEL_PERMUTE_BACKWARD_END_PIPELINED]) ) then
      if ( size(dims) /= 3 ) INTERNAL_ERROR("2-step permutation is only valid for 3d grid")
    endif
#endif
    if ( any(kernel_type == [KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED,                                  &
                             KERNEL_PERMUTE_BACKWARD_END, KERNEL_PERMUTE_BACKWARD_END_PIPELINED]) ) then
#ifdef DTFFT_DEBUG
      if ( .not. present(neighbor_data) ) INTERNAL_ERROR("Neighbor data required")
#endif
      allocate( self%neighbor_data, source=neighbor_data )
    endif
    allocate( self%dims, source=dims )
    if ( size(dims) == 2 .and. kernel_type == KERNEL_PERMUTE_BACKWARD ) then
      self%kernel_type = KERNEL_PERMUTE_FORWARD
    endif
    allocate( self%kernel_string, source=get_kernel_string(self%kernel_type) )

    call self%create_private(effort, base_storage, force_effort)
    self%is_created = .true.
  end subroutine create

  subroutine execute(self, in, out, stream, neighbor)
  !! Executes kernel
    class(abstract_kernel),     intent(inout) :: self             !! Abstract kernel
    real(real32),               intent(in)    :: in(:)            !! Source buffer, can be device or host pointer
    real(real32),               intent(inout) :: out(:)           !! Target buffer, can be device or host pointer
    type(dtfft_stream_t),       intent(in)    :: stream           !! Stream to execute on, used only for device pointers
    integer(int32),   optional, intent(in)    :: neighbor         !! Source rank for pipelined unpacking

    if ( self%is_dummy ) return
    REGION_BEGIN(self%kernel_string, COLOR_EXECUTE)
#ifdef DTFFT_DEBUG
    if ( .not. self%is_created ) INTERNAL_ERROR("`execute` called while plan not created")
    if ( any(self%kernel_type == [KERNEL_UNPACK_PIPELINED, KERNEL_PERMUTE_BACKWARD_END_PIPELINED]) ) then
      if ( .not.present(neighbor) ) INTERNAL_ERROR("Neighbor is not passed")
      if ( neighbor < 1 .or. neighbor > size(self%neighbor_data, dim=2) ) INTERNAL_ERROR("Neighbor index out of bounds")
    endif
#endif
    call self%execute_private(in, out, stream, neighbor)

    REGION_END(self%kernel_string)
  end subroutine execute

  subroutine destroy(self)
  !! Destroys kernel
    class(abstract_kernel), intent(inout) :: self                 !! Abstract kernel

    if ( .not. self%is_created ) return
    if ( self%is_dummy ) then
      self%is_created = .false.
      return
    endif
    call self%destroy_private()
    if ( allocated(self%dims) ) deallocate(self%dims)
    if ( allocated(self%neighbor_data) ) deallocate(self%neighbor_data)
    if ( allocated(self%kernel_string) ) deallocate(self%kernel_string)
    self%is_created = .false.
  end subroutine destroy

  MAKE_EQ_FUN(kernel_type_t, kernel_type_eq)
  MAKE_NE_FUN(kernel_type_t, kernel_type_ne)
  MAKE_VALID_FUN_DTYPE(kernel_type_t, is_transpose_kernel, TRANSPOSE_KERNELS)
  MAKE_VALID_FUN_DTYPE(kernel_type_t, is_unpack_kernel, UNPACK_KERNELS)
end module dtfft_abstract_kernel
