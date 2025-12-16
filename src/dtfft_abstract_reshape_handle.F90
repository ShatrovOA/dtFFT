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
module dtfft_abstract_reshape_handle
!! This module defines `abstract_reshape_handle` type and its type bound procedures.
!!
!! This handle is used to perform data transpositions between distributed pencils.
!! The actual implementation of the handle is deferred to the
!! `create_private`, `execute`, `execute_end`, `destroy` and `get_async_active` procedures.
use iso_fortran_env
use dtfft_abstract_backend
use dtfft_parameters
use dtfft_pencil
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: abstract_reshape_handle
public :: reshape_container
public :: create_args
public :: execute_args

  type :: create_args
  !! Arguments for creating transpose handle
    type(dtfft_platform_t)  :: platform           !! Platform type
    type(backend_helper)    :: helper             !! Backend helper
    type(dtfft_effort_t)    :: effort             !! Effort level for generating transpose kernels
    type(dtfft_backend_t)   :: backend            !! Backend type
    logical                 :: force_effort       !! Should effort be forced or not
    TYPE_MPI_DATATYPE       :: base_type          !! Base MPI Datatype
    integer(int8)           :: datatype_id        !! Type of datatype to use
    integer(int8)           :: comm_id            !! ID of communicator to use
    integer(int64)          :: base_storage
  end type create_args

  type :: execute_args
  !! Arguments for executing transpose handle
    type(dtfft_stream_t)    :: stream             !! Stream to execute on
    type(async_exec_t)      :: exec_type          !! Async execution type
    real(real32), pointer   :: p1(:)              !! `aux` pointer for pipelined operations, `in` pointer for [[execute_end]]
    real(real32), pointer   :: p2(:)              !! `out` pointer for [[execute_end]]
  end type execute_args

  type, abstract :: abstract_reshape_handle
  !! Abstract reshape handle type
    logical                         :: is_transpose !! Is this a transpose operation
  contains
    procedure, non_overridable, pass(self)            :: create           !! Creates reshape handle
    procedure,                  pass(self)            :: get_aux_bytes     !! Returns number of bytes required by aux buffer
    procedure(create_interface),            deferred  :: create_private   !! Creates reshape handle
    procedure(execute_interface),           deferred  :: execute          !! Executes reshape handle
    procedure(execute_end_interface),       deferred  :: execute_end      !! Finishes async reshape
    procedure(destroy_interface),           deferred  :: destroy          !! Destroys reshape handle
    procedure(get_async_active_interface),  deferred  :: get_async_active !! Returns if async reshape is active
  end type abstract_reshape_handle

  type :: reshape_container
  !! This type is a container for allocatable transpose handles
    class(abstract_reshape_handle), allocatable :: p  !! Transpose handle
  end type reshape_container

  abstract interface
    subroutine create_interface(self, comm, send, recv, kwargs)
    !! Creates reshape handle
    import
      class(abstract_reshape_handle),   intent(inout) :: self           !! Abstract reshape handle
      TYPE_MPI_COMM,                    intent(in)    :: comm           !! MPI Communicator
      type(pencil),                     intent(in)    :: send           !! Send pencil
      type(pencil),                     intent(in)    :: recv           !! Recv pencil
      type(create_args),                intent(in)    :: kwargs         !! Additional arguments
    end subroutine create_interface

    subroutine execute_interface(self, in, out, kwargs, error_code)
    !! Executes reshape handle
    import
      class(abstract_reshape_handle),   intent(inout) :: self       !! Abstract reshape Handle
      real(real32),                     intent(inout) :: in(:)      !! Send pointer
      real(real32),                     intent(inout) :: out(:)     !! Recv pointer
      type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
      integer(int32),                   intent(out)   :: error_code !! Error code
    end subroutine execute_interface

    subroutine execute_end_interface(self, kwargs, error_code)
    !! Finishes async reshape
    import
      class(abstract_reshape_handle),   intent(inout) :: self       !! Abstract reshape Handle
      type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
      integer(int32),                   intent(out)   :: error_code !! Error code
    end subroutine execute_end_interface

    subroutine destroy_interface(self)
    !! Destroys reshape handle
    import
      class(abstract_reshape_handle),   intent(inout) :: self       !! Abstract reshape Handle
    end subroutine destroy_interface

    elemental logical function get_async_active_interface(self)
    !! Returns if async reshape is active
    import
      class(abstract_reshape_handle),   intent(in)    :: self       !! Abstract reshape Handle
    end function get_async_active_interface
  end interface

contains

  subroutine create(self, send, recv, kwargs)
  !! Creates reshape handle
    class(abstract_reshape_handle),   intent(inout) :: self           !! Abstract reshape handle
    type(pencil),                     intent(in)    :: send           !! Send pencil
    type(pencil),                     intent(in)    :: recv           !! Recv pencil
    type(create_args),                intent(inout) :: kwargs         !! Additional arguments
    type(dtfft_transpose_t)           :: transpose_type
    type(dtfft_reshape_t)             :: reshape_type
    TYPE_MPI_COMM                     :: comm
    integer(int8)                     :: comm_id

    transpose_type = get_transpose_type(send, recv)
    reshape_type = dtfft_reshape_t(0)
    self%is_transpose = is_valid_transpose_type(transpose_type)
    comm_id = 0
    if ( self%is_transpose ) then
      select case ( abs(transpose_type%val) )
      case ( DTFFT_TRANSPOSE_X_TO_Y%val )
        comm_id = 2
      case ( DTFFT_TRANSPOSE_Y_TO_Z%val )
        comm_id = 3
      case ( DTFFT_TRANSPOSE_X_TO_Z%val )
        comm_id = 1
      endselect
    else
      reshape_type = get_reshape_type(send, recv)
      if ( .not. is_valid_reshape_type(reshape_type) ) then
        INTERNAL_ERROR("abstract_reshape_handle%create: Unable to determine transpose or reshape type.")
      end if
      select case ( reshape_type%val )
      case ( DTFFT_RESHAPE_X_BRICKS_TO_PENCILS%val, DTFFT_RESHAPE_X_PENCILS_TO_BRICKS%val )
        comm_id = 2
      case ( DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS%val, DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS%val )
        comm_id = 3
      end select
    endif

    comm = kwargs%helper%comms(comm_id)
    kwargs%helper%transpose_type = transpose_type
    kwargs%helper%reshape_type = reshape_type
    kwargs%comm_id = comm_id
    call self%create_private(comm, send, recv, kwargs)
  end subroutine create

  pure integer(int64) function get_aux_bytes(self)
  !! Returns number of bytes required by aux buffer
    class(abstract_reshape_handle),   intent(in)    :: self       !! Abstract reshape Handle
    get_aux_bytes = 0
  end function get_aux_bytes
end module dtfft_abstract_reshape_handle