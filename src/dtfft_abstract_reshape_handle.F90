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
use iso_c_binding
use dtfft_abstract_backend
use dtfft_abstract_compressor, only: dtfft_compression_config_t
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_pencil
use dtfft_serializer,           only: serializer
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: abstract_reshape_handle
public :: reshape_container
public :: create_args
public :: execute_args

    character(len=*),   parameter :: RESHAPE_HANDLE = "__reshape_handle"

    type :: create_args
    !! Arguments for creating transpose handle
        type(dtfft_platform_t)              :: platform           !! Platform type
        type(backend_helper)                :: helper             !! Backend helper
        type(dtfft_effort_t)                :: effort             !! Effort level for generating transpose kernels
        type(dtfft_backend_t)               :: backend            !! Backend type
        logical                             :: force_effort       !! Should effort be forced or not
        TYPE_MPI_DATATYPE                   :: base_type          !! Base MPI Datatype
        integer(int8)                       :: comm_id            !! ID of communicator to use
        integer(int64)                      :: base_storage       !! Base storage size in bytes
        type(dtfft_transpose_mode_t)        :: transpose_mode     !! Transpose mode to use
        type(dtfft_compression_config_t)    :: compression_config !! Compression options
    end type create_args

    type :: execute_args
    !! Arguments for executing transpose handle
        type(dtfft_stream_t)    :: stream           !! Stream to execute on
        type(async_exec_t)      :: exec_type        !! Async execution type
        type(c_ptr)             :: p1               !! `aux` pointer for pipelined operations, `in` pointer for [[execute_end]]
        type(c_ptr)             :: p2               !! `out` pointer for [[execute_end]]
        type(c_ptr)             :: p3               !! `in` pointer for unpack-free generic reshape
    end type execute_args

    type, abstract :: abstract_reshape_handle
    !! Abstract reshape handle type
        logical                         :: is_transpose !! Is this a transpose operation
        type(dtfft_transpose_mode_t)    :: transpose_mode
        logical                         :: is_loaded = .false.
        logical                         :: is_created = .false.
    contains
        procedure, non_overridable, pass(self)            :: create           !! Creates reshape handle
        procedure, non_overridable, pass(self)            :: save
        procedure, non_overridable, pass(self)            :: load
        procedure, non_overridable, pass(self)            :: get_backend
        procedure,                  pass(self)            :: get_aux_bytes     !! Returns number of bytes required by aux buffer
        ! procedure,                  pass(self)            :: get_kernel_decisions !! Returns tuning results of local kernels
#ifdef DTFFT_WITH_COMPRESSION
        procedure,                  pass(self)            :: report_compression
#endif
        procedure(create_interface),            deferred  :: create_private   !! Creates reshape handle
        procedure(execute_interface),           deferred  :: execute          !! Executes reshape handle
        procedure(execute_end_interface),       deferred  :: execute_end      !! Finishes async reshape
        procedure(destroy_interface),           deferred  :: destroy          !! Destroys reshape handle
        procedure(get_async_active_interface),  deferred  :: get_async_active !! Returns if async reshape is active
        procedure(get_backend_interface),       deferred  :: get_backend_private
        procedure(save_interface),              deferred  :: save_private
        procedure(load_interface),              deferred  :: load_private
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
            class(abstract_reshape_handle), intent(inout)   :: self         !! Abstract reshape Handle
            type(c_ptr),                    intent(in)      :: in           !! Send pointer
            type(c_ptr),                    intent(in)      :: out          !! Recv pointer
            type(execute_args),             intent(inout)   :: kwargs       !! Additional arguments
            integer(int32),                 intent(out)     :: error_code   !! Error code
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

        elemental type(dtfft_backend_t) function get_backend_interface(self)
        import
            class(abstract_reshape_handle),   intent(in)    :: self       !! Abstract reshape Handle
        end function get_backend_interface

        subroutine save_interface(self, fh)
        import
            class(abstract_reshape_handle),   intent(in)    :: self       !! Abstract reshape Handle
            type(serializer),                 intent(inout) :: fh
        end subroutine save_interface

        integer(int32) function load_interface(self, fh) result(ierr)
        import
            class(abstract_reshape_handle),   intent(inout) :: self       !! Abstract reshape Handle
            type(serializer),                 intent(inout) :: fh
        end function load_interface
    end interface

contains

    ! subroutine get_kernel_decisions(self, pack_decision, unpack_decision)
    ! !! Returns tuning results of the local kernels this handle owns.
    ! !!
    ! !! Default is "no kernels": handles built on MPI derived datatypes do not run local
    ! !! kernels at all. [[reshape_handle_generic]] overrides this.
    !     class(abstract_reshape_handle), intent(in)  :: self             !! Abstract reshape handle
    !     type(kernel_decision),          intent(out) :: pack_decision    !! Pack kernel tuning result
    !     type(kernel_decision),          intent(out) :: unpack_decision  !! Unpack kernel tuning result

    !     pack_decision = kernel_decision()
    !     unpack_decision = kernel_decision()
    ! end subroutine get_kernel_decisions

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
#ifdef DTFFT_DEBUG
            if ( kwargs%transpose_mode /= DTFFT_TRANSPOSE_MODE_PACK .and. kwargs%transpose_mode /= DTFFT_TRANSPOSE_MODE_UNPACK ) then
                INTERNAL_ERROR("kwargs%transpose_mode /= DTFFT_TRANSPOSE_MODE_PACK .and. kwargs%transpose_mode /= DTFFT_TRANSPOSE_MODE_UNPACK.")
            endif
#endif
            if ( .not. self%is_loaded ) then
                self%transpose_mode = kwargs%transpose_mode
            endif
        else
            reshape_type = get_reshape_type(send, recv)
#ifdef DTFFT_DEBUG
            if ( .not. is_valid_reshape_type(reshape_type) ) then
                INTERNAL_ERROR("abstract_reshape_handle%create: Unable to determine transpose or reshape type.")
            end if
#endif
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
! #ifdef DTFFT_DEBUG
!         block
!             integer(int32) :: comm_size, ierr

!             call MPI_Comm_size(comm, comm_size, ierr)

!             if ( self%is_transpose ) then
!                 WRITE_DEBUG("Creating backend: "//dtfft_get_backend_string(kwargs%backend)//", nproc = "//to_str(comm_size)//", transpose mode = "//trim(TRANSPOSE_MODE_NAMES(kwargs%transpose_mode%val)) )
!             else
!                 WRITE_DEBUG("Creating backend: "//dtfft_get_backend_string(kwargs%backend)//", nproc = "//to_str(comm_size))
!             endif

!         endblock
! #endif
        call self%create_private(comm, send, recv, kwargs)
        self%is_created = .true.
    end subroutine create

    subroutine save(self, fh)
        class(abstract_reshape_handle), intent(in)    :: self         !! Abstract reshape Handle
        type(serializer),               intent(inout) :: fh

        if ( .not. self%is_created ) return
        call fh%save(RESHAPE_HANDLE)
        call fh%save(self%is_transpose)
        if ( self%is_transpose ) then
            call fh%save(int(self%transpose_mode%val, int8))
        endif
        call self%save_private(fh)
    end subroutine save

    integer(int32) function load(self, fh) result(ierr)
        class(abstract_reshape_handle), intent(inout) :: self         !! Abstract reshape Handle
        type(serializer),               intent(inout) :: fh
        character(len=len(RESHAPE_HANDLE)) :: handle
        integer(int8) :: temp

        ierr = DTFFT_SUCCESS
        call fh%load(handle)
        if ( handle /= RESHAPE_HANDLE ) then
            ierr = DTFFT_ERROR_FILE_INVALID
            return
        endif
        call fh%load(self%is_transpose)
        if ( self%is_transpose ) then
            call fh%load(temp); self%transpose_mode%val = int(temp, int32)
        endif
        CHECK_CALL( self%load_private(fh), ierr )
        self%is_loaded = .true.
    end function load

    elemental type(dtfft_backend_t) function get_backend(self)
        class(abstract_reshape_handle),   intent(in)    :: self       !! Abstract reshape Handle
        get_backend = BACKEND_DUMMY
        if ( .not. self%is_created ) return
        get_backend = self%get_backend_private()
    end function get_backend

    pure integer(int64) function get_aux_bytes(self)
    !! Returns number of bytes required by aux buffer
        class(abstract_reshape_handle),   intent(in)    :: self         !! Abstract reshape Handle
        get_aux_bytes = 0
    end function get_aux_bytes

#ifdef DTFFT_WITH_COMPRESSION
    subroutine report_compression(self, name)
    !! Reports compression average compression ratio by this reshape handle
        class(abstract_reshape_handle),   intent(in)    :: self         !! Abstract reshape Handle
        character(len=*),                 intent(in)    :: name         !! Name of the reshape handle

        INTERNAL_ERROR("Should have gotten here: "//name)
    end subroutine report_compression
#endif
end module dtfft_abstract_reshape_handle