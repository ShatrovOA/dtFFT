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
module dtfft_api
!------------------------------------------------------------------------------------------------
!< This module is a Fortran part of C/C++ interface
!------------------------------------------------------------------------------------------------
use iso_c_binding
use dtfft_precisions
use dtfft_parameters
use dtfft_core_m
use dtfft_utils
#include "dtfft_mpi.h"
implicit none
private

  type :: dtfft_plan_c
    class(dtfft_core),  allocatable :: p
  end type dtfft_plan_c


contains

  TYPE_MPI_COMM function get_comm(c_comm)
    integer(IP),  intent(in) :: c_comm

#if defined(DTFFT_USE_MPI)
    get_comm = c_comm
#else
    get_comm%MPI_VAL = c_comm
#endif
  end function get_comm

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_create_plan_r2r_c(ndims, dims, kinds, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates R2R dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    type(C_PTR),                  value     :: kinds            !< FFT R2R kinds
    integer(IP),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    integer(IP),                  pointer   :: fkinds(:)        !< Fortran R2R kinds
    type(dtfft_plan_c),           pointer   :: plan

    allocate(plan)
    allocate( dtfft_plan_r2r :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])
    call c_f_pointer(kinds, fkinds, [ndims])
    ! call c_f_pointer(out_kinds, fout_kinds, [ndims])

    select type( p => plan%p )
    type is ( dtfft_plan_r2r )
      call p%create(fdims, fkinds, get_comm(comm), precision, effort_flag, executor_type, dtfft_create_plan_r2r_c)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2r_c

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_create_plan_c2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates C2C dtFFT Plan, allocates all structures and prepares FFT, C/C++ interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    integer(IP),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    type(dtfft_plan_c),           pointer   :: plan

    allocate(plan)
    allocate( dtfft_plan_c2c :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_c2c)
      call p%create(fdims, get_comm(comm), precision, effort_flag, executor_type, dtfft_create_plan_c2c_c)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_c2c_c

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_create_plan_r2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates R2C dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    integer(IP),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    type(dtfft_plan_c),           pointer   :: plan

    allocate(plan)
    allocate( dtfft_plan_r2c :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_r2c)
      call p%create(fdims, get_comm(comm), precision, effort_flag, executor_type, dtfft_create_plan_r2c_c)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2c_c

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_execute_c(plan_ptr, in, out, transpose_type, aux) bind(C)
!------------------------------------------------------------------------------------------------
!< Executes dtFFT Plan, C/C++/Python interface. `Aux` can be NULL. If `in` or `out` are NULL, bad things will happen.
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(in),     value     :: plan_ptr       !< C pointer to Fortran plan
    real(R4P),    intent(inout)             :: in(*)          !< Incomming buffer, not NULL
    real(R4P),    intent(inout)             :: out(*)         !< Outgoing buffer
    integer(IP),  intent(in)                :: transpose_type !< Type of transposition, one of [[DTFFT_TRANSPOSE_OUT]], [[DTFFT_TRANSPOSE_IN]]
    real(R4P),    intent(inout),  optional  :: aux(*)         !< Aux buffer, can be NULL
    type(dtfft_plan_c), pointer :: plan

    if(.not.c_associated(plan_ptr)) then
      dtfft_execute_c = DTFFT_ERROR_PLAN_NOT_CREATED
      return
    endif
    call c_f_pointer(plan_ptr, plan)
    call plan%p%execute(in, out, transpose_type, aux, dtfft_execute_c)
  end function dtfft_execute_c

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_transpose_c(plan_ptr, in, out, transpose_type) bind(C)
!------------------------------------------------------------------------------------------------
!< Executes single transposition, C interface.
!< `in` != `out`
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(in),     value     :: plan_ptr       !< C pointer to Fortran plan
    real(R4P),    intent(inout)             :: in(*)          !< Incomming buffer, not NULL
    real(R4P),    intent(inout)             :: out(*)         !< Outgoing buffer, not NULL
    integer(IP),  intent(in)                :: transpose_type !< Type of transposition, one of [[DTFFT_TRANSPOSE_X_TO_Y]], [[DTFFT_TRANSPOSE_Y_TO_X]],
                                                              !< [[DTFFT_TRANSPOSE_Y_TO_Z]], [[DTFFT_TRANSPOSE_Z_TO_Y]]
    type(dtfft_plan_c),           pointer   :: plan

    if(.not.c_associated(plan_ptr)) then
      dtfft_transpose_c = DTFFT_ERROR_PLAN_NOT_CREATED
      return
    endif
    call c_f_pointer(plan_ptr, plan)
    call plan%p%transpose(in, out, transpose_type, dtfft_transpose_c)
  end function dtfft_transpose_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Destroys dtFFT Plan, C/C++ interface
!------------------------------------------------------------------------------------------------
    type(C_PTR)                             :: plan_ptr       !< C pointer to Fortran plan
    type(dtfft_plan_c),           pointer   :: plan

    ! Do not raise if plan already destroyed
    if ( .not. c_associated(plan_ptr) ) return
    call c_f_pointer(plan_ptr, plan)
    call plan%p%destroy()
    deallocate( plan%p )
    nullify( plan )
    plan_ptr = c_null_ptr
  end subroutine dtfft_destroy_c

!------------------------------------------------------------------------------------------------
  integer(c_int) function dtfft_get_local_sizes_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts in real and Fourier spaces and number of elements to be allocated for `in` and `out` buffers,
!< C/C++ interface. It is necessary for R2C transform  to allocate `out` buffer with `alloc_size` number of elements.
!< Total number of bytes to be allocated :
!<    - C2C: 2 * alloc_size * sizeof(double/float)
!<    - R2R: alloc_size * sizeof(double/float)
!<    - R2C: alloc_size * sizeof(double/float)
!------------------------------------------------------------------------------------------------
    type(C_PTR),                  value     :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(out),    optional  :: in_starts(3)     !< Starts of local portion of data in 'real' space
    integer(IP),  intent(out),    optional  :: in_counts(3)     !< Counts of local portion of data in 'real' space
    integer(IP),  intent(out),    optional  :: out_starts(3)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  intent(out),    optional  :: out_counts(3)    !< Counts of local portion of data in 'fourier' space
    integer(SP),  intent(out)               :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_c),           pointer   :: plan

    dtfft_get_local_sizes_c = DTFFT_SUCCESS
    if ( .not. c_associated(plan_ptr) .or. c_associated(plan_ptr, c_null_ptr) ) dtfft_get_local_sizes_c = DTFFT_ERROR_PLAN_NOT_CREATED
    if ( dtfft_get_local_sizes_c /= DTFFT_SUCCESS ) return
    call c_f_pointer(plan_ptr, plan)
    if ( .not. allocated(plan%p) ) dtfft_get_local_sizes_c = DTFFT_ERROR_PLAN_NOT_CREATED
    if ( dtfft_get_local_sizes_c /= DTFFT_SUCCESS ) return
    call plan%p%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size, dtfft_get_local_sizes_c)
  end function dtfft_get_local_sizes_c

  subroutine dtfft_get_error_string_c(error_code, error_string, error_string_size) bind(C)
    integer(c_int),     intent(in)  :: error_code
    character(c_char),  intent(out) :: error_string(*)
    integer(SP),        intent(out) :: error_string_size
    character(len=:),   allocatable :: error_string_

    call dtfft_get_error_string(error_code, error_string_)
    call dtfft_string_f2c(error_string_, error_string)
    error_string_size = len(error_string_) + 1
    deallocate(error_string_)
  end subroutine dtfft_get_error_string_c
end module dtfft_api