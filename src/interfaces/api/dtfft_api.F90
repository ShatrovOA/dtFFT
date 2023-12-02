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
use dtfft_core_m
#include "../../dtfft.i90"
implicit none
private

  interface
    function Comm_c2f(ccomm) result(fcomm) bind(C, name="DTFFT_Comm_c2f")
      import :: IP, C_PTR
      type(C_PTR), value  :: ccomm
      integer(IP)         :: fcomm
    end function Comm_c2f
  endinterface

  type :: dtfft_plan_c
    class(dtfft_core),  allocatable :: p
  end type dtfft_plan_c


#define CHECK_C_PLAN(ptr) \
  if(.not.c_associated(ptr)) FATAL_ERROR("Recieved NULL as plan pointer")

contains

  function DTFFT_Comm_c2f(ccomm) result(fcomm)
    type(C_PTR)       :: ccomm
    TYPE_MPI_COMM     :: fcomm
    TYPE_MPI_COMM     :: temp_comm
    integer(IP) :: status, ierr

#if defined(DTFFT_USE_MPI)
    temp_comm = Comm_c2f(ccomm)
#else
    temp_comm%MPI_VAL = Comm_c2f(ccomm)
#endif
    call MPI_Topo_test(temp_comm, status, ierr)

    if ( status == MPI_CART ) then
      block
        integer(IP) :: ndims, i
        integer(IP) :: cdims(3), dims(3), ccoords(3)
        logical     :: cperiods(3), periods(3)

        call MPI_Cartdim_get(temp_comm, ndims, ierr)
        if(ndims > 3) call DTFFT_FATAL_ERROR("Number of dimensions in Cartesian comm > 3", "DTFFT_Comm_c2f")

        call MPI_Cart_get(temp_comm, ndims, cdims, cperiods, ccoords, ierr)

        do i = 1, ndims
          dims(i) = cdims(ndims + 1 - i)
          periods(i) = cperiods(ndims + 1 - i)
        enddo

        call MPI_Cart_create(temp_comm, ndims, dims, periods, .true., fcomm, ierr)
      endblock
    elseif(status == MPI_GRAPH) then
      fcomm = MPI_COMM_WORLD
    else
      fcomm = temp_comm
    endif
  end function DTFFT_Comm_c2f

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2r_c(ndims, dims, in_kinds, out_kinds, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates R2R dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    type(C_PTR),                  value     :: in_kinds         !< FFT R2R kinds for DTFFT_TRANSPOSE_OUT transform
    type(C_PTR),                  value     :: out_kinds        !< FFT R2R kinds for DTFFT_TRANSPOSE_IN transform
    type(C_PTR),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    integer(IP),                  pointer   :: fin_kinds(:)     !< Fortran forward kinds
    integer(IP),                  pointer   :: fout_kinds(:)    !< Fortran backward kinds
    type(dtfft_plan_c),           pointer   :: plan

    allocate(plan)
    allocate( dtfft_plan_r2r :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])
    call c_f_pointer(in_kinds, fin_kinds, [ndims])
    call c_f_pointer(out_kinds, fout_kinds, [ndims])

    select type( p => plan%p )
    type is ( dtfft_plan_r2r )
      call p%create(fdims, fin_kinds, fout_kinds, precision, effort_flag, executor_type, DTFFT_Comm_c2f(comm))
    endselect
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2r_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_c2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates C2C dtFFT Plan, allocates all structures and prepares FFT, C/C++ interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    type(C_PTR),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    type(dtfft_plan_c),           pointer   :: plan

    call c_f_pointer(dims, fdims, [ndims])
    allocate(plan)
    allocate( dtfft_plan_c2c :: plan%p )
    select type(p => plan%p)
    class is (dtfft_plan_c2c)
      call p%create(fdims, comm=DTFFT_Comm_c2f(comm), precision=precision, effort_flag=effort_flag, executor_type=executor_type)
    endselect

    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_c2c_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< Creates R2C dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
!------------------------------------------------------------------------------------------------
    integer(IP),  intent(in)                :: ndims            !< Rank of transform. Can be 2 or 3.
    type(C_PTR),                  value     :: dims             !< Global sizes of transform
    type(C_PTR),                  value     :: comm             !< Communicator
    integer(IP),  intent(in)                :: precision        !< Precision of transform
    integer(IP),  intent(in)                :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in)                :: executor_type    !< Type of External FFT Executor
    type(C_PTR),  intent(out)               :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),                  pointer   :: fdims(:)         !< Fortran dims
    type(dtfft_plan_c),           pointer   :: plan

    call c_f_pointer(dims, fdims, [ndims])

    allocate(plan)
    allocate( dtfft_plan_r2c :: plan%p )
    select type(p => plan%p)
    class is (dtfft_plan_r2c)
      call p%create(fdims, comm=DTFFT_Comm_c2f(comm), precision=precision, effort_flag=effort_flag, executor_type=executor_type)
    endselect
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2c_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_c(plan_ptr, in, out, transpose_type, aux) bind(C)
#define __FUNC__ "dtfft_execute"
!------------------------------------------------------------------------------------------------
!< Executes dtFFT Plan, C/C++/Python interface. `Aux` can be NULL. If `in` or `out` are NULL, bad things will happen.
!< `In` can be equal to `out` only on these cases:
!<    - R2R 3d
!<    - C2C 3d
!<    - R2C 2d
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(in),     value     :: plan_ptr       !< C pointer to Fortran plan
    real(R4P),    intent(inout)             :: in(*)          !< Incomming buffer, not NULL
    real(R4P),    intent(inout)             :: out(*)         !< Outgoing buffer
    integer(IP),  intent(in)                :: transpose_type !< Type of transposition, one of [[DTFFT_TRANSPOSE_OUT]], [[DTFFT_TRANSPOSE_IN]]
    real(R4P),    intent(inout),  optional  :: aux(*)         !< Aux buffer, can be NULL
    type(dtfft_plan_c), pointer :: plan

    CHECK_C_PLAN(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%execute(in, out, transpose_type, aux)
#undef __FUNC__
  end subroutine dtfft_execute_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_transpose_c(plan_ptr, in, out, transpose_type) bind(C)
#define __FUNC__ "dtfft_transpose"
!------------------------------------------------------------------------------------------------
!< Executes single transposition, C interface.
!< Executes dtFFT Plan, C interface. `Aux` can be NULL.
!< `in` != `out`
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(in),     value     :: plan_ptr       !< C pointer to Fortran plan
    real(R4P),    intent(inout)             :: in(*)          !< Incomming buffer, not NULL
    real(R4P),    intent(inout)             :: out(*)         !< Outgoing buffer
    integer(IP),  intent(in)                :: transpose_type !< Type of transposition, one of [[DTFFT_TRANSPOSE_X_TO_Y]], [[DTFFT_TRANSPOSE_Y_TO_X]],
                                                              !< [[DTFFT_TRANSPOSE_Y_TO_Z]], [[DTFFT_TRANSPOSE_Z_TO_Y]]
    type(dtfft_plan_c),           pointer   :: plan

    CHECK_C_PLAN(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%transpose(in, out, transpose_type)
#undef __FUNC__
  end subroutine dtfft_transpose_c

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
  subroutine dtfft_get_local_sizes_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
#define __FUNC__ "dtfft_get_local_sizes"
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

    CHECK_C_PLAN(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
#undef __FUNC__
  end subroutine dtfft_get_local_sizes_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_aux_size_c(plan_ptr, alloc_size) bind(C)
#define __FUNC__ "dtfft_get_aux_size"
!------------------------------------------------------------------------------------------------
!< Returns number of elements to be allocated for optional aux buffer, C/C++ interface
!< Total number of bytes to be allocated :
!<    - C2C: 2 * alloc_size * sizeof(double/float)
!<    - R2C: 2 * alloc_size * sizeof(double/float)
!<    - R2R: alloc_size * sizeof(double/float)
!------------------------------------------------------------------------------------------------
    type(C_PTR),                  value     :: plan_ptr         !< C pointer to Fortran plan
    integer(SP),  intent(out)               :: alloc_size       !< Number of elements to be allocated
    type(dtfft_plan_c),           pointer   :: plan

    CHECK_C_PLAN(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    alloc_size = plan%p%get_aux_size()
#undef __FUNC__
  end subroutine dtfft_get_aux_size_c
end module dtfft_api