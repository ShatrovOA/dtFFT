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
module dtfft_executor_fftw_m
!! This module describes FFTW3 based FFT Executor: [[fftw_executor]]
!!
!! http://www.fftw.org
use iso_c_binding,              only: c_ptr, c_loc, c_null_ptr, c_int
use iso_fortran_env,            only: int8, int32, int64, real32
use dtfft_abstract_executor,    only: abstract_executor, FFT_C2C, FFT_R2C, FFT_R2R
use dtfft_pencil,               only: pencil
use dtfft_interface_fftw_m
use dtfft_parameters
use dtfft_utils,                only: get_inverse_kind
implicit none
private
public :: fftw_executor

  integer(int32), parameter :: FFTW3_FLAGS = FFTW_MEASURE + FFTW_DESTROY_INPUT
  !! FFTW3 planner flags

  type, extends(abstract_executor) :: fftw_executor
  !! FFTW3 FFT Executor
  private
    procedure(apply_interface), nopass, pointer :: apply => NULL()          !! Pointer to FFTW3 function that executes FFT plan
    procedure(free_interface),  nopass, pointer :: free => NULL()           !! Pointer to FFTW3 function that destroys FFT plan
    procedure(apply_interface), nopass, pointer :: apply_inverse => NULL()  !! Pointer to FFTW3 function that executes inverse FFT plan
                                                                            !! Used in R2C only
  contains
    procedure :: create_private => create       !! Creates FFT plan via FFTW3 Interface
    procedure :: execute_private => execute     !! Executes FFTW3 plan
    procedure :: destroy_private => destroy     !! Destroys FFTW3 plan
    procedure, nopass :: mem_alloc              !! Allocates FFTW3 memory
    procedure, nopass :: mem_free               !! Frees FFTW3 aligned memory
  end type fftw_executor

  abstract interface
    subroutine apply_interface(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine apply_interface

    subroutine free_interface(plan) bind(C)
    import
      type(c_ptr), value :: plan
    end subroutine free_interface

    type(c_ptr) function create_c2c_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) bind(C)
    import
#include "args_create.i90"
      integer(C_INT), value :: sign
    end function create_c2c_plan

    type(c_ptr) function create_r2c_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) bind(C)
    import
#include "args_create.i90"
    end function create_r2c_plan

    type(c_ptr) function create_r2r_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kinds,flags) bind(C)
    import
#include "args_create.i90"
      integer(C_FFTW_R2R_KIND), intent(in) :: kinds(*)
    end function create_r2r_plan
  end interface

contains
  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
  !! Creates FFT plan via FFTW3 Interface
    class(fftw_executor),             intent(inout) :: self           !! FFTW FFT Executor
    integer(int8),                    intent(in)    :: fft_rank       !! Rank of fft: 1 or 2
    integer(int8),                    intent(in)    :: fft_type       !! Type of fft: r2r, r2c, c2c
    type(dtfft_precision_t),          intent(in)    :: precision      !! Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    integer(int32),                   intent(in)    :: idist          !! Distance between the first element of two consecutive signals in a batch of the input data.
    integer(int32),                   intent(in)    :: odist          !! Distance between the first element of two consecutive signals in a batch of the output data.
    integer(int32),                   intent(in)    :: how_many       !! Number of transforms to create
    integer(int32),                   intent(in)    :: fft_sizes(:)   !! Dimensions of transform
    integer(int32),                   intent(in)    :: inembed(:)     !! Storage dimensions of the input data in memory.
    integer(int32),                   intent(in)    :: onembed(:)     !! Storage dimensions of the output data in memory.
    integer(int32),                   intent(inout) :: error_code     !! Error code to be returned to user
    type(dtfft_r2r_kind_t), optional, intent(in)    :: r2r_kinds(:)   !! Kinds of r2r transform
    real(real32),           target,   allocatable   :: buf(:)         !! Buffer needed to create plan
    integer(int32)                                  :: n_elements     !! Number of elements in `buf`
    type(c_ptr)                                     :: ptr            !! C pointer to `buf`

    error_code = DTFFT_SUCCESS

    n_elements = product(fft_sizes) * how_many
    if ( fft_type == FFT_C2C .or. fft_type == FFT_R2C ) then
      n_elements = n_elements * 2
    endif
    if ( precision == DTFFT_DOUBLE ) then
      n_elements = n_elements * 2
    endif
    allocate( buf(n_elements) )
    ptr = c_loc(buf)

    if ( precision == DTFFT_SINGLE ) then
      self%free => fftwf_destroy_plan
    else
      self%free => fftw_destroy_plan
    endif

    select case (fft_type)
    case (FFT_C2C)
      block
        procedure(create_c2c_plan), pointer  :: constructor

        if ( precision == DTFFT_SINGLE ) then
          constructor => fftwf_plan_many_dft
          self%apply => fftwf_execute_dft
        else
          constructor => fftw_plan_many_dft
          self%apply => fftw_execute_dft
        endif
        self%apply_inverse => NULL()
        self%plan_forward = constructor(int(fft_rank, int32), fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_FORWARD, int32), FFTW3_FLAGS)
        self%plan_backward = constructor(int(fft_rank, int32), fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_BACKWARD, int32), FFTW3_FLAGS)

        nullify(constructor)
      endblock
    case (FFT_R2C)
      block
        procedure(create_r2c_plan), pointer  :: constructor, constructor_inverse

        if ( precision == DTFFT_SINGLE ) then
          constructor => fftwf_plan_many_dft_r2c
          self%apply => fftwf_execute_dft_r2c
          constructor_inverse => fftwf_plan_many_dft_c2r
          self%apply_inverse => fftwf_execute_dft_c2r
        else
          constructor => fftw_plan_many_dft_r2c
          self%apply => fftw_execute_dft_r2c
          constructor_inverse => fftw_plan_many_dft_c2r
          self%apply_inverse => fftw_execute_dft_c2r
        endif
        self%plan_forward = constructor(int(fft_rank, int32), fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, FFTW3_FLAGS)
        self%plan_backward = constructor_inverse(int(fft_rank, int32), fft_sizes, how_many, ptr, onembed, 1, odist, ptr, inembed, 1, idist, FFTW3_FLAGS)

        nullify( constructor, constructor_inverse )
      endblock
    case (FFT_R2R)
      block
        procedure(create_r2r_plan), pointer :: constructor
        type(dtfft_r2r_kind_t),   allocatable :: inverse_kinds(:)
        integer(C_FFTW_R2R_KIND), allocatable :: knds(:)

        if ( precision == DTFFT_SINGLE ) then
          constructor => fftwf_plan_many_r2r
          self%apply => fftwf_execute_r2r
        else
          constructor => fftw_plan_many_r2r
          self%apply => fftw_execute_r2r
        endif
        self%apply_inverse => NULL()

        allocate( knds(size(r2r_kinds)) )
        knds(:) = int(r2r_kinds(:)%val, C_FFTW_R2R_KIND)

        self%plan_forward = constructor(int(fft_rank, int32), fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, knds, FFTW3_FLAGS)

        allocate( inverse_kinds( size(r2r_kinds) ) )
        inverse_kinds(:) = get_inverse_kind(r2r_kinds)
        if ( all( inverse_kinds == r2r_kinds ) ) then
          self%plan_backward = self%plan_forward
          self%is_inverse_copied = .true.
        else
          knds(:) = int(inverse_kinds(:)%val, C_FFTW_R2R_KIND)
          self%plan_backward = constructor(int(fft_rank, int32), fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, knds, FFTW3_FLAGS)
        endif
        deallocate( inverse_kinds, knds )
        nullify(constructor)
      endblock
    endselect

    deallocate( buf )
    ptr = c_null_ptr
  end subroutine create

  subroutine execute(self, a, b, sign)
  !! Executes FFTW3 plan
    class(fftw_executor), intent(in)  :: self                 !! FFTW FFT Executor
    type(c_ptr),          intent(in)  :: a                    !! Source pointer
    type(c_ptr),          intent(in)  :: b                    !! Target pointer
    integer(int8),        intent(in)  :: sign                 !! Sign of transform

    if ( sign == FFT_FORWARD ) then
      call self%apply(self%plan_forward, a, b)
    else
      if ( associated( self%apply_inverse ) ) then
        call self%apply_inverse(self%plan_backward, a, b)
      else
        call self%apply(self%plan_backward, a, b)
      endif
    endif
  end subroutine execute

  subroutine destroy(self)
  !! Destroys FFTW3 plan
    class(fftw_executor), intent(inout) :: self               !! FFTW FFT Executor

    call self%free(self%plan_forward)
    if( .not. self%is_inverse_copied ) call self%free(self%plan_backward)
    self%apply => NULL()
    self%apply_inverse => NULL()
    self%free  => NULL()
  end subroutine destroy

  subroutine mem_alloc(alloc_bytes, ptr)
  !! Allocates FFTW3 memory
    integer(int64),           intent(in)  :: alloc_bytes      !! Number of bytes to allocate
    type(c_ptr),              intent(out) :: ptr              !! Allocated pointer

    ptr = fftw_malloc(alloc_bytes)
  end subroutine mem_alloc

  subroutine mem_free(ptr)
  !! Frees FFTW3 aligned memory
    type(c_ptr),               intent(in)   :: ptr            !! Pointer to free

    call fftw_free(ptr)
  end subroutine mem_free
end module dtfft_executor_fftw_m