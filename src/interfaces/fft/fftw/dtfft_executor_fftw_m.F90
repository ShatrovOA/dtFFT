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
!------------------------------------------------------------------------------------------------
!< This module describes FFTW3 Wrapper to dtFFT: [[fftw_executor]]
!< http://www.fftw.org
!------------------------------------------------------------------------------------------------
use dtfft_parameters
use dtfft_precisions
use dtfft_info_m
use dtfft_interface_fftw_m
use dtfft_abstract_executor_m
use iso_c_binding, only: C_PTR, c_loc
implicit none
private
public :: fftw_executor

  integer(IP), parameter :: FFTW3_FLAGS = FFTW_MEASURE + FFTW_DESTROY_INPUT
  !< FFTW3 planner flags

  type, extends(abstract_executor) :: fftw_executor
  private
    ! type(C_PTR)                   :: plan       !< FFTW plan pointer
    procedure(apply_interface), nopass, pointer :: apply => NULL()
    procedure(free_interface),  nopass, pointer :: free => NULL()
  contains
    procedure, pass(self) :: create
    procedure, pass(self) :: execute
    procedure, pass(self) :: destroy
  end type fftw_executor

  abstract interface
    subroutine apply_interface(plan, a, b) bind(C)
      import
      type(c_ptr), value :: plan
      type(c_ptr), value :: a
      type(c_ptr), value :: b
    end subroutine apply_interface

    subroutine free_interface(plan) bind(C)
      import
      type(c_ptr), value :: plan
    end subroutine free_interface

    type(C_PTR) function create_c2c_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags)
    import
#include "complex_signature.i90"
      integer(C_INT), value :: sign
    end function create_c2c_plan

    type(C_PTR) function create_r2c_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags)
    import
#include "complex_signature.i90"
    end function create_r2c_plan

    type(C_PTR) function create_r2r_plan(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags)
    import
#include "complex_signature.i90"
      integer(C_FFTW_R2R_KIND), intent(in) :: kind(*)
    end function create_r2r_plan
  end interface

contains

  subroutine create(self, fft_type, precision, real_info, complex_info, sign_or_kind)
    class(fftw_executor),     intent(inout) :: self
    integer(IP),              intent(in)    :: fft_type
    integer(IP),              intent(in)    :: precision
    class(info_t), optional,  intent(in)    :: real_info
    class(info_t), optional,  intent(in)    :: complex_info
    integer(IP),   optional,  intent(in)    :: sign_or_kind
    integer(IP)                             :: knd(1)
    complex(C4P), allocatable, target :: cbuf(:)
    real(R4P),    allocatable, target :: rbuf(:)
    integer(IP)              :: n_elements
    type(c_ptr) :: cb, rb

    if(present(real_info) .and. present(sign_or_kind)) knd(1) = sign_or_kind

    if ( present( complex_info ) ) then
      n_elements = product(complex_info%counts)
      if ( precision == DTFFT_DOUBLE ) n_elements = 2 * n_elements
      allocate( cbuf( n_elements ) )
      cb = c_loc( cbuf )
    endif

    if ( present( real_info ) ) then
      n_elements = product(real_info%counts)
      if ( precision == DTFFT_DOUBLE ) n_elements = 2 * n_elements
      allocate( rbuf( n_elements ) )
      rb = c_loc( rbuf )
    endif

    select case (precision)
    case (DTFFT_SINGLE)
      self%free => fftwf_destroy_plan
    case (DTFFT_DOUBLE)
      self%free => fftw_destroy_plan
    endselect

    associate( ci => complex_info, ri => real_info )

    select case (fft_type)
    case (FFT_C2C)
      block
        procedure(create_c2c_plan), pointer  :: constructor

        select case (precision)
        case (DTFFT_SINGLE)
          constructor => fftwf_plan_many_dft
          self%apply => fftwf_execute_dft
        case (DTFFT_DOUBLE)
          constructor => fftw_plan_many_dft
          self%apply => fftw_execute_dft
        endselect

        self%plan = constructor(1, ci%length, ci%how_many, cb, ci%length, 1, ci%counts(1), cb, ci%length, 1, ci%counts(1), sign_or_kind, FFTW3_FLAGS)
        nullify(constructor)
      endblock

      case (FFT_R2C)
        block
          procedure(create_r2c_plan), pointer  :: constructor

          select case (precision)
          case (DTFFT_SINGLE)
            constructor => fftwf_plan_many_dft_r2c
            self%apply => fftwf_execute_dft_r2c
          case (DTFFT_DOUBLE)
            constructor => fftw_plan_many_dft_r2c
            self%apply => fftw_execute_dft_r2c
          endselect

          self%plan = constructor(1, ri%length, ri%how_many, rb, ri%length, 1, ri%counts(1), cb, ci%length, 1, ci%counts(1), FFTW3_FLAGS)
          nullify(constructor)
        endblock
      case (FFT_C2R)
        block
          procedure(create_r2c_plan), pointer  :: constructor

          select case (precision)
          case (DTFFT_SINGLE)
            constructor => fftwf_plan_many_dft_c2r
            self%apply => fftwf_execute_dft_c2r
          case (DTFFT_DOUBLE)
            constructor => fftw_plan_many_dft_c2r
            self%apply => fftw_execute_dft_c2r
          endselect

          self%plan = constructor(1, ri%length, ci%how_many, cb, ci%length, 1, ci%counts(1), rb, ri%length, 1, ri%counts(1), FFTW3_FLAGS)
          nullify(constructor)
        endblock
      case (FFT_R2R)
        block
          procedure(create_r2r_plan), pointer :: constructor

          select case (precision)
          case (DTFFT_SINGLE)
            constructor => fftwf_plan_many_r2r
            self%apply => fftwf_execute_r2r
          case (DTFFT_DOUBLE)
            constructor => fftw_plan_many_r2r
            self%apply => fftw_execute_r2r
          endselect

          self%plan = constructor(1, ri%length, ri%how_many, rb, ri%length, 1, ri%counts(1), rb, ri%length, 1, ri%counts(1), knd, FFTW3_FLAGS)
          nullify(constructor)
        endblock
    endselect

    if ( present( complex_info ) )  deallocate( cbuf )
    if ( present( real_info ) )     deallocate( rbuf )
    endassociate
  end subroutine create

  subroutine execute(self, a, b)
    class(fftw_executor), intent(in)            :: self
    type(*),              intent(inout), target :: a(..)
    type(*),   optional,  intent(inout), target :: b(..)
    type(C_PTR) :: aa, bb

    aa = c_loc(a)
    if(present(b)) then
      bb = c_loc(b)
      call self%apply(self%plan, aa, bb)
      return 
    endif
    call self%apply(self%plan, aa, aa)
  end subroutine execute

  subroutine destroy(self)
    class(fftw_executor),     intent(inout) :: self

    call self%free(self%plan)
    self%apply => NULL()
    self%free  => NULL()
  end subroutine destroy
end module dtfft_executor_fftw_m