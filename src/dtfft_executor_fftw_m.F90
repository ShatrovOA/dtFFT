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
!< This module describes FFTW3 Wrappers to dtFFT: [[fftw_c2c_executor]], [[fftw_r2r_executor]], [[fftw_c2r_executor]] and [[fftw_r2c_executor]]  
!< http://www.fftw.org
!------------------------------------------------------------------------------------------------
#ifndef NO_FFTW3
use dtfft_precisions
use dtfft_info_m
use dtfft_interface_fftw_m
use dtfft_abstract_executor_m
use iso_c_binding, only: C_PTR
implicit none
private
public :: fftw_c2c_executor,  &
          fftw_r2r_executor,  &
          fftw_c2r_executor,  &
          fftw_r2c_executor

  integer(IP), parameter :: FFTW3_FLAGS = FFTW_MEASURE + FFTW_DESTROY_INPUT
  !< FFTW3 planner flags

  type, extends(abstract_c2c_executor) :: fftw_c2c_executor
  !< CPU C2C Executor, FFTW3 library  
  private
    type(C_PTR) :: plan       !< FFTW plan pointer
    integer(IP) :: precision  !< Precision of FFTW
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2c    !< Creates c2c plan
    procedure,  pass(self) :: execute     => execute_c2c        !< Executes c2c plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2c      !< Executes c2c plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2c        !< Destroys c2c plan
  end type fftw_c2c_executor

  type, extends(abstract_c2r_executor) :: fftw_c2r_executor
  !< CPU C2R Executor, FFTW3 library  
  private
    type(C_PTR) :: plan       !< FFTW plan pointer
    integer(IP) :: precision  !< Precision of FFTW
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2r    !< Creates c2r plan
    procedure,  pass(self) :: execute     => execute_c2r        !< Executes c2r plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2r      !< Executes c2r plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2r        !< Destroys c2r plan
  end type fftw_c2r_executor

  type, extends(abstract_r2c_executor) :: fftw_r2c_executor
  !< CPU R2C Executor, FFTW3 library  
  private
    type(C_PTR) :: plan       !< FFTW plan pointer
    integer(IP) :: precision  !< Precision of FFTW
  contains
    procedure,  pass(self) :: create_plan => create_plan_r2c    !< Creates r2c plan
    procedure,  pass(self) :: execute     => execute_r2c        !< Executes r2c plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_r2c      !< Executes r2c plan, single precision
    procedure,  pass(self) :: destroy     => destroy_r2c        !< Destroys r2c plan
  end type fftw_r2c_executor

  type, extends(abstract_r2r_executor) :: fftw_r2r_executor
  !< CPU R2R Executor, FFTW3 library  
  private
    type(C_PTR) :: plan       !< FFTW plan pointer
    integer(IP) :: precision  !< Precision of FFTW
  contains
    procedure,  pass(self) :: create_plan => create_plan_r2r    !< Creates r2r plan
    procedure,  pass(self) :: execute     => execute_r2r        !< Executes r2r plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_r2r      !< Executes r2r plan, single precision
    procedure,  pass(self) :: destroy     => destroy_r2r        !< Destroys r2r plan
  end type fftw_r2r_executor

contains

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2c(self, info, sign, precision)
!------------------------------------------------------------------------------------------------
!< Creates fftw3 c2c plan
!------------------------------------------------------------------------------------------------
    class(fftw_c2c_executor), intent(inout) :: self           !< C2C Executor
    class(info_t),            intent(in)    :: info           !< Buffer info
    integer(IP),              intent(in)    :: sign           !< Sign of transform
    integer(IP),              intent(in)    :: precision      !< Precision of executor
    complex(C8P),             allocatable   :: buf(:)         !< Temporary double precision buffer
    complex(C4P),             allocatable   :: f_buf(:)       !< Temporary single precision buffer
    integer(IP)                             :: tmp(1)         !< Temporary buffer, disables compiler warnings

    tmp(1) = info%counts(1)
    self%precision = precision
    if(precision == C8P) then
      allocate(buf(product(info%counts)))
      self%plan = fftw_plan_many_dft(1, tmp, info%how_many,                                     &
                                      buf, tmp, 1, info%counts(1),                              &
                                      buf, tmp, 1, info%counts(1), sign, FFTW3_FLAGS)
      deallocate(buf)
    elseif(precision == C4P) then
      allocate(f_buf(product(info%counts)))
      self%plan = fftwf_plan_many_dft(1, tmp, info%how_many,                                    &
                                      f_buf, tmp, 1, info%counts(1),                            &
                                      f_buf, tmp, 1, info%counts(1), sign, FFTW3_FLAGS)
      deallocate(f_buf)
    endif
  end subroutine create_plan_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 c2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(fftw_c2c_executor), intent(inout) :: self           !< C2C Executor
    complex(C8P),             intent(inout) :: inout(*)       !< Buffer

    call fftw_execute_dft(self%plan, inout, inout)
  end subroutine execute_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 c2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(fftw_c2c_executor), intent(inout) :: self           !< C2C Executor
    complex(C4P),             intent(inout) :: inout(*)       !< Buffer

    call fftwf_execute_dft(self%plan, inout, inout)  
  end subroutine execute_f_c2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys fftw3 c2c plan
!------------------------------------------------------------------------------------------------
    class(fftw_c2c_executor),   intent(inout) :: self         !< C2C Executor

    call destroy_fftw_plan(self%plan, self%precision)
  end subroutine destroy_c2c

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2r(self, complex_info, real_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates fftw3 c2r plan
!------------------------------------------------------------------------------------------------
    class(fftw_c2r_executor), intent(inout) :: self           !< C2R Executor
    class(info_t),            intent(in)    :: complex_info   !< Complex buffer info
    class(info_t),            intent(in)    :: real_info      !< Real buffer info
    integer(IP),              intent(in)    :: precision      !< Precision of executor
    real(R8P),                allocatable   :: rbuf(:)        !< Temporary real double precision buffer
    real(R4P),                allocatable   :: f_rbuf(:)      !< Temporary real double precision buffer
    complex(C8P),             allocatable   :: cbuf(:)        !< Temporary complex double precision buffer
    complex(C4P),             allocatable   :: f_cbuf(:)      !< Temporary complex single precision buffer
    integer(IP)                             :: tmp1(1)        !< Temporary buffer, disables compiler warnings
    integer(IP)                             :: tmp2(1)        !< Temporary buffer, disables compiler warnings

    tmp1(1) = complex_info%counts(1)
    tmp2(1) = real_info%counts(1)
    self%precision = precision
    if(precision == C8P) then
      allocate(rbuf(product(real_info%counts)))
      allocate(cbuf(product(complex_info%counts)))
      self%plan = fftw_plan_many_dft_c2r(1, tmp2, complex_info%how_many,                        &
                                        cbuf, tmp1, 1, complex_info%counts(1),                  &
                                        rbuf, tmp2, 1, real_info%counts(1), FFTW3_FLAGS)
      deallocate(rbuf, cbuf)
    elseif(precision == C4P) then
      allocate(f_rbuf(product(real_info%counts)))
      allocate(f_cbuf(product(complex_info%counts)))
      self%plan = fftwf_plan_many_dft_c2r(1, tmp2, complex_info%how_many,                       &
                                          f_cbuf, tmp1, 1, complex_info%counts(1),              &
                                          f_rbuf, tmp2, 1, real_info%counts(1), FFTW3_FLAGS)
      deallocate(f_rbuf, f_cbuf)
    endif
  end subroutine create_plan_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 c2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(fftw_c2r_executor), intent(inout) :: self           !< C2R Executor
    complex(C8P),             intent(inout) :: in(*)          !< Complex buffer
    real(R8P),                intent(inout) :: out(*)         !< Real buffer
    
    call fftw_execute_dft_c2r(self%plan, in, out)
  end subroutine execute_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 c2r plan, single precision
!------------------------------------------------------------------------------------------------
    class(fftw_c2r_executor), intent(inout) :: self           !< C2R Executor
    complex(C4P),             intent(inout) :: in(*)          !< Complex buffer
    real(R4P),                intent(inout) :: out(*)         !< Real buffer
    
    call fftwf_execute_dft_c2r(self%plan, in, out)
  end subroutine execute_f_c2r

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2r(self)
!------------------------------------------------------------------------------------------------
!< Destroys fftw3 c2r plan
!------------------------------------------------------------------------------------------------
    class(fftw_c2r_executor),   intent(inout) :: self         !< C2R Executor

    call destroy_fftw_plan(self%plan, self%precision)
  end subroutine destroy_c2r

!------------------------------------------------------------------------------------------------
  subroutine create_plan_r2c(self, real_info, complex_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates fftw3 r2c plan
!------------------------------------------------------------------------------------------------
    class(fftw_r2c_executor), intent(inout) :: self           !< R2C Executor
    class(info_t),            intent(in)    :: real_info      !< Real buffer info
    class(info_t),            intent(in)    :: complex_info   !< Complex buffer info
    integer(IP),              intent(in)    :: precision      !< Precision of executor
    real(R8P),                allocatable   :: rbuf(:)        !< Temporary real double precision buffer
    real(R4P),                allocatable   :: f_rbuf(:)      !< Temporary real double precision buffer
    complex(C8P),             allocatable   :: cbuf(:)        !< Temporary complex double precision buffer
    complex(C4P),             allocatable   :: f_cbuf(:)      !< Temporary complex single precision buffer
    integer(IP)                             :: tmp1(1)        !< Temporary buffer, disables compiler warnings
    integer(IP)                             :: tmp2(1)        !< Temporary buffer, disables compiler warnings

    tmp1(1) = real_info%counts(1)
    tmp2(1) = complex_info%counts(1)
    self%precision = precision
    if(precision == R8P) then
      allocate(rbuf(product(real_info%counts)))
      allocate(cbuf(product(complex_info%counts)))
      self%plan = fftw_plan_many_dft_r2c(1, tmp1, real_info%how_many,                           &
                                        rbuf, tmp1, 1, real_info%counts(1),                     &
                                        cbuf, tmp2, 1, complex_info%counts(1), FFTW3_FLAGS)
      deallocate(rbuf, cbuf)
    elseif(precision == R4P) then
      allocate(f_rbuf(product(real_info%counts)))
      allocate(f_cbuf(product(complex_info%counts)))
      self%plan = fftwf_plan_many_dft_r2c(1, tmp1, real_info%how_many,                          &
                                          f_rbuf, tmp1, 1, real_info%counts(1),                 &
                                          f_cbuf, tmp2, 1, complex_info%counts(1), FFTW3_FLAGS)
      deallocate(f_rbuf, f_cbuf)
    endif
  end subroutine create_plan_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(fftw_r2c_executor), intent(inout) :: self           !< R2C Executor
    real(R8P),                intent(inout) :: in(*)          !< Real buffer
    complex(C8P),             intent(inout) :: out(*)         !< Complex buffer

    call fftw_execute_dft_r2c(self%plan, in, out)
  end subroutine execute_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 r2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(fftw_r2c_executor), intent(inout) :: self           !< R2C Executor
    real(R4P),                intent(inout) :: in(*)          !< Real buffer
    complex(C4P),             intent(inout) :: out(*)         !< Complex buffer

    call fftwf_execute_dft_r2c(self%plan, in, out)
  end subroutine execute_f_r2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_r2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys fftw3 r2c plan
!------------------------------------------------------------------------------------------------
    class(fftw_r2c_executor),   intent(inout) :: self         !< R2C Executor

    call destroy_fftw_plan(self%plan, self%precision)
  end subroutine destroy_r2c

!------------------------------------------------------------------------------------------------
  subroutine create_plan_r2r(self, info, kind, precision)
!------------------------------------------------------------------------------------------------
!< Creates fftw3 r2r plan
!------------------------------------------------------------------------------------------------
    class(fftw_r2r_executor), intent(inout) :: self       !< R2R executor
    class(info_t),            intent(in)    :: info       !< Buffer info
    integer(IP),              intent(in)    :: kind       !< Kind of r2r transform
    integer(IP),              intent(in)    :: precision  !< Precision of executor
    real(R8P),                allocatable   :: buf(:)     !< Temporary double precision buffer
    real(R4P),                allocatable   :: f_buf(:)   !< Temporary single precision buffer
    integer(IP)                             :: tmp(1)     !< Temporary buffer, disables compiler warnings
    integer(IP)                             :: knd(1)     !< Temporary buffer, disables compiler warnings

    tmp(1) = info%counts(1)
    knd(1) = kind
    self%precision = precision
    if(precision == R8P) then
      allocate(buf(product(info%counts)))
      self%plan = fftw_plan_many_r2r(1, tmp, info%how_many,                                     &
                                      buf, tmp, 1, info%counts(1),                              &
                                      buf, tmp, 1, info%counts(1), knd, FFTW3_FLAGS)
      deallocate(buf)
    elseif(precision == R4P) then 
      allocate(f_buf(product(info%counts)))
      self%plan = fftwf_plan_many_r2r(1, tmp, info%how_many,                                    &
                                      f_buf, tmp, 1, info%counts(1),                            &
                                      f_buf, tmp, 1, info%counts(1), knd, FFTW3_FLAGS)
      deallocate(f_buf)
    endif
  end subroutine create_plan_r2r

!------------------------------------------------------------------------------------------------
  subroutine execute_r2r(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 r2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(fftw_r2r_executor), intent(inout) :: self       !< R2R executor
    real(R8P),                intent(inout) :: inout(*)   !< Buffer

    call fftw_execute_r2r(self%plan, inout, inout)  
  end subroutine execute_r2r

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2r(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes fftw3 r2r plan, single precision
!------------------------------------------------------------------------------------------------
    class(fftw_r2r_executor), intent(inout) :: self       !< R2R executor
    real(R4P),                intent(inout) :: inout(*)   !< Buffer

    call fftwf_execute_r2r(self%plan, inout, inout)  
  end subroutine execute_f_r2r

!------------------------------------------------------------------------------------------------
  subroutine destroy_r2r(self)
!------------------------------------------------------------------------------------------------
!< Destroys fftw3 r2r plan
!------------------------------------------------------------------------------------------------
    class(fftw_r2r_executor),   intent(inout) :: self       !< R2R executor

    call destroy_fftw_plan(self%plan, self%precision)
  end subroutine destroy_r2r

  subroutine destroy_fftw_plan(plan, precision)
!------------------------------------------------------------------------------------------------
!< Destroys fftw3 plan
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(inout) :: plan                     !< Plan pointer
    integer(IP),  intent(in)    :: precision                !< Precision of executor

    if(precision == C4P .or. precision == R4P) then
      call fftwf_destroy_plan(plan)
    else
      call fftw_destroy_plan(plan)
    endif
  end subroutine destroy_fftw_plan
#endif 
! NO_FFTW3
end module dtfft_executor_fftw_m