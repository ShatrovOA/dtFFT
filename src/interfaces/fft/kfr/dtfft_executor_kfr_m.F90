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
module dtfft_executor_kfr_m
!------------------------------------------------------------------------------------------------
!< This module describes KFR Wrapper to dtFFT: [[kfr_executor]]
!< http://www.fftw.org
!------------------------------------------------------------------------------------------------
use dtfft_parameters
use dtfft_precisions
use dtfft_info_m
use dtfft_interface_kfr_m
use dtfft_abstract_executor_m
use iso_c_binding, only: C_PTR, c_loc, c_int8_t, c_null_ptr
implicit none
private
public :: kfr_executor

  type, extends(abstract_executor) :: kfr_executor
    integer(SP) :: size, how_many
    integer(IP) :: sign, precision
    ! integer(c_int8_t),  allocatable :: temp(:)
    type(c_ptr) :: temp
    procedure(apply_interface),   nopass, pointer :: apply => NULL()
    procedure(delete_interface),  nopass, pointer :: delete => NULL()
  contains
  private
    procedure,  public, pass(self)  :: create
    procedure,  public, pass(self)  :: execute
    procedure,  public, pass(self)  :: destroy
  end type kfr_executor

  abstract interface
    subroutine apply_interface(plan, precision, in, out, temp, sign, how_many, size) bind(C)
#include "args_execute.i90"
    end subroutine apply_interface

    subroutine delete_interface(plan) bind(C)
    import
      type(c_ptr),  value :: plan
    end subroutine delete_interface
  endinterface

  contains

  subroutine create(self, fft_type, precision, error_code, real_info, complex_info, sign_or_kind)
    class(kfr_executor),      intent(inout) :: self
    integer(IP),              intent(in)    :: fft_type
    integer(IP),              intent(in)    :: precision
    integer(IP),              intent(out)   :: error_code
    class(info_t), optional,  intent(in)    :: real_info
    class(info_t), optional,  intent(in)    :: complex_info
    integer(IP),   optional,  intent(in)    :: sign_or_kind

    error_code = DTFFT_SUCCESS
    self%precision = precision
    select case (fft_type)
    case (FFT_C2C)
      self%sign = sign_or_kind
      self%how_many = complex_info%how_many
      self%size = complex_info%length(1)
      call kfr_create_plan_c2c(self%size, precision, self%temp, self%plan)
      self%apply => kfr_execute_c2c
      select case (precision)
      case ( DTFFT_DOUBLE )
        self%delete => kfr_dft_delete_plan_f64
      case ( DTFFT_SINGLE )
        self%delete => kfr_dft_delete_plan_f32
      endselect
    case (FFT_R2C, FFT_C2R)
      if(fft_type == FFT_R2C) self%sign = DTFFT_FORWARD
      if(fft_type == FFT_C2R) self%sign = DTFFT_BACKWARD
      self%how_many = real_info%how_many
      self%size = real_info%length(1)
      if ( mod(real_info%length(1), 2) /= 0 ) then
        error_code = DTFFT_ERROR_KFR_R2C_SIZE
        return
      endif
      call kfr_create_plan_r2c(self%size, precision, KFR_PACK_CCS, self%temp, self%plan)
      self%apply => kfr_execute_r2c
      select case (precision)
      case ( DTFFT_DOUBLE )
        self%delete => kfr_dft_real_delete_plan_f64
      case ( DTFFT_SINGLE )
        self%delete => kfr_dft_real_delete_plan_f32
      endselect
    case (FFT_R2R)
      if(sign_or_kind /= DTFFT_DCT_2 .and. sign_or_kind /= DTFFT_DCT_3) then
        error_code = DTFFT_ERROR_KFR_R2R_TYPE
        return
      endif
      if(sign_or_kind == DTFFT_DCT_2) self%sign = DTFFT_FORWARD
      if(sign_or_kind == DTFFT_DCT_3) self%sign = DTFFT_BACKWARD
      self%how_many = real_info%how_many
      self%size = int(real_info%length(1), SP)
      call kfr_create_plan_dct(self%size, precision, self%temp, self%plan)
      self%apply => kfr_execute_dct
      select case (precision)
      case ( DTFFT_DOUBLE )
        self%delete => kfr_dct_delete_plan_f64
      case ( DTFFT_SINGLE )
        self%delete => kfr_dct_delete_plan_f32
      endselect
    endselect
  end subroutine create

  subroutine execute(self, a, b)
    class(kfr_executor),  intent(in)    :: self
    type(c_ptr),          intent(in)  :: a
    type(c_ptr),          intent(in)  :: b

    call self%apply(self%plan, self%precision, a, b, self%temp, self%sign, self%how_many, self%size)
  end subroutine execute

  subroutine destroy(self)
    class(kfr_executor),  intent(inout)         :: self

    call self%delete(self%plan)
    ! if ( allocated( self%temp ) ) deallocate( self%temp )
  end subroutine destroy
end module dtfft_executor_kfr_m