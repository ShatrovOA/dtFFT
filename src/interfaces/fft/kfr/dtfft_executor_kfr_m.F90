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
use iso_c_binding, only: C_PTR, c_loc, c_int8_t, c_size_t, c_null_ptr
implicit none
private
public :: kfr_executor

  type, extends(abstract_executor) :: kfr_executor
    integer(IP) :: how_many, stride, sign
    integer(c_int8_t),  pointer :: temp(:)
    type(C_PTR) :: temp_ptr
    procedure(apply_interface), nopass, pointer :: apply => NULL()
  contains
  private
    procedure,  public, pass(self)  :: create
    procedure,  public, pass(self)  :: execute
    procedure,  public, pass(self)  :: destroy
  end type kfr_executor

  abstract interface
    subroutine apply_interface(plan, in, out, temp, sign, how_many, stride) bind(C)
      import :: C_PTR, IP
      type(C_PTR),  value :: plan
      type(C_PTR),  value :: in
      type(C_PTR),  value :: out
      type(C_PTR),  value :: temp
      integer(IP),  value :: sign
      integer(IP),  value :: how_many
      integer(IP),  value :: stride
    end subroutine apply_interface
  endinterface

  contains

  subroutine create(self, fft_type, precision, real_info, complex_info, sign_or_kind)
    class(kfr_executor),      intent(inout) :: self
    integer(IP),              intent(in)    :: fft_type
    integer(IP),              intent(in)    :: precision
    class(info_t), optional,  intent(in)    :: real_info
    class(info_t), optional,  intent(in)    :: complex_info
    integer(IP),   optional,  intent(in)    :: sign_or_kind
    integer(c_size_t)                       :: alloc_size

    alloc_size = 0
    select case (fft_type)
    case (FFT_C2C)
      if(.not. present(sign_or_kind)) error stop "dtFFT Internal Error at 'KFR%create': 'sign_or_kind' is missing"
      self%sign = sign_or_kind
      self%how_many = complex_info%how_many
      self%stride = complex_info%length(1)
      select case (precision)
      case (DTFFT_DOUBLE)
        self%plan = kfr_dft_create_plan_f64(int(complex_info%length(1), c_size_t))
        self%apply => kfr_execute_c2c_64
        alloc_size = kfr_dft_get_temp_size_f64(self%plan)
      case (DTFFT_SINGLE)
        self%plan = kfr_dft_create_plan_f32(int(complex_info%length(1), c_size_t))
        self%apply => kfr_execute_c2c_32
        alloc_size = kfr_dft_get_temp_size_f32(self%plan)
      endselect
    case (FFT_R2C, FFT_C2R)
      self%how_many = real_info%how_many
      self%stride = real_info%length(1)
      select case (precision)
      case (DTFFT_DOUBLE)
        self%plan = kfr_dft_real_create_plan_f64(int(real_info%length(1), c_size_t), KFR_PACK_CCS)
        self%apply => kfr_execute_r2c_64
        alloc_size = kfr_dft_real_get_temp_size_f64(self%plan)
      case (DTFFT_SINGLE)
        self%plan = kfr_dft_real_create_plan_f32(int(real_info%length(1), c_size_t), KFR_PACK_CCS)
        self%apply => kfr_execute_r2c_32
        alloc_size = kfr_dft_real_get_temp_size_f32(self%plan)
      endselect
      if(fft_type == FFT_R2C) self%sign = DTFFT_FORWARD
      if(fft_type == FFT_C2R) self%sign = DTFFT_BACKWARD
    case (FFT_R2R)
      if(.not. present(sign_or_kind)) error stop "dtFFT Internal Error at 'KFR%create': 'sign_or_kind' is missing"
      
    endselect

    self%temp_ptr = c_null_ptr
    if(alloc_size > 0) then
      allocate(self%temp(alloc_size))
      self%temp_ptr = c_loc(self%temp)
    endif

  end subroutine create

  subroutine execute(self, a, b)
    class(kfr_executor),  intent(in)            :: self
    type(*),              intent(inout), target :: a(..)
    type(*),   optional,  intent(inout), target :: b(..)
    type(C_PTR)                                 :: aa, bb

    aa = c_loc(a)
    if(present(b)) then
      bb = c_loc(b)
      call self%apply(self%plan, aa, bb, self%temp_ptr, self%sign, self%how_many, self%stride)
    else
      call self%apply(self%plan, aa, aa, self%temp_ptr, self%sign, self%how_many, self%stride)
    endif
  end subroutine execute

  subroutine destroy(self)
    class(kfr_executor),  intent(inout)         :: self


  end subroutine destroy
end module dtfft_executor_kfr_m