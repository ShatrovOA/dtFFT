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
module dtfft_abstract_executor_m
!------------------------------------------------------------------------------------------------
!< This module describes [[abstract_c2c_executor]], [[abstract_r2r_executor]],
!< [[abstract_r2c_executor]] and [[abstract_c2r_executor]] classes
!------------------------------------------------------------------------------------------------
use dtfft_info_m,     only: info_t
use dtfft_precisions, only: IP
use iso_c_binding,    only: C_PTR
implicit none
private
public :: abstract_executor, &
          FFT_C2C, FFT_R2C, FFT_R2R, FFT_C2R

  integer(IP),      parameter :: FFT_C2C = 0
  integer(IP),      parameter :: FFT_R2C = 1
  integer(IP),      parameter :: FFT_C2R = 2
  integer(IP),      parameter :: FFT_R2R = 3

  type, abstract :: abstract_executor
  !< The "most" abstract executor.
  !< All executors are extending this class.
    type(C_PTR)     :: plan
    ! integer(IP)     :: fft_type
  contains
    procedure(create_interface),  pass(self), deferred  :: create   !< Creates FFT handle
    procedure(execute_interface), pass(self), deferred  :: execute  !< Applies FFT
    procedure(destroy_interface), pass(self), deferred  :: destroy  !< Destroys executor
  end type abstract_executor

  abstract interface
!------------------------------------------------------------------------------------------------
    subroutine create_interface(self, fft_type, precision, real_info, complex_info, sign_or_kind)
!------------------------------------------------------------------------------------------------
!< Creates FFT handle
!------------------------------------------------------------------------------------------------
      import :: abstract_executor, info_t, IP
      class(abstract_executor), intent(inout) :: self
      integer(IP),              intent(in)    :: fft_type
      integer(IP),              intent(in)    :: precision
      class(info_t), optional,  intent(in)    :: real_info
      class(info_t), optional,  intent(in)    :: complex_info
      integer(IP),   optional,  intent(in)    :: sign_or_kind
    end subroutine create_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_interface(self, a, b)
!------------------------------------------------------------------------------------------------
!< Applies FFT
!------------------------------------------------------------------------------------------------
      import :: abstract_executor
      class(abstract_executor), intent(in)            :: self
      type(*),                  intent(inout), target :: a(..)
      type(*),   optional,      intent(inout), target :: b(..)
    end subroutine execute_interface

!------------------------------------------------------------------------------------------------
    subroutine destroy_interface(self)
!------------------------------------------------------------------------------------------------
!< Destroys abstract executor
!------------------------------------------------------------------------------------------------
      import :: abstract_executor
      class(abstract_executor), intent(inout) :: self             !< Abstract executor
    end subroutine destroy_interface
  end interface
end module dtfft_abstract_executor_m