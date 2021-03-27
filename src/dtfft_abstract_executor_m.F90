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
use dtfft_info_m
use dtfft_precisions
implicit none
private
public :: abstract_c2c_executor,  &
          abstract_r2r_executor,  &
          abstract_r2c_executor,  &
          abstract_c2r_executor

  type, abstract :: abstract_executor
  !< The "most" abstract executor.  
  !< All executors have to implement [[destroy]] method
  contains
    procedure(destroy_interface), pass(self), deferred  :: destroy  !< Destroys executor
  end type abstract_executor

  type, extends(abstract_executor), abstract :: abstract_c2c_executor
  !< All C2C Executors will extend this class
  contains
    procedure(create_c2c_plan_interface),   pass(self), deferred :: create_plan   !< Creates C2C Executor
    procedure(execute_c2c_interface),       pass(self), deferred :: execute       !< Executes C2C, double precision
    procedure(execute_f_c2c_interface),     pass(self), deferred :: execute_f     !< Executes C2C, single precision
  end type abstract_c2c_executor

  type, extends(abstract_executor), abstract :: abstract_r2r_executor
  !< All R2R Executors will extend this class
  contains
    procedure(create_r2r_plan_interface),   pass(self), deferred :: create_plan   !< Creates R2R Executor
    procedure(execute_r2r_interface),       pass(self), deferred :: execute       !< Executes R2R, double precision
    procedure(execute_f_r2r_interface),     pass(self), deferred :: execute_f     !< Executes R2R, single precision
  end type abstract_r2r_executor

  type, extends(abstract_executor), abstract :: abstract_r2c_executor
  !< All R2C Executors will extend this class
  contains
    procedure(create_r2c_plan_interface),   pass(self), deferred :: create_plan   !< Creates R2C Executor
    procedure(execute_r2c_interface),       pass(self), deferred :: execute       !< Executes R2C (DTFFT_FORWARD), double precision
    procedure(execute_f_r2c_interface),     pass(self), deferred :: execute_f     !< Executes R2C (DTFFT_FORWARD), single precision
  end type abstract_r2c_executor

  type, extends(abstract_executor), abstract :: abstract_c2r_executor
  !< All C2R Executors will extend this class
  contains
    procedure(create_c2r_plan_interface),   pass(self), deferred :: create_plan   !< Creates C2R Executor
    procedure(execute_c2r_interface),       pass(self), deferred :: execute       !< Executes C2R (DTFFT_BACKWARD), double precision
    procedure(execute_f_c2r_interface),     pass(self), deferred :: execute_f     !< Executes C2R (DTFFT_BACKWARD), single precision
  end type abstract_c2r_executor
  abstract interface 
!------------------------------------------------------------------------------------------------
    subroutine destroy_interface(self)
!------------------------------------------------------------------------------------------------
!< Destroys abstract executor
!------------------------------------------------------------------------------------------------
      import :: abstract_executor
      class(abstract_executor), intent(inout) :: self             !< Abstract executor
    end subroutine destroy_interface

!------------------------------------------------------------------------------------------------
    subroutine create_c2c_plan_interface(self, info, sign, precision)
!------------------------------------------------------------------------------------------------
!< Creates abstract c2c executor
!------------------------------------------------------------------------------------------------
      import :: abstract_c2c_executor, IP, info_t
      class(abstract_c2c_executor), intent(inout) :: self         !< C2C executor
      class(info_t),                intent(in)    :: info         !< Buffer info
      integer(IP),                  intent(in)    :: sign         !< Sign of transform
      integer(IP),                  intent(in)    :: precision    !< Precision of executor
    end subroutine create_c2c_plan_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_c2c_interface(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes abstract c2c executor, double precision
!------------------------------------------------------------------------------------------------
      import :: abstract_c2c_executor, C8P
      class(abstract_c2c_executor), intent(inout) :: self         !< C2C executor
      complex(C8P),                 intent(inout) :: inout(*)     !< Buffer
    end subroutine execute_c2c_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_f_c2c_interface(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes abstract c2c executor, single precision
!------------------------------------------------------------------------------------------------
      import :: abstract_c2c_executor, C4P
      class(abstract_c2c_executor), intent(inout) :: self         !< C2C executor
      complex(C4P),                 intent(inout) :: inout(*)     !< Buffer
    end subroutine execute_f_c2c_interface

!------------------------------------------------------------------------------------------------
    subroutine create_r2r_plan_interface(self, info, kind, precision)
!------------------------------------------------------------------------------------------------
!< Creates abstract r2r executor
!------------------------------------------------------------------------------------------------
      import :: abstract_r2r_executor, IP, info_t
      class(abstract_r2r_executor), intent(inout) :: self         !< R2R executor
      class(info_t),                intent(in)    :: info         !< Buffer info
      integer(IP),                  intent(in)    :: kind         !< Kind of transform
      integer(IP),                  intent(in)    :: precision    !< Precision of executor
    end subroutine create_r2r_plan_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_r2r_interface(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes abstract r2r executor, double precision
!------------------------------------------------------------------------------------------------
      import :: abstract_r2r_executor, R8P
      class(abstract_r2r_executor), intent(inout) :: self         !< R2R executor
      real(R8P),                    intent(inout) :: inout(*)     !< Buffer
    end subroutine execute_r2r_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_f_r2r_interface(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes abstract c2c executor, single precision
!------------------------------------------------------------------------------------------------
      import :: abstract_r2r_executor, R4P
      class(abstract_r2r_executor), intent(inout) :: self         !< R2R executor
      real(R4P),                    intent(inout) :: inout(*)     !< Buffer
    end subroutine execute_f_r2r_interface

!------------------------------------------------------------------------------------------------
    subroutine create_r2c_plan_interface(self, real_info, complex_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates abstract r2c executor
!------------------------------------------------------------------------------------------------
      import :: abstract_r2c_executor, IP, info_t
      class(abstract_r2c_executor), intent(inout) :: self         !< R2C Executor
      class(info_t),                intent(in)    :: real_info    !< Real buffer info
      class(info_t),                intent(in)    :: complex_info !< Complex buffer info
      integer(IP),                  intent(in)    :: precision    !< Precision of executor
    end subroutine create_r2c_plan_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_r2c_interface(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes abstract r2c executor, double precision
!------------------------------------------------------------------------------------------------
      import :: abstract_r2c_executor, R8P, C8P
      class(abstract_r2c_executor), intent(inout) :: self         !< R2C Executor
      real(R8P),                    intent(inout) :: in(*)        !< Real buffer
      complex(C8P),                 intent(inout) :: out(*)       !< Complex buffer
    end subroutine execute_r2c_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_f_r2c_interface(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes abstract r2c executor, single precision
!------------------------------------------------------------------------------------------------
      import :: abstract_r2c_executor, R4P, C4P
      class(abstract_r2c_executor), intent(inout) :: self         !< R2C Executor
      real(R4P),                    intent(inout) :: in(*)        !< Real buffer
      complex(C4P),                 intent(inout) :: out(*)       !< Complex buffer
    end subroutine execute_f_r2c_interface

!------------------------------------------------------------------------------------------------
    subroutine create_c2r_plan_interface(self, complex_info, real_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates abstract c2r executor
!------------------------------------------------------------------------------------------------
      import :: abstract_c2r_executor, IP, info_t
      class(abstract_c2r_executor), intent(inout) :: self         !< C2R Executor
      class(info_t),                intent(in)    :: complex_info !< Complex buffer info
      class(info_t),                intent(in)    :: real_info    !< Real buffer info
      integer(IP),                  intent(in)    :: precision    !< Precision of executor
    end subroutine create_c2r_plan_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_c2r_interface(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes abstract c2r executor, double precision
!------------------------------------------------------------------------------------------------
      import :: abstract_c2r_executor, R8P, C8P
      class(abstract_c2r_executor), intent(inout) :: self         !< C2R Executor
      complex(C8P),                 intent(inout) :: in(*)        !< Complex buffer
      real(R8P),                    intent(inout) :: out(*)       !< Real buffer
    end subroutine execute_c2r_interface

!------------------------------------------------------------------------------------------------
    subroutine execute_f_c2r_interface(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes abstract c2r executor, single precision
!------------------------------------------------------------------------------------------------
      import :: abstract_c2r_executor, R4P, C4P
      class(abstract_c2r_executor), intent(inout) :: self         !< C2R Executor
      complex(C4P),                 intent(inout) :: in(*)        !< Complex buffer
      real(R4P),                    intent(inout) :: out(*)       !< Real buffer
    end subroutine execute_f_c2r_interface
  end interface 
end module dtfft_abstract_executor_m