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
module dtfft_executor_cufft_m
!------------------------------------------------------------------------------------------------
!< This module describes cuFFT Wrappers to dtFFT: [[cufft_c2c_executor]], [[cufft_r2c_executor]] and [[cufft_c2r_executor]]  
!< https://docs.nvidia.com/cuda/cufft/index.html
!------------------------------------------------------------------------------------------------
#ifdef _BUILD_DOCS
#define CUFFT_ENABLED
#endif
#ifdef CUFFT_ENABLED
use dtfft_precisions
use dtfft_info_m
use dtfft_abstract_executor_m
use cufft
use openacc
implicit none
private
public :: cufft_c2c_executor,   &
          cufft_c2r_executor,   &
          cufft_r2c_executor

  type, extends(abstract_c2c_executor) :: cufft_c2c_executor
  !< GPU C2C Executor, cuFFT library  
  private
    integer(IP) :: plan     !< cuFFT plan
    integer(IP) :: sign     !< Sign of c2c transform
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2c    !< Creates plan
    procedure,  pass(self) :: execute     => execute_c2c        !< Executes plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2c      !< Executes plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2c        !< Destroys plan
  end type cufft_c2c_executor

  type, extends(abstract_c2r_executor) :: cufft_c2r_executor
  !< GPU C2R Executor, cuFFT library  
  private
    integer(IP) :: plan     !< cuFFT plan
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2r    !< Creates c2r plan
    procedure,  pass(self) :: execute     => execute_c2r        !< Executes c2r plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2r      !< Executes c2r plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2r        !< Destroys c2r plan
  end type cufft_c2r_executor

  type, extends(abstract_r2c_executor) :: cufft_r2c_executor
  !< GPU R2C Executor, cuFFT library  
  private
    integer(IP) :: plan     !< cuFFT plan
  contains
    procedure,  pass(self) :: create_plan => create_plan_r2c    !< Creates r2c plan
    procedure,  pass(self) :: execute     => execute_r2c        !< Executes r2c plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_r2c      !< Executes r2c plan, single precision
    procedure,  pass(self) :: destroy     => destroy_r2c        !< Destroys r2c plan
  end type cufft_r2c_executor

  contains

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2c(self, info, sign, precision)
!------------------------------------------------------------------------------------------------
!< Creates cuFFT c2c plan
!------------------------------------------------------------------------------------------------
    class(cufft_c2c_executor),  intent(inout) :: self             !< cuFFT Executor
    class(info_t),              intent(in)    :: info             !< Buffer info
    integer(IP),                intent(in)    :: sign             !< Sign of transform
    integer(IP),                intent(in)    :: precision        !< Precision of FFT
    integer(IP)                               :: ierr             !< Error flag
    integer(IP)                               :: cufft_type       !< cuFFT Precision

    self%sign = sign
    if(precision == C8P) then
      cufft_type = CUFFT_Z2Z
    elseif(precision == C4P) then
      cufft_type = CUFFT_C2C
    endif
    ierr = cufftPlan1d(self%plan, info%counts(1), cufft_type, info%how_many)
    ierr = cufftSetStream(self%plan, acc_get_cuda_stream(acc_async_sync))
  end subroutine create_plan_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT plan, double precision
!------------------------------------------------------------------------------------------------
    class(cufft_c2c_executor),  intent(inout) :: self             !< cuFFT Executor
    complex(C8P),               intent(inout) :: inout(*)         !< Buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(inout)
!$acc host_data use_device(inout)
    ierr = cufftExecZ2Z(self%plan, inout, inout, self%sign)
!$acc end host_data
!$acc end data
  end subroutine execute_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT plan, single precision
!------------------------------------------------------------------------------------------------
    class(cufft_c2c_executor),  intent(inout) :: self             !< cuFFT Executor
    complex(C4P),               intent(inout) :: inout(*)         !< Buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(inout) 
!$acc host_data use_device(inout)
    ierr = cufftExecC2C(self%plan, inout, inout, self%sign)
!$acc end host_data
!$acc end data
  end subroutine execute_f_c2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys cuFFT plan
!------------------------------------------------------------------------------------------------
    class(cufft_c2c_executor),  intent(inout) :: self             !< cuFFT Executor
    integer(IP)                               :: ierr             !< Error flag

    ierr = cufftDestroy(self%plan)
  end subroutine destroy_c2c

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2r(self, complex_info, real_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates cuFFT c2r plan
!------------------------------------------------------------------------------------------------
    class(cufft_c2r_executor),  intent(inout) :: self             !< C2R Executor
    class(info_t),              intent(in)    :: complex_info     !< Complex buffer info
    class(info_t),              intent(in)    :: real_info        !< Real buffer info
    integer(IP),                intent(in)    :: precision        !< Precision of executor
    integer(IP)                               :: ierr             !< Error flag
    integer(IP)                               :: cufft_type       !< cuFFT Precision

    if(precision == C8P) then
      cufft_type = CUFFT_Z2D
    elseif(precision == C4P) then
      cufft_type = CUFFT_C2R
    endif
    ierr = cufftPlan1d(self%plan, real_info%counts(1), cufft_type, complex_info%how_many)
    ierr = cufftSetStream(self%plan, acc_get_cuda_stream(acc_async_sync))
  end subroutine create_plan_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT c2r plan, double precision
!------------------------------------------------------------------------------------------------
    class(cufft_c2r_executor),  intent(inout) :: self             !< C2R Executor
    complex(C8P),               intent(inout) :: in(*)            !< Complex buffer
    real(R8P),                  intent(inout) :: out(*)           !< Real buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(in, out) 
!$acc host_data use_device(in, out)
    ierr = cufftExecZ2D(self%plan, in, out)
!$acc end host_data
!$acc end data
  end subroutine execute_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT c2r plan, single precision
!------------------------------------------------------------------------------------------------
    class(cufft_c2r_executor),  intent(inout) :: self             !< C2R Executor
    complex(C4P),               intent(inout) :: in(*)            !< Complex buffer
    real(R4P),                  intent(inout) :: out(*)           !< Real buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(in, out) 
!$acc host_data use_device(in, out)
    ierr = cufftExecC2R(self%plan, in, out)
!$acc end host_data
!$acc end data
  end subroutine execute_f_c2r

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2r(self)
!------------------------------------------------------------------------------------------------
!< Destroys cuFFT c2r plan
!------------------------------------------------------------------------------------------------
    class(cufft_c2r_executor),   intent(inout) :: self            !< C2R Executor
    integer(IP)                                :: ierr            !< Error flag

    ierr = cufftDestroy(self%plan)
  end subroutine destroy_c2r

!------------------------------------------------------------------------------------------------
  subroutine create_plan_r2c(self, real_info, complex_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates cuFFT r2c plan
!------------------------------------------------------------------------------------------------
    class(cufft_r2c_executor),  intent(inout) :: self             !< R2C Executor
    class(info_t),              intent(in)    :: real_info        !< Real buffer info
    class(info_t),              intent(in)    :: complex_info     !< Complex buffer info
    integer(IP),                intent(in)    :: precision        !< Precision of executor
    integer(IP)                               :: ierr             !< Error flag
    integer(IP)                               :: cufft_type       !< cuFFT Precision

    if(precision == C8P) then
      cufft_type = CUFFT_D2Z
    elseif(precision == C4P) then
      cufft_type = CUFFT_R2C
    endif
    ierr = cufftPlan1d(self%plan, real_info%counts(1), cufft_type, real_info%how_many)
    ierr = cufftSetStream(self%plan, acc_get_cuda_stream(acc_async_sync))
  end subroutine create_plan_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(cufft_r2c_executor),  intent(inout) :: self             !< R2C Executor
    real(R8P),                  intent(inout) :: in(*)            !< Real buffer
    complex(C8P),               intent(inout) :: out(*)           !< Complex buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(in, out) 
!$acc host_data use_device(in, out)
    ierr = cufftExecD2Z(self%plan, in, out)
!$acc end host_data
!$acc end data
  end subroutine execute_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes cuFFT r2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(cufft_r2c_executor),  intent(inout) :: self             !< R2C Executor
    real(R4P),                  intent(inout) :: in(*)            !< Real buffer
    complex(C4P),               intent(inout) :: out(*)           !< Complex buffer
    integer(IP)                               :: ierr             !< Error flag

!$acc data present(in, out) 
!$acc host_data use_device(in, out)
    ierr = cufftExecR2C(self%plan, in, out)
!$acc end host_data
!$acc end data
  end subroutine execute_f_r2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_r2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys cuFFT r2c plan
!------------------------------------------------------------------------------------------------
    class(cufft_r2c_executor),   intent(inout) :: self            !< R2C Executor
    integer(IP)                                :: ierr            !< Error flag

    ierr = cufftDestroy(self%plan)
  end subroutine destroy_r2c
#endif
end module dtfft_executor_cufft_m