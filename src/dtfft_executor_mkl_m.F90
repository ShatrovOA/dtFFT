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
module dtfft_executor_mkl_m
!------------------------------------------------------------------------------------------------
!< This module describes MKL Wrappers to dtFFT: [[mkl_c2c_executor]], [[mkl_r2c_executor]] and [[mkl_c2r_executor]]  
!< https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/fourier-transform-functions/fft-functions.html
!------------------------------------------------------------------------------------------------
#ifdef _BUILD_DOCS
#define MKL_ENABLED
#endif
#ifdef MKL_ENABLED
use dtfft_abstract_executor_m
use dtfft_interface_mkl_m
use dtfft_precisions
use dtfft_parameters
use dtfft_info_m
implicit none
private
public :: mkl_c2c_executor,   &
          mkl_r2c_executor,   &
          mkl_c2r_executor

  type, extends(abstract_c2c_executor) :: mkl_c2c_executor
  !< CPU C2C Executor, MKL library
  private
    type(DFTI_DESCRIPTOR), pointer  :: plan   !< MKL plan pointer
    integer(IP)                     :: sign   !< Sign of c2c transform
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2c    !< Creates plan
    procedure,  pass(self) :: execute     => execute_c2c        !< Executes plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2c      !< Executes plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2c        !< Destroys plan
  end type mkl_c2c_executor

  type, extends(abstract_r2c_executor) :: mkl_r2c_executor
  !< CPU R2C Executor, MKL library
  private
    type(DFTI_DESCRIPTOR), pointer  :: plan   !< MKL plan pointer
  contains
    procedure,  pass(self) :: create_plan => create_plan_r2c    !< Creates R2C plan
    procedure,  pass(self) :: execute     => execute_r2c        !< Executes R2C plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_r2c      !< Executes R2C plan, single precision
    procedure,  pass(self) :: destroy     => destroy_r2c        !< Destroys R2C plan
  end type mkl_r2c_executor

  type, extends(abstract_c2r_executor) :: mkl_c2r_executor
  !< CPU C2R Executor, MKL library
  private
    type(DFTI_DESCRIPTOR), pointer  :: plan   !< MKL plan pointer
  contains
    procedure,  pass(self) :: create_plan => create_plan_c2r    !< Creates C2R plan
    procedure,  pass(self) :: execute     => execute_c2r        !< Executes C2R plan, double precision
    procedure,  pass(self) :: execute_f   => execute_f_c2r      !< Executes C2R plan, single precision
    procedure,  pass(self) :: destroy     => destroy_c2r        !< Destroys C2R plan
  end type mkl_c2r_executor

contains

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2c(self, info, sign, precision)
!------------------------------------------------------------------------------------------------
!< Creates MKL c2c plan
!------------------------------------------------------------------------------------------------
    class(mkl_c2c_executor),  intent(inout) :: self           !< C2C Executor
    class(info_t),            intent(in)    :: info           !< Buffer info
    integer(IP),              intent(in)    :: sign           !< Sign of transform
    integer(IP),              intent(in)    :: precision      !< Precision

    self%sign = sign
    self%plan => create_mkl_plan(precision, DFTI_COMPLEX, info%counts(1), info%how_many,          &
                                  DFTI_INPLACE, info%counts(1), info%counts(1))
  end subroutine create_plan_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes MKL plan, double precision
!------------------------------------------------------------------------------------------------
    class(mkl_c2c_executor),  intent(inout) :: self           !< C2C Executor
    complex(C8P),             intent(inout) :: inout(*)       !< Buffer
    integer(IP)                             :: ierr           !< Error flag

    if(self%sign == DTFFT_FORWARD) then 
      ierr = DftiComputeForward(self%plan, inout)
    else
      ierr = DftiComputeBackward(self%plan, inout)
    endif
  end subroutine execute_c2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2c(self, inout)
!------------------------------------------------------------------------------------------------
!< Executes MKL plan, single precision
!------------------------------------------------------------------------------------------------
    class(mkl_c2c_executor),  intent(inout) :: self           !< C2C Executor
    complex(C4P),             intent(inout) :: inout(*)       !< Buffer
    integer(IP)                             :: ierr           !< Error flag

    if(self%sign == DTFFT_FORWARD) then 
      ierr = DftiComputeForward(self%plan, inout)
    else
      ierr = DftiComputeBackward(self%plan, inout)
    endif
  end subroutine execute_f_c2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys MKL plan
!------------------------------------------------------------------------------------------------
    class(mkl_c2c_executor),  intent(inout) :: self           !< C2C Executor

    call destroy_mkl_plan(self%plan)
  end subroutine destroy_c2c

!------------------------------------------------------------------------------------------------
  subroutine create_plan_r2c(self, real_info, complex_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates MKL r2c plan
!------------------------------------------------------------------------------------------------
    class(mkl_r2c_executor),  intent(inout) :: self           !< R2C Executor
    class(info_t),            intent(in)    :: real_info      !< Real buffer info
    class(info_t),            intent(in)    :: complex_info   !< Complex buffer info
    integer(IP),              intent(in)    :: precision      !< Precision of executor

    self%plan => create_mkl_plan(precision, DFTI_REAL, real_info%counts(1), real_info%how_many,   &
                                  DFTI_NOT_INPLACE, real_info%counts(1), complex_info%counts(1))
  end subroutine create_plan_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes MKL r2c plan, double precision
!------------------------------------------------------------------------------------------------
    class(mkl_r2c_executor),  intent(inout) :: self           !< R2C Executor
    real(R8P),                intent(inout) :: in(*)          !< Real buffer
    complex(C8P),             intent(inout) :: out(*)         !< Complex buffer
    integer(IP)                             :: ierr           !< Error flag

    ierr = DftiComputeForward(self%plan, in, out)
  end subroutine execute_r2c

!------------------------------------------------------------------------------------------------
  subroutine execute_f_r2c(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes MKL r2c plan, single precision
!------------------------------------------------------------------------------------------------
    class(mkl_r2c_executor),  intent(inout) :: self           !< R2C Executor
    real(R4P),                intent(inout) :: in(*)          !< Real buffer
    complex(C4P),             intent(inout) :: out(*)         !< Complex buffer
    integer(IP)                             :: ierr           !< Error flag

    ierr = DftiComputeForward(self%plan, in, out)
  end subroutine execute_f_r2c

!------------------------------------------------------------------------------------------------
  subroutine destroy_r2c(self)
!------------------------------------------------------------------------------------------------
!< Destroys MKL r2c plan
!------------------------------------------------------------------------------------------------
    class(mkl_r2c_executor),   intent(inout) :: self          !< R2C Executor

    call destroy_mkl_plan(self%plan)
  end subroutine destroy_r2c

!------------------------------------------------------------------------------------------------
  subroutine create_plan_c2r(self, complex_info, real_info, precision)
!------------------------------------------------------------------------------------------------
!< Creates MKL C2R plan
!------------------------------------------------------------------------------------------------
    class(mkl_c2r_executor),  intent(inout) :: self           !< C2R Executor
    class(info_t),            intent(in)    :: complex_info   !< Complex buffer info
    class(info_t),            intent(in)    :: real_info      !< Real buffer info
    integer(IP),              intent(in)    :: precision      !< Precision of executor
    
    self%plan => create_mkl_plan(precision, DFTI_REAL, real_info%counts(1), real_info%how_many,   &
                                  DFTI_NOT_INPLACE, complex_info%counts(1), real_info%counts(1))
  end subroutine create_plan_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes MKL C2R plan, double precision
!------------------------------------------------------------------------------------------------
    class(mkl_c2r_executor),  intent(inout) :: self           !< C2R Executor
    complex(C8P),             intent(inout) :: in(*)          !< Complex buffer
    real(R8P),                intent(inout) :: out(*)         !< Real buffer
    integer(IP)                             :: ierr           !< Error flag

    ierr = DftiComputeBackward(self%plan, in, out)
  end subroutine execute_c2r

!------------------------------------------------------------------------------------------------
  subroutine execute_f_c2r(self, in, out)
!------------------------------------------------------------------------------------------------
!< Executes MKL C2R plan, single precision
!------------------------------------------------------------------------------------------------
    class(mkl_c2r_executor),  intent(inout) :: self           !< C2R Executor
    complex(C4P),             intent(inout) :: in(*)          !< Complex buffer
    real(C4P),                intent(inout) :: out(*)         !< Real buffer
    integer(IP)                             :: ierr           !< Error flag

    ierr = DftiComputeBackward(self%plan, in, out)
  end subroutine execute_f_c2r

!------------------------------------------------------------------------------------------------
  subroutine destroy_c2r(self)
!------------------------------------------------------------------------------------------------
!< Destroys MKL r2c plan
!------------------------------------------------------------------------------------------------
    class(mkl_c2r_executor),   intent(inout) :: self          !< C2R Executor

    call destroy_mkl_plan(self%plan)
  end subroutine destroy_c2r

!------------------------------------------------------------------------------------------------
  function create_mkl_plan(precision, forward_domain, n, how_many, placement, input, output) result(plan)
!------------------------------------------------------------------------------------------------
!< Creates general MKL plan
!------------------------------------------------------------------------------------------------
    integer(IP),            intent(in)  :: precision          !< MKL Precision
    integer(IP),            intent(in)  :: forward_domain     !< C2C or R2C flag
    integer(IP),            intent(in)  :: n                  !< Size of 1d transform
    integer(IP),            intent(in)  :: how_many           !< Sets DFTI_NUMBER_OF_TRANSFORMS
    integer(IP),            intent(in)  :: placement          !< Sets DFTI_PLACEMENT  
    integer(IP),            intent(in)  :: input              !< Sets DFTI_INPUT_DISTANCE 
    integer(IP),            intent(in)  :: output             !< Sets DFTI_OUTPUT_DISTANCE 
    type(DFTI_DESCRIPTOR),  pointer     :: plan               !< MKL Plan
    integer(IP)                         :: ierr               !< Error flag
    integer(IP)                         :: mkl_precision      !< MKL Precision value

    if(precision == C8P) then
      mkl_precision = DFTI_DOUBLE
    elseif(precision == C4P) then
      mkl_precision = DFTI_SINGLE
    endif

    plan => null()
    ierr = DftiCreateDescriptor(plan, mkl_precision, forward_domain, 1, n)
    ierr = DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, how_many)
    ierr = DftiSetValue(plan, DFTI_PLACEMENT, placement)
    ierr = DftiSetValue(plan, DFTI_INPUT_DISTANCE, input)
    ierr = DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, output)
    ierr = DftiSetValue(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
    ierr = DftiCommitDescriptor(plan)
  end function create_mkl_plan

!------------------------------------------------------------------------------------------------
  subroutine destroy_mkl_plan(plan)
!------------------------------------------------------------------------------------------------
!< Destroys general MKL plan
!------------------------------------------------------------------------------------------------
    type(DFTI_DESCRIPTOR),  pointer  :: plan                  !< MKL Plan
    integer(IP)                      :: ierr                  !< Error flag

    ierr = DftiFreeDescriptor(plan)
  end subroutine destroy_mkl_plan
#endif
end module dtfft_executor_mkl_m