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
!< This module describes MKL Wrappers to dtFFT: [[mkl_executor]]
!< https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/fourier-transform-functions/fft-functions.html
!------------------------------------------------------------------------------------------------
use iso_fortran_env,           only: error_unit
use iso_c_binding,             only: c_long, c_ptr
use dtfft_abstract_executor_m, only: abstract_executor, FFT_C2C, FFT_R2C, FFT_R2R
use dtfft_interface_mkl_m
use dtfft_interface_mkl_native_m
use dtfft_precisions,          only: IP
use dtfft_parameters,          only: DTFFT_SUCCESS, DTFFT_FORWARD, DTFFT_BACKWARD, DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED, DTFFT_DOUBLE
use dtfft_utils,               only: suppress_unused, int_to_str
#include "dtfft_mpi.h"
implicit none
private
public :: mkl_executor

#define MKL_DFTI_CALL(name, func)                                                                                                                                   \
  ierr = func;                                                                                                                                                      \
  if( ierr /= DFTI_NO_ERROR ) then;                                                                                                                                 \
    write(error_unit, '(a)') "Error occured during call to MKL DFTI function '"//name//"': "//DftiErrorMessage(ierr)//" at "//__FILE__//":"//int_to_str(__LINE__);  \
    call MPI_Abort(MPI_COMM_WORLD, int(ierr, c_int));                                                                                                               \
  endif;

  type, extends(abstract_executor)  :: mkl_executor
  !< MKL FFT Executor
  private
    logical                       :: need_reconfigure
    integer(c_long), allocatable  :: istrides(:)
    integer(c_long), allocatable  :: ostrides(:)
    integer(IP)                   :: idist, odist
    ! integer(IP)                     :: sign                         !< Sign of Transform
  contains
    procedure,  pass(self)  :: create_private => create     !< Creates FFT plan via MKL DFTI Interface
    procedure,  pass(self)  :: execute_private => execute   !< Executes MKL plan
    procedure,  pass(self)  :: destroy_private => destroy   !< Destroys MKL plan
  endtype mkl_executor

contains

!------------------------------------------------------------------------------------------------
  subroutine make_plan(fft_rank, fft_sizes, mkl_precision, forward_domain, how_many, idist, odist, plan)
!------------------------------------------------------------------------------------------------
!< Creates general MKL plan
!------------------------------------------------------------------------------------------------
    integer(IP),              intent(in)    :: fft_rank           !< Rank of fft: 1 or 2
    integer(IP),              intent(in)    :: fft_sizes(:)       !< Dimensions of transform
    integer(IP),              intent(in)    :: mkl_precision      !< MKL Precision
    integer(IP),              intent(in)    :: forward_domain     !< C2C or R2C flag
    integer(IP),              intent(in)    :: how_many           !< Sets DFTI_NUMBER_OF_TRANSFORMS
    integer(IP),              intent(in)    :: idist              !< Sets DFTI_INPUT_DISTANCE
    integer(IP),              intent(in)    :: odist              !< Sets DFTI_OUTPUT_DISTANCE
    type(c_ptr),              intent(inout) :: plan               !< Resulting plan
    integer(c_long)                         :: ierr

    MKL_DFTI_CALL( "DftiCreateDescriptor", mkl_dfti_create_desc(mkl_precision, forward_domain, int(fft_rank, c_long), int(fft_sizes, c_long), plan) )
    MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(plan, DFTI_NUMBER_OF_TRANSFORMS, how_many) )
    MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE) )
    MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(plan, DFTI_INPUT_DISTANCE, idist) )
    MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(plan, DFTI_OUTPUT_DISTANCE, odist) )
    MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) )
    MKL_DFTI_CALL( "DftiCommitDescriptor", mkl_dfti_commit_desc(plan) )
  end subroutine make_plan

!------------------------------------------------------------------------------------------------
  subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
!------------------------------------------------------------------------------------------------
!< Creates FFT plan via MKL DFTI Interface
!------------------------------------------------------------------------------------------------
    class(mkl_executor),      intent(inout) :: self           !< MKL FFT Executor
    integer(IP),              intent(in)    :: fft_rank       !< Rank of fft: 1 or 2
    integer(IP),              intent(in)    :: fft_type       !< Type of fft: r2r, r2c, c2r, c2c
    integer(IP),              intent(in)    :: precision      !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
    integer(IP),              intent(in)    :: idist          !< Distance between the first element of two consecutive signals in a batch of the input data.
    integer(IP),              intent(in)    :: odist          !< Distance between the first element of two consecutive signals in a batch of the output data.
    integer(IP),              intent(in)    :: how_many       !< Number of transforms to create
    integer(IP),              intent(in)    :: fft_sizes(:)   !< Dimensions of transform
    integer(IP),              intent(in)    :: inembed(:)     !< Storage dimensions of the input data in memory.
    integer(IP),              intent(in)    :: onembed(:)     !< Storage dimensions of the output data in memory.
    integer(IP),              intent(inout) :: error_code     !< Error code to be returned to user
    integer(IP),   optional,  intent(in)    :: r2r_kinds(:)   !< Kinds of r2r transform
    integer(IP) :: forward_domain, mkl_precision, i, idx
    integer(c_long) :: iprod, oprod

    if ( present(r2r_kinds) ) then
    endif

    error_code = DTFFT_SUCCESS

    self%is_inverse_copied = .true.
    self%need_reconfigure = .false.

    select case (fft_type)
    case (FFT_C2C)
      forward_domain = DFTI_COMPLEX
    case (FFT_R2C)
      forward_domain = DFTI_REAL
      self%need_reconfigure = .true.
      self%idist = idist
      self%odist = odist
      allocate( self%istrides( size(inembed) + 1), source=0_c_long )
      allocate( self%ostrides( size(onembed) + 1), source=0_c_long )
      iprod = 1; oprod = 1
      self%istrides( size(inembed) + 1 ) = iprod
      self%ostrides( size(onembed) + 1 ) = oprod
      do i = 1, size(inembed) - 1
        idx = size(inembed) + 1 - i
        iprod = iprod * inembed( idx )
        oprod = oprod * onembed( idx )
        self%istrides(idx) = iprod
        self%ostrides(idx) = oprod
      enddo
    case (FFT_R2R)
      error_code = DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
      return
    endselect
    if(precision == DTFFT_DOUBLE) then
      mkl_precision = DFTI_DOUBLE
    else
      mkl_precision = DFTI_SINGLE
    endif

    call make_plan(fft_rank, fft_sizes, mkl_precision, forward_domain, how_many, idist, odist, self%plan_forward)
    self%plan_backward = self%plan_forward
  end subroutine create

!------------------------------------------------------------------------------------------------
  subroutine execute(self, a, b, sign)
!------------------------------------------------------------------------------------------------
!< Executes MKL plan
!------------------------------------------------------------------------------------------------
    class(mkl_executor),  intent(in)  :: self                 !< MKL FFT Executor
    type(c_ptr),          intent(in)  :: a                    !< Source pointer
    type(c_ptr),          intent(in)  :: b                    !< Target pointer
    integer(IP),          intent(in)  :: sign
    integer(c_long) :: ierr

    if ( self%need_reconfigure ) then
      if ( sign == DTFFT_FORWARD ) then
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(self%plan_forward, DFTI_INPUT_DISTANCE, self%idist) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(self%plan_forward, DFTI_OUTPUT_DISTANCE, self%odist) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_pointer(self%plan_forward, DFTI_INPUT_STRIDES, self%istrides) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_pointer(self%plan_forward, DFTI_OUTPUT_STRIDES, self%ostrides) )
      else
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(self%plan_forward, DFTI_INPUT_DISTANCE, self%odist) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_value(self%plan_forward, DFTI_OUTPUT_DISTANCE, self%idist) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_pointer(self%plan_forward, DFTI_INPUT_STRIDES, self%ostrides) )
        MKL_DFTI_CALL( "DftiSetValue", mkl_dfti_set_pointer(self%plan_forward, DFTI_OUTPUT_STRIDES, self%istrides) )
      endif
      MKL_DFTI_CALL( "DftiCommitDescriptor", mkl_dfti_commit_desc(self%plan_forward) )
    endif

    MKL_DFTI_CALL( "DftiCompute", mkl_dfti_execute(self%plan_forward, a, b, sign) )
  end subroutine execute

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys MKL plan
!------------------------------------------------------------------------------------------------
    class(mkl_executor),  intent(inout)  :: self              !< MKL FFT Executor
    integer(c_long) :: ierr

    MKL_DFTI_CALL( "DftiFreeDescriptor", mkl_dfti_free_desc(self%plan_forward) )
  end subroutine destroy
end module dtfft_executor_mkl_m