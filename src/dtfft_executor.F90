!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
#include "dtfft_config.h"
module dtfft_executor
use iso_c_binding
use dtfft_abstract_executor
#ifdef DTFFT_WITH_FFTW
use dtfft_executor_fftw_m
#endif
#ifdef DTFFT_WITH_MKL
use dtfft_executor_mkl_m
#endif
#ifdef DTFFT_WITH_CUFFT
use dtfft_executor_cufft_m
#endif
#ifdef DTFFT_WITH_VKFFT
use dtfft_executor_vkfft_m
#endif
use dtfft_utils
#include "_dtfft_private.h"
#include "_dtfft_mpi.h"
implicit none
public :: is_valid_executor, dtfft_get_executor_string
#ifdef DTFFT_WITH_CUDA
public :: is_host_executor, is_cuda_executor
#endif
!------------------------------------------------------------------------------------------------
! External FFT executor types
!------------------------------------------------------------------------------------------------
public :: dtfft_executor_t
    type, bind(C) :: dtfft_executor_t
    !! Type that specifies external FFT executor
        integer(c_int32_t) :: val !! Internal value
    end type dtfft_executor_t

    type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_NONE   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_NONE)
        !! Do not setup any executor. If this type is provided, then `execute` method should not be called.
        !! Use `transpose` method instead
    type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_FFTW3 = dtfft_executor_t(CONF_DTFFT_EXECUTOR_FFTW3)
        !! FFTW3 executor
    type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_MKL   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_MKL)
        !! MKL executor
    type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_CUFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_CUFFT)
        !! cuFFT GPU executor
    type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_VKFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_VKFFT)
        !! VkFFT GPU executor
    type(dtfft_executor_t),  parameter  :: VALID_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE    &
#ifdef DTFFT_WITH_FFTW
                                                                ,DTFFT_EXECUTOR_FFTW3   &
#endif
#ifdef DTFFT_WITH_MKL
                                                                ,DTFFT_EXECUTOR_MKL     &
#endif
#ifdef DTFFT_WITH_CUFFT
                                                                ,DTFFT_EXECUTOR_CUFFT   &
#endif
#ifdef DTFFT_WITH_VKFFT
                                                                ,DTFFT_EXECUTOR_VKFFT   &
#endif
  ]
  !! List of valid executors
#ifdef DTFFT_WITH_CUDA
    type(dtfft_executor_t),  parameter        :: HOST_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_FFTW3, DTFFT_EXECUTOR_MKL]
        !! List of host executors
    type(dtfft_executor_t),  parameter        :: CUDA_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_CUFFT, DTFFT_EXECUTOR_VKFFT]
        !! List of CUDA executors
#endif

public :: operator(==)
    interface operator(==)
        module procedure executor_eq        !! Check if two `dtfft_executor_t` are equal
    end interface


public :: operator(/=)
    interface operator(/=)
        module procedure executor_ne        !! Check if two `dtfft_executor_t` are not equal
    end interface
contains

    subroutine allocate_executor(executor, fft)
        type(dtfft_executor_t), intent(in) :: executor
        class(abstract_executor), allocatable, intent(out) :: fft

        select case ( executor%val )
#ifdef DTFFT_WITH_FFTW
        case ( DTFFT_EXECUTOR_FFTW3%val )
            allocate( fftw_executor :: fft )
#endif
#ifdef DTFFT_WITH_MKL
        case ( DTFFT_EXECUTOR_MKL%val )
            allocate( mkl_executor :: fft )
#endif
#ifdef DTFFT_WITH_CUFFT
        case ( DTFFT_EXECUTOR_CUFFT%val )
            allocate( cufft_executor :: fft )
#endif
#ifdef DTFFT_WITH_VKFFT
        case ( DTFFT_EXECUTOR_VKFFT%val )
            allocate( vkfft_executor :: fft )
#endif
        ! case default
        !     INTERNAL_ERROR("Unknown executor type")
        end select
    end subroutine allocate_executor

    function dtfft_get_executor_string(executor) result(string)
    !! Gets the string description of an executor
        type(dtfft_executor_t), intent(in) :: executor !! Executor type
        character(len=:), allocatable :: string !! Executor string

        select case ( executor%val )
        case ( DTFFT_EXECUTOR_NONE%val )
            allocate(string, source="None")
        case ( DTFFT_EXECUTOR_FFTW3%val )
            allocate(string, source="FFTW3")
        case ( DTFFT_EXECUTOR_MKL%val )
            allocate(string, source="MKL")
        case ( DTFFT_EXECUTOR_CUFFT%val )
            allocate(string, source="CUFFT")
        case ( DTFFT_EXECUTOR_VKFFT%val )
            allocate(string, source="VKFFT")
        case default
            allocate(string, source="Unknown executor")
        endselect
    end function dtfft_get_executor_string


MAKE_EQ_FUN(dtfft_executor_t, executor_eq)
MAKE_NE_FUN(dtfft_executor_t, executor_ne)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_valid_executor, VALID_EXECUTORS)
#ifdef DTFFT_WITH_CUDA
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_host_executor, HOST_EXECUTORS)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_cuda_executor, CUDA_EXECUTORS)
#endif
end module dtfft_executor