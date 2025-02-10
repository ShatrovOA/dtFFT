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
module dtfft_interface_fftw_m
!! This module creates interface with fftw3 library
use iso_c_binding,                 only: c_ptr, c_int
use dtfft_interface_fftw_native_m, only: C_FFTW_R2R_KIND, FFTW_MEASURE, FFTW_DESTROY_INPUT
use dtfft_interface_fftw_native_m, only: fftw_destroy_plan, fftwf_destroy_plan
implicit none
private
public :: C_FFTW_R2R_KIND, FFTW_MEASURE, FFTW_DESTROY_INPUT
public :: fftw_destroy_plan, fftwf_destroy_plan
public :: fftw_execute_dft, fftwf_execute_dft
public :: fftw_execute_dft_r2c, fftwf_execute_dft_r2c
public :: fftw_execute_dft_c2r, fftwf_execute_dft_c2r
public :: fftw_execute_r2r, fftwf_execute_r2r
public :: fftw_plan_many_dft, fftwf_plan_many_dft
public :: fftw_plan_many_dft_r2c, fftwf_plan_many_dft_r2c
public :: fftw_plan_many_dft_c2r, fftwf_plan_many_dft_c2r
public :: fftw_plan_many_r2r, fftwf_plan_many_r2r

  interface
    subroutine fftw_execute_dft(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftw_execute_dft

    subroutine fftwf_execute_dft(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftwf_execute_dft

    subroutine fftw_execute_dft_r2c(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftw_execute_dft_r2c

    subroutine fftwf_execute_dft_r2c(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftwf_execute_dft_r2c

    subroutine fftw_execute_dft_c2r(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftw_execute_dft_c2r

    subroutine fftwf_execute_dft_c2r(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftwf_execute_dft_c2r

    subroutine fftw_execute_r2r(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftw_execute_r2r

    subroutine fftwf_execute_r2r(plan, in, out) bind(C)
#include "args_execute.i90"
    end subroutine fftwf_execute_r2r

    type(C_PTR) function fftw_plan_many_dft(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) bind(C)
#include "args_create.i90"
      integer(C_INT), value :: sign
    end function fftw_plan_many_dft

    type(C_PTR) function fftwf_plan_many_dft(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,sign,flags) bind(C)
#include "args_create.i90"
      integer(C_INT), value :: sign
    end function fftwf_plan_many_dft

    type(C_PTR) function fftw_plan_many_dft_r2c(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) bind(C)
#include "args_create.i90"
    end function fftw_plan_many_dft_r2c

    type(C_PTR) function fftwf_plan_many_dft_r2c(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) bind(C)
#include "args_create.i90"
    end function fftwf_plan_many_dft_r2c

    type(C_PTR) function fftw_plan_many_dft_c2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) bind(C)
#include "args_create.i90"
    end function fftw_plan_many_dft_c2r

    type(C_PTR) function fftwf_plan_many_dft_c2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,flags) bind(C)
#include "args_create.i90"
    end function fftwf_plan_many_dft_c2r

    type(C_PTR) function fftw_plan_many_r2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags) bind(C)
#include "args_create.i90"
      integer(C_FFTW_R2R_KIND), intent(in) :: kind(*)
    end function fftw_plan_many_r2r

    type(C_PTR) function fftwf_plan_many_r2r(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride,odist,kind,flags) bind(C)
#include "args_create.i90"
      integer(C_FFTW_R2R_KIND), intent(in) :: kind(*)
    end function fftwf_plan_many_r2r
  end interface
end module dtfft_interface_fftw_m