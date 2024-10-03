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
module dtfft_interface_kfr_m
use dtfft_precisions
use iso_c_binding
implicit none
private
public :: KFR_PACK_PERM, KFR_PACK_CCS,    &
          kfr_dft_delete_plan_f32,        &
          kfr_dft_delete_plan_f64,        &
          kfr_dft_real_delete_plan_f32,   &
          kfr_dft_real_delete_plan_f64,   &
          kfr_dct_delete_plan_f32,        &
          kfr_dct_delete_plan_f64,        &
          kfr_create_plan_c2c,            &
          kfr_create_plan_r2c,            &
          kfr_create_plan_dct,            &
          kfr_execute_c2c,                &
          kfr_execute_r2c,                &
          kfr_execute_dct



  integer(IP), parameter :: KFR_PACK_PERM = 0
  integer(IP), parameter :: KFR_PACK_CCS = 1

  interface
    subroutine kfr_dft_delete_plan_f32(plan) bind(C, name="kfr_dft_delete_plan_f32")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dft_delete_plan_f32

    subroutine kfr_dft_delete_plan_f64(plan) bind(C, name="kfr_dft_delete_plan_f64")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dft_delete_plan_f64

    subroutine kfr_dft_real_delete_plan_f32(plan) bind(C, name="kfr_dft_real_delete_plan_f32")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dft_real_delete_plan_f32

    subroutine kfr_dft_real_delete_plan_f64(plan) bind(C, name="kfr_dft_real_delete_plan_f64")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dft_real_delete_plan_f64

    subroutine kfr_dct_delete_plan_f32(plan) bind(C, name="kfr_dct_delete_plan_f32")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dct_delete_plan_f32

    subroutine kfr_dct_delete_plan_f64(plan) bind(C, name="kfr_dct_delete_plan_f64")
    import
      type(c_ptr),  value :: plan
    end subroutine kfr_dct_delete_plan_f64

    subroutine kfr_create_plan_c2c(size, precision, temp, plan) bind(C, name="kfr_create_plan_c2c")
    import
      integer(SP),  value :: size
      integer(IP),  value :: precision
      type(c_ptr)         :: temp
      type(c_ptr)         :: plan
    end subroutine kfr_create_plan_c2c

    subroutine kfr_create_plan_r2c(size, precision, pack_format, temp, plan) bind(C, name="kfr_create_plan_r2c")
    import
      integer(SP),  value :: size
      integer(IP),  value :: precision
      integer(IP),  value :: pack_format
      type(c_ptr)         :: temp
      type(c_ptr)         :: plan
    end subroutine kfr_create_plan_r2c

    subroutine kfr_create_plan_dct(size, precision, temp, plan) bind(C, name="kfr_create_plan_dct")
    import
      integer(SP),  value :: size
      integer(IP),  value :: precision
      type(c_ptr)         :: temp
      type(c_ptr)         :: plan
    end subroutine kfr_create_plan_dct

    subroutine kfr_execute_c2c(plan, precision, in, out, temp, sign, how_many, size) bind(C, name="kfr_execute_c2c")
#include "args_execute.i90"
    end subroutine kfr_execute_c2c

    subroutine kfr_execute_r2c(plan, precision, in, out, temp, sign, how_many, size) bind(C, name="kfr_execute_r2c")
#include "args_execute.i90"
    end subroutine kfr_execute_r2c

    subroutine kfr_execute_dct(plan, precision, in, out, temp, sign, how_many, size) bind(C, name="kfr_execute_dct")
#include "args_execute.i90"
    end subroutine kfr_execute_dct
  endinterface
end module dtfft_interface_kfr_m