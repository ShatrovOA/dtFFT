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
!------------------------------------------------------------------------------------------------
!< This module creates interface with fftw3 library
!------------------------------------------------------------------------------------------------
use iso_c_binding, only: C_PTR
use dtfft_interface_fftw_native_m, only: FFTW_MEASURE, FFTW_DESTROY_INPUT
use dtfft_interface_fftw_native_m, only: fftw_plan_many_dft, fftwf_plan_many_dft,           &
                                         fftw_plan_many_dft_r2c, fftwf_plan_many_dft_r2c,   &
                                         fftw_plan_many_dft_c2r, fftwf_plan_many_dft_c2r,   &
                                         fftw_plan_many_r2r, fftwf_plan_many_r2r,           &
                                         fftw_destroy_plan, fftwf_destroy_plan
implicit none
public

  interface
    subroutine fftw_execute_dft(ptr, in, out) bind(C, name="fftw_execute_dft")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftw_execute_dft

    subroutine fftwf_execute_dft(ptr, in, out) bind(C, name="fftwf_execute_dft")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftwf_execute_dft

    subroutine fftw_execute_dft_r2c(ptr, in, out) bind(C, name="fftw_execute_dft_r2c")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftw_execute_dft_r2c

    subroutine fftwf_execute_dft_r2c(ptr, in, out) bind(C, name="fftwf_execute_dft_r2c")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftwf_execute_dft_r2c

    subroutine fftw_execute_dft_c2r(ptr, in, out) bind(C, name="fftw_execute_dft_c2r")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftw_execute_dft_c2r

    subroutine fftwf_execute_dft_c2r(ptr, in, out) bind(C, name="fftwf_execute_dft_c2r")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftwf_execute_dft_c2r

    subroutine fftw_execute_r2r(ptr, in, out) bind(C, name="fftw_execute_r2r")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftw_execute_r2r

    subroutine fftwf_execute_r2r(ptr, in, out) bind(C, name="fftwf_execute_r2r")
    import
      type(C_PTR), value :: ptr
      type(C_PTR), value :: in
      type(C_PTR), value :: out
    end subroutine fftwf_execute_r2r
  end interface
end module dtfft_interface_fftw_m