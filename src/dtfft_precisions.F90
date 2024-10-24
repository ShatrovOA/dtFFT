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
module dtfft_precisions
!------------------------------------------------------------------------------------------------
!< This module defines DTFFT precisions
!------------------------------------------------------------------------------------------------
use iso_c_binding, only: c_int, c_size_t, c_float, c_double, c_float_complex, c_double_complex
implicit none
private

!------------------------------------------------------------------------------------------------
! Integer types
!------------------------------------------------------------------------------------------------
  integer, public, parameter  :: IP  = C_INT
  !< Integer, 4 bytes
  integer, public, parameter  :: SP  = C_SIZE_T
  !< Integer, 4 bytes on 32 bit machine, 8 bytes on 64 bit machine

!------------------------------------------------------------------------------------------------
! Real types
!------------------------------------------------------------------------------------------------
  integer, public, parameter  :: R4P = C_FLOAT
  !< Float, 4 bytes
  integer, public, parameter  :: R8P = C_DOUBLE
  !< Float, 8 bytes

!------------------------------------------------------------------------------------------------
! Complex types
!------------------------------------------------------------------------------------------------
  integer, public, parameter  :: C4P = C_FLOAT_COMPLEX
  !< Complex, 2x4 bytes = 8 bytes
  integer, public, parameter  :: C8P = C_DOUBLE_COMPLEX
  !< Complex, 2x8 bytes = 16 bytes
end module dtfft_precisions