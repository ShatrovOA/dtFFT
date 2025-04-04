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
module dtfft_interface_nvtx
!! nvtx3 Interfaces
use iso_c_binding
use dtfft_utils
implicit none
private
public :: push_nvtx_domain_range, pop_nvtx_domain_range

  type, bind(C) :: nvtxDomainHandle
    type(c_ptr) :: handle
  end type nvtxDomainHandle

  type(nvtxDomainHandle),     save  :: domain_nvtx
  !! NVTX domain handle
  logical,                    save  :: domain_created = .false.
  !! Has domain been created?

  interface
  !! Creates an NVTX domain with the specified name.
    subroutine nvtxDomainCreate_c(name, domain) bind(C, name="nvtxDomainCreate_c")
      import
      character(c_char),  intent(in)  :: name(*)  !! Name of the NVTX domain.
      type(nvtxDomainHandle)          :: domain   !! Handle to the created NVTX domain.
    end subroutine nvtxDomainCreate_c
  end interface

  interface
  !! Pushes a range with a custom message and color onto the specified NVTX domain.
    subroutine nvtxDomainRangePushEx_c(domain, message, color) bind(C, name="nvtxDomainRangePushEx_c")
      import
      type(nvtxDomainHandle), value               :: domain  !! NVTX domain handle.
      character(c_char),              intent(in)  :: message(*)  !! Custom message for the range.
      integer(c_int),         value,  intent(in)  :: color   !! Color for the range.
    end subroutine nvtxDomainRangePushEx_c
  end interface

  interface
  !! Pops a range from the specified NVTX domain.
    subroutine nvtxDomainRangePop_c(domain) bind(C, name="nvtxDomainRangePop_c")
      import
      type(nvtxDomainHandle), value :: domain  !! NVTX domain handle.
    end subroutine nvtxDomainRangePop_c
  end interface

contains

  subroutine create_nvtx_domain
  !! Creates a new NVTX domain
    character(c_char), allocatable :: cstr(:)

    call astring_f2c("dtFFT", cstr)

    call nvtxDomainCreate_c(cstr, domain_nvtx)
    domain_created = .true.
    deallocate(cstr)
  end subroutine create_nvtx_domain

  subroutine push_nvtx_domain_range(message, color)
  !! Pushes a range to the NVTX domain
    character(len=*), intent(in)    :: message    !! Message to push
    integer(c_int),   intent(in)    :: color      !! Color of the range
    character(c_char), allocatable  :: cstr(:)

    if ( .not. domain_created ) call create_nvtx_domain()
    call astring_f2c(message, cstr)
    call nvtxDomainRangePushEx_c(domain_nvtx, cstr, color)
    deallocate(cstr)
  end subroutine push_nvtx_domain_range

  subroutine pop_nvtx_domain_range()
  !! Pops a range from the NVTX domain
    call nvtxDomainRangePop_c(domain_nvtx)
  end subroutine pop_nvtx_domain_range
end module dtfft_interface_nvtx