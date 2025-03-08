module dtfft_interface_nvtx
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

  interface
    subroutine nvtxDomainCreate_c(name, domain) bind(C, name="nvtxDomainCreate_c")
    import
      character(c_char),  intent(in)  :: name(*)
      type(nvtxDomainHandle)          :: domain
    end subroutine nvtxDomainCreate_c

    subroutine nvtxDomainRangePushEx_c(domain, message, color) bind(C, name="nvtxDomainRangePushEx_c")
    import
      type(nvtxDomainHandle), value               :: domain
      character(c_char),              intent(in)  :: message(*)
      integer(c_int),         value,  intent(in)  :: color
    end subroutine nvtxDomainRangePushEx_c

    subroutine nvtxDomainRangePop_c(domain) bind(C, name="nvtxDomainRangePop_c")
    import
      type(nvtxDomainHandle), value               :: domain
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