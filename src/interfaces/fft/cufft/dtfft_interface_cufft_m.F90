module dtfft_interface_cufft_m
use iso_c_binding, only: c_int, c_ptr
use cudafor, only: c_devptr
implicit none
private
public :: cufftPlanMany, cufftXtExec, cufftDestroy

  interface
    integer(c_int) function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, ffttype, batch) bind(C)
    import
      type(c_ptr)           :: plan
      integer(c_int), value :: rank
      integer(c_int)        :: n(*)
      integer(c_int)        :: inembed(*)
      integer(c_int), value :: istride
      integer(c_int), value :: idist
      integer(c_int)        :: onembed(*)
      integer(c_int), value :: ostride
      integer(c_int), value :: odist
      integer(c_int), value :: ffttype
      integer(c_int), value :: batch
    end function cufftPlanMany

    integer(c_int) function cufftXtExec(plan, input, output, direction) bind(C)
    import
      type(c_ptr),    value :: plan
      type(c_devptr), value :: input
      type(c_devptr), value :: output
      integer(c_int), value :: direction
    end function cufftXtExec

    integer(c_int) function cufftDestroy(plan) bind(C)
    import
      type(c_ptr),    value :: plan
    end function cufftDestroy
  end interface
end module dtfft_interface_cufft_m