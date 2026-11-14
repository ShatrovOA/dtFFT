module dtfft_interface_fftw_native
use, intrinsic :: iso_c_binding
implicit none
private
public :: FFTW_MEASURE, FFTW_DESTROY_INPUT, C_FFTW_R2R_KIND
public :: fftw_destroy_plan, fftwf_destroy_plan
public :: fftw_malloc, fftw_free
public :: fftw_init_threads, fftw_plan_with_nthreads
#ifdef FFTW_HAS_PLANNER_NTHREADS
public :: fftw_planner_nthreads
#endif

include "fftw3.f03"
endmodule dtfft_interface_fftw_native