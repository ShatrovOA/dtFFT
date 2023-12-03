module dtfft_interface_fftw_native_m
use iso_c_binding
implicit none
private
public :: FFTW_MEASURE, FFTW_DESTROY_INPUT, C_FFTW_R2R_KIND
public :: fftw_destroy_plan, fftwf_destroy_plan

include "fftw3.f03"
endmodule dtfft_interface_fftw_native_m