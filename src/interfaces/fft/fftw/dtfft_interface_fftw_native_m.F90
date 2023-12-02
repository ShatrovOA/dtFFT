module dtfft_interface_fftw_native_m
use iso_c_binding
implicit none
private
public :: FFTW_MEASURE, FFTW_DESTROY_INPUT
public :: fftw_plan_many_dft, fftwf_plan_many_dft,           &
          fftw_plan_many_dft_r2c, fftwf_plan_many_dft_r2c,   &
          fftw_plan_many_dft_c2r, fftwf_plan_many_dft_c2r,   &
          fftw_plan_many_r2r, fftwf_plan_many_r2r,           &
          fftw_destroy_plan, fftwf_destroy_plan

include "fftw3.f03"
endmodule dtfft_interface_fftw_native_m