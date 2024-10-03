module dtfft_interface_fftw_native_m
use iso_c_binding, only: c_int, c_int32_t, c_intptr_t, c_double_complex, c_double,  &
                         c_ptr, c_funptr, c_size_t, c_float, c_float_complex, c_char
implicit none
private
public :: FFTW_MEASURE, FFTW_DESTROY_INPUT, C_FFTW_R2R_KIND
public :: fftw_destroy_plan, fftwf_destroy_plan

include "fftw3.f03"
endmodule dtfft_interface_fftw_native_m