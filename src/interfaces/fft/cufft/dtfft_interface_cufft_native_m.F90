module dtfft_interface_cufft_native_m
use dtfft_precisions
use cufft
implicit none
private
public :: CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
public :: CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z
end module dtfft_interface_cufft_native_m