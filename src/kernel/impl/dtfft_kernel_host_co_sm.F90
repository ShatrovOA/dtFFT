#include "dtfft_config.h"
submodule (dtfft_kernel_host:dtfft_kernel_host_sm) dtfft_kernel_host_co_sm
#include "_dtfft_private.h"
#include "_dtfft_kernel_defines.inc"
contains

#define MAX_TASKS 128

#define PREC _f32
#define BUFFER_TYPE real(real32)
#define STORAGE_BYTES FLOAT_STORAGE_SIZE
#include "_dtfft_kernel_host_co_routines.inc"

#define PREC _f64
#define BUFFER_TYPE real(real64)
#define STORAGE_BYTES DOUBLE_STORAGE_SIZE
#include "_dtfft_kernel_host_co_routines.inc"

#define PREC _f128
#define BUFFER_TYPE complex(real64)
#define STORAGE_BYTES DOUBLE_COMPLEX_STORAGE_SIZE
#include "_dtfft_kernel_host_co_routines.inc"

end submodule dtfft_kernel_host_co_sm