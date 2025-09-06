#if defined( DTFFT_WITH_PROFILER )
# if defined( DTFFT_WITH_CUDA )
use dtfft_interface_nvtx
#define REGION_BEGIN(name, color) call push_nvtx_domain_range(name, color)
#define REGION_END(name) call pop_nvtx_domain_range()
#define PHASE_BEGIN(name, color) REGION_BEGIN(name, color)
#define PHASE_END(name) REGION_END(name)
# else
use caliper_mod
#define REGION_BEGIN(name, color) call cali_begin_region(name)
#define REGION_END(name) call cali_end_region(name)
#define PHASE_BEGIN(name, color) call cali_begin_phase(name)
#define PHASE_END(name) call cali_end_phase(name)
# endif
#else
#define REGION_BEGIN(name, color)
#define REGION_END(name)
#define PHASE_BEGIN(name, color)
#define PHASE_END(name)
#endif