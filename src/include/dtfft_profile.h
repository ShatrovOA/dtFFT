#if defined( DTFFT_WITH_CALIPER )
use caliper_mod
#define REGION_BEGIN(name) call cali_begin_region(name)
#define REGION_END(name) call cali_end_region(name)
#define PHASE_BEGIN(name) call cali_begin_phase(name)
#define PHASE_END(name) call cali_end_phase(name)
#else
#define REGION_BEGIN(name)
#define REGION_END(name)
#define PHASE_BEGIN(name)
#define PHASE_END(name)
#endif