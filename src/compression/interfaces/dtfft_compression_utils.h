#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
int get_error_bound(const void* data, int32_t mode, size_t n_elements, double tolerance, size_t storage_size, double *absErrBound);
#ifdef __cplusplus
}
#endif