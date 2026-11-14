#include "dtfft_compression_utils.h"
#include "dtfft.hpp"

/** Computes absolute error bound based on requested compression mode */
template<typename T>
inline double get_error_bound(dtfft::CompressionMode mode, const T* data, size_t n_elements, double tolerance)
{
    if ( n_elements == 0 || mode != dtfft::CompressionMode::FIXED_PRECISION ) return tolerance;

    // Relative bound is converted to an absolute one based on value range of the data.
    T min = data[0];
    T max = data[0];
    for ( size_t i = 1; i < n_elements; i++ ) {
        const T v = data[i];
        if ( v < min ) min = v;
        if ( v > max ) max = v;
    }
    double absErrBound = tolerance * (max - min);
    // Constant data has no range to scale with. Any positive bound satisfies
    // requested relative accuracy, while zero one breaks SZp
    if ( absErrBound <= 0 ) absErrBound = tolerance;
    return absErrBound;
}


int get_error_bound(const void* data, int32_t mode, size_t n_elements, double tolerance, size_t storage_size, double *absErrBound)
{
    switch (storage_size) {
        case sizeof(float):
            *absErrBound = get_error_bound(dtfft::CompressionMode(mode), reinterpret_cast<const float*>(data), n_elements, tolerance);
            break;
        case sizeof(double):
            *absErrBound = get_error_bound(dtfft::CompressionMode(mode), reinterpret_cast<const double*>(data), n_elements, tolerance);
            break;
        default:
            return -1; // Invalid data size
    }
    return 0;
}