#include "dtfft.hpp"
#include "dtfft_compression_utils.h"
#include "SZ3/api/sz.hpp"

extern "C"
{
    int32_t _dtfft_sz3_create_config(int32_t x, int32_t y, int32_t z, int32_t mode, double tolerance, void **conf)
    {
        SZ3::Config* config;
        if (z == 0) {
            config = new SZ3::Config(y, x);
        } else {
            config = new SZ3::Config(z, y, x);
        }

#ifdef DTFFT_WITH_OPENMP
        config->openmp = true;
#endif
        switch ( dtfft::CompressionMode(mode) ) {
            case dtfft::CompressionMode::LOSSLESS:
                config->cmprAlgo = SZ3::ALGO_LOSSLESS;
                break;
            case dtfft::CompressionMode::FIXED_PRECISION:
                config->errorBoundMode = SZ3::EB_REL;
                config->relErrorBound = tolerance;
                break;
            case dtfft::CompressionMode::FIXED_ACCURACY:
                config->errorBoundMode = SZ3::EB_ABS;
                config->absErrorBound = tolerance;
                break;
            default:
                delete config;
                return -1; // Invalid compression mode
        }

        *conf = static_cast<void*>(config);
        return 0;
    }

    void _dtfft_sz3_destroy_config(void *conf)
    {
        if(conf == nullptr) return;
        SZ3::Config* config = static_cast<SZ3::Config*>(conf);
        delete config;
    }

    size_t _dtfft_sz3_compress(void *conf, const void* data, char* cmpData, size_t cmpCap, size_t dataSize)
    {
        SZ3::Config* config = static_cast<SZ3::Config*>(conf);
        if (config->errorBoundMode == SZ3::EB_REL) {
            // Convert relative error bound to absolute one based on value range of the data.
            double absErrBound;
            if ( get_error_bound(data, static_cast<int32_t>(dtfft::CompressionMode::FIXED_PRECISION), config->num, config->relErrorBound, dataSize, &absErrBound) != 0 ) {
                return 0; // Failed to compute absolute error bound
            }
            config->errorBoundMode = SZ3::EB_ABS;
            config->absErrorBound = absErrBound;
        }

        switch (dataSize) {
            case sizeof(float):
                return SZ_compress(*config, reinterpret_cast<const float*>(data), cmpData, cmpCap);
            case sizeof(double):
                return SZ_compress(*config, reinterpret_cast<const double*>(data), cmpData, cmpCap);
            default:
                return 0;
        }
    }

    size_t _dtfft_sz3_decompress(void *conf, const char* cmpData, size_t cmpSize, void* decData, int32_t dataSize)
    {
        SZ3::Config* config = static_cast<SZ3::Config*>(conf);
        switch (dataSize) {
            case sizeof(float):
                SZ_decompress(*config, cmpData, cmpSize, reinterpret_cast<float*&>(decData));
                return config->num * sizeof(float);
            case sizeof(double):
                SZ_decompress(*config, cmpData, cmpSize, reinterpret_cast<double*&>(decData));
                return config->num * sizeof(double);
            default:
                return 0;
        }
    }
}