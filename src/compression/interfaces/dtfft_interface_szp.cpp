#include "dtfft_compression_utils.h"

#include "szp.h"
#include "szp_CompressionToolkit.h"

namespace
{
    /**
     * SZp `*_arg` API stores the absolute error bound used during compression in the first
     * `sizeof(T)` big-endian bytes of the compressed stream, while payload starts right after it.
     * `outSize` reported by SZp already accounts for that header.
     */
    template<typename T>
    inline T read_error_bound(unsigned char* cmpData);

    template<>
    inline float read_error_bound<float>(unsigned char* cmpData)
    {
        return szp::bytesToFloat(cmpData);
    }

    template<>
    inline double read_error_bound<double>(unsigned char* cmpData)
    {
        return szp::bytesToDouble(cmpData);
    }

    template<typename T>
    inline void compress_impl(unsigned char* cmpData, T* data, T absErrBound, size_t n_elements, int blockSize, size_t* outSize);

    template<>
    inline void compress_impl<float>(unsigned char* cmpData, float* data, float absErrBound, size_t n_elements, int blockSize, size_t* outSize)
    {
#ifdef DTFFT_WITH_OPENMP
        szp_float_openmp_threadblock_arg(cmpData, data, outSize, absErrBound, n_elements, blockSize);
#else
        szp_float_single_thread_arg(cmpData, data, outSize, absErrBound, n_elements, blockSize);
#endif
    }

    template<>
    inline void compress_impl<double>(unsigned char* cmpData, double* data, double absErrBound, size_t n_elements, int blockSize, size_t* outSize)
    {
#ifdef DTFFT_WITH_OPENMP
        szp_double_openmp_threadblock_arg(cmpData, data, outSize, absErrBound, n_elements, blockSize);
#else
        szp_double_single_thread_arg(cmpData, data, outSize, absErrBound, n_elements, blockSize);
#endif
    }

    template<typename T>
    inline void decompress_impl(T* data, size_t n_elements, T absErrBound, int blockSize, unsigned char* cmpData);

    template<>
    inline void decompress_impl<float>(float* data, size_t n_elements, float absErrBound, int blockSize, unsigned char* cmpData)
    {
#ifdef DTFFT_WITH_OPENMP
        szp_float_decompress_openmp_threadblock_arg(data, n_elements, absErrBound, blockSize, cmpData);
#else
        szp_float_decompress_single_thread_arg(data, n_elements, absErrBound, blockSize, cmpData);
#endif
    }

    template<>
    inline void decompress_impl<double>(double* data, size_t n_elements, double absErrBound, int blockSize, unsigned char* cmpData)
    {
#ifdef DTFFT_WITH_OPENMP
        szp_double_decompress_openmp_threadblock_arg(data, n_elements, absErrBound, blockSize, cmpData);
#else
        // szp_double_decompress_single_thread_arg if bugged
        // TODO: remove when bug is fixed
        const size_t data_offset = sizeof(size_t);
        // SZp's single-thread double decoder omits the offset-table size.
        memcpy(cmpData, &data_offset, sizeof(data_offset));
        szp_double_decompress_single_thread_arg(data, n_elements, absErrBound, blockSize, cmpData);
#endif
    }

    template<typename T>
    inline size_t decompress_typed(size_t n_elements, unsigned char* cmpData, void* data, int blockSize)
    {
        const T absErrBound = read_error_bound<T>(cmpData);
        decompress_impl(reinterpret_cast<T*>(data), n_elements, absErrBound, blockSize, cmpData + sizeof(T));
        return n_elements * sizeof(T);
    }
}

extern "C"
{
    /**
     * Compresses `n_elements` scalars of size `storage_size` into `cmpData`.
     *
     * Returns total number of bytes written to `cmpData` or 0 in case of failure.
     */
    size_t _dtfft_szp_compress(size_t n_elements, void* data, unsigned char* cmpData, size_t storage_size,
        int32_t mode, double tolerance, int32_t blockSize)
    {
        double errBound;
        if ( get_error_bound(data, mode, n_elements, tolerance, storage_size, &errBound) != 0 ) {
            return 0;
        }
        size_t compressedSize = 0;
        switch (storage_size) {
            case sizeof(float):
                compress_impl(cmpData, reinterpret_cast<float*>(data), static_cast<float>(errBound), n_elements, blockSize, &compressedSize);
                break;
            case sizeof(double):
                compress_impl(cmpData, reinterpret_cast<double*>(data), errBound, n_elements, blockSize, &compressedSize);
                break;
        }
        return compressedSize;
    }

    /**
     * Decompresses `n_elements` scalars of size `storage_size` from `cmpData` into `data`.
     *
     * Returns number of decompressed bytes or 0 in case of failure.
     */
    size_t _dtfft_szp_decompress(size_t n_elements, unsigned char* cmpData, void* data, size_t storage_size,
        int32_t blockSize)
    {
        switch (storage_size) {
            case sizeof(float):
                return decompress_typed<float>(n_elements, cmpData, data, blockSize);
            case sizeof(double):
                return decompress_typed<double>(n_elements, cmpData, data, blockSize);
            default:
                return 0;
        }
    }
}
