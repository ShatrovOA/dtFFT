#include "nvcomp/native/bitcomp.h"
#include "dtfft.h"
#include "dtfft_compression_utils.h"

#define BITCOMP_CHECK(func)              \
  do                                     \
  {                                      \
    bitcompResult_t rt = (func);         \
    if (rt != BITCOMP_SUCCESS) return 0; \
  } while (0)

extern "C"
{

size_t _dtfft_compress_bitcomp(void *in, void *out, size_t n, dtfft_compression_mode_t mode, size_t storage_size, double delta, dtfft_stream_t stream)
{
    bitcompHandle_t handle;
    bitcompDataType_t dtype = storage_size == sizeof(float) ? BITCOMP_FP32_DATA : BITCOMP_FP64_DATA;
    bitcompMode_t bmode = mode == DTFFT_COMPRESSION_MODE_LOSSLESS ? BITCOMP_LOSSLESS : BITCOMP_LOSSY_FP_TO_SIGNED;
    BITCOMP_CHECK( bitcompCreatePlan(&handle, n * storage_size, dtype, bmode, BITCOMP_DEFAULT_ALGO ) );
    if ( stream != NULL ) {
        BITCOMP_CHECK( bitcompSetStream(handle, (cudaStream_t)stream) );
        if ( bmode == BITCOMP_LOSSLESS ) {
            BITCOMP_CHECK( bitcompCompressLossless(handle, in, out) );
        } else {
            if ( dtype == BITCOMP_FP32_DATA ) {
                BITCOMP_CHECK( bitcompCompressLossy_fp32(handle, (float*)in, out, (float)delta) );
            } else {
                BITCOMP_CHECK( bitcompCompressLossy_fp64(handle, (double*)in, out, delta) );
            }
        }
        // cudaStreamSynchronize(stream);
    } else {
        double tol;
        int ierr = get_error_bound(in, (int32_t)mode, n, delta, (int32_t)storage_size, &tol);
        if ( ierr != 0 ) return 0;

        if ( bmode == BITCOMP_LOSSLESS ) {
            BITCOMP_CHECK( bitcompHostCompressLossless(handle, in, out) );
        } else {
            if ( dtype == BITCOMP_FP32_DATA ) {
                BITCOMP_CHECK( bitcompHostCompressLossy_fp32(handle, (float*)in, out, (float)tol) );
            } else {
                BITCOMP_CHECK( bitcompHostCompressLossy_fp64(handle, (double*)in, out, tol) );
            }
        }
    }
    size_t compressed_size;
    BITCOMP_CHECK( bitcompGetCompressedSize(out, &compressed_size) );
    BITCOMP_CHECK( bitcompDestroyPlan(handle) );
    return compressed_size;
}

size_t _dtfft_decompress_bitcomp(void *in, void *out, dtfft_stream_t stream)
{
    bitcompHandle_t handle;
    BITCOMP_CHECK( bitcompCreatePlanFromCompressedData(&handle, in) );
    if ( stream != NULL ) {
        BITCOMP_CHECK( bitcompSetStream(handle, (cudaStream_t)stream) );
        BITCOMP_CHECK( bitcompUncompress( handle, in, out ) );
        // cudaStreamSynchronize(stream);
    } else {
        BITCOMP_CHECK( bitcompHostUncompress( handle, in, out ) );
    }
    size_t uncompressed_size;
    BITCOMP_CHECK( bitcompGetUncompressedSize(in, &uncompressed_size) );
    BITCOMP_CHECK( bitcompDestroyPlan(handle) );
    return uncompressed_size;
}

}