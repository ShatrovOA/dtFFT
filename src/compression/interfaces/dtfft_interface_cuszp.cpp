#include "dtfft.h"
#include "cuSZp.h"

static inline void 
prepare(int8_t ndims, int32_t *dims_, size_t storage_size, size_t *nbEle, uint3 *dims, cuszp_dim_t *dim, cuszp_type_t *type, cuszp_mode_t *mode)
{
    *dim = CUSZP_DIM_2D;
    dims->x = (unsigned int)dims_[0];
    dims->y = (unsigned int)dims_[1];
    dims->z = 1;
    if ( ndims == 3 )
    {
        dims->z = (unsigned int)dims_[2];
        *dim = CUSZP_DIM_3D;
    }
    *nbEle = dims->x * dims->y * dims->z;
    *type = storage_size == sizeof(float) ? CUSZP_TYPE_FLOAT : CUSZP_TYPE_DOUBLE;
    *mode = CUSZP_MODE_FIXED;
}

extern "C" {

size_t _dtfft_compress_cuszp(void *in, void *out, int8_t ndims, int32_t *dims, size_t storage_size, float errorBound, dtfft_stream_t stream)
{
    size_t cmpSize;
    uint3 dims_;
    cuszp_dim_t dim;
    cuszp_type_t type;
    cuszp_mode_t mode;
    size_t nbEle;
    prepare(ndims, dims, storage_size, &nbEle, &dims_, &dim, &type, &mode);
    cuSZp_compress(in, reinterpret_cast<unsigned char *>(out), nbEle, &cmpSize, errorBound, dim, dims_, type, mode, (cudaStream_t)stream);
    return cmpSize;
}

void _dtfft_decompress_cuszp(void *in, void *out, int8_t ndims, int32_t *dims, size_t cmpSize, size_t storage_size, float errorBound, dtfft_stream_t stream)
{
    uint3 dims_;
    cuszp_dim_t dim;
    cuszp_type_t type;
    cuszp_mode_t mode;
    size_t nbEle;
    prepare(ndims, dims, storage_size, &nbEle, &dims_, &dim, &type, &mode);
    cuSZp_decompress(out, reinterpret_cast<unsigned char *>(in), nbEle, cmpSize, errorBound, dim, dims_, type, mode, (cudaStream_t)stream);
}

}