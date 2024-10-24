#include <dtfft_private.h>
#include "mkl_dfti.h"


MKL_LONG mkl_dfti_create_desc(int precision, int domain, long int dim, long int *length, void **desc)
{
  DFTI_DESCRIPTOR_HANDLE handle;
  MKL_LONG ierr;
  if ( dim == 1 ) {
    ierr = DftiCreateDescriptor(&handle, precision, domain, dim, length[0]);
  } else {
    ierr = DftiCreateDescriptor(&handle, precision, domain, dim, length);
  }
  *desc = handle;
  return ierr;
}

MKL_LONG mkl_dfti_set_value(void *desc, int param, int value)
{
  return DftiSetValue(desc, param, value);
}

MKL_LONG mkl_dfti_set_pointer(void *desc, int param, MKL_LONG *value)
{
  return DftiSetValue(desc, param, value);
}

MKL_LONG mkl_dfti_commit_desc(void * desc)
{
  return DftiCommitDescriptor(desc);
}

MKL_LONG mkl_dfti_execute(void *desc, void *in, void *out, int sign)
{
  if (sign == FFT_FORWARD) {
    return DftiComputeForward(desc, in, out);
  } else {
    return DftiComputeBackward(desc, in, out);
  }
}

MKL_LONG mkl_dfti_execute_forward(void *desc, void *in, void *out)
{
  return DftiComputeForward(desc, in, out);
}

MKL_LONG mkl_dfti_execute_backward(void *desc, void *in, void *out)
{
  return DftiComputeBackward(desc, in, out);
}

MKL_LONG mkl_dfti_free_desc(DFTI_DESCRIPTOR_HANDLE desc)
{
  return DftiFreeDescriptor(&desc);
}