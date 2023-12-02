#include <stdlib.h>
#include "mkl_dfti.h"


const int C_DFTI_DOUBLE = DFTI_DOUBLE;
const int C_DFTI_SINGLE = DFTI_SINGLE;

const int C_DFTI_NUMBER_OF_TRANSFORMS = DFTI_NUMBER_OF_TRANSFORMS;
const int C_DFTI_PLACEMENT = DFTI_PLACEMENT;
const int C_DFTI_INPUT_DISTANCE = DFTI_INPUT_DISTANCE;
const int C_DFTI_OUTPUT_DISTANCE = DFTI_OUTPUT_DISTANCE;
const int C_DFTI_CONJUGATE_EVEN_STORAGE = DFTI_CONJUGATE_EVEN_STORAGE;
const int C_DFTI_COMPLEX_COMPLEX = DFTI_COMPLEX_COMPLEX;

const int C_DFTI_COMPLEX = DFTI_COMPLEX;
const int C_DFTI_REAL = DFTI_REAL;

const int C_DFTI_INPLACE = DFTI_INPLACE;
const int C_DFTI_NOT_INPLACE = DFTI_NOT_INPLACE;

extern const int C_DTFFT_FORWARD;


void mkl_dfti_create_desc(int precision, int domain, int dim, int length, DFTI_DESCRIPTOR_HANDLE *desc) 
{
  *desc = NULL;
  DFTI_DESCRIPTOR_HANDLE handle;
  DftiCreateDescriptor(&handle, precision, domain, dim, length);
  *desc = handle;
}

void mkl_dfti_set_value(DFTI_DESCRIPTOR_HANDLE desc, int param, int value)
{
  DftiSetValue(desc, param, value);
}

void mkl_dfti_commit_desc(DFTI_DESCRIPTOR_HANDLE desc)
{
  DftiCommitDescriptor(desc);
}

void mkl_dfti_execute(DFTI_DESCRIPTOR_HANDLE desc, void *in, void *out, int sign)
{
  if (sign == C_DTFFT_FORWARD) {
    DftiComputeForward(desc, in, out);
  } else {
    DftiComputeBackward(desc, in, out);
  }
}

void mkl_dfti_free_desc(DFTI_DESCRIPTOR_HANDLE *desc)
{
  DftiFreeDescriptor(desc);
}