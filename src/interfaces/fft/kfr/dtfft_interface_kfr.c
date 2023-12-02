#include <kfr/capi.h>

extern const int C_DTFFT_FORWARD;


void kfr_execute_r2r_32(void *plan, float *in, float *out, uint8_t *temp, int sign, int how_many, int stride)
{
  if ( sign == C_DTFFT_FORWARD ) {
    kfr_dct_execute_f32()
  } else {

  }
}



void kfr_execute_c2c_32(void *plan, float *in, float *out, uint8_t *temp, int sign, int how_many, int stride)
{
  if(sign == C_DTFFT_FORWARD) {
    for (int i = 0; i < 2 * how_many * stride; i += 2 * stride)
      kfr_dft_execute_f32(plan, &out[i], &in[i], temp);
  } else {
    for (int i = 0; i < 2 * how_many * stride; i += 2 * stride)
      kfr_dft_execute_inverse_f32(plan, &out[i], &in[i], temp);
  }
}

void kfr_execute_c2c_64(void *plan, double *in, double *out, uint8_t *temp, int sign, int how_many, int stride)
{
  if(sign == C_DTFFT_FORWARD) {
    for (int i = 0; i < 2 * how_many * stride; i += 2 * stride)
      kfr_dft_execute_f64(plan, &out[i], &in[i], temp);
  } else {
    for (int i = 0; i < 2 * how_many * stride; i += 2 * stride)
      kfr_dft_execute_inverse_f64(plan, &out[i], &in[i], temp);
  }
}

void kfr_execute_r2c_32(void *plan, float *in, float *out, uint8_t *temp, int sign, int how_many, int stride)
{
  if(sign == C_DTFFT_FORWARD) {
    for(int i = 0; i < how_many; i++)
      kfr_dft_real_execute_f32(plan, &out[i * (stride + 2)], &in[i * stride], temp);
  } else {
    for(int i = 0; i < how_many; i++)
      kfr_dft_real_execute_inverse_f32(plan, &out[i * stride], &in[i * (stride + 2)], temp);
  }
}

void kfr_execute_r2c_64(void *plan, double *in, double *out, uint8_t *temp, int sign, int how_many, int stride)
{
  if(sign == C_DTFFT_FORWARD) {
    for(int i = 0; i < how_many; i++){
      kfr_dft_real_execute_f64(plan, &out[i * (stride + 2)], &in[i * stride], temp);
    }
  } else {
    for(int i = 0; i < how_many; i++)
      kfr_dft_real_execute_inverse_f64(plan, &out[i * stride], &in[i * (stride + 2)], temp);
  }
}