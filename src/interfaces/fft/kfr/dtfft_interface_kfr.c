#include <kfr/capi.h>
#include <dtfft_config.h>
#include <dtfft_private.h>
#include <stdio.h>
#include <stdlib.h>



void kfr_create_plan_c2c(size_t size, int precision, uint8_t **temp, void **plan) {
  size_t temp_size;
  void *plan_;
  if ( precision == DTFFT_CONF_DOUBLE ) {
    plan_ = kfr_dft_create_plan_f64(size);
    temp_size = kfr_dft_get_temp_size_f64(plan_);
  } else {
    plan_ = kfr_dft_create_plan_f32(size);
    temp_size = kfr_dft_get_temp_size_f32(plan_);
  }
  uint8_t *temp_ = (uint8_t*)kfr_allocate(temp_size);
  *temp = temp_;
  *plan = plan_;
}

void kfr_create_plan_r2c(size_t size, int precision, int pack_format, uint8_t **temp, void **plan) {
  size_t temp_size;
  void *plan_;
  if ( precision == DTFFT_CONF_DOUBLE ) {
    plan_ = kfr_dft_real_create_plan_f64(size, pack_format);
    temp_size = kfr_dft_real_get_temp_size_f64(plan_);
  } else {
    plan_ = kfr_dft_real_create_plan_f32(size, pack_format);
    temp_size = kfr_dft_real_get_temp_size_f32(plan_);
  }
  uint8_t *temp_ = (uint8_t*)kfr_allocate(temp_size);
  *temp = temp_;
  *plan = plan_;
}

void kfr_create_plan_dct(size_t size, int precision, uint8_t **temp, void **plan) {
  size_t temp_size;
  void *plan_;
  if ( precision == DTFFT_CONF_DOUBLE ) {
    plan_ = kfr_dct_create_plan_f64(size);
    temp_size = kfr_dct_get_temp_size_f64(plan_);
  } else {
    plan_ = kfr_dct_create_plan_f32(size);
    temp_size = kfr_dct_get_temp_size_f32(plan_);
  }
  uint8_t *temp_ = (uint8_t*)kfr_allocate(temp_size);
  *temp = temp_;
  *plan = plan_;
}

void kfr_execute_c2c_32(void *plan, float *in, float *out, uint8_t *temp, int sign, size_t how_many, size_t size) {
  size_t iter;

  if ( sign == FFT_FORWARD ) {
    for (iter = 0; iter < how_many; iter++)
      kfr_dft_execute_f32(plan, &out[2 * iter * size], &in[2 * iter * size], temp);
  } else {
    for (iter = 0; iter < how_many; iter++)
      kfr_dft_execute_inverse_f32(plan, &out[2 * iter * size], &in[2 * iter * size], temp);
  }
}

void kfr_execute_c2c_64(void *plan, double *in, double *out, uint8_t *temp, int sign, size_t how_many, size_t size)
{
  size_t iter;

  if ( sign == FFT_FORWARD ) {
    for (iter = 0; iter < how_many; iter++)
      kfr_dft_execute_f64(plan, &out[2 * iter * size], &in[2 * iter * size], temp);
  } else {
    for (iter = 0; iter < how_many; iter++)
      kfr_dft_execute_inverse_f64(plan, &out[2 * iter * size], &in[2 * iter * size], temp);
  }
}


void kfr_execute_c2c(void *plan, int precision, void *in, void *out, uint8_t *temp, int sign, size_t how_many, size_t size) {
  if ( precision == DTFFT_CONF_DOUBLE ) {
    kfr_execute_c2c_64(plan, (double *) in, (double *) out, temp, sign, how_many, size);
  } else {
    kfr_execute_c2c_32(plan, (float *) in, (float *) out, temp, sign, how_many, size);
  }
}

void kfr_execute_r2c_32(void *plan, float *in, float *out, uint8_t *temp, size_t how_many, size_t size) {
  size_t iter;

  for(iter = 0; iter < how_many; iter++)
    kfr_dft_real_execute_f32(plan, &out[iter * (size + 2)], &in[iter * size], temp);
}

void kfr_execute_c2r_32(void *plan, float *in, float *out, uint8_t *temp, size_t how_many, size_t size) {
  size_t iter;

  for(iter = 0; iter < how_many; iter++)
    kfr_dft_real_execute_inverse_f32(plan, &out[iter * size], &in[iter * (size + 2)], temp);
}

void kfr_execute_r2c_64(void *plan, double *in, double *out, uint8_t *temp, size_t how_many, size_t size) {
  size_t iter;

  for(iter = 0; iter < how_many; iter++)
    kfr_dft_real_execute_f64(plan, &out[iter * (size + 2)], &in[iter * size], temp);
}

void kfr_execute_c2r_64(void *plan, double *in, double *out, uint8_t *temp, size_t how_many, size_t size) {
  size_t iter;

  for(iter = 0; iter < how_many; iter++)
    kfr_dft_real_execute_inverse_f64(plan, &out[iter * size], &in[iter * (size + 2)], temp);
}


void kfr_execute_r2c(void *plan, int precision, void *in, void *out, uint8_t *temp, int sign, size_t how_many, size_t size) {
  if ( precision == DTFFT_CONF_DOUBLE ) {
    if ( sign == FFT_FORWARD ) {
      kfr_execute_r2c_64(plan, (double *)in, (double *)out, temp, how_many, size);
    } else {
      kfr_execute_c2r_64(plan, (double *)in, (double *)out, temp, how_many, size);
    }
  } else {
    if ( sign == FFT_FORWARD ) {
      kfr_execute_r2c_32(plan, (float *)in, (float *)out, temp, how_many, size);
    } else {
      kfr_execute_c2r_32(plan, (float *)in, (float *)out, temp, how_many, size);
    }
  }
}


void kfr_execute_dct_32(void *plan, float *in, float *out, uint8_t *temp, int sign, size_t how_many, size_t size) {
  size_t iter;

  if ( sign == FFT_FORWARD ) {
    for (iter = 0; iter < how_many; iter++) 
      kfr_dct_execute_f32(plan, &out[iter * size], &in[iter * size], temp);
  } else {
    for (iter = 0; iter < how_many; iter++)
      kfr_dct_execute_inverse_f32(plan, &out[iter * size], &in[iter * size], temp);
  }
}

void kfr_execute_dct_64(void *plan, double *in, double *out, uint8_t *temp, int sign, size_t how_many, size_t size)
{
  size_t iter;

  if ( sign == FFT_FORWARD ) {
    for (iter = 0; iter < how_many; iter++) {
      kfr_dct_execute_f64(plan, &out[iter * size], &in[iter * size], temp);
    }
  } else {
    for (iter = 0; iter < how_many; iter++)
      kfr_dct_execute_inverse_f64(plan, &out[iter * size], &in[iter * size], temp);
  }
}

void kfr_execute_dct(void *plan, int precision, void *in, void *out, uint8_t *temp, int sign, size_t how_many, size_t size) {
  if ( precision == DTFFT_CONF_DOUBLE ) {
    kfr_execute_dct_64(plan, (double *) in, (double *) out, temp, sign, how_many, size);
  } else {
    kfr_execute_dct_32(plan, (float *) in, (float *) out, temp, sign, how_many, size);
  }
}
