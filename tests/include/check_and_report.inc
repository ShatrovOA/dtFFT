#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif
#ifndef FUNC_NAME
#error "FUNC_NAME not defined"
#endif
#ifndef FUNC_NAME_STR
#error "FUNC_NAME_STR not defined"
#endif
#ifndef CHECK_FUNC
#error "CHECK_FUNC not defined"
#endif
#ifndef ERROR_PRECISION
#error "ERROR_PRECISION not defined"
#endif

subroutine FUNC_NAME (n_global, tf, tb, buffer, buf_size, check  &
#if defined(DTFFT_WITH_CUDA)
                      , platform                                 &
#endif
) bind(C, name=FUNC_NAME_STR)
  integer(c_size_t), value, intent(in)  :: n_global
  real(c_double),    value, intent(in)  :: tf
  real(c_double),    value, intent(in)  :: tb
  type(c_ptr),       value, intent(in)  :: buffer
  integer(c_size_t), value, intent(in)  :: buf_size
  type(c_ptr),       value, intent(in)  :: check
#if defined(DTFFT_WITH_CUDA)
  integer(c_int32_t),value, intent(in)  :: platform
  type(c_ptr) :: buffer_h
#endif
  real(ERROR_PRECISION) :: local_error

#if defined(DTFFT_WITH_CUDA)
  if ( dtfft_platform_t(platform) == DTFFT_PLATFORM_CUDA ) then
    call mem_alloc_host(STORAGE_BYTES * buf_size, buffer_h)

    CUDA_CALL( "cudaMemcpy", cudaMemcpy(buffer_h, buffer, STORAGE_BYTES * buf_size, cudaMemcpyDeviceToHost) )
    local_error = CHECK_FUNC(check, buffer_h, buf_size)
    call mem_free_host(buffer_h)
  else
    local_error = CHECK_FUNC(check, buffer, buf_size)
  endif
#else
  local_error = CHECK_FUNC(check, buffer, buf_size)
#endif

  call report(tf, tb, local_error, n_global)
end subroutine FUNC_NAME

#undef STORAGE_BYTES
#undef FUNC_NAME
#undef FUNC_NAME_STR
#undef CHECK_FUNC
#undef ERROR_PRECISION