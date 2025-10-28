#ifndef FUNC_NAME
#error "FUNC_NAME not defined"
#endif
#ifndef FUNC_NAME_STR
#error "FUNC_NAME_STR not defined"
#endif

subroutine FUNC_NAME (executor, buffer, buf_size, scale_value      &
#if defined(DTFFT_WITH_CUDA)
                     , platform, stream                           &
#endif
) bind(C, name=FUNC_NAME_STR)
  integer(c_int32_t),   value :: executor
  type(c_ptr),          value :: buffer
  integer(c_size_t),    value :: buf_size
  integer(c_size_t),    value :: scale_value
#if defined(DTFFT_WITH_CUDA)
  integer(c_int32_t),   value :: platform
  type(dtfft_stream_t), value :: stream
#endif

  if ( dtfft_executor_t(executor) == DTFFT_EXECUTOR_NONE ) return

#if defined(DTFFT_WITH_CUDA)
  if ( dtfft_platform_t(platform) == DTFFT_PLATFORM_CUDA ) then
    call CONCAT(FUNC_NAME,CUDA)(buffer, buf_size, scale_value, stream)
    CUDA_CALL( cudaDeviceSynchronize() )
  else
    call CONCAT(FUNC_NAME,Host)(buffer, buf_size, scale_value)
  endif
#else
  call CONCAT(FUNC_NAME,Host)(buffer, buf_size, scale_value)
#endif
end subroutine FUNC_NAME

#undef FUNC_NAME
#undef FUNC_NAME_STR