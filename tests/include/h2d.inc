#ifndef FUNC_NAME
#error "FUNC_NAME not defined"
#endif
#ifndef FUNC_NAME_STR
#error "FUNC_NAME_STR not defined"
#endif
#ifndef DTYPE
#error "DTYPE not defined"
#endif
#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif

#if defined(DTFFT_WITH_CUDA)
  subroutine FUNC_NAME(source, target, count, platform) bind(C, name=FUNC_NAME_STR)
    integer(c_int32_t), intent(in), value :: platform
#else
  subroutine FUNC_NAME(source, target, count) bind(C, name=FUNC_NAME_STR)
#endif
    type(c_ptr),        intent(in), value :: source
    type(c_ptr),        intent(in), value :: target
    integer(c_size_t),  intent(in), value :: count
    DTYPE, pointer, contiguous :: source_(:), target_(:)
    integer(int64) :: i

    call c_f_pointer(source, source_, [count])
    call c_f_pointer(target, target_, [count])
#if defined(DTFFT_WITH_CUDA)
  if ( dtfft_platform_t(platform) == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaMemcpy", cudaMemcpy(target, source, count * STORAGE_BYTES, cudaMemcpyHostToDevice) )
  else
    do i = 1, count
      target_(i) = source_(i)
    enddo
  endif
#else
    do i = 1, count
      target_(i) = source_(i)
    enddo
#endif
  end subroutine FUNC_NAME

#undef FUNC_NAME
#undef FUNC_NAME_STR
#undef DTYPE
#undef STORAGE_BYTES
