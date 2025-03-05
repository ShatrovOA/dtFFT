module dtfft_interface_vkfft_m
use iso_c_binding, only: c_ptr, c_int, c_int8_t, c_int32_t
use dtfft_parameters, only: dtfft_stream_t
implicit none
private
public :: vkfft_create
public :: vkfft_execute
public :: vkfft_destroy

  interface
    subroutine vkfft_create(rank, dims, precision, how_many, r2c, c2r, dct, dst, stream, app_handle) bind(C)
    !! Creates FFT plan via vkFFT Interface
    import
      integer(c_int8_t),          value :: rank         !< Rank of fft: 1 or 2
      integer(c_int)                    :: dims(*)      !< Dimensions of transform
      integer(c_int32_t),         value :: precision    !< Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
      integer(c_int),             value :: how_many     !< Number of transforms to create
      integer(c_int8_t),          value :: r2c          !< Is R2C transform required
      integer(c_int8_t),          value :: c2r          !< Is C2R transform required
      integer(c_int8_t),          value :: dct          !< Is DCT transform required
      integer(c_int8_t),          value :: dst          !< Is DST transform required
      type(dtfft_stream_t),       value :: stream       !< CUDA stream
      type(c_ptr)                       :: app_handle   !< vkFFT application handle
    end subroutine vkfft_create

    subroutine vkfft_execute(app_handle, in, out, sign) bind(C)
    !! Executes vkFFT plan
    import
      type(c_ptr),        value :: app_handle           !< vkFFT application handle
      type(c_ptr),        value :: in                   !< Input data
      type(c_ptr),        value :: out                  !< Output data
      integer(c_int8_t),  value :: sign                 !< Sign of FFT
    end subroutine vkfft_execute

    subroutine vkfft_destroy(app_handle) bind(C)
    !! Destroys vkFFT plan
    import
      type(c_ptr),    value :: app_handle               !< vkFFT application handle
    end subroutine vkfft_destroy
  end interface
end module dtfft_interface_vkfft_m