module dtfft_interface_cufft_m
use iso_c_binding, only: c_int, c_ptr
use cudafor, only: c_devptr, cuda_stream_kind
implicit none
private
public :: cufftPlanMany, cufftXtExec, cufftDestroy, cufftSetStream

  interface
    function cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, ffttype, batch)   &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftPlanMany")
    !! Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
    import
      type(c_ptr)                       :: plan         !< Pointer to an uninitialized cufftHandle object.
      integer(c_int),             value :: rank         !< Dimensionality of the transform (1, 2, or 3).
      integer(c_int)                    :: n(*)         !< Array of size rank, describing the size of each dimension,
                                                        !< n[0] being the size of the outermost 
                                                        !< and n[rank-1] innermost (contiguous) dimension of a transform.
      integer(c_int)                    :: inembed(*)   !< Pointer of size rank that indicates the storage dimensions of the input data in memory. 
                                                        !< If set to NULL all other advanced data layout parameters are ignored.
      integer(c_int),             value :: istride      !< Indicates the distance between two successive input elements in the least 
                                                        !< significant (i.e., innermost) dimension.
      integer(c_int),             value :: idist        !< Indicates the distance between the first element of two consecutive signals 
                                                        !< in a batch of the input data.
      integer(c_int)                    :: onembed(*)   !< Pointer of size rank that indicates the storage dimensions of the output data in memory. 
                                                        !< If set to NULL all other advanced data layout parameters are ignored.
      integer(c_int),             value :: ostride      !< Indicates the distance between two successive output elements in the output array 
                                                        !< in the least significant (i.e., innermost) dimension.
      integer(c_int),             value :: odist        !< Indicates the distance between the first element of two consecutive signals 
                                                        !< in a batch of the output data.
      integer(c_int),             value :: ffttype      !< The transform data type (e.g., CUFFT_R2C for single precision real to complex).
      integer(c_int),             value :: batch        !< Batch size for this transform.
      integer(c_int)                    :: cufftResult  !< The enumerated type cufftResult defines API call result codes.
    end function cufftPlanMany

    function cufftXtExec(plan, input, output, direction)                                                      &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftXtExec")
    !! Executes any cuFFT transform regardless of precision and type.
    !! In case of complex-to-real and real-to-complex transforms direction parameter is ignored.
    import
      type(c_ptr),                value :: plan         !< cufftHandle returned by cufftCreate
      type(c_devptr),             value :: input        !< Pointer to the input data (in GPU memory) to transform.
      type(c_devptr),             value :: output       !< Pointer to the output data (in GPU memory).
      integer(c_int),             value :: direction    !< The transform direction: CUFFT_FORWARD or CUFFT_INVERSE. 
                                                        !< Ignored for complex-to-real and real-to-complex transforms.
      integer(c_int)                    :: cufftResult  !< The enumerated type cufftResult defines API call result codes.
    end function cufftXtExec

    function cufftDestroy(plan)                                                                               &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftDestroy")
    !! Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
    import
      type(c_ptr),                value :: plan         !< Object of the plan to be destroyed.
      integer(c_int)                    :: cufftResult  !< The enumerated type cufftResult defines API call result codes.
    end function cufftDestroy

    function cufftSetStream(plan, stream)                                                                     &
      result(cufftResult)                                                                                     &
      bind(C, name="cufftSetStream")
    !! Associates a CUDA stream with a cuFFT plan.
    import
      type(c_ptr),                value :: plan         !< Object to associate with the stream.
      integer(cuda_stream_kind),  value :: stream       !< A valid CUDA stream
      integer(c_int)                    :: cufftResult  !< The enumerated type cufftResult defines API call result codes.
    end function cufftSetStream
  end interface
end module dtfft_interface_cufft_m