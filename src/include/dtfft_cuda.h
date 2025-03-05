#ifdef DTFFT_WITH_CUDA
# ifdef _DEVELOPMENT
#  define DEVICE_PTR
# else
#  define DEVICE_PTR device,
# endif
# define LOC_FUN c_devloc
# define C_ADDR c_devptr
#else
# define DEVICE_PTR
# define LOC_FUN c_loc
# define C_ADDR c_ptr
#endif

#define GPU_CALL(lib, name, func, getErrorString)                                                                                                                               \
  block;                                                                                                                                                                        \
  use iso_fortran_env, only: error_unit;                                                                                                                                        \
  use iso_c_binding, only: c_int32_t;                                                                                                                                           \
  integer(c_int32_t) :: ierr, mpi_err;                                                                                                                                          \
  ierr = func;                                                                                                                                                                  \
  if( ierr /= cudaSuccess ) then;                                                                                                                                               \
      write(error_unit, '(a)') lib//" Function '"//name//"' returned non-zero error code: '"//trim(getErrorString(ierr))//"' at "//__FILE__//":"//trim(int_to_str(__LINE__));   \
      call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_err);                                                                                                                            \
  endif;                                                                                                                                                                        \
  endblock

#define CUFFT_CALL(name, func) GPU_CALL("cuFFT", name, func, cufftGetErrorString)
#define CUDA_CALL(name, func) GPU_CALL("CUDA", name, func, cudaGetErrorString)
#define NVRTC_CALL(name, func) GPU_CALL("nvRTC", name, func, nvrtcGetErrorString)
#define NVSHMEM_CALL(name, func) GPU_CALL("NVSHMEM", name, func, int_to_str)
#define NCCL_CALL(name, func) GPU_CALL("NCCL", name, func, ncclGetErrorString)
