#ifdef __GFORTRAN__
# define STRINGIFY_(X) "X"
#else /* default stringification */
# define STRINGIFY_(X) #X
#endif

#define GPU_CALL(func,getErrorString) \
block; \
use iso_fortran_env, only: error_unit; \
use iso_c_binding, only: c_int32_t; \
integer(c_int32_t) :: ierr, mpi_err, pos; \
character(len=:), allocatable :: fname; \
    ierr = func; \
    allocate (fname, source=adjustl(STRINGIFY_(func))); \
    pos = index(fname, "("); \
    if (ierr /= 0_c_int32_t) then; \
        write(error_unit, '(a,i0)') "'"//fname(1:pos - 1)//"' returned non-zero error code: '"//trim(getErrorString(ierr))//"' at "//__FILE__//":",__LINE__; \
        call MPI_Abort(MPI_COMM_WORLD, ierr, mpi_err); \
    end if; \
    deallocate (fname); \
end block

#define GPU_CALL_NOCHECK(func) \
block; \
use iso_c_binding, only: c_int32_t; \
    integer(c_int32_t) :: ierr; \
    ierr = func; \
end block

#ifdef DEVICE_NO_ERROR_CHECK
# define CUFFT_CALL(func) GPU_CALL_NOCHECK(func)
# define CUDA_CALL(func) GPU_CALL_NOCHECK(func)
# define NVRTC_CALL(func) GPU_CALL_NOCHECK(func)
# define NVSHMEM_CALL(func) GPU_CALL_NOCHECK(func)
# define NCCL_CALL(func) GPU_CALL_NOCHECK(func)
#else
# define CUFFT_CALL(func) GPU_CALL(func, cufftGetErrorString)
# define CUDA_CALL(func) GPU_CALL(func, cudaGetErrorString)
# define NVRTC_CALL(func) GPU_CALL(func, nvrtcGetErrorString)
# define NVSHMEM_CALL(func) GPU_CALL(func, to_str)
# define NCCL_CALL(func) GPU_CALL(func, ncclGetErrorString)
#endif
