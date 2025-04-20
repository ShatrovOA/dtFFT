/*
  Copyright (c) 2021, Oleg Shatrov
  All rights reserved.
  This file is part of dtFFT library.

  dtFFT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  dtFFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/**
 * @file dtfft.hpp
 * @author Oleg Shatrov
 * @date 2024
 * @brief File containing C++ API functions of dtFFT Library
 */

#ifndef DTFFT_HPP
#define DTFFT_HPP

#include <dtfft.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>


#define DTFFT_THROW_EXCEPTION(msg)                            \
  do {                                                        \
    throw dtfft::Exception(msg, __FILE__, __LINE__);          \
  } while (false);


/** Safe call macro.
 *
 * Should be used to check error codes returned by ``dtFFT``.
 *
 * @details Throws an exception with a message explaining the error if one occurs.
 *
 * **Example**
 * @code
 * DTFFT_CXX_CALL( plan.execute(a, b, dtfft::Execute::FORWARD) )
 * @endcode
 */
#define DTFFT_CXX_CALL( call )                                \
  do {                                                        \
    dtfft::Error ierr = call;                                 \
    if ( ierr != dtfft::Error::SUCCESS ) {                    \
      DTFFT_THROW_EXCEPTION( dtfft::get_error_string(ierr) )  \
    }                                                         \
  } while(false);


namespace dtfft
{
/** dtFFT version information */
class Version {
  public:
    /** dtFFT Major Version */
    static constexpr int32_t MAJOR = DTFFT_VERSION_MAJOR;

    /** dtFFT Minor Version */
    static constexpr int32_t MINOR = DTFFT_VERSION_MINOR;

    /** dtFFT Patch Version */
    static constexpr int32_t PATCH = DTFFT_VERSION_PATCH;

    /** dtFFT Version Code. Can be used for version comparison */
    static constexpr int32_t CODE = DTFFT_VERSION_CODE;

    /** @return Version Code defined during compilation */
    static int32_t get() noexcept { return dtfft_get_version(); }

    /** @return Version Code based on input parameters */
    static constexpr int32_t get(int32_t major, int32_t minor, int32_t patch) noexcept { return DTFFT_VERSION(major, minor, patch); }
  };

/** This enum lists the different error codes that ``dtFFT`` can return. */
  enum class Error {

/** Successful execution */
  SUCCESS = DTFFT_SUCCESS,
/** MPI_Init is not called or MPI_Finalize has already been called */
  MPI_FINALIZED = DTFFT_ERROR_MPI_FINALIZED,
/** Plan not created */
  PLAN_NOT_CREATED = DTFFT_ERROR_PLAN_NOT_CREATED,
/** Invalid `transpose_type` provided */
  INVALID_TRANSPOSE_TYPE = DTFFT_ERROR_INVALID_TRANSPOSE_TYPE,
/** Invalid Number of dimensions provided. Valid options are 2 and 3 */
  INVALID_N_DIMENSIONS = DTFFT_ERROR_INVALID_N_DIMENSIONS,
/** One or more provided dimension sizes <= 0 */
  INVALID_DIMENSION_SIZE = DTFFT_ERROR_INVALID_DIMENSION_SIZE,
/** Invalid communicator type provided */
  INVALID_COMM_TYPE = DTFFT_ERROR_INVALID_COMM_TYPE,
/** Invalid `precision` parameter provided */
  INVALID_PRECISION = DTFFT_ERROR_INVALID_PRECISION,
/** Invalid `effort` parameter provided */
  INVALID_EFFORT = DTFFT_ERROR_INVALID_EFFORT,
/** Invalid `executor` parameter provided */
  INVALID_EXECUTOR = DTFFT_ERROR_INVALID_EXECUTOR,
/** Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine */
  INVALID_COMM_DIMS = DTFFT_ERROR_INVALID_COMM_DIMS,
/** Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1 */
  INVALID_COMM_FAST_DIM = DTFFT_ERROR_INVALID_COMM_FAST_DIM,
/** For R2R plan, `kinds` parameter must be passed if `executor` != `Executor::NONE` */
  MISSING_R2R_KINDS = DTFFT_ERROR_MISSING_R2R_KINDS,
/** Invalid values detected in `kinds` parameter */
  INVALID_R2R_KINDS = DTFFT_ERROR_INVALID_R2R_KINDS,
/** Transpose plan is not supported in R2C, use R2R or C2C plan instead */
  R2C_TRANSPOSE_PLAN = DTFFT_ERROR_R2C_TRANSPOSE_PLAN,
/** Inplace transpose is not supported */
  INPLACE_TRANSPOSE = DTFFT_ERROR_INPLACE_TRANSPOSE,
/** Invalid `aux` buffer provided */
  INVALID_AUX = DTFFT_ERROR_INVALID_AUX,
/** Invalid `dim` passed to `Plan.get_pencil` */
  INVALID_DIM = DTFFT_ERROR_INVALID_DIM,
/** Invalid API Usage. */
  INVALID_USAGE = DTFFT_ERROR_INVALID_USAGE,
/** Trying to create already created plan */
  PLAN_IS_CREATED = DTFFT_ERROR_PLAN_IS_CREATED,
/** Selected `executor` does not support R2R FFTs */
  R2R_FFT_NOT_SUPPORTED = DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED,
/** Internal call of Plan.mem_alloc failed */
  ALLOC_FAILED = DTFFT_ERROR_ALLOC_FAILED,
/** Failed to dynamically load library */
  DLOPEN_FAILED = DTFFT_ERROR_DLOPEN_FAILED,
/** Failed to dynamically load symbol */
  DLSYM_FAILED = DTFFT_ERROR_DLSYM_FAILED,
/** Internal call of Plan.mem_free failed */
  FREE_FAILED = DTFFT_ERROR_FREE_FAILED,
/** Invalid `alloc_bytes` provided */
  INVALID_ALLOC_BYTES = DTFFT_ERROR_INVALID_ALLOC_BYTES,
/** Invalid stream provided */
  GPU_INVALID_STREAM = DTFFT_ERROR_GPU_INVALID_STREAM,
/** Invalid GPU backend provided */
  GPU_INVALID_BACKEND = DTFFT_ERROR_GPU_INVALID_BACKEND,
/** Multiple MPI Processes located on same host share same GPU which is not supported */
  GPU_NOT_SET = DTFFT_ERROR_GPU_NOT_SET,
/** When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions */
  VKFFT_R2R_2D_PLAN = DTFFT_ERROR_VKFFT_R2R_2D_PLAN,
/** Passed `effort` ==  `Effort::PATIENT` but all GPU backends have been disabled by `Config` */
  GPU_BACKENDS_DISABLED = DTFFT_ERROR_GPU_BACKENDS_DISABLED,
/** One of pointers passed to `Plan.execute` or `Plan.transpose` cannot be accessed from device */
  NOT_DEVICE_PTR = DTFFT_ERROR_NOT_DEVICE_PTR,
/** One of pointers passed to `Plan.execute` or `Plan.transpose` is not and `NVSHMEM` pointer */
  NOT_NVSHMEM_PTR = DTFFT_ERROR_NOT_NVSHMEM_PTR,
/** Invalid platform provided */
  INVALID_PLATFORM = DTFFT_ERROR_INVALID_PLATFORM,
/** Invalid executor provided for selected platform */
  INVALID_PLATFORM_EXECUTOR_TYPE = DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE
  };

/** @brief Returns the string description of an error code
 *
 * @param[in] error_code Error code to convert to string
 * @return String representation of `error_code`
 */
  inline std::string get_error_string(Error error_code) noexcept {
    const char* error_str = dtfft_get_error_string(static_cast<dtfft_error_t>(error_code));
    return std::string(error_str);
  }

/** This enum lists valid `execute_type` parameters that can be passed to Plan.execute. */
  enum class Execute {
/** Perform XYZ --> YXZ --> ZXY plan execution (Forward) */
    FORWARD = DTFFT_EXECUTE_FORWARD,

/** Perform ZXY --> YXZ --> XYZ plan execution (Backward) */
    BACKWARD = DTFFT_EXECUTE_BACKWARD
  };

/** This enum lists valid `transpose_type` parameters that can be passed to Plan.transpose */
  enum class Transpose {
/** Transpose from Fortran X aligned to Fortran Y aligned */
    X_TO_Y = DTFFT_TRANSPOSE_X_TO_Y,

/** Transpose from Fortran Y aligned to Fortran X aligned */
    Y_TO_X = DTFFT_TRANSPOSE_Y_TO_X,

/** Transpose from Fortran Y aligned to Fortran Z aligned */
    Y_TO_Z = DTFFT_TRANSPOSE_Y_TO_Z,

/** Transpose from Fortran Z aligned to Fortran Y aligned */
    Z_TO_Y = DTFFT_TRANSPOSE_Z_TO_Y,

/** Transpose from Fortran X aligned to Fortran Z aligned
 * @note This value is valid to pass only in 3D Plan and value returned by Plan.get_z_slab_enabled must be `true`
 */
    X_TO_Z = DTFFT_TRANSPOSE_X_TO_Z,

/** Transpose from Fortran Z aligned to Fortran X aligned
 * @note This value is valid to pass only in 3D Plan and value returned by Plan.get_z_slab_enabled must be `true`
 */
    Z_TO_X = DTFFT_TRANSPOSE_Z_TO_X
  };

/** This enum lists valid `precision` parameters that can be passed to Plan constructors. */
  enum class Precision {
/** Use Single precision */
    SINGLE = DTFFT_SINGLE,

/** Use Double precision */
    DOUBLE = DTFFT_DOUBLE
  };

/** This enum lists valid `effort` parameters that can be passed to Plan constructors. */
  enum class Effort {
/** Create plan as fast as possible */
    ESTIMATE = DTFFT_ESTIMATE,

/** Will attempt to find best MPI Grid decomposition.
 * Passing this flag and MPI Communicator with cartesian topology to any Plan Constructor is same as Effort::ESTIMATE.
 */
    MEASURE = DTFFT_MEASURE,

/** Same as Effort::MEASURE plus cycle through various send and receive MPI_Datatypes.
 * For GPU Build this flag will run autotune procedure to find best backend
 */
    PATIENT = DTFFT_PATIENT
  };

/** This enum lists available FFT executors. */
  enum class Executor {
/** Do not create any FFT plans. Creates transpose only plan. */
    NONE = DTFFT_EXECUTOR_NONE,

/** FFTW3 Executor (Host only) */
    FFTW3 = DTFFT_EXECUTOR_FFTW3,

/** MKL DFTI Executor (Host only) */
    MKL = DTFFT_EXECUTOR_MKL,

/** CUFFT Executor (GPU Only) */
    CUFFT = DTFFT_EXECUTOR_CUFFT,

/** VkFFT Executor (GPU Only) */
    VKFFT = DTFFT_EXECUTOR_VKFFT
  };

/** Real-to-Real FFT kinds available in dtFFT */
  enum class R2RKind {
/** DCT-I (Logical N=2*(n-1), inverse is R2RKind::DCT_1) */
    DCT_1 = DTFFT_DCT_1,

/** DCT-II (Logical N=2*n, inverse is R2RKind::DCT_3) */
    DCT_2 = DTFFT_DCT_2,

/** DCT-III (Logical N=2*n, inverse is R2RKind::DCT_2) */
    DCT_3 = DTFFT_DCT_3,

/** DCT-IV (Logical N=2*n, inverse is R2RKind::DCT_4) */
    DCT_4 = DTFFT_DCT_4,

/** DST-I (Logical N=2*(n+1), inverse is R2RKind::DST_1) */
    DST_1 = DTFFT_DST_1,

/** DST-II (Logical N=2*n, inverse is R2RKind::DST_3) */
    DST_2 = DTFFT_DST_2,

/** DST-III (Logical N=2*n, inverse is R2RKind::DST_2) */
    DST_3 = DTFFT_DST_3,

/** DST-IV (Logical N=2*n, inverse is R2RKind::DST_4) */
    DST_4 = DTFFT_DST_4
  };

#ifdef DTFFT_WITH_CUDA

/** Various Backends available in dtFFT
 *
 * @note This enum is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
  enum class Backend {

/** Backend that uses MPI datatypes
 * @details Not really recommended to use, since it is a million times slower than other backends.
 * It is present here just to show how slow MPI Datatypes are for GPU usage
 */
    MPI_DATATYPE = DTFFT_BACKEND_MPI_DATATYPE,

/** MPI peer-to-peer algorithm */
    MPI_P2P = DTFFT_BACKEND_MPI_P2P,

/** MPI peer-to-peer algorithm with overlapping data copying and unpacking */
    MPI_P2P_PIPELINED = DTFFT_BACKEND_MPI_P2P_PIPELINED,

/** MPI backend using MPI_Alltoallv */
    MPI_A2A = DTFFT_BACKEND_MPI_A2A,

/** NCCL backend */
    NCCL = DTFFT_BACKEND_NCCL,

/** NCCL backend with overlapping data copying and unpacking */
    NCCL_PIPELINED = DTFFT_BACKEND_NCCL_PIPELINED,

/** cuFFTMp backend */
    CUFFTMP = DTFFT_BACKEND_CUFFTMP
  };

/**
 * @brief Returns string with name of backend provided as argument.
 *
 * @param[in]       backend   Backend to represent
 *
 * @return String representation of `backend`.
 * @note This function is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
  std::string get_backend_string(const Backend backend) {
    const char *gpu_str = dtfft_get_backend_string( static_cast<dtfft_backend_t>(backend));
    return std::string(gpu_str);
  }

/** Enum that specifies runtime platform, e.g. Host, CUDA, HIP */
  enum class Platform {
/** Host */
    HOST = DTFFT_PLATFORM_HOST,
/** CUDA */
    CUDA = DTFFT_PLATFORM_CUDA,
  };
#endif

/** Basic exception class */
  class Exception : public std::exception {
  private:
    std::string s;

  public:
    /**
     * @brief Basic exception constructor
     * @param msg   Message describing the error that occurred
     * @param file  Filename
     * @param line  Line number
     */
    Exception(std::string msg, const char* file, int line) {
      s = "dtFFT Exception: '" + std::move(msg) + "' at " + file + ":" + std::to_string(line);
    }

    /** Exception explanation */
    const char* what() const noexcept override { return s.c_str(); }
  };

  /** Class to handle Pencils
   * @see Plan.get_pencil()
  */
  class Pencil {
    private:
      bool is_created;

      dtfft_pencil_t pencil;

      /** Local starts in natural Fortran order */
      std::vector<int32_t> starts;

      /** Local counts in natural Fortran order */
      std::vector<int32_t> counts;

      /** Creates vector from underlying struct pointer */
      inline std::vector<int32_t> create_vector(const int32_t values[], uint8_t size) const noexcept {
        std::vector<int32_t> vec;
        vec.reserve(size);
        std::copy(values, values + size, std::back_inserter(vec));
        return vec;
      }

    public:
      /** Default constructor */
      Pencil() : is_created(false) {}

      /** Constructor used internally by Plan::get_pencil */
      explicit Pencil(dtfft_pencil_t& c_pencil) :
        is_created(true), pencil(c_pencil),
        starts(create_vector(c_pencil.starts, c_pencil.ndims)),
        counts(create_vector(c_pencil.counts, c_pencil.ndims)) {}

    /** @return Number of dimensions in a pencil */
      inline uint8_t get_ndims() const {
        if ( !is_created ) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return pencil.ndims;
      }

    /** @return Aligned dimension id starting from 1 */
      inline uint8_t get_dim() const {
        if ( !is_created ) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return pencil.dim;
      }

    /** @return Local starts in natural Fortran order */
      inline const std::vector<int32_t>& get_starts() const {
        if ( !is_created ) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return starts;
      }

    /** @return Local counts in natural Fortran order */
      inline const std::vector<int32_t>& get_counts() const {
        if ( !is_created ) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return counts;
      }

      /** @return Total number of elements in a pencil */
      inline size_t get_size() const {
        if (!is_created) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return pencil.size;
      }

    /** @return Underlying C structure */
      inline const dtfft_pencil_t& c_struct() const {
        if ( !is_created ) {
          DTFFT_THROW_EXCEPTION(get_error_string(Error::INVALID_USAGE));
        }
        return pencil;
      }
    };

/** Class to set additional configuration parameters to dtFFT
 * @see set_config()
*/
  class Config {
    protected:
    /** Underlying C structure */
      dtfft_config_t config;

    public:
      /** Creates and sets default configuration values */
      Config() {
        DTFFT_CXX_CALL( static_cast<Error>(dtfft_create_config(&config)) );
      }

/**
 * @brief Sets whether dtFFT use Z-slab optimization or not.
 *
 * @details Default is `true`
 *
 * One should consider disabling Z-slab optimization in order to resolve `Error::VKFFT_R2R_2D_PLAN` error
 * OR when underlying FFT implementation of 2D plan is too slow.
 *
 * In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
 */
      inline void set_enable_z_slab(bool enable_z_slab) noexcept { config.enable_z_slab = enable_z_slab; }

#ifdef DTFFT_WITH_CUDA
/**
 * @brief Sets platform to execute plan.
 *
 * Default is Platform::HOST
 *
 * @details This option is only defined with device support build.
 * Even when dtFFT is build with device support it does not necessary means that all plans must be related to device.
 */
      inline void set_platform(Platform platform) noexcept {config.platform = static_cast<dtfft_platform_t>(platform);}
/**
 * @brief Sets Main CUDA stream that will be used in dtFFT.
 *
 * @details This parameter is a placeholder for user to set custom stream.
 * Stream that is actually used by dtFFT plan is returned by Plan.get_stream function.
 * When user sets stream he is responsible of destroying it.
 *
 * Stream must not be destroyed before call to destroy.
 *
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_stream(dtfft_stream_t stream) noexcept {config.stream = stream; }

/**
 * @brief Sets GPU Backend that will be used by dtFFT when `effort` is Effort::ESTIMATE or Effort::MEASURE.
 *
 * @details Default is Backend::NCCL
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_backend(Backend backend) noexcept {config.backend = static_cast<dtfft_backend_t>(backend); }

/**
 * @brief Sets whether MPI GPU Backends be enabled when `effort` is `DTFFT_PATIENT` or not.
 *
 * @details Default is `false`
 *
 * MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
 * It was noticed that during plan autotuning GPU memory not being freed completely.
 *
 * For example:
 * 1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
 * Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.
 *
 * One of the workarounds is to disable MPI Backends by default, which is done here.
 *
 * Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
 * but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_enable_mpi_backends(bool enable_mpi_backends) noexcept {config.enable_mpi_backends = enable_mpi_backends; }

/**
 * @brief Sets whether pipelined GPU backends be enabled when `effort` is Effort::PATIENT or not.
 *
 * @details Default is `true`
 *
 * @note Pipelined backends require additional buffer that user has no control over.
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_enable_pipelined_backends(bool enable_pipelined_backends) noexcept {config.enable_pipelined_backends = enable_pipelined_backends; }

/**
 * @brief Sets whether NCCL Backends be enabled when `effort` is Effort::PATIENT or not.
 * @details Default is `true`.
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_enable_nccl_backends(bool enable_nccl_backends) noexcept {config.enable_nccl_backends = enable_nccl_backends; }

/**
 * @brief Should NVSHMEM Backends be enabled when `effort` is Effort::PATIENT or not.
 * @details Default is `true`.
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline void set_enable_nvshmem_backends(bool enable_nvshmem_backends) noexcept {config.enable_nvshmem_backends = enable_nvshmem_backends; }
#endif

/** @return Underlying C structure */
    dtfft_config_t c_struct() const {
      return config;
    }
  };

/** @brief Sets configuration values to dtFFT. Must be called before plan creation to take effect.
 *
 * @return Error::SUCCESS if the call was successful, error code otherwise
 * @see Config
 */
  inline Error set_config(Config config) noexcept {
    return static_cast<Error>(dtfft_set_config(config.c_struct()));
  }

  /** Abstract plan for all dtFFT plans.
   * @details This class does not have any constructors. To create a plan user should use one of the inherited classes.
  */
  class Plan
  {
    protected:
    /** Underlying C structure */
      dtfft_plan_t _plan;

    public:

/** @brief Checks if plan is using Z-slab optimization.
 * If `true` then flags Transpose::X_TO_Z and Transpose::Z_TO_X will be valid to pass to Plan.transpose method.
 *
 * @param[out]     is_z_slab_enabled       Boolean value if Z-slab is used.
 *
 * @return Error::SUCCESS if call was without error, error code otherwise
 */
      inline
      Error
      get_z_slab_enabled(bool *is_z_slab_enabled) const noexcept
      {
        return static_cast<Error>(dtfft_get_z_slab_enabled(_plan, is_z_slab_enabled));
      }

/**
 * @brief Prints plan-related information to stdout
 *
 * @return Error::SUCCESS if call was without error, error code otherwise
 */
      inline
      Error
      report() const noexcept { return static_cast<Error>(dtfft_report(_plan)); }

/**
 * @brief Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation,
 * that is unavailable in dtFFT.
 *
 * @param[in]     dim             Required dimension:
 *                                  - 0 for XYZ layout (real space, R2C only)
 *                                  - 1 for XYZ layout
 *                                  - 2 for YXZ layout
 *                                  - 3 for ZXY layout
 * @param[out]    pencil          Pencil class
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_pencil(const int8_t dim, Pencil& pencil) const noexcept
      {
        dtfft_pencil_t c_pencil;
        const Error error_code = static_cast<Error>(dtfft_get_pencil(_plan, dim, &c_pencil));
        if ( error_code == Error::SUCCESS ) {
          pencil = Pencil(c_pencil);
        }
        return error_code;
      }

/** @brief Plan execution
 *
 * @param[inout]   in                   Incoming pointer
 * @param[out]     out                  Result pointer
 * @param[in]      execute_type         Type of execution
 * @param[inout]   aux                  Optional auxiliary pointer
 *
 * @return Error::SUCCESS on success or error code on failure.
 * @note
 */
      inline
      Error
      execute(void *in, void *out, const Execute execute_type, void *aux=nullptr) const noexcept
      {
        dtfft_error_t error_code = dtfft_execute(_plan, in, out, static_cast<dtfft_execute_t>(execute_type), aux);
        return static_cast<Error>(error_code);
      }


/** @brief Transpose data in single dimension, e.g. X align -> Y align
 * \attention `in` and `out` cannot be the same pointers
 *
 * @param[inout]   in                    Incoming pointer
 * @param[out]     out                   Transposed pointer
 * @param[in]      transpose_type        Type of transpose to perform.
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      transpose(void *in, void *out, const Transpose transpose_type) const noexcept
      {
        dtfft_error_t error_code = dtfft_transpose(_plan, in, out, static_cast<dtfft_transpose_t>(transpose_type));
        return static_cast<Error>(error_code);
      }


/** @brief Wrapper around `Plan.get_local_sizes` to obtain `alloc_size` only
 *
 * @param[out]     alloc_size       Minimum number of elements to be allocated for `in`, `out` or `aux` buffers.
 *                                    Size of each element in bytes can be obtained by calling `Plan.get_element_size`.
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_alloc_size(size_t *alloc_size) const noexcept
      {return static_cast<Error>(dtfft_get_alloc_size(_plan, alloc_size));}


/** @brief Get grid decomposition information. Results may differ on different MPI processes
 *
 * @param[out]   in_starts              Starts of local portion of data in 'real' space in reversed order
 * @param[out]   in_counts              Sizes  of local portion of data in 'real' space in reversed order
 * @param[out]   out_starts             Starts of local portion of data in 'fourier' space in reversed order
 * @param[out]   out_counts             Sizes  of local portion of data in 'fourier' space in reversed order
 * @param[out]   alloc_size             Minimum number of elements to be allocated for `in`, `out` or `aux` buffers.
 *                                      Size of each element in bytes can be obtained by calling `Plan.get_element_size`.
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_local_sizes(
        std::vector<int32_t>&in_starts,
        std::vector<int32_t>&in_counts,
        std::vector<int32_t>&out_starts,
        std::vector<int32_t>&out_counts,
        size_t *alloc_size) const noexcept
      {return get_local_sizes(in_starts.data(), in_counts.data(), out_starts.data(), out_counts.data(), alloc_size);}


/** @brief Get grid decomposition information. Results may differ on different MPI processes
 *
 * @param[out]   in_starts              Starts of local portion of data in 'real' space in reversed order
 * @param[out]   in_counts              Sizes  of local portion of data in 'real' space in reversed order
 * @param[out]   out_starts             Starts of local portion of data in 'fourier' space in reversed order
 * @param[out]   out_counts             Sizes  of local portion of data in 'fourier' space in reversed order
 * @param[out]   alloc_size             Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers.
 *                                      Size of each element in bytes can be obtained by calling `Plan.get_element_size`.
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_local_sizes(
        int32_t *in_starts=nullptr,
        int32_t *in_counts=nullptr,
        int32_t *out_starts=nullptr,
        int32_t *out_counts=nullptr,
        size_t *alloc_size=nullptr) const noexcept
      {return static_cast<Error>(dtfft_get_local_sizes(_plan, in_starts, in_counts, out_starts, out_counts, alloc_size));}


/**
 * @brief Obtains number of bytes required to store single element by this plan.
 *
 * @param[out]    element_size    Size of element in bytes
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_element_size(size_t *element_size) const noexcept
      {return static_cast<Error>(dtfft_get_element_size(_plan, element_size));}


/**
 * @brief Returns minimum number of bytes required to execute plan.
 *
 * This function is a combination of two calls: Plan.get_alloc_size and Plan.get_element_size
 *
 * @param[out]    alloc_bytes    Number of bytes required
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      get_alloc_bytes(size_t *alloc_bytes) const noexcept
      {return static_cast<Error>(dtfft_get_alloc_bytes(_plan, alloc_bytes));}

/**
 * @brief Allocates memory specific for this plan
 *
 * @param[in]     alloc_bytes     Number of bytes to allocate
 * @param[out]    ptr             Allocated pointer
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      mem_alloc(size_t alloc_bytes, void **ptr) const noexcept
      {return static_cast<Error>(dtfft_mem_alloc(_plan, alloc_bytes, ptr));}


/**
 * @brief Frees memory specific for this plan
 *
 * @param[inout]  ptr             Allocated pointer
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error
      mem_free(void *ptr) const noexcept
      {return static_cast<Error>(dtfft_mem_free(_plan, ptr));}


/** @brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
 *
 * @return Error::SUCCESS on success or error code on failure.
 */
      inline
      Error destroy() noexcept
      {
        return static_cast<Error>(dtfft_destroy(&_plan));
      }


#ifdef DTFFT_WITH_CUDA
/**
 * @brief Returns stream associated with current Plan.
 * This can either be stream passed by Config.set_stream followed by set_config() or stream created internally.
 * Returns NULL pointer if plan's platform is Platform::HOST.
 *
 * @param[out]     stream         CUDA stream associated with plan
 *
 * @return Error::SUCCESS on success or error code on failure.
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline
      Error
      get_stream(dtfft_stream_t *stream) const noexcept
      {return static_cast<Error>(dtfft_get_stream(_plan, stream));}

/**
 * @brief Returns selected GPU backend during autotune if `effort` is Effort::PATIENT.
 *
 * If `effort` passed to any create function is Effort::ESTIMATE or Effort::MEASURE
 * returns value set by Config.set_backend followed by set_config() or default value, which is Backend::NCCL.
 *
 * @return Error::SUCCESS on success or error code on failure.
 * @note This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.
 */
      inline
      Error
      get_backend(Backend *backend) const noexcept
      {
        dtfft_backend_t backend_;
        dtfft_error_t error_code = dtfft_get_backend(_plan, &backend_);
        *backend = static_cast<Backend>(backend_);
        return static_cast<Error>(error_code);
      }

/**
 * @brief Returns plan execution platform.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
      inline
      Error
      get_platform(Platform *platform) const noexcept
      {
        dtfft_platform_t platform_;
        dtfft_error_t error_code = dtfft_get_platform(_plan, &platform_);
        *platform = static_cast<Platform>(platform_);
        return static_cast<Error>(error_code);
      }
#endif

  /** @return Underlying C structure */
  dtfft_plan_t c_struct() const {
    return _plan;
  }

/** Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize */
    virtual ~Plan() noexcept = 0;
  };

  inline Plan::~Plan() noexcept { destroy(); }

/** Complex-to-Complex Plan */
  class PlanC2C final: public Plan
  {
    public:
/** @brief Complex-to-Complex Plan constructor.
 *
 * @param[in]    dims                   Vector with global dimensions in reversed order.
 *                                        `dims.size()` must be 2 or 3
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]    executor               Type of external FFT executor
 *
 * @throws Exception In case error occurs during plan creation
 */
      explicit PlanC2C(
        const std::vector<int32_t> &dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE,
        const Executor executor=Executor::NONE
      ):PlanC2C(dims.size(), dims.data(), comm, precision, effort, executor) {}


/** @brief Complex-to-Complex Transpose-only Plan constructor.
 *
 * @param[in]    dims                   Vector with global dimensions in reversed order.
 *                                        `dims.size()` must be 2 or 3
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 *
 * @throws Exception In case error occurs during plan creation
 */
      explicit PlanC2C(
        const std::vector<int32_t> &dims,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE
      ): PlanC2C(dims.size(), dims.data(), MPI_COMM_WORLD, precision, effort) {}


/** @brief Complex-to-Complex Generic Plan constructor.
 *
 * @param[in]    ndims                  Number of dimensions: 2 or 3
 * @param[in]    dims                   Buffer of size `ndims` with global dimensions in reversed order.
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]    executor               Type of external FFT executor
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanC2C(
        const int8_t ndims,
        const int32_t *dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE,
        const Executor executor=Executor::NONE
      ){
        dtfft_error_t error_code = dtfft_create_plan_c2c(ndims, dims, comm,
          static_cast<dtfft_precision_t>(precision),
          static_cast<dtfft_effort_t>(effort),
          static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL( static_cast<Error>(error_code) )
      }
  };

#ifndef DTFFT_TRANSPOSE_ONLY

/** Real-to-Complex Plan
 *
 * @note This class is only present in the API when ``dtFFT`` was compiled with any external FFT.
 */
  class PlanR2C final: public Plan
  {
    public:

/** @brief Real-to-Complex Plan constructor.
 *
 * @param[in]    dims                   Vector with global dimensions in reversed order.
 *                                        `dims.size()` must be 2 or 3
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    executor               Type of external FFT executor
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 *
 * @note Parameter `executor` cannot be Executor::NONE. PlanC2C should be used instead.
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanR2C(
        const std::vector<int32_t> &dims,
        const Executor executor,
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE
      ):PlanR2C(dims.size(), dims.data(), executor, comm, precision, effort) {}


/** Real-to-Complex Generic Plan constructor.
 *
 * @param[in]    ndims                  Number of dimensions: 2 or 3
 * @param[in]    dims                   Buffer of size `ndims` with global dimensions in reversed order.
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]    executor               Type of external FFT executor
 *
 * @note Parameter `executor` cannot be Executor::NONE. PlanC2C should be used instead.
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanR2C(
        const int8_t ndims,
        const int32_t *dims,
        const Executor executor,
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE
      ){
        dtfft_error_t error_code = dtfft_create_plan_r2c(ndims, dims, comm,
          static_cast<dtfft_precision_t>(precision),
          static_cast<dtfft_effort_t>(effort),
          static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL( static_cast<Error>(error_code) )
      }
  };
#endif


/** Real-to-Real Plan */
  class PlanR2R final: public Plan
  {
    public:
/** @brief Real-to-Real Plan constructor.
 *
 * @param[in]    dims                   Vector with global dimensions in reversed order.
 *                                        `dims.size()` must be 2 or 3
 * @param[in]    kinds                  Real FFT kinds in reversed order.
 *                                      Can be empty vector if `executor` == Executor::NONE
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]    executor               Type of external FFT executor.
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanR2R(
        const std::vector<int32_t> &dims,
        const std::vector<R2RKind> &kinds=std::vector<R2RKind>(),
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE,
        const Executor executor=Executor::NONE
      ):PlanR2R(dims.size(), dims.data(), kinds.data(), comm, precision, effort, executor) {}


/** @brief Real-to-Real Transpose-only Plan constructor.
 *
 * @param[in]    dims                   Vector with global dimensions in reversed order.
 *                                        `dims.size()` must be 2 or 3
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanR2R(
        const std::vector<int32_t> &dims,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE
      ):PlanR2R(dims.size(), dims.data(), nullptr, MPI_COMM_WORLD, precision, effort) {}


/** @brief Real-to-Real Generic Plan constructor.
 *
 * @param[in]    ndims                  Number of dimensions: 2 or 3
 * @param[in]    dims                   Buffer of size `ndims` with global dimensions in reversed order.
 * @param[in]    kinds                  Buffer of size `ndims` with Real FFT kinds in reversed order.
 *                                        Can be nullptr if `executor` == Executor::NONE
 * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]    precision              Precision of transform.
 * @param[in]    effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]    executor               Type of external FFT executor.
 *
 * @throws Exception In case error occurs during plan creation
 */
      PlanR2R(
        const int8_t ndims,
        const int32_t *dims,
        const R2RKind *kinds=nullptr,
        MPI_Comm comm=MPI_COMM_WORLD,
        const Precision precision=Precision::DOUBLE,
        const Effort effort=Effort::ESTIMATE,
        const Executor executor=Executor::NONE
      ) {
        dtfft_error_t error_code = dtfft_create_plan_r2r(ndims, dims, (dtfft_r2r_kind_t*)(kinds), comm,
          static_cast<dtfft_precision_t>(precision),
          static_cast<dtfft_effort_t>(effort),
          static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL( static_cast<Error>(error_code) )
      }
  };
}
// DTFFT_HPP
#endif