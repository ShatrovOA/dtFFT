/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
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
 * @date 2024 - 2025
 * @brief File containing C++ API of dtFFT Library
 */

#ifndef DTFFT_HPP
#define DTFFT_HPP

#include "dtfft.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/** Throwing exception macro */
#define DTFFT_THROW_EXCEPTION(error_code, msg)                       \
    do {                                                             \
        throw dtfft::Exception(error_code, msg, __FILE__, __LINE__); \
    } while (false);

/** Safe call macro.
 *
 * Should be used to check error codes returned by ``dtFFT``.
 *
 * @details Throws an exception with a message explaining the error if one
 * occurs.
 *
 * **Example**
 * @code
 * DTFFT_CXX_CALL( plan.execute(a, b, dtfft::Execute::FORWARD) )
 * @endcode
 */
#define DTFFT_CXX_CALL(call)                                           \
    do {                                                               \
        dtfft::Error ierr = call;                                      \
        if (ierr != dtfft::Error::SUCCESS) {                           \
            DTFFT_THROW_EXCEPTION(ierr, dtfft::get_error_string(ierr)) \
        }                                                              \
    } while (false);

namespace dtfft {
/** dtFFT version information */
struct Version {
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
    static constexpr int32_t get(int32_t major, int32_t minor,
        int32_t patch) noexcept
    {
        return DTFFT_VERSION(major, minor, patch);
    }
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
    /** Number of dimensions in provided Cartesian communicator > Number of
       dimension passed to `create` subroutine */
    INVALID_COMM_DIMS = DTFFT_ERROR_INVALID_COMM_DIMS,
    /** Passed Cartesian communicator with number of processes in 1st (fastest
       varying) dimension > 1 */
    INVALID_COMM_FAST_DIM = DTFFT_ERROR_INVALID_COMM_FAST_DIM,
    /** For R2R plan, `kinds` parameter must be passed if `executor` !=
       `Executor::NONE` */
    MISSING_R2R_KINDS = DTFFT_ERROR_MISSING_R2R_KINDS,
    /** Invalid values detected in `kinds` parameter */
    INVALID_R2R_KINDS = DTFFT_ERROR_INVALID_R2R_KINDS,
    /** Transpose plan is not supported in R2C, use R2R or C2C plan instead */
    R2C_TRANSPOSE_PLAN = DTFFT_ERROR_R2C_TRANSPOSE_PLAN,
    /** Inplace transpose is not supported */
    INPLACE_TRANSPOSE = DTFFT_ERROR_INPLACE_TRANSPOSE,
    /** Invalid `aux` buffer provided */
    INVALID_AUX = DTFFT_ERROR_INVALID_AUX,
    /** Invalid `layout` passed to `Plan.get_pencil` */
    INVALID_LAYOUT = DTFFT_ERROR_INVALID_LAYOUT,
    /** Invalid API Usage. */
    INVALID_USAGE = DTFFT_ERROR_INVALID_USAGE,
    /** Trying to create already created plan */
    PLAN_IS_CREATED = DTFFT_ERROR_PLAN_IS_CREATED,
    /** Selected `executor` does not support R2R FFTs */
    R2R_FFT_NOT_SUPPORTED = DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED,
    /** Internal call of Plan.mem_alloc failed */
    ALLOC_FAILED = DTFFT_ERROR_ALLOC_FAILED,
    /** Internal call of Plan.mem_free failed */
    FREE_FAILED = DTFFT_ERROR_FREE_FAILED,
    /** Invalid `alloc_bytes` provided */
    INVALID_ALLOC_BYTES = DTFFT_ERROR_INVALID_ALLOC_BYTES,
    /** Failed to dynamically load library */
    DLOPEN_FAILED = DTFFT_ERROR_DLOPEN_FAILED,
    /** Failed to dynamically load symbol */
    DLSYM_FAILED = DTFFT_ERROR_DLSYM_FAILED,
    /** Deprecated/unused: R2C transpose call restriction (kept for backward compatibility of error code numbering) */
    // R2C_TRANSPOSE_CALLED = DTFFT_ERROR_R2C_TRANSPOSE_CALLED,
    /** Sizes of `starts` and `counts` arrays passed to Pencil constructor do not
       match */
    PENCIL_ARRAYS_SIZE_MISMATCH = DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH,
    /** Sizes of `starts` and `counts` < 2 or > 3 provided to Pencil constructor
     */
    PENCIL_ARRAYS_INVALID_SIZES = DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES,
    /** Invalid `counts` provided to Pencil constructor */
    PENCIL_INVALID_COUNTS = DTFFT_ERROR_PENCIL_INVALID_COUNTS,
    /** Invalid `starts` provided to Pencil constructor */
    PENCIL_INVALID_STARTS = DTFFT_ERROR_PENCIL_INVALID_STARTS,
    /** Processes have same lower bounds (starts) but different sizes in some
       dimensions */
    PENCIL_SHAPE_MISMATCH = DTFFT_ERROR_PENCIL_SHAPE_MISMATCH,
    /** Pencil overlap detected, i.e. two processes share same part of global
       space */
    PENCIL_OVERLAP = DTFFT_ERROR_PENCIL_OVERLAP,
    /** Local pencils do not cover the global space without gaps */
    PENCIL_NOT_CONTINUOUS = DTFFT_ERROR_PENCIL_NOT_CONTINUOUS,
    /** Pencil is not initialized, i.e. `constructor` subroutine was not called */
    PENCIL_NOT_INITIALIZED = DTFFT_ERROR_PENCIL_NOT_INITIALIZED,
    /** Invalid `n_measure_warmup_iters` provided */
    INVALID_MEASURE_WARMUP_ITERS = DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS,
    /** Invalid `n_measure_iters` provided */
    INVALID_MEASURE_ITERS = DTFFT_ERROR_INVALID_MEASURE_ITERS,
    /** Invalid `dtfft_request_t` provided */
    INVALID_REQUEST = DTFFT_ERROR_INVALID_REQUEST,
    /** Attempting to execute already active transposition */
    TRANSPOSE_ACTIVE = DTFFT_ERROR_TRANSPOSE_ACTIVE,
    /** Attempting to finalize non-active transposition */
    TRANSPOSE_NOT_ACTIVE = DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE,
    /** Invalid `reshape_type` provided */
    INVALID_RESHAPE_TYPE = DTFFT_ERROR_INVALID_RESHAPE_TYPE,
    /** Attempting to execute already active reshape */
    RESHAPE_ACTIVE = DTFFT_ERROR_RESHAPE_ACTIVE,
    /** Attempting to finalize non-active reshape */
    RESHAPE_NOT_ACTIVE = DTFFT_ERROR_RESHAPE_NOT_ACTIVE,
    /** Inplace reshape is not supported */
    INPLACE_RESHAPE = DTFFT_ERROR_INPLACE_RESHAPE,
    /** R2C reshape was called */
    // R2C_RESHAPE_CALLED = DTFFT_ERROR_R2C_RESHAPE_CALLED,
    /** Invalid `execute_type` provided */
    INVALID_EXECUTE_TYPE = DTFFT_ERROR_INVALID_EXECUTE_TYPE,
    /** Reshape is not supported for this plan */
    RESHAPE_NOT_SUPPORTED = DTFFT_ERROR_RESHAPE_NOT_SUPPORTED,
    /** Execute called for transpose-only R2C Plan */
    R2C_EXECUTE_CALLED = DTFFT_ERROR_R2C_EXECUTE_CALLED,
    /** Invalid cartesian communicator provided */
    INVALID_CART_COMM = DTFFT_ERROR_INVALID_CART_COMM,
    /** Invalid stream provided */
    GPU_INVALID_STREAM = DTFFT_ERROR_GPU_INVALID_STREAM,
    /** Invalid backend provided */
    INVALID_BACKEND = DTFFT_ERROR_INVALID_BACKEND,
    /** Multiple MPI Processes located on same host share same GPU which is not
       supported */
    GPU_NOT_SET = DTFFT_ERROR_GPU_NOT_SET,
    /** When using R2R FFT and executor type is vkFFT and plan uses Z-slab
       optimization, it is required that types of R2R transform are same in X and
       Y directions */
    VKFFT_R2R_2D_PLAN = DTFFT_ERROR_VKFFT_R2R_2D_PLAN,
    /** Passed `effort` ==  `Effort::PATIENT` but all GPU backends have been
       disabled by `Config` */
    BACKENDS_DISABLED = DTFFT_ERROR_BACKENDS_DISABLED,
    /** One of pointers passed to `Plan.execute` or `Plan.transpose` cannot be
       accessed from device */
    NOT_DEVICE_PTR = DTFFT_ERROR_NOT_DEVICE_PTR,
    /** One of pointers passed to `Plan.execute` or `Plan.transpose` is not an
       `NVSHMEM` pointer */
    NOT_NVSHMEM_PTR = DTFFT_ERROR_NOT_NVSHMEM_PTR,
    /** Invalid platform provided */
    INVALID_PLATFORM = DTFFT_ERROR_INVALID_PLATFORM,
    /** Invalid executor provided for selected platform */
    INVALID_PLATFORM_EXECUTOR = DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR,
    /** Invalid backend provided for selected platform */
    INVALID_PLATFORM_BACKEND = DTFFT_ERROR_INVALID_PLATFORM_BACKEND
};

/** @brief Returns the string description of an error code
 *
 * @param[in] error_code Error code to convert to string
 * @return String representation of `error_code`
 */
std::string get_error_string(Error error_code) noexcept;

/** This enum lists valid `execute_type` parameters that can be passed to
 * Plan.execute. */
enum class Execute {
    /** Perform XYZ --> YZX --> ZXY plan execution (Forward) */
    FORWARD = DTFFT_EXECUTE_FORWARD,

    /** Perform ZXY --> YZX --> XYZ plan execution (Backward) */
    BACKWARD = DTFFT_EXECUTE_BACKWARD
};

/** This enum lists valid `transpose_type` parameters that can be passed to
 * Plan.transpose */
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
     * @note This value is valid only for 3D plans, and Plan.get_z_slab_enabled()
     * must return `true`
     */
    X_TO_Z = DTFFT_TRANSPOSE_X_TO_Z,

    /** Transpose from Fortran Z aligned to Fortran X aligned
     * @note This value is valid only for 3D plans, and Plan.get_z_slab_enabled()
     * must return `true`
     */
    Z_TO_X = DTFFT_TRANSPOSE_Z_TO_X
};

/** This enum lists valid `reshape_type` parameters that can be passed to
 * Plan.reshape */
enum class Reshape {
    /** Reshape from X bricks to X pencils */
    X_BRICKS_TO_PENCILS = DTFFT_RESHAPE_X_BRICKS_TO_PENCILS,
    /** Reshape from X pencils to X bricks */
    X_PENCILS_TO_BRICKS = DTFFT_RESHAPE_X_PENCILS_TO_BRICKS,
    /** Reshape from Z bricks to Z pencils */
    Z_BRICKS_TO_PENCILS = DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS,
    /** Reshape from Z pencils to Z bricks */
    Z_PENCILS_TO_BRICKS = DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS,
    /** Reshape from Y-bricks to Y-pencils
     * This is to be used in 2D Plans.
     */
    Y_BRICKS_TO_PENCILS = DTFFT_RESHAPE_Y_BRICKS_TO_PENCILS,
    /** Reshape from Y-pencils to Y-bricks
     * This is to be used in 2D Plans.
     */
    Y_PENCILS_TO_BRICKS = DTFFT_RESHAPE_Y_PENCILS_TO_BRICKS,
};

/** This enum represents different data layouts used in dtFFT and it should be
 * used to retrieve layout information from plans. */
enum class Layout {
    /** X-brick layout: data is distributed along all dimensions */
    X_BRICKS = DTFFT_LAYOUT_X_BRICKS,
    /** X-pencil layout: data is distributed along Y and Z dimensions */
    X_PENCILS = DTFFT_LAYOUT_X_PENCILS,
    /** X-pencil layout obtained after executing FFT for R2C plan: data is
       distributed along Y and Z dimensions */
    X_PENCILS_FOURIER = DTFFT_LAYOUT_X_PENCILS_FOURIER,
    /** Y-pencil layout: data is distributed along X and Z dimensions */
    Y_PENCILS = DTFFT_LAYOUT_Y_PENCILS,
    /** Z-pencil layout: data is distributed along X and Y dimensions */
    Z_PENCILS = DTFFT_LAYOUT_Z_PENCILS,
    /** Z-brick layout: data is distributed along all dimensions */
    Z_BRICKS = DTFFT_LAYOUT_Z_BRICKS
};

/** This enum lists valid `precision` parameters that can be passed to Plan
 * constructors.
 * @see get_precision_string
 */
enum class Precision {
    /** Use Single precision */
    SINGLE = DTFFT_SINGLE,
    /** Use Double precision */
    DOUBLE = DTFFT_DOUBLE
};

/**
 * @brief Returns the string representation of a Precision value.
 *
 * @param[in]       precision      Precision level to convert to string
 * @return String representation of Precision.
 */
std::string get_precision_string(Precision precision) noexcept;

/** This enum lists valid `effort` parameters that can be passed to Plan
 * constructors. */
enum class Effort {
    /** Create plan as fast as possible */
    ESTIMATE = DTFFT_ESTIMATE,

    /** Will attempt to find best MPI Grid decomposition.
     * Passing this flag and MPI Communicator with cartesian topology to any Plan
     * Constructor is same as Effort::ESTIMATE.
     */
    MEASURE = DTFFT_MEASURE,

    /** Same as Effort::MEASURE plus autotune will try to find best backend
     */
    PATIENT = DTFFT_PATIENT,

    /** Same as Effort::PATIENT plus will autotune all possible kernels
     * and reshape backends to find best configuration.
     */
    EXHAUSTIVE = DTFFT_EXHAUSTIVE
};

/** This enum lists available FFT executors.
 * @see get_executor_string
 */
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

/**
 * @brief Returns the string representation of an Executor value.
 *
 * @param[in]       executor       Executor type to convert to string
 * @return String representation of Executor.
 */
std::string get_executor_string(Executor executor) noexcept;

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

/** Various Backends available in dtFFT */
enum class Backend {
    /** @brief Backend that uses MPI datatypes.
     * @details This is default backend for Host build.
     *
     * Not really recommended to use for GPU usage, since it is a 'million' times
     * slower than other backends. Not available for autotune when `effort` is
     * Effort::PATIENT in GPU build.
     */
    MPI_DATATYPE = DTFFT_BACKEND_MPI_DATATYPE,

    /** MPI peer-to-peer algorithm */
    MPI_P2P = DTFFT_BACKEND_MPI_P2P,

    /** MPI peer-to-peer algorithm with overlapping data copying and unpacking */
    MPI_P2P_PIPELINED = DTFFT_BACKEND_MPI_P2P_PIPELINED,

    /** MPI backend using MPI_Alltoallv */
    MPI_A2A = DTFFT_BACKEND_MPI_A2A,

    /** MPI backend using one-sided communications */
    MPI_RMA = DTFFT_BACKEND_MPI_RMA,

    /** MPI backend using pipelined one-sided communications */
    MPI_RMA_PIPELINED = DTFFT_BACKEND_MPI_RMA_PIPELINED,

    /** MPI peer-to-peer algorithm with scheduled communication */
    MPI_P2P_SCHEDULED = DTFFT_BACKEND_MPI_P2P_SCHEDULED,

    /** NCCL backend */
    NCCL = DTFFT_BACKEND_NCCL,

    /** NCCL backend with overlapping data copying and unpacking */
    NCCL_PIPELINED = DTFFT_BACKEND_NCCL_PIPELINED,

    /** cuFFTMp backend */
    CUFFTMP = DTFFT_BACKEND_CUFFTMP,

    /** cuFFTMp backend that uses additional buffer to avoid extra copy and gain
       performance */
    CUFFTMP_PIPELINED = DTFFT_BACKEND_CUFFTMP_PIPELINED
};

/**
 * @brief Returns string with name of backend provided as argument.
 *
 * @param[in]       backend   Backend to represent
 *
 * @return String representation of `backend`.
 */
std::string get_backend_string(Backend backend);

/**
 * @brief Returns true if passed backend is pipelined and false otherwise.
 *
 * @param[in]         backend    Backend to check
 *
 * @return Logical flag
 */
bool get_backend_pipelined(Backend backend);

#ifdef DTFFT_WITH_CUDA
/** Enum that specifies runtime platform, e.g. Host, CUDA, HIP */
enum class Platform {
    /** Host */
    HOST = DTFFT_PLATFORM_HOST,
    /** CUDA */
    CUDA = DTFFT_PLATFORM_CUDA,
};
#endif

/** Basic exception class */
class Exception final : public std::exception {
private:
    Error _error_code;
    std::string _message;
    std::string _file;
    int _line;
    mutable std::string _what_cache;

public:
    /**
     * @brief Basic exception constructor
     * @param[in] error_code Error code
     * @param[in] msg        Message describing the error that occurred
     * @param[in] file       Filename where the exception was thrown
     * @param[in] line       Line number where the exception was thrown
     */
    Exception(Error error_code, std::string msg, const char* file, int line);

    /** Exception explanation */
    const char* what() const noexcept override;

    /** Returns error code of exception */
    Error get_error_code() const noexcept;

    /** Returns error message of exception */
    const std::string& get_message() const noexcept;

    /** Returns file name where exception occurred */
    const std::string& get_file() const noexcept;

    /** Returns line number where exception occurred */
    int get_line() const noexcept;
};

/** Class to handle Pencils. This is wrapper around `dtfft_pencil_t` C
 * structure.
 *
 * There are two ways users might find pencils useful inside dtFFT:
 * 1. To create a Plan using users's own grid decomposition, you can pass Pencil
 * to Plan constructor.
 * 2. To obtain Pencil from Plan in all possible layouts, in order to run FFT
 * not available in dtFFT.
 *
 * @see Plan.get_pencil()
 */
struct Pencil {
private:
    bool is_created;

    bool is_obtained;

    dtfft_pencil_t pencil;

    /** Constructor used internally by Plan::get_pencil */
    explicit Pencil(dtfft_pencil_t& c_pencil);
    friend class Plan;

public:
    /** Default constructor, does not actually initialize anything */
    Pencil();

    /** Pencil constructor. After calling this constructor, this pencil can be
     * used to create Plan
     *
     * @param[in]    n_dims                 Number of dimensions in pencil, must
     *                                      be 2 or 3
     * @param[in]    starts                 Local starts in natural Fortran order
     * @param[in]    counts                 Local counts in natural Fortran order
     */
    explicit Pencil(int32_t n_dims, const int32_t* starts, const int32_t* counts);

    /** Pencil constructor. After calling this constructor, this pencil can be
     * used to create Plan
     *
     * @param[in]    starts                 Local starts in natural Fortran order
     * @param[in]    counts                 Local counts in natural Fortran order
     */
    explicit Pencil(const std::vector<int32_t>& starts,
        const std::vector<int32_t>& counts);

    /** @return Number of dimensions in a pencil */
    uint8_t get_ndims() const;

    /** @return Aligned dimension ID starting from 1 */
    uint8_t get_dim() const;

    /** @return Local starts in natural Fortran order */
    std::vector<int32_t> get_starts() const;

    /** @return Local counts in natural Fortran order */
    std::vector<int32_t> get_counts() const;

    /** @return Total number of elements in a pencil */
    size_t get_size() const;

    /** @return Underlying C structure */
    const dtfft_pencil_t& c_struct() const;
};

/** Class to set additional configuration parameters to dtFFT
 * @see set_config()
 */
struct Config {
protected:
    /** Underlying C structure */
    dtfft_config_t config;

public:
    /** Creates and sets default configuration values */
    explicit Config() {
        DTFFT_CXX_CALL(static_cast<Error>(dtfft_create_config(&config)))
    }

    /**
     * @brief Sets whether dtFFT should print additional information or not.
     *
     * @details Default is `false`
     */
    Config& set_enable_log(const bool enable_log) noexcept
    {
        config.enable_log = enable_log;
        return *this;
    }

    /**
     * @brief Sets whether dtFFT use Z-slab optimization or not.
     *
     * @details Default is `true`
     *
     * One should consider disabling Z-slab optimization in order to resolve
     * `Error::VKFFT_R2R_2D_PLAN` error or when underlying FFT implementation of
     * 2D plan is too slow.
     *
     * In all other cases, Z-slab is considered to be always faster.
     */
    Config& set_enable_z_slab(bool enable_z_slab) noexcept
    {
        config.enable_z_slab = enable_z_slab;
        return *this;
    }

    /**
     * @brief Sets whether dtFFT should use Y-slab optimization or not.
     *
     * @details Default is `false`
     *
     * If `true` then `dtFFT` will skip the transpose step between Y and Z aligned
     * layouts during call to Plan.execute(). One should consider disabling Y-slab
     * optimization in order to resolve `Error::VKFFT_R2R_2D_PLAN` error or when
     * underlying FFT implementation of 2D plan is too slow.
     *
     * In all other cases, Y-slab is considered to be always faster.
     */
    Config& set_enable_y_slab(bool enable_y_slab) noexcept
    {
        config.enable_y_slab = enable_y_slab;
        return *this;
    }

    /**
     * @brief Sets number of warmup iterations to underlying C structure
     *
     * @param[in] n_measure_warmup_iters    Number of warmup iterations to execute during backend and kernel autotuning when effort level is Effort::MEASURE or higher.
     */
    Config& set_measure_warmup_iters(int32_t n_measure_warmup_iters) noexcept
    {
        config.n_measure_warmup_iters = n_measure_warmup_iters;
        return *this;
    }
    /**
     * @brief Sets number of actual iterations to underlying C structure
     * @param[in] n_measure_iters           Number of iterations to execute during backend and kernel autotuning when effort level is Effort::MEASURE or higher.
     */
    Config& set_measure_iters(int32_t n_measure_iters) noexcept
    {
        config.n_measure_iters = n_measure_iters;
        return *this;
    }

#ifdef DTFFT_WITH_CUDA
    /**
     * @brief Sets platform to execute plan.
     *
     * @details Default is Platform::HOST.
     *
     * This option is only available when dtFFT is built with device support.
     * Even when dtFFT is built with device support, it does not necessarily mean that all plans must be device-related.
     * This enables a single library installation to support both host and CUDA plans.
     */
    Config& set_platform(Platform platform) noexcept
    {
        config.platform = static_cast<dtfft_platform_t>(platform);
        return *this;
    }
    /**
     * @brief Sets Main CUDA stream that will be used in dtFFT.
     *
     * @details This parameter is a placeholder for user to set custom stream.
     * Stream that is actually used by dtFFT plan is returned by Plan.get_stream
     * function. When user sets stream he is responsible of destroying it.
     *
     * Stream must not be destroyed before call to destroy.
     *
     * @note This method is only present in the API when ``dtFFT`` was compiled
     * with CUDA Support.
     */
    Config& set_stream(dtfft_stream_t stream) noexcept
    {
        config.stream = stream;
        return *this;
    }
#endif

    /**
     * @brief Sets Backend that will be used by dtFFT when `effort` is
     * Effort::ESTIMATE or Effort::MEASURE.
     *
     * @details Default for HOST platform is Backend::MPI_DATATYPE.
     *
     * Default for CUDA platform is Backend::NCCL if NCCL is enabled, otherwise Backend::MPI_P2P.
     */
    Config& set_backend(Backend backend) noexcept
    {
        config.backend = static_cast<dtfft_backend_t>(backend);
        return *this;
    }

    /**
     * @brief Sets Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when `effort` is Effort::ESTIMATE or Effort::MEASURE.
     *
     * @details Default for HOST platform is Backend::MPI_DATATYPE.
     *
     * Default for CUDA platform is Backend::NCCL if NCCL is enabled, otherwise Backend::MPI_P2P.
     */
    Config& set_reshape_backend(Backend backend) noexcept
    {
        config.reshape_backend = static_cast<dtfft_backend_t>(backend);
        return *this;
    }

    /**
     * @brief Should Backend::MPI_DATATYPE be considered for autotuning when `effort` is
     * Effort::PATIENT or Effort::EXHAUSTIVE.
     *
     * @details Default is `true`.
     *
     * This option only works when `platform` is Platform::HOST.
     * When `platform` is Platform::CUDA, Backend::MPI_DATATYPE is always disabled during autotuning.
     */
    Config& set_enable_datatype_backend(bool enable_datatype_backend) noexcept
    {
        config.enable_datatype_backend = enable_datatype_backend;
        return *this;
    }

    /**
     * @brief Should MPI Backends be enabled when `effort` is Effort::PATIENT or Effort::EXHAUSTIVE.
     *
     * @details Default is `false`.
     *
     * The following applies only to CUDA builds.
     * MPI Backends are disabled by default during autotuning process due to
     * OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849 It was noticed
     * that during plan autotuning GPU memory not being freed completely.
     *
     * For example:
     * 1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization,
     * with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
     * Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of
     * the GPUs.
     *
     * One of the workarounds is to disable MPI Backends by default, which is done
     * here.
     *
     * Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
     * but it was noticed that disabling CUDA IPC seriously affects overall
     * performance of MPI algorithms
     */
    Config& set_enable_mpi_backends(bool enable_mpi_backends) noexcept
    {
        config.enable_mpi_backends = enable_mpi_backends;
        return *this;
    }

    /**
     * @brief Sets whether pipelined backends be enabled when `effort` is
     * Effort::PATIENT or Effort::EXHAUSTIVE.
     *
     * @details Default is `true`.
     */
    Config&
    set_enable_pipelined_backends(bool enable_pipelined_backends) noexcept
    {
        config.enable_pipelined_backends = enable_pipelined_backends;
        return *this;
    }

#ifdef DTFFT_WITH_CUDA
    /**
     * @brief Sets whether NCCL backends be enabled when `effort` is
     * Effort::PATIENT or Effort::EXHAUSTIVE.
     * @details Default is `true`.
     * @note This method is only present in the API when ``dtFFT`` was compiled
     * with CUDA Support.
     */
    Config& set_enable_nccl_backends(bool enable_nccl_backends) noexcept
    {
        config.enable_nccl_backends = enable_nccl_backends;
        return *this;
    }

    /**
     * @brief Should NVSHMEM backends be enabled when `effort` is Effort::PATIENT
     * or Effort::EXHAUSTIVE.
     * @details Default is `true`.
     * @note This method is only present in the API when ``dtFFT`` was compiled
     * with CUDA Support.
     */
    Config& set_enable_nvshmem_backends(bool enable_nvshmem_backends) noexcept
    {
        config.enable_nvshmem_backends = enable_nvshmem_backends;
        return *this;
    }
#endif
    /**
     * @brief Should dtFFT try to optimize kernel launch parameters during plan creation when `effort` is below Effort::EXHAUSTIVE.
     *
     * @details Default is `false`.
     *
     * Kernel optimization is always enabled for Effort::EXHAUSTIVE effort level.
     * Setting this option to true enables kernel optimization for lower effort levels (Effort::ESTIMATE, Effort::MEASURE, Effort::PATIENT).
     * This may increase plan creation time but can improve runtime performance.
     * Since kernel optimization is performed without data transfers, the time increase is usually minimal.
     */
    Config&
    set_enable_kernel_autotune(bool enable_kernel_autotune) noexcept
    {
        config.enable_kernel_autotune = enable_kernel_autotune;
        return *this;
    }

    /**
     * @brief Should dtFFT execute reshapes from pencils to bricks and vice versa in Fourier space during calls to execute.
     *
     * @details Default is `false`.
     *
     * When enabled, data will be in brick layout in Fourier space, which may be useful for certain operations
     * between forward and backward transforms. However, this requires additional data transpositions
     * and will reduce overall FFT performance.
     */
    Config&
    set_enable_fourier_reshape(bool enable_fourier_reshape) noexcept
    {
        config.enable_fourier_reshape = enable_fourier_reshape;
        return *this;
    }

    /** @return Underlying C structure */
    dtfft_config_t c_struct() const { return config; }
};

/** @brief Sets configuration values to dtFFT. Must be called before plan
 * creation to take effect.
 *
 * @return Error::SUCCESS if the call was successful, error code otherwise
 * @see Config
 */
Error set_config(const Config& config) noexcept;

/** Abstract plan for all dtFFT plans.
 * @details This class does not have any constructors. To create a plan user
 * should use one of the inherited classes.
 */
class Plan {
protected:
    /** Underlying C structure */
    dtfft_plan_t _plan;

public:
    /** @brief Checks if plan is using Z-slab optimization.
     * If `true` then flags Transpose::X_TO_Z and Transpose::Z_TO_X will be valid
     * to pass to Plan.transpose method.
     *
     * @param[out]     is_z_slab_enabled       Boolean value if Z-slab is used.
     *
     * @return Error::SUCCESS if call was without error, error code otherwise
     */
    Error get_z_slab_enabled(bool* is_z_slab_enabled) const noexcept;

    /**
     * @brief Checks if plan is using Z-slab optimization.
     *
     * @return `true` if Z-slab is enabled, false otherwise
     *
     * @throws Exception if underlying call fails
     */
    bool get_z_slab_enabled() const;

    /** @brief Checks if plan is using Y-slab optimization.
     * If `true` then during call to Plan.execute the transpose between Y and Z
     * aligned layouts will be skipped.
     *
     * @param[out]     is_y_slab_enabled       Boolean value if Y-slab is used.
     * @return Error::SUCCESS if call was without error, error code otherwise
     */
    Error get_y_slab_enabled(bool* is_y_slab_enabled) const noexcept;

    /**
     * @brief Checks if plan is using Y-slab optimization.
     *
     * @return `true` if Y-slab is enabled, false otherwise
     *
     * @throws Exception if underlying call fails
     */
    bool get_y_slab_enabled() const;

    /**
     * @brief Prints plan-related information to stdout
     *
     * @return Error::SUCCESS if call was without error, error code otherwise
     */
    Error report() const noexcept;

    /**
     * @brief Obtains pencil information from plan. This can be useful when user
     * wants to use own FFT implementation, that is unavailable in dtFFT.
     *
     * @param[in]     layout          Required layout of the pencil
     * @param[out]    pencil          Created Pencil object
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_pencil(Layout layout, Pencil& pencil) const noexcept;

    /**
     * @brief Get the pencil object
     *
     * @param[in]     layout          Required layout of the pencil
     * @return Created Pencil object
     *
     * @throws Exception if underlying call fails
     */
    Pencil get_pencil(Layout layout) const;

    /** @brief Plan execution
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Result pointer
     * @param[in]      execute_type         Direction of execution
     * @param[inout]   aux                  Optional Auxiliary pointer. If provided, must be at least get_aux_bytes() bytes.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error execute(void* in, void* out, Execute execute_type,
        void* aux = nullptr) const noexcept;

    /**
     * @brief In-place plan execution
     * @details This template allows user to cast result pointer to desired type.
     *
     * @code{.cpp}
     * float *data = ...; // Pointer to data
     *
     * PlanR2C plan = ...; // Create plan
     *
     * auto fourier_data = plan.execute<std::complex<float>>(data,
     * Execute::FORWARD);
     * // `fourier_data` is still pointing to `data`, but is of type
     * std::complex<float>*
     * @endcode
     *
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[in]         execute_type  Direction of execution
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename Tr>
    Tr* execute(void* inout, const Execute execute_type,
        void* aux = nullptr) const
    {
        DTFFT_CXX_CALL(execute(inout, inout, execute_type, aux))
        return static_cast<Tr*>(inout);
    }

    /**
     * @brief In-place plan execution
     * @details This template allows user to keep result pointer of the same type
     * as input pointer.
     *
     * @code{.cpp}
     * float *data = ...; // Pointer to data
     *
     * PlanR2R plan = ...; // Create plan
     *
     * auto fourier_data = plan.execute(data, Execute::FORWARD);
     * // `fourier_data` is still pointing to `data` and is still of type float*
     * @endcode
     *
     * @tparam            T             Type of input/output data. This should be
     *                                  a basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[in]         execute_type  Direction of execution
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename T, typename Tr = T>
    Tr* execute(T* inout, const Execute execute_type, void* aux = nullptr) const
    {
        return execute<Tr>(static_cast<void*>(inout), execute_type, aux);
    }

    /** @brief Forward plan execution
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Result pointer
     * @param[inout]   aux                  Auxiliary pointer. Can be `nullptr`. If provided, must be at least get_aux_bytes() bytes.
     *
     * @return Error::SUCCESS on success or error code on failure.
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    Error forward(void* in, void* out, void* aux) const noexcept;

    /**
     * @brief In-place forward plan execution
     * @details This template allows user to cast result pointer to desired type.
     *
     * @code{.cpp}
     * float *data = ...; // Pointer to data
     *
     * PlanR2C plan = ...; // Create plan
     *
     * auto fourier_data = plan.forward<std::complex<float>>(data);
     * // `fourier_data` is still pointing to `data`, but is of type
     * std::complex<float>*
     * @endcode
     *
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename Tr>
    Tr* forward(void* inout, void* aux = nullptr) const
    {
        return execute<Tr>(inout, Execute::FORWARD, aux);
    }

    /**
     * @brief In-place forward plan execution
     * @details This template allows user to keep result pointer of the same type
     * as input pointer.
     *
     * @code{.cpp}
     * float *data = ...; // Pointer to data
     *
     * PlanR2R plan = ...; // Create plan
     *
     * auto fourier_data = plan.forward(data);
     * // `fourier_data` is still pointing to `data` and is still of type float*
     * @endcode
     *
     * @tparam            T             Type of input/output data. This should be
     *                                  a basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename T, typename Tr = T>
    Tr* forward(T* inout, void* aux = nullptr) const
    {
        return forward<Tr>(static_cast<void*>(inout), aux);
    }

    /** @brief Backward plan execution
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Result pointer
     * @param[inout]   aux                  Auxiliary pointer. Can be `nullptr`
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error backward(void* in, void* out, void* aux) const noexcept;

    /**
     * @brief In-place backward plan execution
     * @details This template allows user to cast result pointer to desired type.
     *
     * @code{.cpp}
     * std::complex<float> *fourier_data = ...; // Pointer to data
     *
     * PlanR2C plan = ...; // Create plan
     *
     * auto real_data = plan.backward<float>(fourier_data);
     * // `real_data` is still pointing to `fourier_data`, but is of type float*
     * @endcode
     *
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename Tr>
    Tr* backward(void* inout, void* aux = nullptr) const
    {
        return execute<Tr>(inout, Execute::BACKWARD, aux);
    }

    /**
     * @brief In-place backward plan execution
     * @details This template allows user to keep result pointer of the same type
     * as input pointer.
     *
     * @code{.cpp}
     * float *fourier_data = ...; // Pointer to data
     *
     * PlanR2R plan = ...; // Create plan
     *
     * auto real_data = plan.backward(fourier_data);
     * // `real_data` is still pointing to `fourier_data` and is still of type
     * float *
     * @endcode
     *
     * @tparam            T             Type of input/output data. This should be
     *                                  a basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @tparam            Tr            Type of returned data. This should be a
     *                                  basic pointer type, e.g. float, double
     *                                  or std::complex of any of those
     * @param[inout]      inout         Input/output pointer
     * @param[inout]      aux           Optional Auxiliary pointer
     *
     * @return Pointer to the processed data casted to type `Tr`
     * @throws Exception if underlying call fails
     * @note Not all plans support in-place plan executing. Refer to the manual
     * for list of unsupported cases.
     */
    template <typename T, typename Tr = T>
    Tr* backward(T* inout, void* aux = nullptr) const
    {
        return backward<Tr>(static_cast<void*>(inout), aux);
    }

    /** @brief Transpose data in single dimension, e.g. X align -> Y align
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                    Input pointer
     * @param[out]     out                   Pointer of transposed data
     * @param[in]      transpose_type        Type of transpose to perform.
     * @param[inout]   aux                   Auxiliary pointer. Can be `nullptr`. If provided, must be at least get_alloc_size() elements.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error transpose(void* in, void* out, Transpose transpose_type, void* aux = nullptr) const noexcept;

    /**
     * @brief Starts an asynchronous transpose operation in single dimension, e.g.
     * X align -> Y align
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Output pointer
     * @param[in]      transpose_type       Type of transpose to perform
     * @param[inout]   aux                  Auxiliary pointer. Can be `nullptr`. If provided, must be at least get_alloc_bytes() bytes.
     * @param[out]     request              Handle to manage the asynchronous
     *                                      operation
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error transpose_start(void* in, void* out, Transpose transpose_type, void* aux,
        dtfft_request_t* request) const noexcept;

    /** @brief Starts an asynchronous transpose operation in single dimension, e.g.
     * X align -> Y align
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Output pointer
     * @param[in]      transpose_type       Type of transpose to perform
     * @param[out]     request              Handle to manage the asynchronous
     *                                      operation
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error transpose_start(void* in, void* out, Transpose transpose_type,
        dtfft_request_t* request) const noexcept;

    /**
     * @brief Ends an asynchronous transpose operation.
     *
     * @param[inout] request Handle to manage the asynchronous operation
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error transpose_end(dtfft_request_t request) const noexcept;

    /** @brief Reshape data from bricks to pencils and vice versa
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                    Input pointer
     * @param[out]     out                   Pointer of reshaped data
     * @param[in]      reshape_type          Type of reshape to perform.
     * @param[inout]   aux                   Auxiliary pointer. Can be `nullptr`. If provided, must be at least get_alloc_size() elements.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error reshape(void* in, void* out, Reshape reshape_type, void* aux=nullptr) const noexcept;

    /** @brief Starts an asynchronous reshape operation from bricks to pencils
     * and vice versa
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Output pointer
     * @param[in]      reshape_type         Type of reshape to perform
     * @param[inout]   aux                  Auxiliary pointer. Can be `nullptr`. If provided, must be at least get_alloc_size() elements.
     * @param[out]     request              Handle to manage the asynchronous
     *                                      operation
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error reshape_start(void* in, void* out, Reshape reshape_type, void* aux, dtfft_request_t* request) const noexcept;

    /** @brief Starts an asynchronous reshape operation from bricks to pencils
     * and vice versa
     * @attention `in` and `out` cannot be the same pointers
     *
     * @param[inout]   in                   Input pointer
     * @param[out]     out                  Output pointer
     * @param[in]      reshape_type         Type of reshape to perform
     * @param[out]     request              Handle to manage the asynchronous
     *                                      operation
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error reshape_start(void* in, void* out, Reshape reshape_type, dtfft_request_t* request) const noexcept;

    /**
     * @brief Ends an asynchronous reshape operation.
     *
     * @param[inout] request Handle to manage the asynchronous operation
     * @return  Error::SUCCESS on success or error code on failure.
     */
    Error reshape_end(dtfft_request_t request) const noexcept;

    /** @brief Wrapper around `Plan.get_local_sizes` to obtain `alloc_size` only
     *
     * @param[out]     alloc_size       Minimum number of elements to be allocated
     *                                  for `in` and `out` buffers required by `Plan.execute`.
     *                                  This also returns minimum number of elements required for `aux` buffer
     *                                  required by `Plan.transpose` and `Plan.reshape`.
     *                                  Minimum number of `aux` elements required by `Plan.execute` can be obtained
     *                                  by calling `Plan.get_aux_size`.
     *                                  Size of each element in bytes can be obtained by calling `Plan.get_element_size`.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_alloc_size(size_t* alloc_size) const noexcept;

    /** @brief Wrapper around Plan.get_local_sizes to obtain `alloc_size` only
     *
     * @return Minimum number of elements to be allocated for `in` and `out` buffers required by `Plan.execute`.
     *         This also returns minimum number of elements required for `aux` buffer required by `Plan.transpose` and `Plan.reshape`.
     *         Minimum number of `aux` elements required by `Plan.execute` can be obtained by calling `Plan.get_aux_size`.
     *
     * @throws Exception if underlying call fails
     */
    std::size_t get_alloc_size() const;

    /** @brief Get auxiliary buffer size required to execute the plan
     *
     * @param[out]   aux_size              Number of elements required for
     *                                     auxiliary buffer. Size of each element in bytes can be obtained by calling
     *                                     `Plan.get_element_size`.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_aux_size(std::size_t* aux_size) const noexcept;

    /** @brief Get auxiliary buffer size required to execute the plan
     *
     * @return Number of elements required for auxiliary buffer.
     *
     * @throws Exception if underlying call fails
     */
    std::size_t get_aux_size() const;

    /** @brief Get auxiliary buffer size in bytes required to execute the plan
     *
     * @param[out]   aux_bytes              Number of bytes required for
     *                                     auxiliary buffer.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_aux_bytes(std::size_t* aux_bytes) const noexcept;

    /** @brief Get auxiliary buffer size in bytes required to execute the plan
     *
     * @return Number of bytes required for auxiliary buffer.
     *
     * @throws Exception if underlying call fails
     */
    std::size_t get_aux_bytes() const;

    /** @brief Get number of elements required by Plan.reshape
     * 
     * @param[out]  aux_size    Number of elements required for auxiliary buffer during reshape operation. 
     *                          Size of each element in bytes can be obtained by calling `Plan.get_element_size`.
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_aux_size_reshape(std::size_t* aux_size) const noexcept;

    /** @brief Get number of elements required by Plan.reshape
     * 
     * @return Number of elements required for auxiliary buffer during reshape operation.
     * @throws Exception if underlying call fails
     */
    std::size_t get_aux_size_reshape() const;

    /** @brief Get number of bytes required by Plan.reshape
     * 
     * @param[out]  aux_bytes   Number of bytes required for auxiliary buffer during reshape operation.
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_aux_bytes_reshape(std::size_t* aux_bytes) const noexcept;

    /** @brief Get number of bytes required by Plan.reshape
     * 
     * @return Number of bytes required for auxiliary buffer during reshape operation.
     * @throws Exception if underlying call fails
     */
    std::size_t get_aux_bytes_reshape() const;

    /** @brief Get grid decomposition information. Results may differ on different
     * MPI processes
     *
     * @param[out]   in_starts              Starts of local portion of data in
     * 'real' space in reversed order
     * @param[out]   in_counts              Sizes  of local portion of data in
     * 'real' space in reversed order
     * @param[out]   out_starts             Starts of local portion of data in
     * 'fourier' space in reversed order
     * @param[out]   out_counts             Sizes  of local portion of data in
     * 'fourier' space in reversed order
     * @param[out]   alloc_size             Minimum number of elements to be
     * allocated for `in`, `out` or `aux` buffers. Size of each element in bytes
     * can be obtained by calling `Plan.get_element_size`.
     *
     * @return Error::SUCCESS on success or error code on failure.
     *
     * @note Before calling this function, user must ensure that `in_starts`,
     * `in_counts`, `out_starts` and `out_counts` vectors are large enough to hold
     * the data.
     */
    Error get_local_sizes(std::vector<int32_t>& in_starts,
        std::vector<int32_t>& in_counts,
        std::vector<int32_t>& out_starts,
        std::vector<int32_t>& out_counts,
        std::size_t* alloc_size) const noexcept;

    /** @brief Get grid decomposition information. Results may differ on different
     * MPI processes
     *
     * @param[out]   in_starts              Starts of local portion of data in
     * 'real' space in reversed order
     * @param[out]   in_counts              Sizes  of local portion of data in
     * 'real' space in reversed order
     * @param[out]   out_starts             Starts of local portion of data in
     * 'fourier' space in reversed order
     * @param[out]   out_counts             Sizes  of local portion of data in
     * 'fourier' space in reversed order
     * @param[out]   alloc_size             Minimum number of elements needs to be
     * allocated for `in`, `out` or `aux` buffers. Size of each element in bytes
     * can be obtained by calling `Plan.get_element_size`.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_local_sizes(int32_t* in_starts = nullptr,
        int32_t* in_counts = nullptr,
        int32_t* out_starts = nullptr,
        int32_t* out_counts = nullptr,
        size_t* alloc_size = nullptr) const noexcept;

    /**
     * @brief Obtains number of bytes required to store single element by this
     * plan.
     *
     * @param[out]    element_size    Size of element in bytes
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_element_size(size_t* element_size) const noexcept;

    /**
     * @brief Obtains number of bytes required to store single element by this
     * plan.
     *
     * @return Size of element in bytes
     *
     * @throws Exception if underlying call fails
     */
    size_t get_element_size() const;

    /**
     * @brief Returns minimum number of bytes required for in and out buffers.
     *
     * This function is a combination of two calls: Plan.get_alloc_size and
     * Plan.get_element_size.
     * Returns minimum number of bytes to be allocated for `in` and `out` buffers required by `Plan.execute`.
     * This also returns minimum number of bytes required for `aux` buffer required by `Plan.transpose` and `Plan.reshape`.
     * Minimum number of `aux` bytes required by `Plan.execute` can be obtained by calling `Plan.get_aux_bytes`.
     *
     * @param[out]    alloc_bytes    Number of bytes required
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_alloc_bytes(size_t* alloc_bytes) const noexcept;

    /**
     * @brief Returns minimum number of bytes required for in and out buffers.
     *
     * This function is a combination of two calls: Plan.get_alloc_size and
     * Plan.get_element_size.
     * Returns minimum number of bytes to be allocated for `in` and `out` buffers required by `Plan.execute`.
     * This also returns minimum number of bytes required for `aux` buffer required by `Plan.transpose` and `Plan.reshape`.
     * Minimum number of `aux` bytes required by `Plan.execute` can be obtained by calling `Plan.get_aux_bytes`.
     *
     * @return  Number of bytes of each buffer required to execute plan
     *
     * @throws Exception if underlying call fails
     */
    size_t get_alloc_bytes() const;

    /**
     * @brief  Returns executor used by this plan.
     * @param[out]    executor   Executor used by this plan.
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_executor(Executor* executor) const noexcept;

    /**
     * @brief Returns executor used by this plan.
     * @return Executor used by this plan.
     * @throws Exception if underlying call fails
     */
    Executor get_executor() const;

    /**
     * @brief Returns precision of the plan.
     *
     * @param[out]    precision   Precision of the plan.
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_precision(Precision* precision) const noexcept;

    /**
     * @brief Returns precision of the plan.
     * @return Precision of the plan.
     * @throws Exception if underlying call fails
     */
    Precision get_precision() const;

    /**
     * @brief Returns global dimensions of the plan.
     * @param[out]    ndims     Number of dimensions in the plan. User can pass
     *                          nullptr if this value is not needed.
     * @param[out]    dims      Array of dimensions in natural Fortran order. User
     *                          can pass nullptr if this value is not needed.
     *
     * @note Do not free the array, it is freed when the Plan is destroyed.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_dims(int8_t* ndims, const int32_t* dims[]) const noexcept;

    /**
     * @brief Returns global dimensions of the plan.
     *
     * @return Vector of dimensions in natural Fortran order. Size of vector is
     * equal to number of dimensions in the plan.
     * @throws Exception if underlying call fails
     */
    std::vector<int32_t> get_dims() const;

    /**
     * @brief Returns grid decomposition dimensions of the plan.
     *
     * @param[out]     ndims            Number of dimensions in plan. User can pass
     *                                  nullptr if this value is not needed.
     * @param[out]     grid_dims        Pointer of size `ndims` containing grid
     *                                  decomposition dimensions in reverse order:
     *                                  grid_dims[0] is the fastest varying and is always equal to 1.
     *                                  User can pass nullptr if this value is not needed.
     *
     * @note Do not free `grid_dims` array, it is freed when the Plan is
     * destroyed.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_grid_dims(int8_t* ndims, const int32_t* grid_dims[]) const noexcept;

    /**
     * @brief Returns grid decomposition dimensions of the plan.
     *
     * @return Vector of grid decomposition dimensions in natural Fortran order.
     * Size of vector is equal to number of dimensions in the plan. First value is
     * always equal to 1.
     * @throws Exception if underlying call fails
     */
    std::vector<int32_t> get_grid_dims() const;

    /**
     * @brief Allocates memory specific for this plan
     *
     * @param[in]     alloc_bytes     Number of bytes to allocate
     * @param[out]    ptr             Allocated pointer
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error mem_alloc(size_t alloc_bytes, void** ptr) const noexcept;

    /**
     * @brief Allocates memory specific for this plan
     *
     * @param alloc_bytes Number of bytes to allocate
     *
     * @return Pointer to allocated memory
     * @throws Exception if underlying call fails
     */
    void* mem_alloc(size_t alloc_bytes) const;

    /**
     * @brief Allocates memory for an array of elements of type T
     *
     * @tparam      T           Type of elements
     * @param[in]   alloc_size  Number of elements to allocate
     *
     * @return Pointer to allocated memory
     * @throws Exception if underlying call fails
     */
    template <typename T>
    T* mem_alloc(const size_t alloc_size) const
    {
        return static_cast<T*>(mem_alloc(alloc_size * sizeof(T)));
    }

    /**
     * @brief Frees memory specific for this plan
     *
     * @param[inout]  ptr             Allocated pointer
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error mem_free(void* ptr) const noexcept;

    /** @brief Plan Destructor. To fully clean all internal memory, this should be
     * called before MPI_Finalize
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error destroy() noexcept;

    /**
     * @brief Returns selected backend during autotune if `effort` is
     * Effort::PATIENT.
     *
     * If `effort` passed to any create function is Effort::ESTIMATE or
     * Effort::MEASURE returns value set by Config.set_backend followed by
     * set_config() or default value, which is Backend::NCCL.
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_backend(Backend& backend) const noexcept;

    /**
     * @brief Returns selected backend during autotune if `effort` is
     * Effort::PATIENT. If `effort` passed to any create function is
     * Effort::ESTIMATE or Effort::MEASURE returns value set by Config.set_backend
     * followed by set_config() or default value, which is Backend::NCCL.
     *
     * @return Backend used by this plan.
     *
     * @throws Exception if underlying call fails
     */
    Backend get_backend() const;

    /**
     * @brief Returns backend used for reshape operations.
     *
     * @param[out]    backend         Backend used for reshape operations
     *
     * @return Error::SUCCESS on success or error code on failure.
     */
    Error get_reshape_backend(Backend& backend) const noexcept;

    /**
     * @brief Returns backend used for reshape operations.
     *
     * @return Backend used for reshape operations
     *
     * @throws Exception if underlying call fails
     */
    Backend get_reshape_backend() const;

#ifdef DTFFT_WITH_CUDA
    /**
     * @brief Returns stream associated with current Plan.
     * This can either be stream passed by Config.set_stream followed by
     * set_config() or stream created internally. Returns NULL pointer if plan's
     * platform is Platform::HOST.
     *
     * @param[out]     stream         CUDA stream associated with plan
     *
     * @return Error::SUCCESS on success or error code on failure.
     * @note This method is only present in the API when ``dtFFT`` was compiled
     * with CUDA Support.
     */
    Error get_stream(dtfft_stream_t* stream) const noexcept;

    /**
     * @brief Returns stream associated with current Plan.
     * This can either be stream passed by Config.set_stream followed by
     * set_config() or stream created internally. Returns NULL pointer if plan's
     * platform is Platform::HOST.
     *
     * @return dtFFT stream associated with plan
     * @note This method is only present in the API when ``dtFFT`` was compiled
     * with CUDA Support.
     * @throws Exception if underlying call fails
     */
    dtfft_stream_t get_stream() const;

    /**
     * @brief Returns plan execution platform.
     *
     * @return `::DTFFT_SUCCESS` on success or error code on failure.
     */
    Error get_platform(Platform& platform) const noexcept;

    /**
     * @brief Returns plan execution platform.
     * @return Platform::HOST if plan is executed on host, Platform::CUDA if plan
     * is executed on CUDA device.
     * @throws Exception if underlying call fails
     */
    Platform get_platform() const;
#endif

    /** @return Underlying C structure */
    dtfft_plan_t c_struct() const { return _plan; }

    /** Plan Destructor. To fully clean all internal memory, this should be called
     * before MPI_Finalize */
    virtual ~Plan() noexcept = 0;
};

inline Plan::~Plan() noexcept { destroy(); }

/** Complex-to-Complex Plan */
class PlanC2C final : public Plan {
public:
    /** @brief Complex-to-Complex Plan constructor.
     *
     * @param[in]    dims                   Vector with global dimensions in
     *                                      reversed order. `dims.size()` must be 2 or 3
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanC2C(const std::vector<int32_t>& dims,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);

    /** @brief Complex-to-Complex Transpose-only Plan constructor.
     *
     * @param[in]    dims                   Vector with global dimensions in
     *                                      reversed order. `dims.size()` must be 2 or 3
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanC2C(const std::vector<int32_t>& dims, Precision precision,
        Effort effort = Effort::ESTIMATE);

    /** @brief Complex-to-Complex Generic Plan constructor.
     *
     * @param[in]    ndims                  Number of dimensions: 2 or 3
     * @param[in]    dims                   Buffer of size `ndims` with global
     *                                      dimensions in reversed order.
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanC2C(int8_t ndims, const int32_t* dims,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);

    /** @brief Complex-to-Complex Plan constructor using pencil decomposition
     * information.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @note Parameter `executor` cannot be Executor::NONE. PlanC2C should be used
     * instead.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanC2C(const Pencil& pencil, Precision precision,
        Effort effort = Effort::ESTIMATE);

    /** @brief Complex-to-Complex Plan constructor using pencil decomposition
     * information.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor
     *
     * @note Parameter `executor` cannot be Executor::NONE. PlanC2C should be used
     * instead.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanC2C(const Pencil& pencil, MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);
};

/** Real-to-Complex Plan */
class PlanR2C final : public Plan {
public:
    /** @brief Real-to-Complex Plan constructor.
     *
     * @param[in]    dims                   Vector with global dimensions in
     *                                      reversed order. `dims.size()` must be 2 or 3
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    executor               Type of external FFT executor
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2C(const std::vector<int32_t>& dims, Executor executor,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE);

    /** Real-to-Complex Generic Plan constructor.
     *
     * @param[in]    ndims                  Number of dimensions: 2 or 3
     * @param[in]    dims                   Buffer of size `ndims` with global
     *                                      dimensions in reversed order.
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2C(int8_t ndims, const int32_t* dims, Executor executor,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE);

    /** @brief Real-to-Complex Plan constructor.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    executor               Type of external FFT executor
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2C(const Pencil& pencil, Executor executor,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE);
};

/** Real-to-Real Plan */
class PlanR2R final : public Plan {
public:
    /** @brief Real-to-Real Plan constructor.
     *
     * @param[in]    dims                   Vector with global dimensions in
     *                                      reversed order. `dims.size()` must be 2 or 3
     * @param[in]    kinds                  Real FFT kinds in reversed order.
     *                                      Can be empty vector if `executor` == Executor::NONE
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(const std::vector<int32_t>& dims,
        const std::vector<R2RKind>& kinds = std::vector<R2RKind>(),
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);

    /** @brief Real-to-Real Transpose-only Plan constructor.
     *
     * @param[in]    dims                   Vector with global dimensions in
     *                                      reversed order. `dims.size()` must be 2 or 3
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(const std::vector<int32_t>& dims, Precision precision,
        Effort effort = Effort::ESTIMATE);

    /** @brief Real-to-Real Generic Plan constructor.
     *
     * @param[in]    ndims                  Number of dimensions: 2 or 3
     * @param[in]    dims                   Buffer of size `ndims` with global
     *                                      dimensions in reversed order.
     * @param[in]    kinds                  Buffer of size `ndims` with Real FFT
     *                                      kinds in reversed order.
     *                                      Can be nullptr if `executor` == Executor::NONE
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(int8_t ndims, const int32_t* dims,
        const R2RKind* kinds = nullptr,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);

    /** @brief Real-to-Real Transpose-only Plan constructor.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(const Pencil& pencil, Precision precision,
        Effort effort = Effort::ESTIMATE);

    /** @brief Real-to-Real Plan constructor.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    kinds                  Real FFT kinds in reversed order.
     *                                      Can be empty vector if `executor` == Executor::NONE
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(const Pencil& pencil, const std::vector<R2RKind>& kinds,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);

    /** @brief Real-to-Real Generic Plan constructor.
     *
     * @param[in]    pencil                 Initialized Pencil object.
     * @param[in]    kinds                  Buffer of size `ndims` with Real FFT
     *                                      kinds in reversed order. Can be nullptr if `executor` == Executor::NONE
     * @param[in]    comm                   MPI communicator: `MPI_COMM_WORLD` or
     *                                      Cartesian communicator
     * @param[in]    precision              Precision of transform.
     * @param[in]    effort                 How thoroughly `dtFFT` searches for
     *                                      the optimal plan
     * @param[in]    executor               Type of external FFT executor.
     *
     * @throws Exception In case error occurs during plan creation
     */
    explicit PlanR2R(const Pencil& pencil, const R2RKind* kinds = nullptr,
        MPI_Comm comm = MPI_COMM_WORLD,
        Precision precision = Precision::DOUBLE,
        Effort effort = Effort::ESTIMATE,
        Executor executor = Executor::NONE);
};
} // namespace dtfft

#endif // DTFFT_HPP
