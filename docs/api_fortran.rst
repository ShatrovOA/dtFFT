.. _f_link:

#####################
Fortran API Reference
#####################

The ``dtFFT`` library provides a header file, ``dtfft.f03``, which can be included in Fortran source files using ``#include "dtfft.f03"``.
This file defines the ``DTFFT_CHECK`` macro for checking error codes returned by library functions:

.. code-block:: fortran

  call plan%execute(a, b, DTFFT_EXECUTE_FORWARD, aux, error_code)
  DTFFT_CHECK(error_code)  ! Checks for execution errors

All library functionality is contained in the Fortran module ``dtfft``, which should be imported with the ``use dtfft`` statement in your code.

Error Codes
===========

.. f:function:: dtfft_get_error_string(error_code)

  Gets the string description of an error code

  :p integer(int32) error_code [in]: Error code to convert to string
  :r character(len=:), allocatable: Error string explaining error.

All error codes that ``dtFFT`` can return are listed below.

.. f:variable:: DTFFT_SUCCESS

  Successful execution

.. f:variable:: DTFFT_ERROR_MPI_FINALIZED

  MPI_Init is not called or MPI_Finalize has already been called

.. f:variable:: DTFFT_ERROR_PLAN_NOT_CREATED

  Plan not created

.. f:variable:: DTFFT_ERROR_INVALID_TRANSPOSE_TYPE

  Invalid ``transpose_type`` provided

.. f:variable:: DTFFT_ERROR_INVALID_N_DIMENSIONS

  Invalid Number of dimensions provided. Valid options are 2 and 3

.. f:variable:: DTFFT_ERROR_INVALID_DIMENSION_SIZE

  One or more provided dimension sizes <= 0

.. f:variable:: DTFFT_ERROR_INVALID_COMM_TYPE

  Invalid communicator type provided

.. f:variable:: DTFFT_ERROR_INVALID_PRECISION

  Invalid ``precision`` parameter provided

.. f:variable:: DTFFT_ERROR_INVALID_EFFORT

  Invalid ``effort`` parameter provided

.. f:variable:: DTFFT_ERROR_INVALID_EXECUTOR

  Invalid ``executor`` parameter provided

.. f:variable:: DTFFT_ERROR_INVALID_COMM_DIMS

  Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine

.. f:variable:: DTFFT_ERROR_INVALID_COMM_FAST_DIM

  Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1

.. f:variable:: DTFFT_ERROR_MISSING_R2R_KINDS

  For R2R plan, ``kinds`` parameter must be passed if ``executor`` != :f:var:`DTFFT_EXECUTOR_NONE`

.. f:variable:: DTFFT_ERROR_INVALID_R2R_KINDS

  Invalid values detected in ``kinds`` parameter

.. f:variable:: DTFFT_ERROR_R2C_TRANSPOSE_PLAN

  Transpose plan is not supported in R2C, use C2C plan instead

.. f:variable:: DTFFT_ERROR_INPLACE_TRANSPOSE

  Inplace transpose is not supported

.. f:variable:: DTFFT_ERROR_INVALID_AUX

  Invalid ``aux`` buffer provided

.. f:variable:: DTFFT_ERROR_INVALID_DIM

  Invalid ``dim`` passed to :f:func:`get_pencil`

.. f:variable:: DTFFT_ERROR_INVALID_USAGE

  Invalid API Usage. Probably passed NULL pointer

.. f:variable:: DTFFT_ERROR_PLAN_IS_CREATED

  Trying to create already created plan

.. f:variable:: DTFFT_ERROR_ALLOC_FAILED

  Internal allocation failed

.. f:variable:: DTFFT_ERROR_FREE_FAILED

  Internal memory free failed

.. f:variable:: DTFFT_ERROR_INVALID_ALLOC_BYTES

  Invalid ``alloc_bytes`` provided

.. f:variable:: DTFFT_ERROR_DLOPEN_FAILED

  Dynamic library loading failed

.. f:variable:: DTFFT_ERROR_DLSYM_FAILED

  Dynamic library symbol lookup failed

.. f:variable:: DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH

  Sizes of starts and counts arrays passed to dtfft_pencil_t constructor do not match.

.. f:variable:: DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES

  Sizes of starts and counts < 2 or > 3 provided to dtfft_pencil_t constructor.

.. f:variable:: DTFFT_ERROR_PENCIL_INVALID_COUNTS

  Invalid counts provided to dtfft_pencil_t constructor.

.. f:variable:: DTFFT_ERROR_PENCIL_INVALID_STARTS

  Invalid starts provided to dtfft_pencil_t constructor.

.. f:variable:: DTFFT_ERROR_PENCIL_SHAPE_MISMATCH

  Processes have same lower bounds but different sizes in some dimensions.

.. f:variable:: DTFFT_ERROR_PENCIL_OVERLAP

  Pencil overlap detected, i.e. two processes share same part of global space

.. f:variable:: DTFFT_ERROR_PENCIL_NOT_CONTINUOUS

  Local pencils do not cover the global space without gaps.

.. f:variable:: DTFFT_ERROR_PENCIL_NOT_INITIALIZED

  Pencil is not initialized, i.e. constructor subroutine was not called

.. f:variable:: DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS

  Invalid `n_measure_warmup_iters` provided

.. f:variable:: DTFFT_ERROR_INVALID_MEASURE_ITERS

  Invalid `n_measure_iters` provided

.. f:variable:: DTFFT_ERROR_INVALID_REQUEST

  Invalid `dtfft_request_t` provided

.. f:variable:: DTFFT_ERROR_TRANSPOSE_ACTIVE

  Attempting to execute already active transposition

.. f:variable:: DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE

  Attempting to finalize non-active transposition

.. f:variable:: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED

  Selected ``executor`` do not support R2R FFTs

.. f:variable:: DTFFT_ERROR_GPU_INVALID_STREAM

  Invalid stream provided

.. f:variable:: DTFFT_ERROR_INVALID_BACKEND

  Invalid backend provided

.. f:variable:: DTFFT_ERROR_GPU_NOT_SET

  Multiple MPI Processes located on same host share same GPU which is not supported

.. f:variable:: DTFFT_ERROR_VKFFT_R2R_2D_PLAN

  When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions

.. f:variable:: DTFFT_ERROR_BACKENDS_DISABLED

  Passed ``effort`` ==  :f:var:`DTFFT_PATIENT` but all Backends has been disabled by :f:type:`dtfft_config_t`.

.. f:variable:: DTFFT_ERROR_NOT_DEVICE_PTR

  One of pointers passed to :f:func:`execute` or :f:func:`transpose` cannot be accessed from device

.. f:variable:: DTFFT_ERROR_NOT_NVSHMEM_PTR

  One of pointers passed to :f:func:`execute` or :f:func:`transpose` is not and ``NVSHMEM`` pointer

.. f:variable:: DTFFT_ERROR_INVALID_PLATFORM

  Invalid platform provided

.. f:variable:: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR

  Invalid executor provided for selected platform

.. f:variable:: DTFFT_ERROR_INVALID_PLATFORM_BACKEND

  Invalid backend provided for selected platform

Basic types
===========

dtfft_execute_t
---------------------

.. f:type:: dtfft_execute_t

  Enumerated type used to specify the direction of execution in the :f:func:`execute` method.

Type Parameters
_____________________

.. f:variable:: DTFFT_EXECUTE_FORWARD

  Forward execution: Performs the sequence XYZ to YXZ to ZXY.

.. f:variable:: DTFFT_EXECUTE_BACKWARD

  Backward execution: Performs the sequence ZXY to YXZ to XYZ.

------

dtfft_transpose_t
-----------------------

.. f:type:: dtfft_transpose_t

  Enumerated type used to specify the transposition direction in the :f:func:`transpose` method.

Type Parameters
_____________________

.. f:variable:: DTFFT_TRANSPOSE_X_TO_Y

  Transpose from Fortran X-aligned to Fortran Y-aligned

.. f:variable:: DTFFT_TRANSPOSE_Y_TO_X

  Transpose from Fortran Y-aligned to Fortran X-aligned

.. f:variable:: DTFFT_TRANSPOSE_Y_TO_Z

  Transpose from Fortran Y-aligned to Fortran Z aligned

.. f:variable:: DTFFT_TRANSPOSE_Z_TO_Y

  Transpose from Fortran Z-aligned to Fortran Y-aligned

.. f:variable:: DTFFT_TRANSPOSE_X_TO_Z

  Transpose from Fortran X-aligned to Fortran Z-aligned

.. note:: This value is valid to pass only in 3D Plan and value returned by :f:func:`get_z_slab_enabled` must be ``.true.``

.. f:variable:: DTFFT_TRANSPOSE_Z_TO_X

  Transpose from Fortran Z aligned to Fortran X aligned

.. note:: This value is valid to pass only in 3D Plan and value returned by :f:func:`get_z_slab_enabled` must be ``.true.``

------

dtfft_reshape_t
-----------------------

.. f:type:: dtfft_reshape_t

  Enumerated type used to specify the reshape direction in the :f:func:`reshape` method.

Type Parameters
_____________________

.. f:variable:: DTFFT_RESHAPE_X_BRICKS_TO_PENCILS

  Perform reshape to X-aligned pencils from X-aligned bricks

.. f:variable:: DTFFT_RESHAPE_X_PENCILS_TO_BRICKS

  Perform reshape to X-aligned bricks from X-aligned pencils

.. f:variable:: DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS

  Perform reshape to Z-aligned bricks from Z-aligned pencils

.. f:variable:: DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS

  Perform reshape to Z-aligned pencils from Z-aligned bricks

.. f:variable:: DTFFT_RESHAPE_Y_PENCILS_TO_BRICKS

  Perform reshape to Y-aligned bricks from Y-aligned pencils. This is an alias to :f:var:`DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS` for 2D plans

.. f:variable:: DTFFT_RESHAPE_Y_BRICKS_TO_PENCILS

  Perform reshape to Y-aligned pencils from Y-aligned bricks. This is an alias to :f:var:`DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS` for 2D plans

------

dtfft_executor_t
-----------------------

.. f:type:: dtfft_executor_t

  Type that specifies external FFT executor

Type Parameters
_____________________

.. f:variable:: DTFFT_EXECUTOR_NONE

  Do not create any FFT plans. Creates transpose only plan.

.. f:variable:: DTFFT_EXECUTOR_FFTW3

  FFTW3 Executor (Host only)

.. f:variable:: DTFFT_EXECUTOR_MKL

  MKL DFTI Executor (Host only)

.. f:variable:: DTFFT_EXECUTOR_CUFFT

  CUFFT Executor (GPU Only)

.. f:variable:: DTFFT_EXECUTOR_VKFFT

  VkFFT Executor (GPU Only)

Related Type functions
______________________

.. f:function:: dtfft_get_executor_string(executor)

  Gets the string description of an error code

  :p dtfft_executor_t executor [in]: Executor type to convert to string
  :r character(len=:), allocatable: String representation of dtfft_executor_t

------

dtfft_effort_t
-----------------------

.. f:type:: dtfft_effort_t

  Type that specifies effort that ``dtFFT`` should use when creating plan

Type Parameters
_____________________

.. f:variable:: DTFFT_ESTIMATE

  Create plan as fast as possible

.. f:variable:: DTFFT_MEASURE

  Will attempt to find best MPI Grid decomposition.
  Passing this flag and MPI Communicator with cartesian topology to any plan constructor is same as :f:var:`DTFFT_ESTIMATE`

.. f:variable:: DTFFT_PATIENT

  Same as :f:var:`DTFFT_MEASURE` plus cycle through various send and recieve MPI_Datatypes.

  For GPU Build of the library this value will cycle through enabled GPU Backend in order to find the fastest.

------

dtfft_precision_t
-----------------------

.. f:type:: dtfft_precision_t

  Type that specifies precision of ``dtFFT`` plan

Type Parameters
_____________________

.. f:variable:: DTFFT_SINGLE

  Use Single precision

.. f:variable:: DTFFT_DOUBLE

  Use Double precision

Related Type functions
______________________

.. f:function:: dtfft_get_precision_string(precision)

  Gets the string description of an error code

  :p dtfft_precision_t precision [in]: Precision level to convert to string
  :r character(len=:), allocatable: String representation of dtfft_precision_t

------

dtfft_r2r_kind_t
-----------------------

.. f:type:: dtfft_r2r_kind_t

  Type that specifies various kinds of R2R FFTs

Type Parameters
_____________________

.. f:variable:: DTFFT_DCT_1

  DCT-I (Logical N=2*(n-1), inverse is :f:var:`DTFFT_DCT_1`)

.. f:variable:: DTFFT_DCT_2

  DCT-II (Logical N=2*n, inverse is :f:var:`DTFFT_DCT_3`)

.. f:variable:: DTFFT_DCT_3

  DCT-III (Logical N=2*n, inverse is :f:var:`DTFFT_DCT_2`)

.. f:variable:: DTFFT_DCT_4

  DCT-IV (Logical N=2*n, inverse is :f:var:`DTFFT_DCT_4`)

.. f:variable:: DTFFT_DST_1

  DST-I (Logical N=2*(n+1), inverse is :f:var:`DTFFT_DST_1`)

.. f:variable:: DTFFT_DST_2

  DST-II (Logical N=2*n, inverse is :f:var:`DTFFT_DST_3`)

.. f:variable:: DTFFT_DST_3

  DST-III (Logical N=2*n, inverse is :f:var:`DTFFT_DST_2` )

.. f:variable:: DTFFT_DST_4

  DST-IV (Logical N=2*n, inverse is :f:var:`DTFFT_DST_4`)

------

dtfft_backend_t
-----------------------

.. f:type:: dtfft_backend_t

  Type that specifies various GPU Backend present in ``dtFFT``

Type Parameters
_____________________

.. f:variable:: DTFFT_BACKEND_MPI_DATATYPE

  Backend that uses MPI datatypes.

  Not really recommended to use when ``platform`` is :f:var:`DTFFT_PLATFORM_CUDA`, since it is a million times slower than other backends. It is present here just to show how slow MPI Datatypes are for GPU usage

.. f:variable:: DTFFT_BACKEND_MPI_P2P

  MPI peer-to-peer algorithm

.. f:variable:: DTFFT_BACKEND_MPI_P2P_PIPELINED

  MPI peer-to-peer algorithm with overlapping data copying and unpacking

.. f:variable:: DTFFT_BACKEND_MPI_A2A

  MPI backend using MPI_Alltoallv

.. f:variable:: DTFFT_BACKEND_MPI_RMA

  MPI RMA backend

.. f:variable:: DTFFT_BACKEND_MPI_RMA_PIPELINED

  Pipelined MPI RMA backend

.. f:variable:: DTFFT_BACKEND_MPI_P2P_SCHEDULED

  MPI peer-to-peer algorithm with scheduled communication

.. f:variable:: DTFFT_BACKEND_NCCL

  NCCL backend

.. f:variable:: DTFFT_BACKEND_NCCL_PIPELINED

  NCCL backend with overlapping data copying and unpacking

.. f:variable:: DTFFT_BACKEND_CUFFTMP

  cuFFTMp backend

.. f:variable:: DTFFT_BACKEND_CUFFTMP_PIPELINED

  cuFFTMp backend that uses additional buffer to avoid extra copy and gain performance.

Related Type functions
_______________________

.. f:function:: dtfft_get_backend_string(backend)

  Gets the string description of a GPU backend

  :p dtfft_backend_t backend [in]:
    Backend
  :r character(len=:), allocatable string:
    Backend string

.. f:function:: dtfft_get_backend_pipelined(backend)

  Check if backend is pipelined

  :p dtfft_backend_t backend [in]: Backend to check
  :r logical:
    ``.true.`` if backend is pipelined, ``.false.`` otherwise

------

dtfft_config_t
-----------------------

.. f:type:: dtfft_config_t

  Type that can be used to set additional configuration parameters to ``dtFFT``

  :f logical enable_log:
    Should dtFFT print additional information during plan creation or not.

    Default is ``.false.``

  :f logical enable_z_slab:
    Should ``dtFFT`` use Z-slab optimization or not.

    Default is ``.true.``

    One should consider disabling Z-slab optimization in order to resolve :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error or when underlying FFT implementation of 2D plan is too slow.

    In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.

  :f logical enable_y_slab:
    Should ``dtFFT`` use Y-slab optimization or not.

    Default is ``.false.``

    One should consider disabling Y-slab optimization in order to resolve :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error or when underlying FFT implementation of 2D plan is too slow.

    In all other cases it is considered that Y-slab is always faster, since it reduces number of data transpositions.

  :f integer(int32) n_measure_warmup_iters:
    Number of warmup iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.

    Default is 2

  :f integer(int32) n_measure_iters:
    Number of iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.

    Default is 5

    When ``dtFFT`` is built with CUDA support, this value also used to determine number
    of iterations when selecting block of threads for NVRTC transpose kernel

  :f type(dtfft_platform_t) platform:

    Selects platform to execute plan.

    Default is :f:var:`DTFFT_PLATFORM_HOST`

    This option is only defined in a build with device support. Even when dtFFT is built with device support, it does not necessarily mean that all plans must be device-related.

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f type(dtfft_stream_t) stream:

    Main CUDA stream that will be used in dtFFT.

    This parameter is a placeholder for user to set custom stream.

    Stream that is actually used by dtFFT plan is returned by f:func:`get_stream` function.

    When user sets stream he is responsible of destroying it.

    Stream must not be destroyed before call to :f:func:`destroy`.

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f type(dtfft_backend_t) backend:

    Backend that will be used by dtFFT when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.

    Default for HOST platform is :f:var:`DTFFT_BACKEND_MPI_DATATYPE`.

    Default for CUDA platform is :f:var:`DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise :f:var:`DTFFT_BACKEND_MPI_P2P`.

  :f type(dtfft_backend_t) reshape_backend:

    Backend that will be used by dtFFT for reshape operations.

    Defaults are same as `backend`.

  :f logical enable_datatype_backend:
    Should ``DTFFT_BACKEND_MPI_DATATYPE`` be considered for autotuning when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.

    Default is ``.true.``

    This option only works when ``platform`` is ``DTFFT_PLATFORM_HOST``.
    When ``platform`` is ``DTFFT_PLATFORM_CUDA``, ``DTFFT_BACKEND_MPI_DATATYPE`` is always disabled during autotuning.

  :f logical enable_mpi_backends:

    Should MPI Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.

    Default is ``.false.``

    The following applies only to CUDA builds.
    MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
    It was noticed that during plan autotuning GPU memory not being freed completely.
    For example:
    1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
    Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.

    One of the workarounds is to disable MPI Backends by default, which is done here.

    Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to ``mpiexec``,
    but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms

  :f logical enable_pipelined_backends:

    Should pipelined backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.

    Default is ``.true.``

  :f logical enable_nccl_backends:
    Should NCCL Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.

    Default is ``.true.``

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_nvshmem_backends:
    Should NVSHMEM Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.

    Default is ``.true.``

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_kernel_autotune:
    Should dtFFT try to optimize kernel launch parameters during plan creation when ``effort`` is below ``DTFFT_EXHAUSTIVE``.

    Default is ``.false.``

    Kernel optimization is always enabled for ``DTFFT_EXHAUSTIVE`` effort level.
    Setting this option to true enables kernel optimization for lower effort levels (``DTFFT_ESTIMATE``, ``DTFFT_MEASURE``, ``DTFFT_PATIENT``).
    This may increase plan creation time but can improve runtime performance.
    Since kernel optimization is performed without data transfers, the time increase is usually minimal.

    This option can also be controlled using the ``DTFFT_ENABLE_KERNEL_AUTOTUNE`` environment variable.

  :f logical enable_fourier_reshape:
    Should dtFFT execute reshapes from pencils to bricks and vice versa in Fourier space during calls to ``execute``.

    Default is ``.false.``

    When enabled, data will be in brick layout in Fourier space, which may be useful for certain operations
    between forward and backward transforms. However, this requires additional data transpositions
    and will reduce overall FFT performance.

    This option can also be controlled using the ``DTFFT_ENABLE_FOURIER_RESHAPE`` environment variable.

Related Type functions
_______________________

.. f:function:: dtfft_create_config(config)

  Creates dtfft_config_t object and sets default values to it.

  :p dtfft_config_t config [out]: Constructed ``dtFFT`` config ready to be set by call to :f:func:`dtfft_set_config`

------

.. f:function:: dtfft_config_t([enable_log, enable_z_slab, enable_y_slab, n_measure_warmup_iters, n_measure_iters, backend, reshape_backend, enable_datatype_backend, enable_mpi_backends, enable_pipelined_backends, enable_kernel_autotune, enable_fourier_reshape])

  Type bound constructor

  .. note:: This version of constructor is only present in the API when ``dtFFT`` was compiled without CUDA Support.

  :o logical enable_log [in, optional]:
    Should dtFFT print additional information during plan creation or not.
  :o logical enable_z_slab [in, optional]:
    Should dtFFT use Z-slab optimization or not.
  :o logical enable_y_slab [in, optional]:
    Should dtFFT use Y-slab optimization or not.
  :o integer(int32) n_measure_warmup_iters [in, optional]:
    Number of warmup iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.
  :o integer(int32) n_measure_iters [in, optional]:
    Number of iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.
  :o dtfft_backend_t backend [in, optional]:
    Backend that will be used by dtFFT when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  :o dtfft_backend_t reshape_backend [in, optional]:
    Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  :o logical enable_datatype_backend [in, optional]:
    Should ``DTFFT_BACKEND_MPI_DATATYPE`` be considered for autotuning when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_mpi_backends [in, optional]:
    Should MPI Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_pipelined_backends [in, optional]:
    Should pipelined backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_kernel_autotune [in, optional]:
    Should dtFFT try to optimize kernel launch parameters during plan creation when ``effort`` is below ``DTFFT_EXHAUSTIVE``.
  :o logical enable_fourier_reshape [in, optional]:
    Should dtFFT execute reshapes from pencils to bricks in Fourier space.
  :r dtfft_config_t: Constructed ``dtFFT`` config ready to be set by call to :f:func:`dtfft_set_config`

------

.. f:function:: dtfft_config_t([enable_log, enable_z_slab, enable_y_slab, n_measure_warmup_iters, n_measure_iters, platform, stream, backend, reshape_backend, enable_datatype_backend, enable_mpi_backends, enable_pipelined_backends, enable_nccl_backends, enable_nvshmem_backends, enable_kernel_autotune, enable_fourier_reshape])

  Type bound constructor

  .. note:: This version of constructor is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :o logical enable_log [in, optional]:
    Should dtFFT print additional information during plan creation or not.
  :o logical enable_z_slab [in, optional]:
    Should dtFFT use Z-slab optimization or not.
  :o logical enable_y_slab [in, optional]:
    Should dtFFT use Y-slab optimization or not.
  :o integer(int32) n_measure_warmup_iters [in, optional]:
    Number of warmup iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.
  :o integer(int32) n_measure_iters [in, optional]:
    Number of iterations to execute during backend and kernel autotuning when effort level is ``DTFFT_MEASURE`` or higher.
  :o dtfft_platform_t platform [in, optional]:
    Selects platform to execute plan.
  :o dtfft_stream_t stream [in, optional]:
    Main CUDA stream that will be used in dtFFT.
  :o dtfft_backend_t backend [in, optional]:
    Backend that will be used by dtFFT when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  :o dtfft_backend_t reshape_backend [in, optional]:
    Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  :o logical enable_datatype_backend [in, optional]:
    Should ``DTFFT_BACKEND_MPI_DATATYPE`` be considered for autotuning when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_mpi_backends [in, optional]:
    Should MPI Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_pipelined_backends [in, optional]:
    Should pipelined backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_nccl_backends [in, optional]:
    Should NCCL Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_nvshmem_backends [in, optional]:
    Should NVSHMEM Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or ``DTFFT_EXHAUSTIVE``.
  :o logical enable_kernel_autotune [in, optional]:
    Should dtFFT try to optimize kernel launch parameters during plan creation when ``effort`` is below ``DTFFT_EXHAUSTIVE``.
  :o logical enable_fourier_reshape [in, optional]:
    Should dtFFT execute reshapes from pencils to bricks in Fourier space.
  :r dtfft_config_t: Constructed ``dtFFT`` config ready to be set by call to :f:func:`dtfft_set_config`

------

.. f:subroutine:: dtfft_set_config(config[, error_code])

  Set configuration values to ``dtFFT``.

  In order to take effect should be called before plan creation

  :p dtfft_config_t config [in]:
    Config to set
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

dtfft_pencil_t
--------------

.. f:type:: dtfft_pencil_t

  Type used to hold pencil decomposition info.

  There are two ways users might find pencils useful inside dtFFT:

  1. To create a Plan using users's own grid decomposition, you can pass Pencil to Plan constructors.
  2. To obtain Pencil from Plan in all possible layouts, in order to run FFT not available in dtFFT.

  When pencil is returned from :f:func:`get_pencil`, all pencil properties are defined.

  :f int(int8) dim: Aligned dimension id starting from 1
  :f int(int8) ndims: Number of dimensions in a pencil
  :f int(int32) starts(:) [allocatable]: Local starts in natural Fortran order
  :f int(int32) counts(:) [allocatable]: Local counts in natural Fortran order
  :f int(int64) size: Total number of elements in a pencil

Related Type functions
_______________________

.. f:function:: dtfft_pencil_t(starts, counts)

  Type bound constructor

  :p int(int32) starts(:) [in]: Local starts in natural Fortran order
  :p int(int32) counts(:) [in]: Local counts in natural Fortran order

------

dtfft_platform_t
----------------

.. f:type:: dtfft_platform_t

  Type that specifies the execution platform, such as Host, CUDA, or HIP

Type Parameters
_______________

.. f:variable:: DTFFT_PLATFORM_HOST

  Create HOST-related plan

.. f:variable:: DTFFT_PLATFORM_CUDA

  Create CUDA-related plan

------

dtfft_stream_t
--------------

.. f:type:: dtfft_stream_t

  ``dtFFT`` stream representation.

  :f type(c_ptr) stream:
    Actual stream pointer

Related Type functions
______________________

.. f:function:: dtfft_stream_t(stream)

  C-pointer constructor

  :p type(c_ptr) stream [in]: Stream pointer
  :r dtfft_stream_t: Stream object

.. f:function:: dtfft_stream_t(stream)

  CUDA-Fortran stream constructor

  :p integer(cuda_stream_kind) stream [in]: CUDA-Fortran stream
  :r dtfft_stream_t: Stream object

.. f:function:: dtfft_get_cuda_stream(stream)

  Gets CUDA stream from dtfft_stream_t object

  :p dtfft_stream_t stream [in]: Stream object
  :r integer(cuda_stream_kind): CUDA-Fortran stream

dtfft_request_t
---------------

.. f:type:: dtfft_request_t

  Helper type to manage asynchronous operations

  See :f:func:`transpose_start`, :f:func:`transpose_end`, :f:func:`reshape_start`, :f:func:`reshape_end`


Version handling
================

Parameters
----------

.. f:variable:: DTFFT_VERSION_MAJOR

  ``dtFFT`` Major Version

.. f:variable:: DTFFT_VERSION_MINOR

  ``dtFFT`` Minor Version

.. f:variable:: DTFFT_VERSION_PATCH

  ``dtFFT`` Patch Version

.. f:variable:: DTFFT_VERSION_CODE

  ``dtFFT`` Version Code. Can be used in Version comparison

------

Functions
---------

.. f:function:: dtfft_get_version

  :r integer(int32):
    Version Code defined during compilation

.. f:function:: dtfft_get_version(major, minor, patch)

  Computes Version Code based on Major, Minor and Patch versions

  :p integer(int32) major: Major version
  :p integer(int32) minor: Minor version
  :p integer(int32) patch: Patch version
  :r integer(int32):
    Requested Version Code

------

Abstract plan
=============

.. f:type:: dtfft_plan_t

  Abstract class for all ``dtFFT`` plans

Type bound procedures
-----------------------

reshape
_______

.. f:subroutine:: reshape(in, out, reshape_type [, aux, error_code])

  Performs reshape from bricks to pencils layout or vice versa

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_reshape_t reshape_type [in]:
    Type of reshape
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

reshape_ptr
___________

.. f:subroutine:: reshape_ptr(in, out, reshape_type, aux [, error_code])

  Performs reshape from bricks to pencils layout or vice versa using pointers

  :p type(c_ptr) in [in]:
    Incoming pointer
  :p type(c_ptr) out [in]:
    Resulting pointer
  :p dtfft_reshape_t reshape_type [in]:
    Type of reshape
  :p type(c_ptr) aux [in]:
    Auxiliary pointer. Not optional. Must pass ``c_null_ptr`` if not used. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

reshape_start
_____________

.. f:function:: reshape_start(in, out, reshape_type [, aux, error_code]) result(request)

  Starts asynchronous reshape operation

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_reshape_t reshape_type [in]:
    Type of reshape
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_request_t request:
    Asynchronous handle describing started reshape operation

------

reshape_start_ptr
_________________

.. f:function:: reshape_start_ptr(in, out, reshape_type, aux [, error_code]) result(request)

  Starts asynchronous reshape operation using pointers

  :p type(c_ptr) in [in]:
    Incoming pointer
  :p type(c_ptr) out [in]:
    Resulting pointer
  :p dtfft_reshape_t reshape_type [in]:
    Type of reshape
  :p type(c_ptr) aux [in]:
    Auxiliary pointer. Not optional. Must pass ``c_null_ptr`` if not used. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_request_t request:
    Asynchronous handle describing started reshape operation

------

reshape_end
___________

.. f:subroutine:: reshape_end(request [, error_code])

  Ends previously started reshape operation

  :p dtfft_request_t request [inout]:
    Asynchronous handle describing started reshape operation
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

transpose
_________

.. f:subroutine:: transpose(in, out, transpose_type [, aux, error_code])

  Performs single transposition

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_transpose_t transpose_type [in]:
    Type of transposition
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

transpose_ptr
_____________

.. f:subroutine:: transpose_ptr(in, out, transpose_type, aux [, error_code])

  Performs single transposition

  :p type(c_ptr) in [in]:
    Incoming pointer
  :p type(c_ptr) out [in]:
    Resulting pointer
  :p dtfft_transpose_t transpose_type [in]:
    Type of transposition
  :p type(c_ptr) aux [in]:
    Auxiliary pointer. Not optional. Must pass ``c_null_ptr`` if not used. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

transpose_start
_______________

.. f:function:: transpose_start(in, out, transpose_type [, aux, error_code])

  Starts an asynchronous transpose operation.

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_transpose_t transpose_type [in]:
    Type of transposition
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_request_t request:
    Asynchronous handle describing started transpose operation

------

transpose_start_ptr
___________________

.. f:function:: transpose_start_ptr(in, out, transpose_type, aux [, error_code])

  Starts an asynchronous transpose operation using type(c_ptr) pointers instead of buffers

  :p type(c_ptr) in [in]:
    Incoming pointer
  :p type(c_ptr) out [in]:
    Resulting pointer
  :p dtfft_transpose_t transpose_type [in]:
    Type of transposition
  :p type(c_ptr) aux [in]:
    Auxiliary pointer. Not optional. Must pass ``c_null_ptr`` if not used. If provided, size must be at least ``alloc_size`` from :f:func:`get_local_sizes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_request_t request:
    Asynchronous handle describing started transpose operation

------

transpose_end
_____________

.. f:subroutine:: transpose_end(request [, error_code])

  Ends previously started transposition

  :p dtfft_request_t request [inout]:
    Asynchronous handle describing started transpose operation
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

execute
_______

.. f:subroutine:: execute(in, out, execute_type [, aux, error_code])

  Executes plan

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_execute_t execute_type [in]:
    Type of execution
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer. If provided, size must be at least the value returned by :f:func:`get_aux_size`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

execute_ptr
___________

.. f:subroutine:: execute_ptr(in, out, execute_type, aux [, error_code])

  Executes plan

  :p type(c_ptr) in [in]:
    Incoming pointer
  :p type(c_ptr) out [in]:
    Resulting pointer
  :p dtfft_execute_t execute_type [in]:
    Type of execution
  :p type(c_ptr) aux [in]:
    Auxiliary pointer. Not optional. Must pass ``c_null_ptr`` if not used. If provided, size must be at least the value returned by :f:func:`get_aux_bytes`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

destroy
_______

.. f:subroutine:: destroy( [error_code] )

  Destroys plan, frees all memory

  :o integer(int32) error_code [out]: Optional error code returned to user

------

get_local_sizes
_______________

.. f:subroutine:: get_local_sizes([in_starts, in_counts, out_starts, out_counts, alloc_size, error_code])

  Obtain local starts and counts in `real` and `fourier` spaces

  :o integer(int32) in_starts(:) [out, optional]:
    Start indexes in `real` space (0-based)
  :o integer(int32) in_counts(:) [out, optional]:
    Number of elements in `real` space
  :o integer(int32) out_starts(:) [out, optional]:
    Start indexes in `fourier` space (0-based)
  :o integer(int32) out_counts(:) [out, optional]:
    Number of elements in `fourier` space
  :o integer(int64) alloc_size(:) [out, optional]:
    Minimum number of elements to be allocated for ``in``, ``out`` buffers required by :f:func:`execute`.
    This also returns minimum number of elements required for ``aux`` buffer required by :f:func:`transpose` and :f:func:`reshape` in case underlying backend is pipelined.
    Minimum number of ``aux`` elements required by :f:func:`execute` can be obtained by calling :f:func:`get_aux_size`.
    Size of each element in bytes can be obtained by calling :f:func:`get_element_size`.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

get_alloc_size
______________

.. f:function:: get_alloc_size([error_code])

  Wrapper around :f:func:`get_local_sizes` to obtain number of elements only

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64):
    Minimum number of elements to be allocated for ``in``, ``out`` buffers required by :f:func:`execute`.
    This also returns minimum number of elements required for ``aux`` buffer required by :f:func:`transpose` and :f:func:`reshape`.
    Minimum number of ``aux`` elements required by :f:func:`execute` can be obtained by calling :f:func:`get_aux_size`.

------

get_element_size
________________

.. f:function:: get_element_size( [error_code] )

  Returns number of bytes required to store single element.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64): Size of element in bytes

------

get_alloc_bytes
_______________

.. f:function:: get_alloc_bytes([error_code])

  Returns minimum number of bytes required for in and out buffers.
  This also returns minimum number of bytes required for aux buffer in transpose and reshape operations.
  Minimum number of aux bytes required by execute can be obtained by calling get_aux_bytes.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64):
    Minimum number of bytes to be allocated for ``in`` and ``out`` buffers required by execute.
    This also returns minimum number of bytes required for ``aux`` buffer required by transpose and reshape.

------

get_aux_size
____________

.. f:function:: get_aux_size([error_code])

  Returns minimum number of elements required for auxiliary buffer which may be different from ``alloc_size`` when backend is pipelined.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64):
    Minimum number of elements required for auxiliary buffer.

------

get_aux_bytes
_____________

.. f:function:: get_aux_bytes([error_code])

  Returns minimum number of bytes required for auxiliary buffer.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64):
    Minimum number of bytes required for auxiliary buffer.

------

mem_alloc
_________

Allocates memory tailored to the specific needs of the plan.

.. f:subroutine:: mem_alloc(alloc_size, ptr [, lbound, error_code])

  :p integer(int64) alloc_size [in]: Number of elements to allocate
  :p type(*) ptr(:) [pointer, out]: 1D pointer to allocate
  :o integer(int32) lbound [in, optional]: Lower boundary of allocated pointer
  :o integer(int32) error_code [out, optional]: Optional error code returned to user

------

.. f:subroutine:: mem_alloc(alloc_size, ptr, sizes [, lbounds, error_code])

  :p integer(int64) alloc_size [in]: Number of elements to allocate
  :p type(*) ptr(..) [pointer, out]: 2D or 3D pointer to allocate
  :p integer(int32) sizes(:) [in]: Sizes of each dimension in natural Fortran order. Size of ``sizes`` must match rank of pointer.
  :o integer(int32) lbounds(:) [in, optional]: Lower boundaries of allocated pointer. Size of ``lbounds`` must match rank of pointer.
  :o integer(int32) error_code [out, optional]: Optional error code returned to user

------

mem_alloc_ptr
_____________

Allocates memory tailored to the specific needs of the plan.

.. f:function:: mem_alloc_ptr(alloc_bytes [, error_code])

  :p integer(int64) alloc_bytes [in]: Number of bytes to allocate
  :o integer(int32) error_code [out, optional]: Optional error code returned to user
  :r type(c_ptr): Allocated pointer

------

mem_free
________

Frees memory previously allocated by :f:func:`mem_alloc`.


.. f:subroutine:: mem_free(ptr[, error_code])

  :p type(*) ptr(..) [inout]: Pointer allocated with ``mem_alloc``
  :o integer(int32) error_code [out, optional]: Optional error code returned to user

------

mem_free_ptr
____________

Frees memory previously allocated by :f:func:`mem_alloc_ptr`.


.. f:subroutine:: mem_free_ptr(ptr[, error_code])

  :p type(c_ptr) ptr [in]: Pointer allocated with ``mem_alloc_ptr``
  :o integer(int32) error_code [out, optional]: Optional error code returned to user

------

get_z_slab_enabled
__________________

.. f:function:: get_z_slab_enabled([error_code])

  Returns logical value is Z-slab optimization enabled internally

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r logical:
    Boolean value if Z-slab is used.

------

get_y_slab_enabled
__________________

.. f:function:: get_y_slab_enabled([error_code])

  Returns logical value is Y-slab optimization enabled internally

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r logical:
    Boolean value if Y-slab is used.

------

get_pencil
__________

.. f:function:: get_pencil(layout[, error_code])

  Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation,
  that is unavailable in ``dtFFT``.

  :p type(dtfft_layout_t) layout [in]:
    Required layout
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_pencil_t: Pencil data

------

report
______

.. f:subroutine:: report([error_code])

  Prints plan-related information to stdout

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

get_executor
____________

.. f:function:: get_executor([error_code])

  Returns FFT Executor associated with plan

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_executor_t: FFT Executor used by this plan.

------

get_precision
_____________

.. f:function:: get_precision([error_code])

  Returns precision of the plan

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r dtfft_precision_t: Precision of the plan.

------

get_dims
________

.. f:subroutine:: get_dims(dims [, error_code])

  Returns global dimensions of the plan.

  :p integer(int32) dims [out, pointer]:
    Global dimensions of the plan.

    Users should not attempt to change values in this pointer.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

get_grid_dims
_____________

.. f:subroutine:: get_grid_dims(dims [, error_code])

  Returns grid decomposition dimensions

  :p integer(int32) dims [out, pointer]:
    Grid dimensions of the plan.

    Users should not attempt to change values in this pointer.
  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

------

get_backend
___________

.. f:function:: get_backend([error_code])

  Returns backend that is used during data transpositions.

  If ``effort`` is :f:var:`DTFFT_ESTIMATE` or :f:var:`DTFFT_MEASURE`, returns the value set by :f:func:`dtfft_set_config`
  or via environment variable ``DTFFT_BACKEND``.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

  :r dtfft_backend_t: Selected backend

------

get_reshape_backend
___________________

.. f:function:: get_reshape_backend([error_code])

  Returns backend that is used during data reshapes from bricks to pencils and vice versa.

  If ``effort`` is :f:var:`DTFFT_ESTIMATE` or :f:var:`DTFFT_MEASURE`, returns the value set by :f:func:`dtfft_set_config`
  or via environment variable ``DTFFT_RESHAPE_BACKEND``.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

  :r dtfft_backend_t: Selected backend

------

get_platform
____________

.. f:function:: get_platform([error_code])

  Returns execution platform of the plan (HOST or CUDA)

  .. note:: This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

  :r dtfft_platform_t: Execution platform

------

get_stream
__________

This method is overloaded to support both CUDA and dtFFT streams.

.. f:subroutine:: get_stream(stream[, error_code])

  Returns CUDA stream associated with plan

  .. note:: This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :p integer(cuda_stream_kind): CUDA stream associated with plan

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

.. f:subroutine:: get_stream(stream[, error_code])

  Returns dtFFT stream associated with plan

  .. note:: This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :p type(dtfft_stream_t): dtFFT stream associated with plan

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

Real-to-Real plan
=================

.. f:type:: dtfft_plan_r2r_t

  Real-to-real plan class

  Extends :f:type:`dtfft_plan_t`

Type bound procedures
---------------------

.. _create_r2r:

create
______

.. f:subroutine:: create(dims [, kinds, comm, precision, effort, executor, error_code])

  R2R Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o dtfft_r2r_kind_t kinds(:) [in, optional]: Kinds of R2R transforms, default = empty.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user

------

.. f:subroutine:: create(pencil [, kinds, comm, precision, effort, executor, error_code])

  R2R Plan Constructor using local pencil information

  :p dtfft_pencil_t pencil[in]: Local pencil of data to be transformed
  :o dtfft_r2r_kind_t kinds(:) [in, optional]: Kinds of R2R transforms, default = empty.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user

------

Complex-to-Complex plan
=======================

.. f:type:: dtfft_plan_c2c_t

  Complex-to-complex plan class

  Extends :f:type:`dtfft_plan_t`

Type bound procedures
---------------------

.. _create_c2c:

create
______

.. f:subroutine:: create(dims [, comm, precision, effort, executor, error_code])

  C2C Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user

------

.. f:subroutine:: create(pencil [, comm, precision, effort, executor, error_code])

  C2C Plan Constructor using local pencil information

  :p dtfft_pencil_t pencil[in]: Local pencil of data to be transformed
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user

------

Real-to-Complex plan
====================

.. f:type:: dtfft_plan_r2c_t

  Real-to-complex plan class

  Extends :f:type:`dtfft_plan_t`

Type bound procedures
---------------------

.. _create_r2c:

create
______

.. f:subroutine:: create(dims [, comm, precision, effort, executor, error_code])

  R2C Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user

------

.. f:subroutine:: create(pencil[, comm, precision, effort, executor, error_code])

  R2C Plan Constructor using local pencil information

  :p dtfft_pencil_t pencil[in]: Local pencil of data to be transformed
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user



