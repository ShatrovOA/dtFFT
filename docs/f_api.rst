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

.. f:variable:: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED

  Selected ``executor`` do not support R2R FFTs

.. f:variable:: DTFFT_ERROR_GPU_INVALID_STREAM

  Invalid stream provided

.. f:variable:: DTFFT_ERROR_GPU_INVALID_BACKEND

  Invalid GPU backend provided

.. f:variable:: DTFFT_ERROR_GPU_NOT_SET

  Multiple MPI Processes located on same host share same GPU which is not supported

.. f:variable:: DTFFT_ERROR_VKFFT_R2R_2D_PLAN

  When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions

.. f:variable:: DTFFT_ERROR_GPU_BACKENDS_DISABLED

  Passed ``effort`` ==  :f:var:`DTFFT_PATIENT` but all GPU Backends has been disabled by :f:type:`dtfft_config_t`.

.. f:variable:: DTFFT_ERROR_NOT_DEVICE_PTR

  One of pointers passed to :f:func:`execute` or :f:func:`transpose` cannot be accessed from device

.. f:variable:: DTFFT_ERROR_NOT_NVSHMEM_PTR

  One of pointers passed to :f:func:`execute` or :f:func:`transpose` is not and ``NVSHMEM`` pointer

.. f:variable:: DTFFT_ERROR_INVALID_PLATFORM

  Invalid platform provided

.. f:variable:: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE

  Invalid executor provided for selected platform

Basic types
===========

dtfft_execute_type_t
---------------------

.. f:type:: dtfft_execute_type_t

  Enumerated type used to specify the direction of execution in the :f:func:`execute` method.

Type Parameters
_____________________

.. f:variable:: DTFFT_EXECUTE_FORWARD

  Forward execution: Performs the sequence XYZ to YXZ to ZXY.

.. f:variable:: DTFFT_EXECUTE_BACKWARD

  Backward execution: Performs the sequence ZXY to YXZ to XYZ.

------

dtfft_transpose_type_t
-----------------------

.. f:type:: dtfft_transpose_type_t

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

dtfft_gpu_backend_t
-----------------------

.. f:type:: dtfft_gpu_backend_t

  Type that specifies various GPU Backend present in ``dtFFT``

.. note:: This type is only present in the API when ``dtFFT`` was compiled with CUDA Support.

Type Parameters
_____________________

.. f:variable:: DTFFT_GPU_BACKEND_MPI_DATATYPE

  Backend that uses MPI datatypes.

  Not really recommended to use, since it is a million times slower than other backends.
  It is present here just to show how slow MPI Datatypes are for GPU usage

.. f:variable:: DTFFT_GPU_BACKEND_MPI_P2P

  MPI peer-to-peer algorithm

.. f:variable:: DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED

  MPI peer-to-peer algorithm with overlapping data copying and unpacking

.. f:variable:: DTFFT_GPU_BACKEND_MPI_A2A

  MPI backend using MPI_Alltoallv

.. f:variable:: DTFFT_GPU_BACKEND_NCCL

  NCCL backend

.. f:variable:: DTFFT_GPU_BACKEND_NCCL_PIPELINED

  NCCL backend with overlapping data copying and unpacking

.. f:variable:: DTFFT_GPU_BACKEND_CUFFTMP

  cuFFTMp backend

Related Type functions
_______________________

.. f:function:: dtfft_get_gpu_backend_string(gpu_backend)

  Gets the string description of a GPU backend

  This function is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :p dtfft_gpu_backend_t gpu_backend [in]:
    GPU backend
  :r character(len=:), allocatable string:
    Backend string

------

dtfft_config_t
-----------------------

.. f:type:: dtfft_config_t

  Type that can be used to set additional configuration parameters to ``dtFFT``

  :f logical enable_z_slab:
    Should ``dtFFT`` use Z-slab optimization or not.

    Default is ``.true.``

    One should consider disabling Z-slab optimization in order to resolve :f:var:`DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error
    OR when underlying FFT implementation of 2D plan is too slow.

    In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.

  :f integer(cuda_stream_kind) stream:

    Main CUDA stream that will be used in dtFFT.

    This parameter is a placeholder for user to set custom stream.

    Stream that is actually used by dtFFT plan is returned by f:func:`get_stream` function.

    When user sets stream he is responsible of destroying it.

    Stream must not be destroyed before call to :f:func:`destroy`.

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f type(dtfft_gpu_backend_t) gpu_backend:

    Backend that will be used by dtFFT when ``effort`` is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.

    Default is ``DTFFT_GPU_BACKEND_NCCL``

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_mpi_backends:

    Should MPI GPU Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or not.

    Default is ``.false.``

    MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
    It was noticed that during plan autotuning GPU memory not being freed completely.
    For example:
    1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
    Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.

    One of the workarounds is to disable MPI Backends by default, which is done here.

    Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to ``mpiexec``,
    but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_pipelined_backends:

    Should pipelined GPU backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or not.

    Default is ``.true.``

    Pipelined backends require additional buffer that user has no control over.

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_nccl_backends:
    Should NCCL Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or not.

    Default is ``.true.``

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :f logical enable_nvshmem_backends:
    Should NCCL Backends be enabled when ``effort`` is ``DTFFT_PATIENT`` or not.

    Default is ``.true.``

    Unused. Reserved for future.

    .. note:: This field is only present in the API when ``dtFFT`` was compiled with CUDA Support.

Related Type functions
_______________________

.. f:function:: dtfft_create_config(config)

  Creates dtfft_config_t objects and sets default values to it.

  :p dtfft_config_t config [out]: Configuration object

------

.. f:function:: dtfft_config_t()

  Type bound constructor

  :r dtfft_config_t: Configuration object

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
-----------------------

.. f:type:: dtfft_pencil_t

  Type used to hold pencil decomposition info

  :f int(int8) dim: Aligned dimension id starting from 1
  :f int(int8) ndims: Number of dimensions in a pencil
  :f int(int32) starts(3): Local starts in natural Fortran order
  :f int(int32) counts(3): Local counts in natural Fortran order

.. seealso:: :f:func:`get_pencil`

------

dtfft_platform_t
-----------------------

  Type that specifies the execution platform, such as Host, CUDA, or HIP

Type Parameters
_____________________

.. f:variable:: DTFFT_PLATFORM_HOST

  Create HOST-related plan

.. f:variable:: DTFFT_PLATFORM_CUDA

  Create CUDA-related


Version handling
================

.. f:variable:: DTFFT_VERSION_MAJOR

  ``dtFFT`` Major Version

.. f:variable:: DTFFT_VERSION_MINOR

  ``dtFFT`` Minor Version

.. f:variable:: DTFFT_VERSION_PATCH

  ``dtFFT`` Patch Version

.. f:variable:: DTFFT_VERSION_CODE

  ``dtFFT`` Version Code. Can be used in Version comparison

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

transpose
_________

.. f:subroutine:: transpose(in, out, transpose_type [, error_code])

  Performs single transposition

  :p type(*), dimension(..) in [inout]:
    Incoming buffer of any rank and kind.
  :p type(*), dimension(..) out [inout]:
    Resulting buffer of any rank and kind
  :p dtfft_transpose_type_t transpose_type [in]:
    Type of transposition
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
  :p dtfft_execute_type_t execute_type [in]:
    Type of execution
  :o type(*), dimension(..) aux [inout, optional]:
    Optional auxiliary buffer.
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
    Minimum number of elements needs to be allocated for ``in``, ``out`` or ``aux`` buffers.
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
    Minimum number of elements needs to be allocated for ``in``, ``out`` or ``aux`` buffers.
    Size of each element in bytes can be obtained by calling :f:func:`get_element_size`.

------

get_element_size
________________

.. f:function:: get_element_size( [error_code] )

  Returns number of bytes required to store single element.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user
  :r integer(int64): Size of element in bytes

------

mem_alloc
_________

Allocates memory tailored to the specific needs of the plan.  

.. f:function:: mem_alloc(alloc_bytes[, error_code])

  :p integer(int64) alloc_bytes [in]: Number of bytes to allocate  
  :o integer(int32) error_code [out, optional]: Optional error code returned to user  
  :r type(c_ptr): Allocated pointer


------

mem_free
________

Frees memory previously allocated by :f:func:`mem_alloc`.  


.. f:subroutine:: mem_free(ptr[, error_code])

  :p type(c_ptr) ptr [in]: Pointer allocated with ``mem_alloc``
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

get_pencil
__________

.. f:function:: get_pencil(dim[, error_code])

  Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation,
  that is unavailable in ``dtFFT``.

  :p integer(int8) dim [in]:
    Required dimension:
      - 1 for XYZ layout
      - 2 for YXZ layout
      - 3 for ZXY layout
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

get_gpu_backend
_______________

.. f:function:: get_gpu_backend([error_code])

  Returns the fastest detected GPU backend if ``effort`` is :f:var:`DTFFT_PATIENT`.

  If ``effort`` is :f:var:`DTFFT_ESTIMATE` or :f:var:`DTFFT_MEASURE`, returns the value set by :f:func:`dtfft_set_config`
  or the default, :f:var:`DTFFT_GPU_BACKEND_NCCL`.

  .. note:: This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

  :r dtfft_gpu_backend_t: Selected GPU backend

------

get_stream
__________

.. f:function:: get_stream([error_code])

  Returns CUDA stream associated with plan

  .. note:: This method is only present in the API when ``dtFFT`` was compiled with CUDA Support.

  :o integer(int32) error_code [out, optional]:
    Optional error code returned to user

  :r integer(cuda_stream_kind): CUDA stream associated with plan

------

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
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.

  :from: :ref:`constructor<constructor_r2r>`

------

.. _constructor_r2r:

constructor
___________

.. f:function:: dtfft_plan_r2r_t(dims [, kinds, comm, precision, effort, executor, error_code])

  R2R Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o dtfft_r2r_kind_t kinds(:) [in, optional]: Kinds of R2R transforms, default = empty.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.
  :r dtfft_plan_r2r_t: Plan ready for execution

  :to: :ref:`create<create_r2r>`

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
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.

  :from: :ref:`constructor<constructor_c2c>`

------

.. _constructor_c2c:

constructor
___________

.. f:function:: dtfft_plan_c2c_t(dims [, kinds, comm, precision, effort, executor, error_code])

  C2C Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.
  :r dtfft_plan_c2c_t: Plan ready for execution

  :to: :ref:`create<create_c2c>`

------

Real-to-Complex plan
====================

.. f:type:: dtfft_plan_r2c_t

  Real-to-complex plan class

  Extends :f:type:`dtfft_plan_t`

.. note:: This type is only present in the API when ``dtFFT`` is compiled with FFT support.

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
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = ``undefined``. Must not be :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.

  :from: :ref:`constructor<constructor_r2c>`

------

.. _constructor_r2c:

constructor
___________

.. f:function:: dtfft_plan_r2c_t(dims [, comm, precision, effort, executor, error_code])

  R2C Plan Constructor.

  :p integer(int32) dims(:)[in]: Global dimensions of the transform as an integer array.
  :o MPI_Comm comm [in, optional]: Communicator for parallel execution, default = MPI_COMM_WORLD.
  :o dtfft_precision_t precision [in, optional]: Precision of the transform, default = :f:var:`DTFFT_DOUBLE`.
  :o dtfft_effort_t effort [in, optional]: How hard ``dtFFT`` should look for best plan, default = :f:var:`DTFFT_ESTIMATE`.
  :o dtfft_executor_t executor [in, optional]: Type of external FFT executor, default = undefined. Must not be :f:var:`DTFFT_EXECUTOR_NONE`.
  :o integer(int32) error_code [out, optional]: Optional error code returned to the user, default = not returned.
  :r dtfft_plan_r2c_t: Plan ready for execution

  :to: :ref:`create<create_r2c>`

