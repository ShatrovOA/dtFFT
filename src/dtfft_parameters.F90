!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
#include "dtfft_config.h"
module dtfft_parameters
!! This module defines common ``dtFFT`` parameters
use iso_c_binding,    only: c_int32_t, c_null_ptr, c_ptr
use iso_fortran_env,  only: int8, int32, int64, real32, real64
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: dtfft_execute_t, dtfft_transpose_t, dtfft_reshape_t
public :: dtfft_executor_t, dtfft_effort_t, dtfft_layout_t
public :: dtfft_precision_t, dtfft_r2r_kind_t
public :: is_valid_execute_type, is_valid_transpose_type, is_valid_reshape_type
public :: is_valid_executor, is_valid_effort, is_valid_layout
public :: is_valid_precision, is_valid_r2r_kind
public :: is_valid_dimension, is_valid_comm_type
public :: is_valid_transpose_mode, is_valid_access_mode
public :: dtfft_get_version
public :: dtfft_get_precision_string, dtfft_get_executor_string
public :: dtfft_backend_t, dtfft_stream_t
public :: dtfft_get_backend_string
public :: is_backend_pipelined, is_backend_mpi, is_backend_fused, is_backend_rma
public :: is_valid_backend, is_backend_nccl, is_backend_cufftmp, is_backend_nvshmem
public :: is_backend_compressed
public :: dtfft_get_backend_pipelined
#ifdef DTFFT_WITH_CUDA
public :: is_valid_platform
public :: is_host_executor, is_cuda_executor
public :: dtfft_get_cuda_stream
#endif

  integer(int32), parameter, public :: DTFFT_VERSION_MAJOR = CONF_DTFFT_VERSION_MAJOR
    !! dtFFT Major Version
  integer(int32), parameter, public :: DTFFT_VERSION_MINOR = CONF_DTFFT_VERSION_MINOR
    !! dtFFT Minor Version
  integer(int32), parameter, public :: DTFFT_VERSION_PATCH = CONF_DTFFT_VERSION_PATCH
    !! dtFFT Patch Version
  integer(int32), parameter, public :: DTFFT_VERSION_CODE  = CONF_DTFFT_VERSION_CODE
    !! dtFFT Version Code. Can be used in Version comparison

  interface dtfft_get_version
  !! Get dtFFT version
    module procedure :: dtfft_get_version_current  !! Get current version
    module procedure :: dtfft_get_version_required !! Get required version
  end interface dtfft_get_version

  real(real32),   parameter,  public :: MAX_REAL32  =  huge(1._real32)
    !! Maximum value of real32
  integer(int32), parameter,  public :: MAX_INT32 = huge(1_int32)
    !! Maximum value of int32

!------------------------------------------------------------------------------------------------
! Execute types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_execute_t
  !! Type that is used during call to `execute` method
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_execute_t

  type(dtfft_execute_t), parameter, public :: DTFFT_EXECUTE_FORWARD = dtfft_execute_t(CONF_DTFFT_EXECUTE_FORWARD)
    !! Perform XYZ --> YZX --> ZXY plan execution (Forward)
  type(dtfft_execute_t), parameter, public :: DTFFT_EXECUTE_BACKWARD  = dtfft_execute_t(CONF_DTFFT_EXECUTE_BACKWARD)
    !! Perform ZXY --> YZX --> XYZ plan execution (Backward)
  type(dtfft_execute_t), parameter :: VALID_EXECUTE_TYPES(*) = [DTFFT_EXECUTE_FORWARD, DTFFT_EXECUTE_BACKWARD]
    !! Valid execute types

!------------------------------------------------------------------------------------------------
! Transpose types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_transpose_t
  !! Type that is used during call to [[dtfft_plan_t(type):transpose]] method
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_transpose_t

  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_X_TO_Y = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_X_TO_Y)
    !! Perform single transposition, from X aligned to Y aligned
  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_Y_TO_X = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_Y_TO_X)
    !! Perform single transposition, from Y aligned to X aligned
  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_X_TO_Z = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_X_TO_Z)
    !! Perform single transposition, from X aligned to Z aligned
  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_Y_TO_Z = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_Y_TO_Z)
    !! Perform single transposition, from Y aligned to Z aligned
  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_Z_TO_Y = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_Z_TO_Y)
    !! Perform single transposition, from Z aligned to Y aligned
  type(dtfft_transpose_t), parameter,  public :: DTFFT_TRANSPOSE_Z_TO_X = dtfft_transpose_t(CONF_DTFFT_TRANSPOSE_Z_TO_X)
    !! Perform single transposition, from Z aligned to X aligned
  type(dtfft_transpose_t), parameter :: VALID_TRANSPOSE_TYPES(*) = [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_Y, DTFFT_TRANSPOSE_X_TO_Z, DTFFT_TRANSPOSE_Z_TO_X]
    !! Types of transpose that are valid to pass to `transpose` method
  character(len=*), parameter,  public :: TRANSPOSE_NAMES(-3:3) = ["Z_TO_X", "Z_TO_Y", "Y_TO_X", " NULL ", "X_TO_Y", "Y_TO_Z", "X_TO_Z"]
    !! String representation of `dtfft_transpose_t`

  type, bind(C) :: dtfft_reshape_t
  !! Type that is used during call to [[dtfft_plan_t(type):reshape]] method
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_reshape_t

  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_X_BRICKS_TO_PENCILS  = dtfft_reshape_t(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS)
    !! Perform reshape to X-aligned pencils (forward) from X-aligned bricks
  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_X_PENCILS_TO_BRICKS = dtfft_reshape_t(CONF_DTFFT_RESHAPE_X_PENCILS_TO_BRICKS)
    !! Perform reshape to X-aligned bricks (backward) from X-aligned pencils
  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS  = dtfft_reshape_t(CONF_DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS)
    !! Perform reshape to Z-aligned bricks (forward) from Z-aligned pencils
  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS = dtfft_reshape_t(CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS)
    !! Perform reshape to Z-aligned pencils (backward) from Z-aligned bricks
  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_Y_BRICKS_TO_PENCILS = dtfft_reshape_t(CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS)
    !! Reshape from Y-bricks to Y-pencils
    !! This is to be used in 2D Plans.
  type(dtfft_reshape_t), parameter,  public :: DTFFT_RESHAPE_Y_PENCILS_TO_BRICKS = dtfft_reshape_t(CONF_DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS)
    !! Reshape from Y-pencils to Y-bricks
    !! This is to be used in 2D Plans.
  type(dtfft_reshape_t), parameter :: VALID_RESHAPE_TYPES(*) = [DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, DTFFT_RESHAPE_X_PENCILS_TO_BRICKS, DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS, DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS]
    !! Types of reshape that are valid to pass to `reshape` method
  character(len=*), parameter,  public :: RESHAPE_NAMES(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS:CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS) = ["X_BRICKS_TO_PENCILS", "X_PENCILS_TO_BRICKS", "Z_PENCILS_TO_BRICKS", "Z_BRICKS_TO_PENCILS"]
    !! String representations of `dtfft_reshape_t`

  type, bind(C) :: dtfft_layout_t
  !! Type that specifies data layout in distributed array
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_layout_t

  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_X_BRICKS = dtfft_layout_t(CONF_DTFFT_LAYOUT_X_BRICKS)
    !! X-brick layout: data is distributed along all dimensions
  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_Z_BRICKS = dtfft_layout_t(CONF_DTFFT_LAYOUT_Z_BRICKS)
    !! Z-brick layout: data is distributed along all dimensions
  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_X_PENCILS = dtfft_layout_t(CONF_DTFFT_LAYOUT_X_PENCILS)
    !! X-pencil layout: data is distributed along Y and Z dimensions
  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_X_PENCILS_FOURIER = dtfft_layout_t(CONF_DTFFT_LAYOUT_X_PENCILS_FOURIER)
    !! X-pencil layout obtained after executing FFT for R2C plan: data is distributed along Y and Z dimensions
  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_Y_PENCILS = dtfft_layout_t(CONF_DTFFT_LAYOUT_Y_PENCILS)
    !! Y-pencil layout: data is distributed along X and Z dimensions
  type(dtfft_layout_t),  parameter,  public :: DTFFT_LAYOUT_Z_PENCILS = dtfft_layout_t(CONF_DTFFT_LAYOUT_Z_PENCILS)
    !! Z-pencil layout: data is distributed along X and Y dimensions
  type(dtfft_layout_t),  parameter :: VALID_LAYOUTS(*) = [DTFFT_LAYOUT_X_BRICKS, DTFFT_LAYOUT_Z_BRICKS, DTFFT_LAYOUT_X_PENCILS, DTFFT_LAYOUT_X_PENCILS_FOURIER, DTFFT_LAYOUT_Y_PENCILS, DTFFT_LAYOUT_Z_PENCILS]
    !! Valid layouts

!------------------------------------------------------------------------------------------------
! External FFT executor types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_executor_t
  !! Type that specifies external FFT executor
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_executor_t

  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_NONE   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_NONE)
    !! Do not setup any executor. If this type is provided, then `execute` method should not be called.
    !! Use `transpose` method instead
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_FFTW3 = dtfft_executor_t(CONF_DTFFT_EXECUTOR_FFTW3)
    !! FFTW3 executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_MKL   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_MKL)
    !! MKL executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_CUFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_CUFFT)
    !! cuFFT GPU executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_VKFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_VKFFT)
    !! VkFFT GPU executor
  type(dtfft_executor_t),  parameter  :: VALID_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE    &
#ifdef DTFFT_WITH_FFTW
    ,DTFFT_EXECUTOR_FFTW3                                                             &
#endif
#ifdef DTFFT_WITH_MKL
    ,DTFFT_EXECUTOR_MKL                                                               &
#endif
#ifdef DTFFT_WITH_CUFFT
    ,DTFFT_EXECUTOR_CUFFT                                                             &
#endif
#ifdef DTFFT_WITH_VKFFT
    ,DTFFT_EXECUTOR_VKFFT                                                             &
#endif
  ]
  !! List of valid executors
#ifdef DTFFT_WITH_CUDA
  type(dtfft_executor_t),  parameter        :: HOST_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_FFTW3, DTFFT_EXECUTOR_MKL]
    !! List of host executors
  type(dtfft_executor_t),  parameter        :: CUDA_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_CUFFT, DTFFT_EXECUTOR_VKFFT]
    !! List of CUDA executors
#endif
!------------------------------------------------------------------------------------------------
! FFT Execution directions
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: FFT_FORWARD   = CONF_FFT_FORWARD
    !! Forward c2c transform
  integer(int8),  parameter,  public :: FFT_BACKWARD  = CONF_FFT_BACKWARD
    !! Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags.
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_effort_t
  !! Type that specifies effort that dtFFT should use when creating plan
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_effort_t

  type(dtfft_effort_t), parameter,  public :: DTFFT_ESTIMATE = dtfft_effort_t(CONF_DTFFT_ESTIMATE)
    !! Estimate flag. ``dtFFT`` will use default decomposition provided by MPI_Dims_create
  type(dtfft_effort_t), parameter,  public :: DTFFT_MEASURE  = dtfft_effort_t(CONF_DTFFT_MEASURE)
    !! Measure flag. ``dtFFT`` will run transpose routines to find the best grid decomposition.
    !! Passing this flag and MPI Communicator with Cartesian topology to `plan%create` makes dtFFT do nothing.
  type(dtfft_effort_t), parameter,  public :: DTFFT_PATIENT  = dtfft_effort_t(CONF_DTFFT_PATIENT)
    !! Patient flag. Same as `DTFFT_MEASURE`, but different MPI datatypes will also be tested
  type(dtfft_effort_t), parameter,  public :: DTFFT_EXHAUSTIVE  = dtfft_effort_t(CONF_DTFFT_EXHAUSTIVE)
    !! Exhaustive flag. Same as `DTFFT_PATIENT`, but all possible backends and reshape backends will be tested to find the best plan.
  type(dtfft_effort_t), parameter :: VALID_EFFORTS(*) = [DTFFT_ESTIMATE, DTFFT_MEASURE, DTFFT_PATIENT, DTFFT_EXHAUSTIVE]
    !! Valid effort flags

!------------------------------------------------------------------------------------------------
! Precision flags
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_precision_t
  !! Type that specifies precision of dtFFT plan
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_precision_t

  type(dtfft_precision_t),  parameter,  public :: DTFFT_SINGLE = dtfft_precision_t(CONF_DTFFT_SINGLE)
    !! Use single precision
  type(dtfft_precision_t),  parameter,  public :: DTFFT_DOUBLE = dtfft_precision_t(CONF_DTFFT_DOUBLE)
    !! Use double precision
  type(dtfft_precision_t),  parameter :: VALID_PRECISIONS(*) = [DTFFT_SINGLE, DTFFT_DOUBLE]
    !! Valid precision flags

!------------------------------------------------------------------------------------------------
! R2R Transform kinds
! This parameters matches FFTW definitions. Hope they will never change there.
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_r2r_kind_t
  !! Type that specifies various kinds of R2R FFTs
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_r2r_kind_t

  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_1 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_1)
    !! DCT-I (Logical N=2*(n-1), inverse is `DTFFT_DCT_1`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_2 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_2)
    !! DCT-II (Logical N=2*n, inverse is `DTFFT_DCT_3`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_3 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_3)
    !! DCT-III (Logical N=2*n, inverse is `DTFFT_DCT_2`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_4 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_4)
    !! DCT-IV (Logical N=2*n, inverse is `DTFFT_DCT_4`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_1 = dtfft_r2r_kind_t(CONF_DTFFT_DST_1)
    !! DST-I (Logical N=2*(n+1), inverse is `DTFFT_DST_1`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_2 = dtfft_r2r_kind_t(CONF_DTFFT_DST_2)
    !! DST-II (Logical N=2*n, inverse is `DTFFT_DST_3`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_3 = dtfft_r2r_kind_t(CONF_DTFFT_DST_3)
    !! DST-III (Logical N=2*n, inverse is `DTFFT_DST_2`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_4 = dtfft_r2r_kind_t(CONF_DTFFT_DST_4)
    !! DST-IV (Logical N=2*n, inverse is `DTFFT_DST_4`)
  type(dtfft_r2r_kind_t),   parameter :: VALID_R2R_KINDS(*) = [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_4, DTFFT_DST_1, DTFFT_DST_2, DTFFT_DST_3, DTFFT_DST_4]
    !! Array of valid R2R kinds

public :: operator(==)
  interface operator(==)
    module procedure execute_type_eq    !! Check if two `dtfft_execute_t` are equal
    module procedure transpose_type_eq  !! Check if two `dtfft_transpose_t` are equal
    module procedure reshape_type_eq    !! Check if two `dtfft_reshape_t` are equal
    module procedure executor_eq        !! Check if two `dtfft_executor_t` are equal
    module procedure effort_eq          !! Check if two `dtfft_effort_t` are equal
    module procedure precision_eq       !! Check if two `dtfft_precision_t` are equal
    module procedure r2r_kind_eq        !! Check if two `dtfft_r2r_kind_t` are equal
    module procedure platform_eq        !! Check if two `dtfft_platform_t` are equal
    module procedure exec_eq            !! Check if two `async_exec_t` are equal
    module procedure backend_eq         !! Check if two `dtfft_backend_t` are equal
    module procedure layout_eq          !! Check if two `dtfft_layout_t` are equal
    module procedure transpose_mode_eq  !! Check if two `dtfft_transpose_mode_t` are equal
    module procedure access_mode_eq     !! Check if two `dtfft_access_mode_t` are equal
  end interface

public :: operator(/=)
  interface operator(/=)
    module procedure execute_type_ne    !! Check if two `dtfft_execute_t` are not equal
    module procedure transpose_type_ne  !! Check if two `dtfft_transpose_t` are not equal
    module procedure reshape_type_ne    !! Check if two `dtfft_reshape_t` are not equal
    module procedure executor_ne        !! Check if two `dtfft_executor_t` are not equal
    module procedure effort_ne          !! Check if two `dtfft_effort_t` are not equal
    module procedure precision_ne       !! Check if two `dtfft_precision_t` are not equal
    module procedure r2r_kind_ne        !! Check if two `dtfft_r2r_kind_t` are not equal
    module procedure platform_ne        !! Check if two `dtfft_platform_t` are not equal
    module procedure backend_ne         !! Check if two `dtfft_backend_t` are not equal
    module procedure layout_ne          !! Check if two `dtfft_layout_t` are not equal
    module procedure transpose_mode_ne  !! Check if two `dtfft_transpose_mode_t` are not equal
    module procedure access_mode_ne     !! Check if two `dtfft_access_mode_t` are not equal
  end interface

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(int64), parameter,  public :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._real64, 1._real64)) / 8_int64
    !! Number of bytes to store single double precision complex element
  integer(int64), parameter,  public :: COMPLEX_STORAGE_SIZE          = storage_size((1._real32, 1._real32)) / 8_int64
    !! Number of bytes to store single float precision complex element
  integer(int64), parameter,  public :: DOUBLE_STORAGE_SIZE           = storage_size(1._real64) / 8_int64
    !! Number of bytes to store single double precision real element
  integer(int64), parameter,  public :: FLOAT_STORAGE_SIZE            = storage_size(1._real32) / 8_int64
    !! Number of bytes to store single single precision real element


  integer(int8),    parameter :: VALID_DIMENSIONS(*) = [2_int8, 3_int8]
    !! Valid dimensions for `plan.create`
  integer(int32),   parameter :: VALID_COMM_TYPES(*) = [MPI_UNDEFINED, MPI_CART]
    !! Valid communicator types for `plan.create`


  integer(int32), parameter,  public :: COLOR_CREATE        = int(Z'00FAB53C')
    !! Color for `plan.create`
  integer(int32), parameter,  public :: COLOR_EXECUTE       = int(Z'00E25DFC')
    !! Color for `plan.execute`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE     = int(Z'00B175BD')
    !! Color for `plan.transpose`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_XY  = int(Z'005DFCCA')
    !! Color for `DTFFT_TRANSPOSE_X_TO_Y`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_YX  = int(Z'0076A797')
    !! Color for `DTFFT_TRANSPOSE_Y_TO_X`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_YZ  = int(Z'00E3CF9F')
    !! Color for `DTFFT_TRANSPOSE_Y_TO_Z`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_ZY  = int(Z'008C826A')
    !! Color for `DTFFT_TRANSPOSE_Z_TO_Y`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_XZ  = int(Z'00546F66')
    !! Color for `DTFFT_TRANSPOSE_X_TO_Z`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_ZX  = int(Z'007A6D7D')
    !! Color for `DTFFT_TRANSPOSE_Z_TO_X`
  integer(int32), parameter,  public :: COLOR_FFT           = int(Z'00FCD05D')
    !! Color for FFT
  integer(int32), parameter,  public :: COLOR_AUTOTUNE      = int(Z'006075FF')
    !! Color for Autotune
  integer(int32), parameter,  public :: COLOR_AUTOTUNE2     = int(Z'0056E874')
    !! Color for Autotune2
  integer(int32), parameter,  public :: COLOR_DESTROY       = int(Z'00000000')
    !! Color for `plan.destroy`
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_PALLETTE(-3:3) = [COLOR_TRANSPOSE_ZX, COLOR_TRANSPOSE_ZY, COLOR_TRANSPOSE_YX, 0, COLOR_TRANSPOSE_XY, COLOR_TRANSPOSE_YZ, COLOR_TRANSPOSE_XZ]
    !! Color pallete for `plan.transpose`

    ! Extended color palette for profiling
  integer(int32), parameter,  public :: COLOR_CORAL         = int(Z'00FF7F50')
    !! Coral
  integer(int32), parameter,  public :: COLOR_TOMATO        = int(Z'00FF6347')
    !! Tomato
  integer(int32), parameter,  public :: COLOR_SALMON        = int(Z'00FA8072')
    !! Salmon
  integer(int32), parameter,  public :: COLOR_LIME          = int(Z'0000FF00')
    !! Lime
  integer(int32), parameter,  public :: COLOR_CHARTREUSE    = int(Z'007FFF00')
    !! Chartreuse
  integer(int32), parameter,  public :: COLOR_CYAN          = int(Z'0000FFFF')
    !! Cyan
  integer(int32), parameter,  public :: COLOR_DODGER_BLUE   = int(Z'001E90FF')
    !! Dodger Blue
  integer(int32), parameter,  public :: COLOR_STEEL_BLUE    = int(Z'004682B4')
    !! Steel Blue
  integer(int32), parameter,  public :: COLOR_ORCHID        = int(Z'00DA70D6')
    !! Orchid
  integer(int32), parameter,  public :: COLOR_VIOLET        = int(Z'00EE82EE')
    !! Violet
  integer(int32), parameter,  public :: COLOR_MAGENTA       = int(Z'00FF00FF')
    !! Magenta
  integer(int32), parameter,  public :: COLOR_INDIGO        = int(Z'004B0082')
    !! Indigo
  integer(int32), parameter,  public :: COLOR_MAROON        = int(Z'00800000')
    !! Maroon
  integer(int32), parameter,  public :: COLOR_OLIVE         = int(Z'00808000')
    !! Olive
  integer(int32), parameter,  public :: COLOR_TEAL          = int(Z'00008080')
    !! Teal
  integer(int32), parameter,  public :: COLOR_NAVY          = int(Z'00000080')
    !! Navy
  integer(int32), parameter,  public :: COLOR_SIENNA        = int(Z'00A0522D')
    !! Sienna
  integer(int32), parameter,  public :: COLOR_PERU          = int(Z'00CD853F')
    !! Peru
  integer(int32), parameter,  public :: COLOR_KHAKI         = int(Z'00F0E68C')
    !! Khaki
  integer(int32), parameter,  public :: COLOR_PLUM          = int(Z'00DDA0DD')
    !! Plum
  integer(int32), parameter,  public :: COLOR_LAVENDER      = int(Z'00E6E6FA')
    !! Lavender
  integer(int32), parameter,  public :: COLOR_MINT          = int(Z'0098FF98')
    !! Mint
  integer(int32), parameter,  public :: COLOR_PEACH         = int(Z'00FFDAB9')
    !! Peach Puff
  integer(int32), parameter,  public :: COLOR_SKY_BLUE      = int(Z'0087CEEB')
    !! Sky Blue
  integer(int32), parameter,  public :: COLOR_SLATE_BLUE    = int(Z'006A5ACD')
    !! Slate Blue
  integer(int32), parameter,  public :: COLOR_SEA_GREEN     = int(Z'002E8B57')
    !! Sea Green
  integer(int32), parameter,  public :: COLOR_FOREST_GREEN  = int(Z'00228B22')
    !! Forest Green
  integer(int32), parameter,  public :: COLOR_YELLOW_GREEN  = int(Z'009ACD32')
    !! Yellow Green
  integer(int32), parameter,  public :: COLOR_ORANGE        = int(Z'00FFA500')
    !! Orange
  integer(int32), parameter,  public :: COLOR_RESHAPE_PALLETTE(CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS:CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS) = [COLOR_LIME, COLOR_MAGENTA, COLOR_INDIGO, COLOR_PERU]


  integer(int32),  parameter,  public  :: VARIABLE_NOT_SET = -111
  !! Default value when environ is not set

!------------------------------------------------------------------------------------------------
! Backends that are responsible for transfering data between processes
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_backend_t
  !! Type that specifies various backends present in dtFFT
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_backend_t


  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_DATATYPE = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_DATATYPE)
    !! Backend that uses MPI datatypes
    !! This is default backend for Host build.
    !! Not really recommended to use for GPU usage, since it is a 'million' times slower than other backends.
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_P2P = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_P2P)
    !! MPI peer-to-peer algorithm
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_A2A = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_A2A)
    !! MPI backend using MPI_Alltoallv
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_RMA = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_RMA)
    !! MPI RMA backend
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_RMA_PIPELINED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_RMA_PIPELINED)
    !! MPI Pipelined RMA backend
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_P2P_PIPELINED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_P2P_PIPELINED)
    !! MPI peer-to-peer algorithm with overlapping data exchange and unpacking
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_P2P_SCHEDULED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_P2P_SCHEDULED)
    !! MPI peer-to-peer algorithm with scheduled communication
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_P2P_FUSED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_P2P_FUSED)
    !! MPI peer-to-peer pipelined algorithm with overlapping packing, exchange and unpacking with scheduled communication
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_RMA_FUSED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_RMA_FUSED)
    !! MPI RMA pipelined algorithm with overlapping packing, exchange and unpacking with scheduled communication
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_P2P_COMPRESSED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_P2P_COMPRESSED)
    !! Extension of ``DTFFT_BACKEND_MPI_P2P_FUSED``. Performs compression before sending and decomporession after recieving.
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_MPI_RMA_COMPRESSED = dtfft_backend_t(CONF_DTFFT_BACKEND_MPI_RMA_COMPRESSED)
    !! Extension of ``DTFFT_BACKEND_MPI_RMA_FUSED``. Performs compression before sending and decomporession after recieving.
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_NCCL = dtfft_backend_t(CONF_DTFFT_BACKEND_NCCL)
    !! NCCL backend
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_NCCL_PIPELINED = dtfft_backend_t(CONF_DTFFT_BACKEND_NCCL_PIPELINED)
    !! NCCL backend with overlapping data exchange and unpacking
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_NCCL_COMPRESSED = dtfft_backend_t(CONF_DTFFT_BACKEND_NCCL_COMPRESSED)
    !! Extension of ``DTFFT_BACKEND_NCCL``. Performs compression before sending and decomporession after recieving.
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_CUFFTMP = dtfft_backend_t(CONF_DTFFT_BACKEND_CUFFTMP)
    !! cuFFTMp backend
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_CUFFTMP_PIPELINED = dtfft_backend_t(CONF_DTFFT_BACKEND_CUFFTMP_PIPELINED)
    !! cuFFTMp backend that uses extra buffer to gain performance
  type(dtfft_backend_t),  parameter,  public  :: DTFFT_BACKEND_ADAPTIVE = dtfft_backend_t(CONF_DTFFT_BACKEND_ADAPTIVE)
    !! Adaptive backend that selects best available backend at runtime
    !! Currently only available for HOST execution platform
  type(dtfft_backend_t),  parameter,  public  :: BACKEND_NOT_SET = dtfft_backend_t(VARIABLE_NOT_SET)
    !! Backend is not used
  type(dtfft_backend_t),  parameter,  public  :: BACKEND_DUMMY = dtfft_backend_t(2 * VARIABLE_NOT_SET)
  type(dtfft_backend_t),  parameter :: PIPELINED_BACKENDS(*) = [DTFFT_BACKEND_MPI_P2P_PIPELINED, DTFFT_BACKEND_NCCL_PIPELINED, DTFFT_BACKEND_CUFFTMP_PIPELINED, DTFFT_BACKEND_MPI_RMA_PIPELINED]
    !! List of pipelined backends
  type(dtfft_backend_t),  parameter :: FUSED_BACKENDS(*) = [DTFFT_BACKEND_MPI_P2P_FUSED, DTFFT_BACKEND_MPI_RMA_FUSED]
    !! List of fused backends
  type(dtfft_backend_t),  parameter :: RMA_BACKENDS(*) = [DTFFT_BACKEND_MPI_RMA, DTFFT_BACKEND_MPI_RMA_PIPELINED, DTFFT_BACKEND_MPI_RMA_FUSED, DTFFT_BACKEND_MPI_RMA_COMPRESSED]
    !! List of RMA backends
  type(dtfft_backend_t),  parameter :: COMPRESSED_BACKENDS(*) = [DTFFT_BACKEND_MPI_P2P_COMPRESSED, DTFFT_BACKEND_MPI_RMA_COMPRESSED, DTFFT_BACKEND_NCCL_COMPRESSED]
    !! List of backends that support compression
  type(dtfft_backend_t),  parameter :: MPI_BACKENDS(*) = [DTFFT_BACKEND_MPI_DATATYPE, DTFFT_BACKEND_MPI_P2P, DTFFT_BACKEND_MPI_A2A, DTFFT_BACKEND_MPI_P2P_PIPELINED, DTFFT_BACKEND_MPI_RMA, DTFFT_BACKEND_MPI_RMA_PIPELINED, DTFFT_BACKEND_MPI_P2P_SCHEDULED, DTFFT_BACKEND_MPI_P2P_FUSED, DTFFT_BACKEND_MPI_RMA_FUSED, DTFFT_BACKEND_MPI_P2P_COMPRESSED, DTFFT_BACKEND_MPI_RMA_COMPRESSED]
    !! List of MPI backends
  type(dtfft_backend_t),  parameter :: NCCL_BACKENDS(*) = [DTFFT_BACKEND_NCCL, DTFFT_BACKEND_NCCL_PIPELINED, DTFFT_BACKEND_NCCL_COMPRESSED]
    !! List of NCCL backends
  type(dtfft_backend_t),  parameter :: CUFFTMP_BACKENDS(*) = [DTFFT_BACKEND_CUFFTMP, DTFFT_BACKEND_CUFFTMP_PIPELINED]
    !! List of cuFFTMp backends
  type(dtfft_backend_t),  parameter :: NVSHMEM_BACKENDS(*) = [DTFFT_BACKEND_CUFFTMP, DTFFT_BACKEND_CUFFTMP_PIPELINED]
    !! List of NVSHMEM-based backends
  type(dtfft_backend_t),  parameter,  public :: VALID_BACKENDS(*) = [DTFFT_BACKEND_MPI_DATATYPE         &
                                                                    ,DTFFT_BACKEND_MPI_P2P              &
                                                                    ,DTFFT_BACKEND_MPI_A2A              &
                                                                    ,DTFFT_BACKEND_MPI_P2P_PIPELINED    &
                                                                    ,DTFFT_BACKEND_MPI_P2P_SCHEDULED    &
                                                                    ,DTFFT_BACKEND_MPI_P2P_FUSED        &
#ifdef DTFFT_WITH_COMPRESSION
                                                                    ,DTFFT_BACKEND_MPI_P2P_COMPRESSED   &
#endif
#ifdef DTFFT_WITH_RMA
                                                                    ,DTFFT_BACKEND_MPI_RMA              &
                                                                    ,DTFFT_BACKEND_MPI_RMA_PIPELINED    &
                                                                    ,DTFFT_BACKEND_MPI_RMA_FUSED        &
# ifdef DTFFT_WITH_COMPRESSION
                                                                    ,DTFFT_BACKEND_MPI_RMA_COMPRESSED   &
# endif
#endif
#ifdef DTFFT_WITH_NCCL
                                                                    ,DTFFT_BACKEND_NCCL_PIPELINED       &
                                                                    ,DTFFT_BACKEND_NCCL                 &
# ifdef DTFFT_WITH_COMPRESSION
                                                                    ,DTFFT_BACKEND_NCCL_COMPRESSED      &
# endif
#endif
#ifdef DTFFT_WITH_NVSHMEM
                                                                    ,DTFFT_BACKEND_CUFFTMP              &
                                                                    ,DTFFT_BACKEND_CUFFTMP_PIPELINED    &
#endif
                                                                    ,DTFFT_BACKEND_ADAPTIVE             &
                                                                    ]
    !! List of valid backends that `dtFFT` was compiled for

  type, bind(C) :: dtfft_stream_t
  !! `dtFFT` stream representation.
    type(c_ptr) :: stream !! Actual stream
  end type dtfft_stream_t

  type(dtfft_stream_t), parameter,  public :: NULL_STREAM = dtfft_stream_t(c_null_ptr)

#ifdef DTFFT_WITH_CUDA
  interface dtfft_stream_t
  !! Creates [[dtfft_stream_t]] from integer(cuda_stream_kind)
    module procedure stream_from_int64
  end interface dtfft_stream_t
#endif

public :: dtfft_platform_t
  type, bind(C) :: dtfft_platform_t
  !! Type that specifies runtime platform, e.g. Host, CUDA, HIP
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_platform_t

  type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_HOST = dtfft_platform_t(CONF_DTFFT_PLATFORM_HOST)
    !! Host platform
  type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_CUDA = dtfft_platform_t(CONF_DTFFT_PLATFORM_CUDA)
    !! CUDA platform
#ifdef DTFFT_WITH_CUDA
  type(dtfft_platform_t),         parameter :: VALID_PLATFORMS(*) = [DTFFT_PLATFORM_HOST, DTFFT_PLATFORM_CUDA]
    !! Valid platforms
#endif
  ! type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_HIP = dtfft_platform_t(3)

  type(dtfft_platform_t), public, parameter :: PLATFORM_NOT_SET = dtfft_platform_t(VARIABLE_NOT_SET)

public :: dtfft_request_t
  type, bind(C) :: dtfft_request_t
    type(c_ptr) :: val = c_null_ptr
  end type dtfft_request_t

public :: async_exec_t
  type :: async_exec_t
    integer(int32) :: val
  end type async_exec_t

  type(async_exec_t), parameter, public :: EXEC_BLOCKING = async_exec_t(1)
    !! Blocking execution
  type(async_exec_t), parameter, public :: EXEC_NONBLOCKING = async_exec_t(2)
    !! Non-blocking execution

  integer(int32),     parameter, public :: DEF_TILE_SIZE = 32
    !! Default tile size for CUDA kernels

public :: dtfft_transpose_mode_t
  type, bind(C) :: dtfft_transpose_mode_t
  !! This type specifies at which stage the local transposition is performed during global exchange.
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_transpose_mode_t

  type(dtfft_transpose_mode_t), parameter, public :: DTFFT_TRANSPOSE_MODE_PACK = dtfft_transpose_mode_t(CONF_DTFFT_TRANSPOSE_MODE_PACK)
    !! Perform transposition during the packing stage (Sender side).
  type(dtfft_transpose_mode_t), parameter, public :: DTFFT_TRANSPOSE_MODE_UNPACK = dtfft_transpose_mode_t(CONF_DTFFT_TRANSPOSE_MODE_UNPACK)
    !! Perform transposition during the unpacking stage (Receiver side).
  type(dtfft_transpose_mode_t), parameter :: VALID_TRANSPOSE_MODES(*) = [DTFFT_TRANSPOSE_MODE_PACK, DTFFT_TRANSPOSE_MODE_UNPACK]
    !! Valid transpose modes

  type(dtfft_transpose_mode_t), parameter, public :: TRANSPOSE_MODE_NOT_SET = dtfft_transpose_mode_t(VARIABLE_NOT_SET)
    !! Transpose mode not set
  character(len=*), parameter,  public :: TRANSPOSE_MODE_NAMES(CONF_DTFFT_TRANSPOSE_MODE_PACK:CONF_DTFFT_TRANSPOSE_MODE_UNPACK) = ["PACK  ", "UNPACK"]
    !! String representations of `dtfft_transpose_mode_t`

public :: dtfft_access_mode_t
  type, bind(C) :: dtfft_access_mode_t
  !! This type specifies which access pattern should be used for host kernels.
    integer(c_int32_t) :: val !! Internal value
  end type dtfft_access_mode_t

  type(dtfft_access_mode_t), parameter, public :: DTFFT_ACCESS_MODE_WRITE = dtfft_access_mode_t(CONF_DTFFT_ACCESS_MODE_WRITE)
    !! Optimize for write access (Aligned writing)
  type(dtfft_access_mode_t), parameter, public :: DTFFT_ACCESS_MODE_READ = dtfft_access_mode_t(CONF_DTFFT_ACCESS_MODE_READ)
    !! Optimize for read access (Aligned reading)
  type(dtfft_access_mode_t), parameter :: VALID_ACCESS_MODES(*) = [DTFFT_ACCESS_MODE_WRITE, DTFFT_ACCESS_MODE_READ]
    !! Valid access modes

  type(dtfft_access_mode_t), parameter, public :: ACCESS_MODE_NOT_SET = dtfft_access_mode_t(VARIABLE_NOT_SET)
    !! Access mode not set


  integer(int32), parameter,  public  :: DIMS_PERMUTE_FORWARD = -1
    !! For backward permutation
  integer(int32), parameter,  public  :: DIMS_PERMUTE_BACKWARD = +1
    !! For forward permutation
  integer(int32), parameter,  public  :: DIMS_PERMUTE_NONE = 0
    !! For reshape

contains

MAKE_EQ_FUN(dtfft_execute_t, execute_type_eq)
MAKE_EQ_FUN(dtfft_transpose_t, transpose_type_eq)
MAKE_EQ_FUN(dtfft_reshape_t, reshape_type_eq)
MAKE_EQ_FUN(dtfft_executor_t, executor_eq)
MAKE_EQ_FUN(dtfft_effort_t, effort_eq)
MAKE_EQ_FUN(dtfft_precision_t, precision_eq)
MAKE_EQ_FUN(dtfft_r2r_kind_t, r2r_kind_eq)
MAKE_EQ_FUN(dtfft_platform_t, platform_eq)
MAKE_EQ_FUN(async_exec_t, exec_eq)
MAKE_EQ_FUN(dtfft_layout_t, layout_eq)
MAKE_EQ_FUN(dtfft_transpose_mode_t, transpose_mode_eq)
MAKE_EQ_FUN(dtfft_access_mode_t, access_mode_eq)

MAKE_NE_FUN(dtfft_execute_t, execute_type_ne)
MAKE_NE_FUN(dtfft_transpose_t, transpose_type_ne)
MAKE_NE_FUN(dtfft_reshape_t, reshape_type_ne)
MAKE_NE_FUN(dtfft_executor_t, executor_ne)
MAKE_NE_FUN(dtfft_effort_t, effort_ne)
MAKE_NE_FUN(dtfft_precision_t, precision_ne)
MAKE_NE_FUN(dtfft_r2r_kind_t, r2r_kind_ne)
MAKE_NE_FUN(dtfft_platform_t, platform_ne)
MAKE_NE_FUN(dtfft_layout_t, layout_ne)
MAKE_NE_FUN(dtfft_transpose_mode_t, transpose_mode_ne)
MAKE_NE_FUN(dtfft_access_mode_t, access_mode_ne)

MAKE_VALID_FUN_DTYPE(dtfft_execute_t, is_valid_execute_type, VALID_EXECUTE_TYPES)
MAKE_VALID_FUN_DTYPE(dtfft_transpose_t, is_valid_transpose_type, VALID_TRANSPOSE_TYPES)
MAKE_VALID_FUN_DTYPE(dtfft_reshape_t, is_valid_reshape_type, VALID_RESHAPE_TYPES)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_valid_executor, VALID_EXECUTORS)
MAKE_VALID_FUN_DTYPE(dtfft_effort_t, is_valid_effort, VALID_EFFORTS)
MAKE_VALID_FUN_DTYPE(dtfft_precision_t, is_valid_precision, VALID_PRECISIONS)
MAKE_VALID_FUN_DTYPE(dtfft_r2r_kind_t, is_valid_r2r_kind, VALID_R2R_KINDS)
MAKE_VALID_FUN_DTYPE(dtfft_layout_t, is_valid_layout, VALID_LAYOUTS)
MAKE_VALID_FUN_DTYPE(dtfft_transpose_mode_t, is_valid_transpose_mode, VALID_TRANSPOSE_MODES)
MAKE_VALID_FUN_DTYPE(dtfft_access_mode_t, is_valid_access_mode, VALID_ACCESS_MODES)

MAKE_VALID_FUN(integer(int8), is_valid_dimension, VALID_DIMENSIONS)
MAKE_VALID_FUN(integer(int32), is_valid_comm_type, VALID_COMM_TYPES)

  integer(c_int32_t) function dtfft_get_version_current() bind(C)
  !! Returns the current version code
    dtfft_get_version_current = DTFFT_VERSION_CODE
  end function dtfft_get_version_current

  integer(int32) function dtfft_get_version_required(major, minor, patch)
  !! Returns the version code required by the user
    integer(int32), intent(in) :: major !! Major version
    integer(int32), intent(in) :: minor !! Minor version
    integer(int32), intent(in) :: patch !! Patch version

    dtfft_get_version_required = CONF_DTFFT_VERSION(major,minor,patch)
  end function dtfft_get_version_required

  function dtfft_get_precision_string(precision) result(string)
  !! Gets the string description of a precision
    type(dtfft_precision_t), intent(in) :: precision !! Precision type
    character(len=:), allocatable :: string !! Precision string

    select case ( precision%val )
    case ( DTFFT_SINGLE%val )
      allocate(string, source="Single")
    case ( DTFFT_DOUBLE%val )
      allocate(string, source="Double")
    case default
      allocate(string, source="Unknown precision")
    endselect
  end function dtfft_get_precision_string

  function dtfft_get_executor_string(executor) result(string)
  !! Gets the string description of an executor
    type(dtfft_executor_t), intent(in) :: executor !! Executor type
    character(len=:), allocatable :: string !! Executor string

    select case ( executor%val )
    case ( DTFFT_EXECUTOR_NONE%val )
      allocate(string, source="None")
    case ( DTFFT_EXECUTOR_FFTW3%val )
      allocate(string, source="FFTW3")
    case ( DTFFT_EXECUTOR_MKL%val )
      allocate(string, source="MKL")
    case ( DTFFT_EXECUTOR_CUFFT%val )
      allocate(string, source="CUFFT")
    case ( DTFFT_EXECUTOR_VKFFT%val )
      allocate(string, source="VKFFT")
    case default
      allocate(string, source="Unknown executor")
    endselect
  end function dtfft_get_executor_string

  function dtfft_get_backend_string(backend) result(string)
  !! Gets the string description of a backend
    type(dtfft_backend_t),  intent(in)  :: backend    !! Backend ID
    character(len=:),       allocatable :: string     !! Backend string

    select case ( backend%val )
    case ( DTFFT_BACKEND_MPI_DATATYPE%val )
      allocate(string, source="MPI_DATATYPE")
    case ( DTFFT_BACKEND_MPI_P2P%val )
      allocate(string, source="MPI_P2P")
    case ( DTFFT_BACKEND_MPI_A2A%val )
      allocate(string, source="MPI_A2A")
    case ( DTFFT_BACKEND_MPI_RMA%val )
      allocate(string, source="MPI_RMA")
    case ( DTFFT_BACKEND_MPI_RMA_PIPELINED%val )
      allocate(string, source="MPI_RMA_PIPELINED")
    case ( DTFFT_BACKEND_MPI_P2P_SCHEDULED%val )
      allocate(string, source="MPI_P2P_SCHEDULED")
    case ( DTFFT_BACKEND_NCCL%val )
      allocate(string, source="NCCL")
    case ( DTFFT_BACKEND_CUFFTMP%val )
      allocate(string, source="CUFFTMP")
    case ( DTFFT_BACKEND_MPI_P2P_PIPELINED%val )
      allocate(string, source="MPI_P2P_PIPELINED")
    case ( DTFFT_BACKEND_NCCL_PIPELINED%val )
      allocate(string, source="NCCL_PIPELINED")
    case ( DTFFT_BACKEND_CUFFTMP_PIPELINED%val )
      allocate(string, source="CUFFTMP_PIPELINED")
    case ( DTFFT_BACKEND_MPI_P2P_FUSED%val )
      allocate(string, source="MPI_P2P_FUSED")
    case ( DTFFT_BACKEND_MPI_RMA_FUSED%val )
      allocate(string, source="MPI_RMA_FUSED")
    case ( DTFFT_BACKEND_MPI_P2P_COMPRESSED%val )
      allocate(string, source="MPI_P2P_COMPRESSED")
    case ( DTFFT_BACKEND_MPI_RMA_COMPRESSED%val )
      allocate(string, source="MPI_RMA_COMPRESSED")
    case ( DTFFT_BACKEND_NCCL_COMPRESSED%val )
      allocate(string, source="NCCL_COMPRESSED")
    case ( DTFFT_BACKEND_ADAPTIVE%val )
      allocate(string, source="ADAPTIVE")
    case ( BACKEND_NOT_SET%val )
      allocate(string, source="None")
    case default
      allocate(string, source="Unknown backend")
    endselect
  end function dtfft_get_backend_string

  MAKE_EQ_FUN(dtfft_backend_t, backend_eq)
  MAKE_NE_FUN(dtfft_backend_t, backend_ne)

  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, dtfft_get_backend_pipelined, PIPELINED_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_pipelined, PIPELINED_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_mpi, MPI_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_fused, FUSED_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_rma, RMA_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_compressed, COMPRESSED_BACKENDS)

  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_valid_backend, VALID_BACKENDS)

  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_nccl, NCCL_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_cufftmp, CUFFTMP_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_backend_t, is_backend_nvshmem, NVSHMEM_BACKENDS)

#ifdef DTFFT_WITH_CUDA
  MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_host_executor, HOST_EXECUTORS)
  MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_cuda_executor, CUDA_EXECUTORS)
  MAKE_VALID_FUN_DTYPE(dtfft_platform_t, is_valid_platform, VALID_PLATFORMS)

  function stream_from_int64(cuda_stream) result(stream)
  !! Creates [[dtfft_stream_t]] from integer(cuda_stream_kind)
    integer(int64), intent(in)  :: cuda_stream  !! CUDA stream
    type(dtfft_stream_t)        :: stream       !! dtfft Stream

    stream = transfer(cuda_stream, stream)
  end function stream_from_int64

  function dtfft_get_cuda_stream(stream) result(cuda_stream)
  !! Returns the CUDA stream from [[dtfft_stream_t]]
    type(dtfft_stream_t), intent(in) :: stream      !! dtfft stream
    integer(int64)                   :: cuda_stream !! CUDA stream

    cuda_stream = transfer(stream, int64)
  end function dtfft_get_cuda_stream
#endif
end module dtfft_parameters