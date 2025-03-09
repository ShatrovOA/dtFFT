!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
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
#include "dtfft.f03"
module dtfft_transpose_plan_cuda
!! This module describes [[transpose_plan_cuda]] class
use iso_fortran_env
use iso_c_binding
use dtfft_interface_cuda_runtime
use dtfft_config
use dtfft_abstract_transpose_plan,        only: abstract_transpose_plan, create_cart_comm, alloc_mem, free_mem
use dtfft_abstract_backend,               only: backend_helper
use dtfft_nvrtc_kernel,                   only: clean_unused_cache
use dtfft_parameters
use dtfft_pencil,                         only: pencil, get_local_sizes
use dtfft_transpose_handle_cuda,          only: transpose_handle_cuda
use dtfft_utils
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
#include "dtfft_cuda.h"
#include "dtfft_private.h"
implicit none
private
public :: transpose_plan_cuda

  real(real32),    parameter :: MaxR4P  =  huge(1._real32)

  type, extends(abstract_transpose_plan) :: transpose_plan_cuda
  private
    type(dtfft_stream_t)                      :: stream
    type(c_ptr)                               :: aux
    logical                                   :: is_aux_alloc
    type(transpose_handle_cuda), allocatable  :: in_plans(:)
    type(transpose_handle_cuda), allocatable  :: out_plans(:)
  contains
    procedure :: create_private => create_cuda
    procedure :: execute_private => execute_cuda
    procedure :: destroy => destroy_cuda
  end type transpose_plan_cuda

contains

  integer(int32) function create_cuda(self, dims, transposed_dims, base_comm, comm_dims, effort, base_dtype, base_storage, is_custom_cart_comm, cart_comm, comms, pencils)
    class(transpose_plan_cuda),     intent(inout) :: self                 !! GPU transpose plan
    integer(int32),                 intent(in)    :: dims(:)              !! Global sizes of the transform requested
    integer(int32),                 intent(in)    :: transposed_dims(:,:) !! Transposed dimensions
    TYPE_MPI_COMM,                  intent(in)    :: base_comm            !! Base communicator
    integer(int32),                 intent(in)    :: comm_dims(:)         !! Number of processors in each dimension
    type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner type of effort
    TYPE_MPI_DATATYPE,              intent(in)    :: base_dtype           !! Base MPI_Datatype
    integer(int8),                  intent(in)    :: base_storage         !! Number of bytes needed to store single element
    logical,                        intent(in)    :: is_custom_cart_comm  !! is custom Cartesian communicator provided by user
    TYPE_MPI_COMM,                  intent(out)   :: cart_comm            !! Cartesian communicator
    TYPE_MPI_COMM,                  intent(out)   :: comms(:)             !! Array of 1d communicators
    type(pencil),                   intent(out)   :: pencils(:)           !! Data distributing meta
    integer(int8) :: n_transpose_plans
    integer(int8) :: d, ndims!, b
    integer(int32) :: comm_size, ierr
    ! integer(cuda_count_kind) :: free, total
    integer(int32), allocatable :: best_decomposition(:)
    ! real(real32) :: execution_time
    logical :: pencils_created
    real(real64) :: ts, te

    create_cuda = DTFFT_SUCCESS
    if ( .not.get_mpi_enabled() .and. .not.get_nccl_enabled() .and. .not.get_nvshmem_enabled() .and. effort == DTFFT_PATIENT) then
      create_cuda = DTFFT_ERROR_GPU_BACKENDS_DISABLED
      return
    endif

    ndims = size(dims, kind=int8)
    allocate( best_decomposition(ndims) )

    self%stream = get_user_stream()
    self%gpu_backend = get_user_gpu_backend()

    best_decomposition(:) = comm_dims(:)
    call MPI_Comm_size(base_comm, comm_size, ierr)
    if ( comm_size == 1 ) self%gpu_backend = BACKEND_NOT_SET

    pencils_created = .false.
    if ( ndims == 2 .or. is_custom_cart_comm .or. self%is_z_slab ) then
      pencils_created = .true.
      call create_cart_comm(base_comm, best_decomposition, cart_comm, comms)
      do d = 1, ndims
        call pencils(d)%create(ndims, d, transposed_dims(:,d), comms)
      enddo
    endif

    ts = MPI_Wtime()

    if ( effort == DTFFT_PATIENT .and. comm_size > 1) then
      if ( pencils_created ) then
        call run_autotune_backend(comms, cart_comm, pencils, base_storage, self%stream, self%is_z_slab, best_backend_id=self%gpu_backend)
      else
        call autotune_grid_decomposition(dims, transposed_dims, base_comm, base_storage, self%stream, best_decomposition, best_backend_id=self%gpu_backend)
      endif
    else if ( ndims == 3                                &
      .and. .not.is_custom_cart_comm                    &
      .and. .not.self%is_z_slab                         &
      .and. effort == DTFFT_MEASURE                     &
      .and. comm_size > 1 ) then

      call autotune_grid_decomposition(dims, transposed_dims, base_comm, base_storage,self%stream, best_decomposition, gpu_backend=self%gpu_backend)
    endif
    te = MPI_Wtime()

    if ( effort%val >= DTFFT_MEASURE%val .and. ndims > 2 .and. comm_size > 1 ) then
      WRITE_INFO(repeat("*", 50))
      if ( self%is_z_slab ) then
        WRITE_INFO("Skipped search of MPI processor grid due to Z-slab optimization enabled")
      else if ( is_custom_cart_comm ) then
        WRITE_INFO("Skipped search of MPI processor grid due to custom grid provided")
      else
        WRITE_INFO("DTFFT_MEASURE: Selected MPI processor grid 1x"//int_to_str(best_decomposition(2))//"x"//int_to_str(best_decomposition(3)))
      endif
    endif
    if ( effort == DTFFT_PATIENT .and. comm_size > 1 ) then
      WRITE_INFO("DTFFT_PATIENT: Selected backend is "//dtfft_get_gpu_backend_string(self%gpu_backend))
    endif
    if ( effort%val >= DTFFT_MEASURE%val .and. comm_size > 1 ) then
      WRITE_INFO("Time spent on autotune: "//double_to_str(te - ts)//" [s]")
    endif

    if ( .not.pencils_created ) then
      call create_cart_comm(base_comm, best_decomposition, cart_comm, comms)
      do d = 1, ndims
        call pencils(d)%create(ndims, d, transposed_dims(:,d), comms)
      enddo
    endif
    n_transpose_plans = ndims - 1_int8; if( self%is_z_slab ) n_transpose_plans = n_transpose_plans + 1_int8
    allocate( self%out_plans(n_transpose_plans), self%in_plans(n_transpose_plans) )

    call self%helper%create(cart_comm, comms, is_backend_nccl(self%gpu_backend), pencils)

    do d = 1_int8, ndims - 1_int8
      call self%out_plans(d)%create(self%helper, pencils(d), pencils(d + 1), base_storage, self%gpu_backend)
      call self%in_plans (d)%create(self%helper, pencils(d + 1), pencils(d), base_storage, self%gpu_backend)
    enddo
    if ( self%is_z_slab ) then
      call self%out_plans(3)%create(self%helper, pencils(1), pencils(3), base_storage, self%gpu_backend)
      call self%in_plans (3)%create(self%helper, pencils(3), pencils(1), base_storage, self%gpu_backend)
    endif
    self%is_aux_alloc = alloc_and_set_aux(self%helper, self%gpu_backend, cart_comm, self%aux, self%in_plans, self%out_plans)

    call clean_unused_cache()
    deallocate( best_decomposition )
    create_cuda = DTFFT_SUCCESS
  end function create_cuda

  subroutine execute_cuda(self, in, out, transpose_type)
  !! Executes single transposition
    class(transpose_plan_cuda),    intent(inout) :: self          !! Transposition class
    type(*),              target,  intent(inout) :: in(..)        !! Incoming buffer of any rank and kind
    type(*),              target,  intent(inout) :: out(..)       !! Resulting buffer of any rank and kind
    type(dtfft_transpose_type_t),  intent(in)    :: transpose_type
    real(real32),   pointer :: pin(:)
    real(real32),   pointer :: pout(:)

    pin => convert_pointer(c_loc(in), 1_int64)
    pout => convert_pointer(c_loc(out), 1_int64)

    if ( transpose_type%val > 0 ) then
      call self%out_plans(transpose_type%val)%execute(pin, pout, self%stream)
    else
      call self%in_plans(abs(transpose_type%val))%execute(pin, pout, self%stream)
    endif
  end subroutine execute_cuda

  subroutine destroy_cuda(self)
  !! Destroys transposition plans
    class(transpose_plan_cuda),    intent(inout) :: self         !! Transposition class
    integer(int8) :: i
    integer(int32) :: ierr

    if ( self%is_aux_alloc ) then
      call self%mem_free(self%aux, ierr)
      self%is_aux_alloc = .false.
    endif

    do i = 1, size(self%in_plans, kind=int8)
      call self%out_plans(i)%destroy()
      call self% in_plans(i)%destroy()
    enddo

    call self%helper%destroy()
    deallocate(self%out_plans)
    deallocate(self% in_plans)
  end subroutine destroy_cuda

  subroutine autotune_grid_decomposition(dims, transposed_dims, base_comm, base_storage, stream, best_decomposition, gpu_backend, min_execution_time, best_backend_id)
    integer(int32),                       intent(in)    :: dims(:)              !! Global sizes of the transform requested
    integer(int32),                       intent(in)    :: transposed_dims(:,:)
    TYPE_MPI_COMM,                        intent(in)    :: base_comm            !! 3D comm
    integer(int8),                        intent(in)    :: base_storage         !! Number of bytes needed to store Basic MPI Datatype
    type(dtfft_stream_t),                 intent(in)    :: stream
    integer(int32),                       intent(out)   :: best_decomposition(:)
    type(dtfft_gpu_backend_t),  optional, intent(in)    :: gpu_backend    !! ID of transpose name (from -3 to 3, except 0)
    real(real32),               optional, intent(out)   :: min_execution_time         !! Elapsed time for best plans selected
    type(dtfft_gpu_backend_t),  optional, intent(out)   :: best_backend_id
    integer(int8)   :: ndims
    type(dtfft_gpu_backend_t) :: best_backend_id_
    integer(int32)  :: comm_size, square_root, i, current_timer, k, ierr
    real(real32) :: current_time, elapsed_time
    real(real32), allocatable :: timers(:)
    integer(int32),   allocatable :: decomps(:,:)
    type(dtfft_gpu_backend_t), allocatable :: backends(:)

    call MPI_Comm_size(base_comm, comm_size, ierr)
    ndims = size(dims, kind=int8)

    square_root = int(sqrt(real(comm_size, real64))) + 1
    allocate(timers(2 * square_root))
    allocate(decomps(2, 2 * square_root))
    allocate(backends(2 * square_root))

    current_timer = 0
    do i = 1, square_root - 1
      if ( mod( comm_size, i ) /= 0 ) cycle

      call autotune_grid(dims, transposed_dims, base_comm, [1, i, comm_size / i], base_storage, .false., stream, gpu_backend=gpu_backend, best_time=current_time,best_backend_id=best_backend_id_)
      if ( current_time > 0.0 ) then
        current_timer = current_timer + 1
        timers(current_timer) = current_time
        decomps(1, current_timer) = i
        decomps(2, current_timer) = comm_size / i
        backends(current_timer) = best_backend_id_
      endif
      if ( i /= comm_size / i) then
        call autotune_grid(dims, transposed_dims, base_comm, [1, comm_size / i, i], base_storage, .false., stream, gpu_backend=gpu_backend, best_time=current_time,best_backend_id=best_backend_id_)
        if ( current_time > 0.0 ) then
          current_timer = current_timer + 1
          timers(current_timer) = current_time
          decomps(1, current_timer) = comm_size / i
          decomps(2, current_timer) = i
          backends(current_timer) = best_backend_id_
        endif
      endif
    enddo

    elapsed_time = MaxR4P
    k = 1
    do i = 1, current_timer
      if ( timers(i) < elapsed_time ) then
        elapsed_time = timers(i)
        k = i
      endif
    enddo

    best_decomposition(1) = 1
    best_decomposition(2) = decomps(1, k)
    best_decomposition(3) = decomps(2, k)
    if ( present(best_backend_id) ) best_backend_id = backends(k)
    if ( present(min_execution_time) ) min_execution_time = elapsed_time
    deallocate( timers, decomps, backends )
  end subroutine autotune_grid_decomposition

  subroutine autotune_grid(dims, transposed_dims, base_comm, comm_dims, base_storage, is_z_slab, stream, gpu_backend, best_time, best_backend_id)
    integer(int32),                       intent(in)    :: dims(:)              !! Global sizes of the transform requested
    integer(int32),                       intent(in)    :: transposed_dims(:,:)
    TYPE_MPI_COMM,                        intent(in)    :: base_comm            !! 3D comm
    integer(int32),                       intent(in)    :: comm_dims(:)
    integer(int8),                        intent(in)    :: base_storage         !! Number of bytes needed to store Basic MPI Datatype
    logical,                              intent(in)    :: is_z_slab
    type(dtfft_stream_t),                 intent(in)    :: stream
    type(dtfft_gpu_backend_t),  optional, intent(in)    :: gpu_backend           !! ID of transpose name (from -3 to 3, except 0)
    type(dtfft_gpu_backend_t),  optional, intent(out)   :: best_backend_id
    real(real32),               optional, intent(out)   :: best_time         !! Elapsed time for best plans selected
    type(pencil), allocatable :: pencils(:)
    character(len=:),             allocatable   :: phase_name           !! Caliper phase name
    integer(int8) :: d, ndims
    TYPE_MPI_COMM, allocatable :: comms(:)
    TYPE_MPI_COMM :: cart_comm
    integer(int32) :: mpi_ierr

    best_time = -1.0
    ndims = size(dims, kind=int8)

    if ( ndims == 3 ) then
      if ( comm_dims(2) > dims(2) .or. comm_dims(3) > dims(3) ) return
      allocate( phase_name, source = "Testing grid 1x"//int_to_str(comm_dims(2))//"x"//int_to_str(comm_dims(3)) )
    else
      allocate( phase_name, source = "Testing grid 1x"//int_to_str(comm_dims(2)) )
    endif

    WRITE_INFO("")
    WRITE_INFO(phase_name)
    PHASE_BEGIN(phase_name, COLOR_AUTOTUNE)

    allocate( comms(ndims), pencils(ndims) )
    call create_cart_comm(base_comm, comm_dims, cart_comm, comms)
    do d = 1, ndims
      call pencils(d)%create(ndims, d, transposed_dims(:,d), comms)
    enddo

    ! elapsed_time = autotune_backend_id(comms, cart_comm, pencils, base_storage, gpu_backend, stream)

    call run_autotune_backend(comms, cart_comm, pencils, base_storage, stream, is_z_slab, gpu_backend=gpu_backend, best_time=best_time, best_backend_id=best_backend_id)

    do d = 1, ndims
      call pencils(d)%destroy()
      call MPI_Comm_free(comms(d), mpi_ierr)
    enddo
    call MPI_Comm_free(cart_comm, mpi_ierr)
    deallocate( comms, pencils )
    PHASE_END(phase_name)
  end subroutine autotune_grid

  subroutine run_autotune_backend(comms, cart_comm, pencils, base_storage, stream, is_z_slab, gpu_backend, best_time, best_backend_id)
    TYPE_MPI_COMM,                        intent(in)    :: comms(:)              !! 1D comms
    TYPE_MPI_COMM,                        intent(in)    :: cart_comm            !! 3D Cartesian comm
    type(pencil),                         intent(in)    :: pencils(:)           !! Source meta
    integer(int8),                        intent(in)    :: base_storage         !! Number of bytes needed to store Basic MPI Datatype
    type(dtfft_stream_t),                 intent(in)    :: stream
    logical,                              intent(in)    :: is_z_slab
    type(dtfft_gpu_backend_t),  optional, intent(in)    :: gpu_backend           !! ID of transpose name (from -3 to 3, except 0)
    real(real32),               optional, intent(out)   :: best_time
    type(dtfft_gpu_backend_t),  optional, intent(out)   :: best_backend_id
    type(dtfft_gpu_backend_t),  allocatable :: backends_to_run(:)
    type(dtfft_gpu_backend_t) :: current_backend_id, best_backend_id_
    logical :: is_udb !! Used defined backend
    real(real32) :: execution_time, min_execution_time, max_execution_time, avg_execution_time, best_time_, total_time
    integer(int32) :: iter, comm_size, mpi_ierr, b, ierr
    type(transpose_handle_cuda),  allocatable   :: plans(:)
    integer(int8) :: i, n_transpose_plans
    type(c_ptr) :: in, out, aux
    real(real32), pointer :: pin(:), pout(:)
    logical :: is_aux_alloc
    ! , need_aux
    integer(int64)         :: alloc_size
    type(cudaEvent) :: timer_start, timer_stop
    character(len=:), allocatable :: testing_phase
    type(backend_helper)                      :: helper
    integer(int32) :: n_warmup_iters, n_iters
    ! integer(int64)                                :: scaler         !! Scaling data amount to float size
    ! integer(cuda_count_kind) :: free, total

    if ( present(gpu_backend) ) then
      allocate( backends_to_run(1) )
      backends_to_run(1) = gpu_backend
      is_udb = .true.
    else
      allocate(backends_to_run(size(VALID_GPU_BACKENDS(2:))))
      do b = 1, size(VALID_GPU_BACKENDS(2:))
        backends_to_run(b) = VALID_GPU_BACKENDS(b + 1)
      enddo
      is_udb = .false.
    endif
    best_backend_id_ = backends_to_run(1)

    if ( is_z_slab ) then
      n_transpose_plans = 1
    else
      n_transpose_plans = size(pencils, kind=int8) - 1_int8
    endif

    allocate( plans(2 * n_transpose_plans) )

    call get_local_sizes(pencils, alloc_size=alloc_size)
    alloc_size = alloc_size * int(base_storage, int64)

    CUDA_CALL( "cudaEventCreate", cudaEventCreate(timer_start) )
    CUDA_CALL( "cudaEventCreate", cudaEventCreate(timer_stop) )

    call MPI_Comm_size(cart_comm, comm_size, mpi_ierr)

    call helper%create(cart_comm, comms, any(is_backend_nccl(backends_to_run)), pencils)

    best_time_ = MaxR4P

    do b = 1, size(backends_to_run)
      current_backend_id = backends_to_run(b)

      if ( (is_backend_pipelined(current_backend_id) .and. .not.get_pipelined_enabled()         &
            .or.is_backend_mpi(current_backend_id) .and. .not.get_mpi_enabled()                 &
            .or.is_backend_nvshmem(current_backend_id) .and. .not.get_nvshmem_enabled())        &
            .and. .not.is_udb) cycle

      if ( is_z_slab ) then
        call plans(1)%create(helper, pencils(1), pencils(3), base_storage, current_backend_id)
        call plans(2)%create(helper, pencils(3), pencils(1), base_storage, current_backend_id)
      else
        do i = 1, n_transpose_plans
          call plans(i)%create(helper, pencils(i), pencils(i + 1), base_storage, current_backend_id)
          call plans(i + n_transpose_plans)%create(helper, pencils(i + 1), pencils(i), base_storage, current_backend_id)
        enddo
      endif

      call alloc_mem(helper, current_backend_id, cart_comm, alloc_size, in, ierr); DTFFT_CHECK(ierr)
      call alloc_mem(helper, current_backend_id, cart_comm, alloc_size, out, ierr); DTFFT_CHECK(ierr)

      pin => convert_pointer(in, 1_int64)
      pout => convert_pointer(out, 1_int64)

      is_aux_alloc = alloc_and_set_aux(helper, current_backend_id, cart_comm, aux, plans)

      testing_phase = "Testing backend "//dtfft_get_gpu_backend_string(current_backend_id)
      PHASE_BEGIN(testing_phase, COLOR_AUTOTUNE)
      WRITE_INFO(testing_phase)

      n_warmup_iters = get_iters_from_env(.true.)

      PHASE_BEGIN("Warmup, "//int_to_str(n_warmup_iters)//" iterations", COLOR_TRANSPOSE)
      do iter = 1, n_warmup_iters
        do i = 1, 2_int8 * n_transpose_plans
          call plans(i)%execute(pin, pout, stream)
        enddo
      enddo
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
      PHASE_END("Warmup, "//int_to_str(n_warmup_iters)//" iterations")

      call MPI_Barrier(cart_comm, mpi_ierr)

      n_iters = get_iters_from_env(.false.)

      PHASE_BEGIN("Testing, "//int_to_str(n_iters)//" iterations", COLOR_EXECUTE)
      total_time = 0.0
    ! do i = 1, 2_int8 * n_transpose_plans
      CUDA_CALL( "cudaEventRecord", cudaEventRecord(timer_start, stream) )
      do iter = 1, n_iters
        do i = 1, 2_int8 * n_transpose_plans
          call plans(i)%execute(pin, pout, stream)
        enddo
      enddo
      CUDA_CALL( "cudaEventRecord", cudaEventRecord(timer_stop, stream) )
      CUDA_CALL( "cudaEventSynchronize", cudaEventSynchronize(timer_stop) )
      CUDA_CALL( "cudaEventElapsedTime", cudaEventElapsedTime(execution_time, timer_start, timer_stop) )
      execution_time = execution_time / real(n_iters, real32)
      total_time = total_time + execution_time
      ! WRITE_INFO( TRANSPOSE_NAMES(plans(i)%get_tranpose_id())//" : "//double_to_str(real(execution_time, real64))//" [ms]")
    ! enddo

      PHASE_END("Testing, "//int_to_str(n_iters)//" iterations")

      call MPI_Allreduce(total_time, min_execution_time, 1, MPI_REAL4, MPI_MIN, cart_comm, mpi_ierr)
      call MPI_Allreduce(total_time, max_execution_time, 1, MPI_REAL4, MPI_MAX, cart_comm, mpi_ierr)
      call MPI_Allreduce(total_time, avg_execution_time, 1, MPI_REAL4, MPI_SUM, cart_comm, mpi_ierr)

      avg_execution_time = avg_execution_time / real(comm_size, real32)

      WRITE_INFO("  max: "//double_to_str(real(max_execution_time, real64))//" [ms]")
      WRITE_INFO("  min: "//double_to_str(real(min_execution_time, real64))//" [ms]")
      WRITE_INFO("  avg: "//double_to_str(real(avg_execution_time, real64))//" [ms]")

      if ( avg_execution_time < best_time_ ) then
        best_time_ = avg_execution_time
        best_backend_id_ = current_backend_id
      endif

      call free_mem(helper, current_backend_id, in, ierr)
      call free_mem(helper, current_backend_id, out, ierr)
      if ( is_aux_alloc ) call free_mem(helper, current_backend_id, aux, ierr)


      do i = 1, 2_int8 * n_transpose_plans
        call plans(i)%destroy()
      enddo

      PHASE_END("Testing backend "//dtfft_get_gpu_backend_string(current_backend_id))
    enddo

    ! CUDA_CALL( "cudaFree", cudaFree(in) )
    ! CUDA_CALL( "cudaFree", cudaFree(out) )
    ! if ( is_aux_alloc ) then
    !   CUDA_CALL( "cudaFree", cudaFree(aux) )
    ! endif

    deallocate( plans )
    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(timer_start) )
    CUDA_CALL( "cudaEventDestroy", cudaEventDestroy(timer_stop) )

    call helper%destroy()

    if ( present(best_time)) best_time = best_time_
    if ( present(best_backend_id) ) best_backend_id = best_backend_id_
  end subroutine run_autotune_backend

  function alloc_and_set_aux(helper, gpu_backend, cart_comm, aux, plans, out_plans) result(is_aux_alloc)
    type(backend_helper),         intent(inout)             :: helper
    type(dtfft_gpu_backend_t),    intent(in)                :: gpu_backend
    TYPE_MPI_COMM,                intent(in)                :: cart_comm
    type(c_ptr),                  intent(inout)             :: aux
    type(transpose_handle_cuda),  intent(inout)             :: plans(:)
    type(transpose_handle_cuda),  intent(inout),  optional  :: out_plans(:)
    logical                                                 :: is_aux_alloc
    integer(int64), allocatable :: worksizes(:)
    integer(int64) :: max_work_size_local, max_work_size_global
    integer(int32)  :: mpi_ierr, n_transpose_plans, i, n_in_plans, n_out_plans
    integer(int32) :: alloc_ierr
    real(real32),   pointer :: paux(:)

    n_in_plans = size(plans)
    n_out_plans = 0;  if ( present(out_plans) ) n_out_plans = size(out_plans)
    n_transpose_plans = n_in_plans + n_out_plans

    allocate( worksizes( n_transpose_plans ) )

    do i = 1, n_in_plans
      worksizes(i) = plans(i)%get_aux_size()
    enddo
    do i = 1, n_out_plans
      worksizes(i + n_in_plans) = out_plans(i)%get_aux_size()
    enddo

    max_work_size_local = maxval(worksizes)
    call MPI_Allreduce(max_work_size_local, max_work_size_global, 1, MPI_INTEGER8, MPI_MAX, cart_comm, mpi_ierr)

    is_aux_alloc = .false.
    if ( max_work_size_global > 0 ) then
      call alloc_mem(helper, gpu_backend, cart_comm, max_work_size_global, aux, alloc_ierr)
      if ( alloc_ierr /= DTFFT_SUCCESS ) error stop "Failed to allocate aux memory"
      paux => convert_pointer(aux, 1_int64)
      do i = 1, n_in_plans
        call plans(i)%set_aux(paux)
      enddo
      do i = 1, n_out_plans
        call out_plans(i)%set_aux(paux)
      enddo
      is_aux_alloc = .true.
    endif

    deallocate( worksizes )
  end function alloc_and_set_aux

  function convert_pointer(in, size) result(out)
    type(c_ptr),            intent(in)  :: in
    integer(int64),         intent(in)  :: size
    real(real32),     pointer :: out(:)

    call c_f_pointer(in, out, [size])
  end function convert_pointer
end module dtfft_transpose_plan_cuda