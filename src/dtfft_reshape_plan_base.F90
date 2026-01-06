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
#include "dtfft.f03"
module dtfft_reshape_plan_base
use iso_c_binding
use iso_fortran_env
use dtfft_abstract_reshape_handle,      only: reshape_container, execute_args
use dtfft_abstract_backend,             only: backend_helper
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_pencil,                       only: pencil_init, pencil
#ifdef DTFFT_WITH_CUDA
# ifdef NCCL_HAVE_COMMREGISTER
use dtfft_abstract_backend,             only: NCCL_REGISTER_PREALLOC_SIZE
# endif
use dtfft_interface_cuda,               only: load_cuda
use dtfft_interface_cuda_runtime
use dtfft_interface_nvrtc,              only: load_nvrtc
# ifdef DTFFT_WITH_NCCL
use dtfft_interface_nccl
# endif
# ifdef DTFFT_WITH_NVSHMEM
use dtfft_interface_nvshmem
# endif
#endif
use dtfft_reshape_handle_generic,       only: reshape_handle_generic
use dtfft_reshape_handle_datatype,      only: reshape_handle_datatype
use dtfft_utils
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: reshape_plan_base
public :: allocate_plans, destroy_plans
public :: get_aux_bytes_generic
public :: execute_autotune
public :: report_timings

type :: reshape_plan_base
    type(dtfft_backend_t)     :: backend
        !! Backend
    type(backend_helper)      :: helper
        !! Backend helper
    integer(int64)            :: min_buffer_size
        !! Minimal buffer size for transposition
    type(dtfft_platform_t)    :: platform
        !! Platform used for transposition
    type(dtfft_stream_t)      :: stream
        !! CUDA stream
    type(reshape_container), allocatable :: plans(:)
        !! Plans for each reshape operation
    type(string), allocatable :: names(:)
        !! Names of each reshape operation
contains
procedure,  pass(self), non_overridable :: init
procedure,  pass(self), non_overridable :: get_async_active
procedure,  pass(self), non_overridable :: get_aux_bytes      !! Returns auxiliary buffer size
procedure,  pass(self), non_overridable :: is_aux_needed
procedure,  pass(self), non_overridable :: get_backend       !! Returns backend id
procedure,  pass(self)                  :: destroy
procedure,  pass(self), non_overridable :: execute
procedure,  pass(self), non_overridable :: execute_end
procedure,  pass(self), non_overridable :: mem_alloc
procedure,  pass(self), non_overridable :: mem_free
end type reshape_plan_base

contains

function init(self, platform, effort) result(error_code)
class(reshape_plan_base),   intent(inout) :: self
type(dtfft_platform_t),     intent(in)    :: platform
type(dtfft_effort_t),       intent(in)    :: effort
integer(int32)                            :: error_code

    error_code = DTFFT_SUCCESS
    self%stream = NULL_STREAM
    self%backend = BACKEND_NOT_SET
    self%platform = platform
    if ( platform == DTFFT_PLATFORM_HOST ) then
        if ( .not. get_conf_datatype_enabled() .and. .not. get_conf_mpi_enabled() .and. effort%val >= DTFFT_PATIENT%val) then
            error_code = DTFFT_ERROR_BACKENDS_DISABLED
            return
        endif
#ifdef DTFFT_WITH_CUDA
    else
        if ( .not.get_conf_mpi_enabled() .and. .not.get_conf_nccl_enabled() .and. .not.get_conf_nvshmem_enabled() .and. effort%val >= DTFFT_PATIENT%val) then
            error_code = DTFFT_ERROR_BACKENDS_DISABLED
            return
        endif

        CHECK_CALL( load_cuda(), error_code )
        CHECK_CALL( load_nvrtc(), error_code )
        self%stream = get_conf_stream()
#endif
    endif
end function init

subroutine execute(self, in, out, reshape_type, exec_type, aux, error_code)
!! Executes transposition
class(reshape_plan_base),     intent(inout) :: self           !! Transposition class
type(c_ptr),                  intent(in)    :: in             !! Incoming buffer
type(c_ptr),                  intent(in)    :: out            !! Resulting buffer
integer(int32),               intent(in)    :: reshape_type   !! Type of reshape to execute
type(c_ptr),                  intent(in)    :: aux            !! Optional auxiliary buffer
type(async_exec_t),           intent(in)    :: exec_type      !! Type of execution (sync/async)
integer(int32),    optional,  intent(out)   :: error_code     !! Error code
real(real32), pointer :: pin(:)   !! Source buffer
real(real32), pointer :: pout(:)  !! Destination buffer
real(real32), pointer :: paux(:)  !! Auxiliary buffer
type(execute_args)    :: kwargs   !! Additional arguments for execution
integer(int32)        :: ierr     !! Error code

#ifdef DTFFT_DEBUG
    if ( is_same_ptr(in, out) .or. is_same_ptr(in, aux) .or. is_same_ptr(out, aux) ) INTERNAL_ERROR("reshape_plan_base: wrong pointers for reshape "//self%names(reshape_type)%raw)
    if ( is_null_ptr(in) .or. is_null_ptr(out) ) INTERNAL_ERROR("reshape_plan_base: null pointer detected for reshape "//self%names(reshape_type)%raw)
#endif

    REGION_BEGIN('Reshape '//self%names(reshape_type)%raw, COLOR_AUTOTUNE2)
    call c_f_pointer(in, pin, [self%min_buffer_size])
    call c_f_pointer(out, pout, [self%min_buffer_size])
    kwargs%exec_type = exec_type
    kwargs%stream = self%stream

    if ( is_null_ptr(aux) ) then
        kwargs%p1 => pin
    else
        call c_f_pointer(aux, paux, [self%min_buffer_size])
        kwargs%p1 => paux
    endif
    call self%plans(reshape_type)%p%execute(pin, pout, kwargs, ierr)
    if( present( error_code ) ) error_code = ierr
    REGION_END('Reshape '//self%names(reshape_type)%raw)
end subroutine execute

subroutine execute_end(self, in, out, reshape_type, aux, error_code)
!! Finishes asynchronous reshape
class(reshape_plan_base),   intent(inout) :: self           !! Reshape class
type(c_ptr),                intent(in)    :: in             !! Incoming buffer
type(c_ptr),                intent(in)    :: out            !! Resulting buffer
integer(int32),             intent(in)    :: reshape_type   !! Type of reshape to execute
type(c_ptr),                intent(in)    :: aux
integer(int32),             intent(out)   :: error_code     !! Error code
real(real32),   pointer :: pin(:)   !! Source buffer
real(real32),   pointer :: pout(:)  !! Destination buffer
real(real32),   pointer :: paux(:)  !! Aux buffer
type(execute_args)      :: kwargs   !! Additional arguments for execution

    REGION_BEGIN('Reshape '//self%names(reshape_type)%raw//' end', COLOR_AUTOTUNE2)
    call c_f_pointer(in, pin, [self%min_buffer_size])
    call c_f_pointer(out, pout, [self%min_buffer_size])

    kwargs%p1 => pin
    kwargs%p2 => pout
    if ( .not. is_null_ptr(aux) ) then
        call c_f_pointer(aux, paux, [self%min_buffer_size])
        kwargs%p3 => paux
    else
        kwargs%p3 => pin
    endif
    kwargs%stream = self%stream
    call self%plans(reshape_type)%p%execute_end(kwargs, error_code)
    REGION_END('Reshape '//self%names(reshape_type)%raw//' end')
end subroutine execute_end

logical function get_async_active(self)
!! Returns .true. if any of the plans is running asynchronously
class(reshape_plan_base),   intent(in)    :: self           !! Transposition class
integer(int32)  :: i

    get_async_active = .false.
    do i = lbound(self%plans, dim=1), ubound(self%plans, dim=1)
        if ( allocated( self%plans(i)%p ) ) then
            get_async_active = get_async_active .or. self%plans(i)%p%get_async_active()
        endif
    enddo
end function get_async_active

pure integer(int64) function get_aux_bytes(self)
!! Returns maximum auxiliary memory size needed by transpose plan
class(reshape_plan_base), intent(in)    :: self  !! Transposition class
    get_aux_bytes = 0_int64
    if ( .not. allocated( self%plans ) ) return
    get_aux_bytes = get_aux_bytes_generic(self%plans)
end function get_aux_bytes

pure logical function is_aux_needed(self)
!! Returns true if aux is needed. false otherwise
class(reshape_plan_base), intent(in)    :: self  !! Transposition class
    is_aux_needed = self%get_aux_bytes() > 0
end function is_aux_needed

type(dtfft_backend_t) function get_backend(self)
!! Returns plan backend
class(reshape_plan_base), intent(in)    :: self           !! Transposition class
    get_backend = self%backend
end function get_backend

subroutine destroy(self)
class(reshape_plan_base),   intent(inout) :: self
integer(int32) :: i

    if ( allocated( self%plans ) ) then
        call destroy_plans(self%plans)
        deallocate( self%plans )
    endif
    call self%helper%destroy()
    if ( allocated(self%names) ) then
        do i = lbound(self%names, dim=1), ubound(self%names, dim=1)
            call self%names(i)%destroy()
        enddo
        deallocate(self%names)
    endif
end subroutine destroy

subroutine mem_alloc(self, comm, alloc_bytes, ptr, error_code)
!! Allocates memory based on selected backend
class(reshape_plan_base),  intent(inout) :: self           !! Transposition class
TYPE_MPI_COMM,          intent(in)    :: comm           !! MPI communicator
integer(int64),         intent(in)    :: alloc_bytes    !! Number of bytes to allocate
type(c_ptr),            intent(out)   :: ptr            !! Pointer to the allocated memory
integer(int32),         intent(out)   :: error_code     !! Error code

    call alloc_mem(self%platform, self%helper, self%backend, comm, alloc_bytes, ptr, error_code)
end subroutine mem_alloc

subroutine mem_free(self, ptr, error_code)
!! Frees memory allocated with mem_alloc
class(reshape_plan_base),  intent(inout) :: self           !! Transposition class
type(c_ptr),            intent(in)    :: ptr            !! Pointer to the memory to free
integer(int32),         intent(out)   :: error_code     !! Error code

    call free_mem(self%platform, self%helper, self%backend, ptr, error_code)
end subroutine mem_free

subroutine alloc_mem(platform, helper, backend, comm, alloc_bytes, ptr, error_code)
!! Allocates memory based on ``backend``
type(dtfft_platform_t), intent(in)    :: platform
type(backend_helper),   intent(inout) :: helper         !! Backend helper
type(dtfft_backend_t),  intent(in)    :: backend        !! GPU backend
TYPE_MPI_COMM,          intent(in)    :: comm           !! MPI communicator
integer(int64),         intent(in)    :: alloc_bytes    !! Number of bytes to allocate
type(c_ptr),            intent(out)   :: ptr            !! Pointer to the allocated memory
integer(int32),         intent(out)   :: error_code     !! Error code
#ifdef DTFFT_WITH_CUDA
integer(int64)  :: free_mem_avail, total_mem_avail
#endif

    error_code = DTFFT_SUCCESS
    if ( platform == DTFFT_PLATFORM_HOST ) then
        ptr = mem_alloc_host(alloc_bytes)
#ifdef DTFFT_WITH_CUDA
    else
        CUDA_CALL( cudaMemGetInfo(free_mem_avail, total_mem_avail) )
# ifdef DTFFT_DEBUG
        block
        integer(int64) :: min_mem, max_mem, min_free_mem, max_free_mem
        integer(int32) :: mpi_err

            call MPI_Allreduce(alloc_bytes, max_mem, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
            call MPI_Allreduce(alloc_bytes, min_mem, 1, MPI_INTEGER8, MPI_MIN, comm, mpi_err)
            call MPI_Allreduce(free_mem_avail, max_free_mem, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
            call MPI_Allreduce(free_mem_avail, min_free_mem, 1, MPI_INTEGER8, MPI_MIN, comm, mpi_err)
            WRITE_DEBUG("Trying to allocate "//to_str(min_mem)//"/"//to_str(max_mem)//" (min/max) bytes for backend: '"//dtfft_get_backend_string(backend)//"'")
            WRITE_DEBUG("Free memory available: "//to_str(min_free_mem)//"/"//to_str(max_free_mem)//" (min/max) bytes")
        endblock
# endif
        if ( alloc_bytes > free_mem_avail ) then
            error_code = DTFFT_ERROR_ALLOC_FAILED
            return
        endif
        if ( is_backend_nccl(backend) ) then
# ifdef DTFFT_WITH_NCCL
#   ifdef NCCL_HAVE_MEMALLOC
            error_code = ncclMemAlloc(ptr, alloc_bytes)
#   else
            error_code = cudaMalloc(ptr, alloc_bytes)
#   endif
#   ifdef NCCL_HAVE_COMMREGISTER
            if ( error_code == cudaSuccess .and. helper%should_register ) then
                block
                type(c_ptr), allocatable :: temp(:,:)
                type(c_ptr) :: handle

                    if ( size(helper%nccl_register, dim=2) == helper%nccl_register_size ) then
                        allocate( temp(2, helper%nccl_register_size + NCCL_REGISTER_PREALLOC_SIZE) )
                        temp(2, 1:helper%nccl_register_size) = helper%nccl_register(2, 1:helper%nccl_register_size)
                        deallocate( helper%nccl_register )
                        call move_alloc(temp, helper%nccl_register)
                    endif
                    helper%nccl_register_size = helper%nccl_register_size + 1

                    NCCL_CALL( ncclCommRegister(helper%nccl_comm, ptr, alloc_bytes, handle) )
                    helper%nccl_register(1, helper%nccl_register_size) = ptr
                    helper%nccl_register(2, helper%nccl_register_size) = handle
                    WRITE_DEBUG("Registered pointer "//to_str(transfer(ptr, int64)))
                endblock
            endif
#   endif
# else
        INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
        else if ( is_backend_nvshmem(backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
            block
            integer(int64)  :: max_alloc_bytes
            integer(int32)  :: mpi_err

                call MPI_Allreduce(alloc_bytes, max_alloc_bytes, 1, MPI_INTEGER8, MPI_MAX, comm, mpi_err)
                ptr = nvshmem_malloc(max_alloc_bytes)
                if ( is_null_ptr(ptr) ) error_code = DTFFT_ERROR_ALLOC_FAILED
            endblock
# else
      INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
        else
            error_code = cudaMalloc(ptr, alloc_bytes)
        endif
#endif
    endif
    if ( error_code /= DTFFT_SUCCESS ) error_code = DTFFT_ERROR_ALLOC_FAILED
end subroutine alloc_mem

subroutine free_mem(platform, helper, backend, ptr, error_code)
!! Frees memory based on ``backend``
type(dtfft_platform_t),         intent(in)    :: platform
type(backend_helper),           intent(inout) :: helper         !! Backend helper
type(dtfft_backend_t),          intent(in)    :: backend        !! GPU backend
type(c_ptr),                    intent(in)    :: ptr            !! Pointer to the memory to free
integer(int32),                 intent(out)   :: error_code     !! Error code

    error_code = DTFFT_SUCCESS
    if ( platform == DTFFT_PLATFORM_HOST ) then
        call mem_free_host(ptr)
#ifdef DTFFT_WITH_CUDA
    else
        if ( is_backend_nccl(backend) ) then
# ifdef NCCL_HAVE_COMMREGISTER
            if ( helper%should_register ) then
                block
                integer(int32) :: i

                    do i = 1, size(helper%nccl_register, dim=2)
                        if ( .not. is_same_ptr(ptr, helper%nccl_register(1, i)) ) cycle
                        NCCL_CALL( ncclCommDeregister(helper%nccl_comm, helper%nccl_register(2, i)) )
                        helper%nccl_register(1, i) = c_null_ptr
                        helper%nccl_register(2, i) = c_null_ptr
                        helper%nccl_register_size = helper%nccl_register_size - 1
                        WRITE_DEBUG("Pointer "//to_str(transfer(ptr, int64))//" has been removed from registry")
                    enddo
                endblock
            endif
# endif
# ifdef DTFFT_WITH_NCCL
#   ifdef NCCL_HAVE_MEMALLOC
            error_code = ncclMemFree(ptr)
#   else
            error_code = cudaFree(ptr)
#   endif
# else
            INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
        else if ( is_backend_nvshmem(backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
            call nvshmem_free(ptr)
# else
            INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
        else
            error_code = cudaFree(ptr)
        endif
#endif
    endif
    if ( error_code /= DTFFT_SUCCESS ) error_code = DTFFT_ERROR_FREE_FAILED
end subroutine free_mem

subroutine allocate_plans(plans, backend)
!! Allocates array of plans
type(reshape_container),    intent(inout) :: plans(:)   !! Plans to allocate
type(dtfft_backend_t),      intent(in)    :: backend    !! Backend to use
integer(int32) :: i

    do i = 1, size(plans)
        if ( backend == DTFFT_BACKEND_MPI_DATATYPE ) then
            allocate( reshape_handle_datatype :: plans(i)%p )
        else
            allocate( reshape_handle_generic :: plans(i)%p )
        endif
    enddo
end subroutine allocate_plans

pure integer(int64) function get_aux_bytes_generic(plans)
!! Returns maximum auxiliary memory size needed by plans
type(reshape_container),   intent(in)  :: plans(:)       !! Transpose plans
integer(int32)  :: i

    get_aux_bytes_generic = 0_int64
    do i = 1, size(plans)
        if ( allocated( plans(i)%p ) ) then
            get_aux_bytes_generic = max( get_aux_bytes_generic, plans(i)%p%get_aux_bytes() )
        endif
    enddo
end function get_aux_bytes_generic

subroutine destroy_plans(plans)
!! Destroys array of plans
type(reshape_container),           intent(inout) :: plans(:) !! Plans to destroy
integer(int32) :: i

    do i = 1, size(plans)
        if( allocated(plans(i)%p) ) then
            call plans(i)%p%destroy()
            deallocate(plans(i)%p)
        endif
    enddo
end subroutine destroy_plans

function execute_autotune(plans, comm, backend, platform, helper, stream, buffer_size, report_space_count) result(execution_time)
!! Destroys array of plans
type(reshape_container),    intent(inout) :: plans(:)       !! Allocated plans
TYPE_MPI_COMM,              intent(in)    :: comm           !! Communicator
type(dtfft_backend_t),      intent(in)    :: backend        !! Backend to use
type(dtfft_platform_t),     intent(in)    :: platform       !! Platform used
type(backend_helper),       intent(inout) :: helper         !! Helper to use
type(dtfft_stream_t),       intent(in)    :: stream         !! Stream to use
integer(int64),             intent(in)    :: buffer_size
integer(int32), optional,   intent(in)    :: report_space_count
real(real32)                              :: execution_time !! Execution time
type(execute_args) :: execute_kwargs
#ifdef DTFFT_WITH_CUDA
type(cudaEvent) :: timer_start, timer_stop
#endif
type(c_ptr) :: in, out, aux
real(real32), pointer :: pin(:)   !! Source buffer
real(real32), pointer :: pout(:)  !! Destination buffer
real(real32), pointer :: paux(:)  !! Auxiliary buffer
integer(int32) :: i, iter, ierr
integer(int64) :: float_buffer_size
! character(len=:), allocatable :: testing_phase
logical :: is_aux_alloc
integer(int32) :: n_warmup_iters, n_iters
real(real64) :: ts, te


    n_warmup_iters = get_conf_measure_warmup_iters()
    n_iters = get_conf_measure_iters()

#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaEventCreate(timer_start) )
      CUDA_CALL( cudaEventCreate(timer_stop) )
    endif
#endif
    float_buffer_size = buffer_size / FLOAT_STORAGE_SIZE

    call alloc_mem(platform, helper, backend, comm, buffer_size, in, ierr); DTFFT_CHECK(ierr)
    call alloc_mem(platform, helper, backend, comm, buffer_size, out, ierr); DTFFT_CHECK(ierr)

    call c_f_pointer(in, pin, [float_buffer_size])
    call c_f_pointer(out, pout, [float_buffer_size])

    ! testing_phase = "Testing backend "//dtfft_get_backend_string(backend)
    ! REGION_BEGIN(testing_phase, COLOR_AUTOTUNE2)
    ! WRITE_INFO(testing_phase)

    call alloc_and_set_aux(platform, helper, backend, comm, aux, paux, plans, is_aux_alloc)
    if( is_aux_alloc ) then
        execute_kwargs%p1 => paux
    else
        execute_kwargs%p1 => pout
    endif
    execute_kwargs%exec_type = EXEC_BLOCKING
    execute_kwargs%stream = stream

    ! REGION_BEGIN("Warmup", COLOR_TRANSPOSE)
    do iter = 1, n_warmup_iters
        do i = 1, size(plans)
            call plans(i)%p%execute(pin, pout, execute_kwargs, ierr)
        enddo
    enddo
#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
        CUDA_CALL( cudaStreamSynchronize(stream) )
    endif
#endif
    ! REGION_END("Warmup")

    ! REGION_BEGIN("Measure", COLOR_EXECUTE)
    if ( platform == DTFFT_PLATFORM_HOST ) then
        ts = MPI_Wtime()
#ifdef DTFFT_WITH_CUDA
    else
        CUDA_CALL( cudaEventRecord(timer_start, stream) )
#endif
    endif
    do iter = 1, n_iters
        do i = 1, size(plans)
            call plans(i)%p%execute(pin, pout, execute_kwargs, ierr)
        enddo
    enddo
    if ( platform == DTFFT_PLATFORM_HOST ) then
        te = MPI_Wtime()
        execution_time = real(te - ts, real32) * 1000._real32
#ifdef DTFFT_WITH_CUDA
    else
        CUDA_CALL( cudaEventRecord(timer_stop, stream) )
        CUDA_CALL( cudaEventSynchronize(timer_stop) )
        CUDA_CALL( cudaEventElapsedTime(execution_time, timer_start, timer_stop) )
#endif
    endif
    ! REGION_END("Measure")

    ! REGION_END(testing_phase)

    execution_time = report_timings(comm, execution_time, n_iters, report_space_count)

#ifdef DTFFT_WITH_CUDA
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaEventDestroy(timer_start) )
      CUDA_CALL( cudaEventDestroy(timer_stop) )
    endif
#endif

    call free_mem(platform, helper, backend, in, ierr)
    call free_mem(platform, helper, backend, out, ierr)
    if ( is_aux_alloc ) then
        call free_mem(platform, helper, backend, aux, ierr)
    endif

    ! if ( platform == DTFFT_PLATFORM_CUDA .and. is_backend_mpi(backend)) then
    !     call try_free_mpi_handles(comm, platform, helper)
    ! endif
end function execute_autotune

function report_timings(comm, elapsed_time, n_iters, space_count) result(max_time)
TYPE_MPI_COMM,            intent(in)  :: comm
real(real32),             intent(in)  :: elapsed_time
integer(int32),           intent(in)  :: n_iters
integer(int32), optional, intent(in)  :: space_count
real(real32)                          :: max_time
real(real32) :: execution_time, min_time, avg_time
integer(int32) :: ierr, comm_size, space_count_

    execution_time = elapsed_time / real(n_iters, real32)
    space_count_ = 2; if ( present(space_count) ) space_count_ = space_count

    call MPI_Allreduce(execution_time, min_time, 1, MPI_REAL4, MPI_MIN, comm, ierr)
    call MPI_Allreduce(execution_time, max_time, 1, MPI_REAL4, MPI_MAX, comm, ierr)
    call MPI_Allreduce(execution_time, avg_time, 1, MPI_REAL4, MPI_SUM, comm, ierr)
    call MPI_Comm_size(comm, comm_size, ierr)

    avg_time = avg_time / real(comm_size, real32)

    WRITE_INFO(repeat(" ", space_count_)//"max: "//to_str(real(max_time, real64))//" [ms]")
    WRITE_INFO(repeat(" ", space_count_)//"min: "//to_str(real(min_time, real64))//" [ms]")
    WRITE_INFO(repeat(" ", space_count_)//"avg: "//to_str(real(avg_time, real64))//" [ms]")
end function report_timings

subroutine alloc_and_set_aux(platform, helper, backend, cart_comm, aux, paux, plans, is_aux_alloc)
!! Allocates auxiliary memory according to the backend and sets it to the plans
type(dtfft_platform_t),       intent(in)                :: platform
type(backend_helper),         intent(inout)             :: helper       !! Backend helper
type(dtfft_backend_t),        intent(in)                :: backend      !! GPU backend
TYPE_MPI_COMM,                intent(in)                :: cart_comm    !! Cartesian communicator
type(c_ptr),                  intent(inout)             :: aux          !! Allocatable auxiliary memory
real(real32),     pointer,    intent(inout)             :: paux(:)      !! Pointer to auxiliary memory
type(reshape_container),      intent(in)                :: plans(:)
logical                                                 :: is_aux_alloc !! Is auxiliary memory allocated
integer(int64) :: max_work_size_local, max_work_size_global
integer(int32)  :: mpi_ierr
integer(int32) :: alloc_ierr

    max_work_size_local = get_aux_bytes_generic(plans)
    call MPI_Allreduce(max_work_size_local, max_work_size_global, 1, MPI_INTEGER8, MPI_MAX, cart_comm, mpi_ierr)

    is_aux_alloc = .false.
    paux => null()
    if ( max_work_size_global > 0 ) then
        call alloc_mem(platform, helper, backend, cart_comm, max_work_size_global, aux, alloc_ierr)
        DTFFT_CHECK(alloc_ierr)
        call c_f_pointer(aux, paux, [max_work_size_global / 4_int64])
        is_aux_alloc = .true.
    endif
end subroutine alloc_and_set_aux

! subroutine try_free_mpi_handles(comm, platform, helper)
! TYPE_MPI_COMM,          intent(in)      :: comm
! type(dtfft_platform_t), intent(in)      :: platform
! type(backend_helper),   intent(inout)   :: helper
! #ifdef DTFFT_WITH_CUDA
! integer(int64) :: alloc_size
! integer(int32) :: base_size, comm_size, ierr
! type(c_ptr) :: ptr1, ptr2, ptr3
! integer(int64)  :: free_mem_before, free_mem_after, total_mem_avail
! real(real32), pointer, contiguous :: rptr1(:), rptr2(:), rptr3(:)
! #endif

!     if ( platform == DTFFT_PLATFORM_HOST ) return
! #ifdef DTFFT_WITH_CUDA

!     call MPI_Comm_size(comm, comm_size, ierr)

!     base_size = 1024 * 1024 * 30    ! 10Mb per process
!     alloc_size = int(base_size, int64) * comm_size

!     CUDA_CALL( cudaMemGetInfo(free_mem_before, total_mem_avail) )

!     call alloc_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, comm, alloc_size, ptr1, ierr);  call c_f_pointer(ptr1, rptr1, [1])
!     call alloc_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, comm, alloc_size, ptr2, ierr);  call c_f_pointer(ptr2, rptr2, [1])
!     call alloc_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, comm, alloc_size, ptr3, ierr);  call c_f_pointer(ptr3, rptr3, [1])

!     call MPI_Alltoall(rptr1, base_size, MPI_BYTE, rptr2, base_size, MPI_BYTE, comm, ierr)
!     call MPI_Alltoall(rptr1, base_size, MPI_BYTE, rptr3, base_size, MPI_BYTE, comm, ierr)
!     call MPI_Alltoall(rptr2, base_size, MPI_BYTE, rptr3, base_size, MPI_BYTE, comm, ierr)

!     call free_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, ptr1, ierr)
!     call free_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, ptr2, ierr)
!     call free_mem(platform, helper, DTFFT_BACKEND_MPI_P2P, ptr3, ierr)

!     CUDA_CALL( cudaMemGetInfo(free_mem_after, total_mem_avail) )

!     WRITE_INFO("Tried to release internal MPI handles. Free mem before: "//to_str(free_mem_before)//", after = "//to_str(free_mem_after))

! #endif
! end subroutine try_free_mpi_handles

end module dtfft_reshape_plan_base