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
module dtfft_plan_base_m
!------------------------------------------------------------------------------------------------
!< This module describes [[dtfft_base_plan]] class
!------------------------------------------------------------------------------------------------
use dtfft_info_m
use dtfft_parameters
use dtfft_precisions
use dtfft_transpose_m
#include "dtfft_mpi.h"
use iso_fortran_env, only: output_unit
implicit none
private
public :: dtfft_base_plan

  type :: dtfft_base_plan
  !< Base plan for all DTFFT plans
    TYPE_MPI_COMM                     :: comm                       !< Grid communicator
    TYPE_MPI_COMM,      allocatable   :: comms(:)                   !< Local 1d communicators
    integer(IP),        allocatable   :: comm_dims(:)               !< Dimensions of grid comm
    integer(IP),        allocatable   :: comm_coords(:)             !< Coordinates of grod comm
    integer(IP)                       :: dims                       !< Number of global dimensions
    integer(IP)                       :: precision                  !< Precision of transform
    integer(IP)                       :: comm_size, comm_rank       !< Size of comm, rank of comm
    type(transpose_t),  allocatable   :: transpose_out(:)           !< Classes that perform TRANSPOSED_OUT transposes: XYZ --> YXZ --> ZXY
    type(transpose_t),  allocatable   :: transpose_in(:)            !< Classes that perform TRANSPOSED_IN transposes: ZXY --> YXZ --> XYZ
    type(info_t),       allocatable   :: info(:)                    !< Information about data aligment and datatypes
    logical                           :: is_created = .false.       !< Plan creation flag
  contains
  private
    procedure,  pass(self),   public  :: init_base_plan             !< Creates base plan
    procedure,  pass(self),   public  :: transpose                  !< Performs transposition with timers
    procedure,  pass(self),   public  :: get_local_sizes_internal   !< Returns local sizes, starts and number of elements to be allocated
    procedure,  pass(self),   public  :: get_worker_size_internal   !< Returns local sizes, starts and number of elements to be allocated for the optional worker buffer
    procedure,  pass(self),   public  :: check_plan                 !< Checks if plan is created with proper precision
    procedure,  pass(self),   public  :: destroy_base_plan          !< Destroys base plan
    procedure,  pass(self)            :: create_transpose_plans     !< Creates transposition plans
    procedure,  pass(self)            :: create_cart_comm           !< Creates cartesian communicator
    procedure,  pass(self)            :: base_alloc                 !< Alocates classes needed by base plan
  end type dtfft_base_plan

  contains

!------------------------------------------------------------------------------------------------
  subroutine check_plan(self, precision)
!------------------------------------------------------------------------------------------------
!< Checks if plan is created with proper precision
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan),  intent(in)  :: self                  !< Base class
    integer(IP),             intent(in)  :: precision             !< Precision

    if(.not. self%is_created) then 
      error stop 'DTFFT: Plan was not properly created'
    endif
    if(self%precision /= precision) then
      error stop 'DTFFT: Trying to execute with different precision then it was created'
    endif
  end subroutine check_plan

!------------------------------------------------------------------------------------------------
  subroutine destroy_base_plan(self)
!------------------------------------------------------------------------------------------------
!< Destroys base plan
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan),  intent(inout)  :: self               !< Base class
    integer(IP)                             :: d                  !< Counter

    if(allocated(self%transpose_in) .and. allocated(self%transpose_out)) then
      do d = 1, self%dims - 1
        call self%transpose_in(d)%destroy()
        call self%transpose_out(d)%destroy()
      enddo
      deallocate(self%transpose_in)
      deallocate(self%transpose_out)
    endif
    if(allocated(self%info)) then 
      do d = 1, self%dims
        call self%info(d)%destroy()
      enddo
      deallocate(self%info)
    endif
    if(allocated(self%comms)) then 
      do d = 1, self%dims
        call MPI_Comm_free(self%comms(d), IERROR)
      enddo
      deallocate(self%comms)
    endif
    if(allocated(self%comm_dims))   deallocate(self%comm_dims)
    if(allocated(self%comm_coords)) deallocate(self%comm_coords)
    call MPI_Comm_free(self%comm, IERROR)
    self%dims = -1
    self%is_created = .false.
  end subroutine destroy_base_plan

!------------------------------------------------------------------------------------------------
  pure subroutine base_alloc(self)
!------------------------------------------------------------------------------------------------
!< Alocates classes needed by base plan
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class

    allocate(self%info(self%dims))
    allocate(self%transpose_in(self%dims - 1))
    allocate(self%transpose_out(self%dims - 1))
    allocate(self%comm_dims(self%dims))
    allocate(self%comms(self%dims))
    allocate(self%comm_coords(self%dims))
  end subroutine base_alloc

!------------------------------------------------------------------------------------------------
  subroutine init_base_plan(self, comm, precision, counts, base_type, base_storage, effort_flag)
!------------------------------------------------------------------------------------------------
!< Creates base plan
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class
    TYPE_MPI_COMM,          intent(in)    :: comm                 !< User communicator
    integer(IP),            intent(in)    :: precision            !< Precision of base plan
    integer(IP),            intent(in)    :: counts(:)            !< Counts of the transform requested
    TYPE_MPI_DATATYPE,      intent(in)    :: base_type            !< Base MPI_Datatype
    integer(IP),            intent(in)    :: base_storage         !< Number of bytes needed to store single element
    integer(IP),  optional, intent(in)    :: effort_flag          !< DTFFT planner effort flag
    integer(IP)                           :: planner_flag         !< DTFFT planner effort flag
    integer(IP)                           :: d                    !< Counter
    integer(IP)                           :: ndims                !< Number of dims in user defined cartesian communicator
    integer(IP)                           :: status               !< MPI_Topo_test flag
    integer(IP),            allocatable   :: tr_counts(:,:)       !< Global counts in transposed coordinates
    integer(IP),            allocatable   :: temp_dims(:)         !< Temporary dims needed by MPI_Cart_get
    integer(IP),            allocatable   :: temp_coords(:)       !< Temporary coordinates needed by MPI_Cart_get
    logical,                allocatable   :: temp_periods(:)      !< Temporary periods needed by MPI_Cart_get

    self%dims = size(counts)
    self%precision = precision

    call self%base_alloc()
    allocate(tr_counts(self%dims, self%dims))
    if(self%dims == 2) then
      ! Nx x Ny
      tr_counts(1, :) = counts(:) 
      ! Ny x Nx
      tr_counts(2, 1) = counts(2) 
      tr_counts(2, 2) = counts(1) 
    else
      ! Nx x Ny x Nz
      tr_counts(1, :) = counts(:)
      ! Ny x Nx x Nz
      tr_counts(2, 1) = counts(2)
      tr_counts(2, 2) = counts(1) 
      tr_counts(2, 3) = counts(3) 
      ! Nz x Nx x Ny
      tr_counts(3, 1) = counts(3)
      tr_counts(3, 2) = counts(1) 
      tr_counts(3, 3) = counts(2) 
    endif

    call MPI_Comm_size(comm, self%comm_size, IERROR)
    call MPI_Comm_rank(comm, self%comm_rank, IERROR)
    call MPI_Topo_test(comm, status, IERROR)

    if(status == MPI_UNDEFINED) then
      planner_flag = DTFFT_MEASURE
      if(present(effort_flag)) planner_flag = effort_flag

!TODO Implement effort_flag algorithms
        self%comm_dims(:) = 0
        self%comm_dims(1) = 1
        call MPI_Dims_create(self%comm_size, self%dims, self%comm_dims, IERROR)
 
    elseif(status == MPI_CART) then
      call MPI_Cartdim_get(comm, ndims, IERROR)
      if(ndims > self%dims) error stop "DTFFT: Number of cartesian dims > size of transpose"
      self%comm_dims(:) = 1
      allocate(temp_dims(ndims), temp_periods(ndims), temp_coords(ndims))
      call MPI_Cart_get(comm, ndims, temp_dims, temp_periods, temp_coords, IERROR)
      if(ndims == self%dims) then 
        self%comm_dims = temp_dims
      elseif(ndims == self%dims - 1) then
        self%comm_dims(2:) = temp_dims
      elseif(ndims == self%dims - 2) then
        self%comm_dims(3) = temp_dims(1)
      endif
      deallocate(temp_dims, temp_periods, temp_coords)
    else
      error stop 'DTFFT: User must provide cartesian communicator or communicator with no topology'
    endif

    call self%create_cart_comm(self%comm_dims, self%comm, self%comm_coords, self%comms)
    do d = 1, self%dims
      call self%info(d)%init(self%dims, d, tr_counts(d,:), self%comms, self%comm_dims, self%comm_coords)
    enddo

    call self%create_transpose_plans(self%comms, base_type, base_storage)

    deallocate(tr_counts)
  end subroutine init_base_plan

!------------------------------------------------------------------------------------------------
  subroutine create_cart_comm(self, comm_dims, comm, comm_coords, local_comms)
!------------------------------------------------------------------------------------------------
!< Creates cartesian communicator
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class
    integer(IP),            intent(in)    :: comm_dims(:)         !< Dims in cartesian communcator
    TYPE_MPI_COMM,          intent(out)   :: comm                 !< Cartesian communcator
    integer(IP),            intent(out)   :: comm_coords(:)       !< Coordinates of current process in cartesian communcator
    TYPE_MPI_COMM,          intent(out)   :: local_comms(:)       !< 1d communicators in cartesian communcator
    logical,                allocatable   :: periods(:)           !< Grid is not periodic
    logical,                allocatable   :: remain_dims(:)       !< Needed by MPI_Cart_sub
    integer(IP)                           :: d                    !< Counter
    integer(IP)                           :: comm_rank            !< Rank of current process in cartesian communcator

    allocate(periods(self%dims), source = .false.)
    call MPI_Cart_create(MPI_COMM_WORLD, self%dims, comm_dims, periods, .true., comm, IERROR)
    call MPI_Comm_rank(comm, comm_rank, IERROR)
    call MPI_Cart_coords(comm, comm_rank, self%dims, comm_coords, IERROR)

    allocate(remain_dims(self%dims), source = .false.)
    do d = 1, self%dims
      remain_dims(d) = .true.
      call MPI_Cart_sub(comm, remain_dims, local_comms(d), IERROR)
      remain_dims(d) = .false.
    enddo
    deallocate(remain_dims, periods)
  end subroutine 

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_plans(self, comms, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates transpose plans: XYZ -> YXZ -> ZXY -> YXZ -> XYZ
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class
    TYPE_MPI_COMM,          intent(in)    :: comms(:)             !< Array of 1d communicators
    TYPE_MPI_DATATYPE,      intent(in)    :: base_type            !< Base MPI_Datatype
    integer(IP),            intent(in)    :: base_storage         !< Number of bytes needed to store single element

    call self%transpose_out(1)%init(comms(2), self%info(1), self%info(2), base_type, base_storage)
    call self%transpose_in(1)%init(comms(2), self%info(2), self%info(1), base_type, base_storage)

    if(self%dims == 3) then
      call self%transpose_out(2)%init(comms(3), self%info(2), self%info(3), base_type, base_storage)
      call self%transpose_in(2)%init(comms(3), self%info(3), self%info(2), base_type, base_storage)
    endif
  end subroutine create_transpose_plans

!------------------------------------------------------------------------------------------------
  subroutine get_local_sizes_internal(self, in_starts, in_counts, out_starts, out_counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(in)  :: self                   !< Base class
    integer(IP),  optional, intent(out) :: in_starts(:)           !< Starts of local portion of data in 'real' space
    integer(IP),  optional, intent(out) :: in_counts(:)           !< Counts of local portion of data in 'real' space
    integer(IP),  optional, intent(out) :: out_starts(:)          !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional, intent(out) :: out_counts(:)          !< Counts of local portion of data in 'fourier' space
    integer(IP),  optional, intent(out) :: alloc_size             !< Maximum number of elements needs to be allocated

    if(self%is_created) then
      if(present(in_starts))  in_starts   = self%info(1)%starts
      if(present(in_counts))  in_counts   = self%info(1)%counts
      if(present(out_starts)) out_starts  = self%info(self%dims)%starts
      if(present(out_counts)) out_counts  = self%info(self%dims)%counts
      if(present(alloc_size)) alloc_size  = max(product(self%info(1)%counts), product(self%info(self%dims)%counts))
    else
      error stop 'DTFFT: error in "get_local_sizes", plan has not been created'
    endif
  end subroutine get_local_sizes_internal

!------------------------------------------------------------------------------------------------
  subroutine get_worker_size_internal(self, work_id, starts, counts, alloc_size)
!------------------------------------------------------------------------------------------------
!< Returns local sizes, counts and allocation size of optional worker buffer
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class
    integer(IP),            intent(in)    :: work_id              !< Id of aligned info
    integer(IP),  optional, intent(out)   :: starts(:)            !< Starts of local portion of Y pencil
    integer(IP),  optional, intent(out)   :: counts(:)            !< Counts of local portion of Y pencil
    integer(IP),  optional, intent(out)   :: alloc_size           !< Number of elements to be allocated

    if(self%is_created) then 
      if(present(starts))     starts      = self%info(work_id)%starts
      if(present(counts))     counts      = self%info(work_id)%counts
      if(present(alloc_size)) alloc_size  = product(self%info(work_id)%counts)
    else
      error stop 'DTFFT: error in "get_worker_size", plan has not been created'
    endif
  end subroutine get_worker_size_internal

!------------------------------------------------------------------------------------------------
  subroutine transpose(self, plan, in, out, from, to)
!------------------------------------------------------------------------------------------------
!< Performs global transposition from buffer in to out. In-place transposition is not supported
!------------------------------------------------------------------------------------------------
    class(dtfft_base_plan), intent(inout) :: self                 !< Base class
    class(transpose_t),     intent(in)    :: plan                 !< Transpose plan
    type(*),                intent(in)    :: in(..)               !< Send buffer
    type(*),                intent(inout) :: out(..)              !< Recv buffer
    character(len=1),       intent(in)    :: from                 !< Character to display in case of debug, transpose from
    character(len=1),       intent(in)    :: to                   !< Character to display in case of debug, transpose to
#ifdef __DEBUG
    real(R8P)                             :: elapsed              !< Mean time of transposition
    real(R8P)                             :: t_sum                !< Total time by all processes

    elapsed = 0._R8P - MPI_Wtime()
#endif
    call plan%transpose(in, out)
#ifdef __DEBUG
    elapsed = elapsed + MPI_Wtime()
    call MPI_Allreduce(elapsed, t_sum, 1, MPI_REAL8, MPI_SUM, self%comm, IERROR)
    elapsed = t_sum / real(self%comm_size, R8P)
    if(self%comm_rank == 0) then 
      write(output_unit, '(a)') 'Transpose '//from//' --> '//to
      write(output_unit, '(a,E23.15E3)') 'Elapsed time = ', elapsed
    endif
#endif
  end subroutine transpose
end module dtfft_plan_base_m