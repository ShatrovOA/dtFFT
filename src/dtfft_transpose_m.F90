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
#include "dtfft_cuda.h"
module dtfft_transpose_m
!------------------------------------------------------------------------------------------------
!< This module describes `transpose_t` class
!------------------------------------------------------------------------------------------------
use dtfft_info_m
use dtfft_precisions
#include "dtfft_mpi.h"
#include "dtfft_profile.h"
implicit none
private
public :: transpose_t

  integer(MPI_ADDRESS_KIND), parameter :: LB = 0
  !< Lower bound for all derived datatypes

  type :: handle_t
  !< Transposition handle class
    TYPE_MPI_DATATYPE,  allocatable :: dtypes(:)              !< Datatypes buffer
    integer(IP),        allocatable :: counts(:)              !< Number of datatypes (always equals 1)
    integer(IP),        allocatable :: displs(:)              !< Displacements is bytes
  contains
    procedure, pass(self) :: init => init_handle              !< Creates transposition handle
    procedure, pass(self) :: destroy => destroy_handle        !< Destroys transposition handle
  end type handle_t

  type :: transpose_t
  !< Transposition class
  private
    TYPE_MPI_COMM                   :: comm                   !< 1d communicator
    logical                         :: is_even                !< Is decomposition even
    type(handle_t)                  :: send                   !< Handle to send data
    type(handle_t)                  :: recv                   !< Handle to recieve data
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    TYPE_MPI_REQUEST                :: request
    logical                         :: is_request_created
#endif
  contains
  private
    procedure, pass(self),  public  :: init                   !< Initializes class
    procedure, pass(self),  public  :: transpose              !< Performs MPI_Alltoall(w)
    procedure, pass(self),  public  :: destroy                !< Destroys class
    procedure, pass(self)           :: create_transpose_2d    !< Creates two-dimensional transposition datatypes
    procedure, pass(self)           :: create_transpose_XY    !< Creates three-dimensional X --> Y, Y --> X transposition datatypes
    procedure, pass(self)           :: create_transpose_YZ    !< Creates three-dimensional Y --> Z, Z --> Y transposition datatypes
    procedure, pass(self)           :: create_transpose_XZ    !< Creates three-dimensional X --> Z datatype, only slab!
    procedure, pass(self)           :: create_transpose_ZX    !< Creates three-dimensional Z --> X datatype, only slab!
  end type transpose_t

contains

!------------------------------------------------------------------------------------------------
  subroutine init_handle(self, n)
!------------------------------------------------------------------------------------------------
!< Creates transposition handle
!------------------------------------------------------------------------------------------------
    class(handle_t),  intent(inout) :: self   !< Transposition handle
    integer(IP),      intent(in)    :: n      !< Number of datatypes to be created

    call self%destroy()
    allocate(self%dtypes(n))
    allocate(self%counts(n), source = 1_IP)
    allocate(self%displs(n), source = 0_IP)
  end subroutine init_handle

!------------------------------------------------------------------------------------------------
  subroutine destroy_handle(self)
!------------------------------------------------------------------------------------------------
!< Destroys transposition handle
!------------------------------------------------------------------------------------------------
    class(handle_t),  intent(inout) :: self   !< Transposition handle
    integer(IP)                     :: i      !< Counter
    integer(IP)                     :: ierr

    if ( allocated(self%dtypes) ) then
      do i = 1, size(self%dtypes)
        call MPI_Type_free(self%dtypes(i), ierr)
      enddo
      deallocate(self%dtypes)
    endif
    if ( allocated(self%displs) ) deallocate(self%displs)
    if ( allocated(self%counts) ) deallocate(self%counts)
  end subroutine destroy_handle

!------------------------------------------------------------------------------------------------
  subroutine init(self, comm, send, recv, base_type, base_storage, transpose_id)
!------------------------------------------------------------------------------------------------
!< Creates `transpose_t` class
!------------------------------------------------------------------------------------------------
    class(transpose_t),     intent(inout) :: self               !< Transposition class
    TYPE_MPI_COMM,          intent(in)    :: comm               !< 1d communicator
    class(info_t),          intent(in)    :: send               !< Information about send buffer
    class(info_t),          intent(in)    :: recv               !< Information about recv buffer
    TYPE_MPI_DATATYPE,      intent(in)    :: base_type          !< Base MPI Datatype
    integer(IP),            intent(in)    :: base_storage       !< Number of bytes needed to store single element
    integer(IP),            intent(in)    :: transpose_id       !< Type of transpose
    integer(IP)                       :: comm_size              !< Size of 1d communicator
    integer(IP)                       :: n_neighbors            !< Number of datatypes to be created
    integer(IP),        allocatable   :: recv_counts(:,:)       !< Each processor should know how much data each processor recieves
    integer(IP),        allocatable   :: send_counts(:,:)       !< Each processor should know how much data each processor sends
    integer(IP)                       :: i                      !< Counter
    integer(IP)                       :: ierr

    self%comm = comm
    call MPI_Comm_size(self%comm, comm_size, ierr)
    self%is_even = send%is_even .and. recv%is_even
    n_neighbors = comm_size;  if ( self%is_even ) n_neighbors = 1

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    self%is_request_created = .false.
#endif

    call self%send%init(n_neighbors)
    call self%recv%init(n_neighbors)

    allocate(recv_counts(recv%rank, comm_size), source = 0_IP)
    allocate(send_counts, source = recv_counts)
    call MPI_Allgather(recv%counts, recv%rank, MPI_INTEGER, recv_counts, recv%rank, MPI_INTEGER, self%comm, ierr)
    call MPI_Allgather(send%counts, send%rank, MPI_INTEGER, send_counts, send%rank, MPI_INTEGER, self%comm, ierr)
    if ( send%rank == 2 ) then
      do i = 1, n_neighbors
        call self%create_transpose_2d(n_neighbors, i, send, send_counts(:,i), recv, recv_counts(:,i), transpose_id, base_type, base_storage)
      enddo
    else
      if ( (send%aligned_dim == 1 .and. recv%aligned_dim == 2)          &
      .or. (recv%aligned_dim == 1 .and. send%aligned_dim == 2) ) then
        ! self%sendcount = recv%counts(2) * recv%counts(3) * send%counts(2)
        ! self%recvcount = recv%counts(2) * recv%counts(3) * send%counts(2)
        do i = 1, n_neighbors
          call self%create_transpose_XY(n_neighbors, i, send, send_counts(:,i), recv, recv_counts(:,i), transpose_id, base_type, base_storage)
        enddo
      elseif ( (send%aligned_dim == 2 .and. recv%aligned_dim == 3)      &
          .or. (recv%aligned_dim == 2 .and. send%aligned_dim == 3) )then
        ! self%sendcount = recv%counts(2) * recv%counts(3) * send%counts(3)
        ! self%recvcount = recv%counts(2) * recv%counts(3) * send%counts(3)
        do i = 1, n_neighbors
          call self%create_transpose_YZ(n_neighbors, i, send, send_counts(:,i), recv, recv_counts(:,i), transpose_id, base_type, base_storage)
        enddo
      else
        do i = 1, n_neighbors
          ! Since XZ transpose is not symmetric, different DataTypes are needed
          if ( send%aligned_dim == 1 ) then
            call self%create_transpose_XZ(n_neighbors, i, send, send_counts(:,i), recv, recv_counts(:,i), transpose_id, base_type, base_storage)
          else
            call self%create_transpose_ZX(n_neighbors, i, send, send_counts(:,i), recv, recv_counts(:,i), transpose_id, base_type, base_storage)
          endif
        enddo
      endif
    endif
    deallocate(recv_counts, send_counts)
  end subroutine init

!------------------------------------------------------------------------------------------------
  subroutine transpose(self, send, recv)
!------------------------------------------------------------------------------------------------
!< Executes transposition
!------------------------------------------------------------------------------------------------
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    class(transpose_t),     intent(inout) :: self         !< Transposition class
#else
    class(transpose_t),     intent(in)    :: self         !< Transposition class
#endif
    type(*),                intent(in)    &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: send(..)     !< Incoming buffer of any rank and kind
    type(*),                intent(inout) &
#ifdef DTFFT_WITH_CUDA
      , device                            &
#endif
                                          :: recv(..)     !< Resulting buffer of any rank and kind
    integer(IP)                           :: ierr

#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    if ( .not. self%is_request_created ) then
      if ( self%is_even ) then
        call MPI_Alltoall_init(send, 1, self%send%dtypes(1), recv, 1, self%recv%dtypes(1), self%comm, MPI_INFO_NULL, self%request, ierr)
      else
        call MPI_Alltoallw_init(send, self%send%counts, self%send%displs, self%send%dtypes,                                             &
                                recv, self%recv%counts, self%recv%displs, self%recv%dtypes, self%comm, MPI_INFO_NULL, self%request, ierr)
      endif
      self%is_request_created = .true.
    endif
    call MPI_Start(self%request, ierr)
    call MPI_Wait(self%request, MPI_STATUS_IGNORE, ierr)
#else
    if ( self%is_even ) then
      call MPI_Alltoall(send, 1, self%send%dtypes(1), recv, 1, self%recv%dtypes(1), self%comm, ierr)
    else
      call MPI_Alltoallw(send, self%send%counts, self%send%displs, self%send%dtypes,                &
                         recv, self%recv%counts, self%recv%displs, self%recv%dtypes, self%comm, ierr)
    endif
#endif
  end subroutine transpose

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys `transpose_t` class
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self       !< Transposition class

    call self%send%destroy()
    call self%recv%destroy()
#ifdef DTFFT_ENABLE_PERSISTENT_COMM
    block
      integer(IP)                       :: ierr

      if( self%is_request_created ) call MPI_Request_free(self%request, ierr)
      self%is_request_created = .false.
    endblock
#endif
  end subroutine destroy

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_2d(self, n_neighbors, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates two-dimensional transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: n_neighbors        !< Size of 1d comm
    integer(IP),        intent(in)    :: i                  !< Counter
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    integer(IP),        intent(in)    :: send_counts(:)     !< Rank i is sending this counts
    class(info_t),      intent(in)    :: recv               !< Information about send buffer
    integer(IP),        intent(in)    :: recv_counts(:)     !< Rank i is recieving this counts
    integer(IP),        intent(in)    :: transpose_id       !< Id of transpose plan. Reserved for future effort flag development.
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI_Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    TYPE_MPI_DATATYPE                 :: temp1              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp2              !< Temporary datatype
    integer(IP)                       :: displ              !< Displacement in bytes
    integer(IP)                       :: ierr


    if ( transpose_id == 1 ) then
      call MPI_Type_vector(send%counts(2), recv_counts(2), send%counts(1), base_type, temp1, ierr)
      displ = recv_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1)

      call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send_counts(2), temp2, self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + send_counts(2) * base_storage
      call free_datatypes(temp1, temp2)
    else
      call MPI_Type_vector(send%counts(2), 1, send%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(2), temp2, self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      displ = recv_counts(2) * base_storage
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1, temp2)

      call MPI_Type_vector(recv%counts(2), send_counts(2), recv%counts(1), base_type, temp1, ierr)
      displ = send_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)

      ! call MPI_Type_contiguous(recv_counts(2) * send_counts(2), base_type, self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + displ
      call free_datatypes(temp1)
    endif
  end subroutine create_transpose_2d

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_XY(self, n_neighbors, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional X --> Y, Y --> X transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: n_neighbors        !< Size of 1d comm
    integer(IP),        intent(in)    :: i                  !< Counter
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    integer(IP),        intent(in)    :: send_counts(:)     !< Rank i is sending this counts
    class(info_t),      intent(in)    :: recv               !< Information about send buffer
    integer(IP),        intent(in)    :: recv_counts(:)     !< Rank i is recieving this counts
    integer(IP),        intent(in)    :: transpose_id       !< Id of transpose plan. Reserved for future effort flag development.
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI_Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    TYPE_MPI_DATATYPE                 :: temp1              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp2              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp3              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp4              !< Temporary datatype
    integer(IP)                       :: displ              !< Rank i is sending / recieving with this displacement in bytes
    integer(IP)                       :: ierr

    if ( transpose_id == 1 ) then
    ! This transpose_id has "contiguous" send and strided recieve datatype
      call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(2), send%counts(1), base_type, temp1, ierr)
      displ = recv_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1)

      call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send_counts(2), temp2, temp3, ierr)
      call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = send_counts(2) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3, temp4)
    elseif ( transpose_id == 2 ) then
    ! This transpose_id has strided send and "contiguous" recieve datatypes
      call MPI_Type_vector(send%counts(2) * send%counts(3), 1, send%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(2), temp2, self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) +  recv_counts(2) * base_storage
      call free_datatypes(temp1, temp2)

      call MPI_Type_vector(recv%counts(3), send_counts(2), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      displ = send_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_create_hvector(recv%counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      call MPI_Type_create_resized(temp3, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3)
    endif
  end subroutine create_transpose_XY

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_YZ(self, n_neighbors, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional Y --> Z, Z --> Y transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: n_neighbors        !< Size of 1d comm
    integer(IP),        intent(in)    :: i                  !< Counter
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    integer(IP),        intent(in)    :: send_counts(:)     !< Rank i is sending this counts
    class(info_t),      intent(in)    :: recv               !< Information about send buffer
    integer(IP),        intent(in)    :: recv_counts(:)     !< Rank i is recieving this counts
    integer(IP),        intent(in)    :: transpose_id       !< Id of transpose plan. Reserved for future effort flag development.
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI_Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    TYPE_MPI_DATATYPE                 :: temp1              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp2              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp3              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp4              !< Temporary datatype
    integer(IP)                       :: displ              !< Rank i is sending / recieving with this displacement in bytes
    integer(IP)                       :: ierr

    if ( transpose_id == 1 ) then
      call MPI_Type_vector(send%counts(3), 1, send%counts(1) * send%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(3), temp2, temp3, ierr)
      call MPI_Type_create_hvector(send%counts(2), 1, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = recv_counts(3) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3, temp4)

      call MPI_Type_vector(recv%counts(3), send_counts(3), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_create_hvector(send_counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      call MPI_Type_create_resized(temp3, LB, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + send_counts(3) * base_storage
      call free_datatypes(temp1, temp2, temp3)
    else
      call MPI_Type_vector(send%counts(2), 1, send%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(3), temp2, temp3, ierr)
      call MPI_Type_create_hvector(send%counts(3), 1, int(send%counts(1) * send%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      call MPI_Type_create_resized(temp4, LB, int(recv_counts(3) * base_storage, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) +  recv_counts(3) * base_storage
      call free_datatypes(temp1, temp2, temp3, temp4)

      ! call MPI_Type_contiguous(send%counts(2) * send%counts(3) * recv_counts(3), base_type, self%send%dtypes(i), ierr)
      ! call MPI_Type_commit(self%send%dtypes(i), ierr)

      ! call MPI_Type_contiguous(send%counts(2) * send%counts(3) * recv_counts(3), base_type, self%recv%dtypes(i), ierr)
      ! call MPI_Type_commit(self%recv%dtypes(i), ierr)

      call MPI_Type_vector(recv%counts(2) * recv%counts(3), 1, recv%counts(1), base_type, temp1, ierr)

      ! call MPI_Type_vector(send_counts(3), 1, recv%counts(2) * recv%counts(3), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      ! call MPI_Type_create_hvector(recv%counts(2) * recv%counts(3), 1, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      ! call MPI_Type_create_resized(temp3, LB, int(send_counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_contiguous(send_counts(3), temp2,  self%recv%dtypes(i), ierr)
      ! call MPI_Type_contiguous(recv%counts(2) * recv%counts(3), temp2,  self%recv%dtypes(i), ierr)
      ! call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      ! call MPI_Type_create_resized(temp4, LB, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + send_counts(3) * base_storage
      call free_datatypes(temp1, temp2)
    endif
    end subroutine create_transpose_YZ

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_XZ(self, n_neighbors, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional X --> Z transposition datatypes
!< Can only be used with 3D slab decomposition when slabs are distributed in Z direction
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: n_neighbors        !< Size of 1d comm
    integer(IP),        intent(in)    :: i                  !< Counter
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    integer(IP),        intent(in)    :: send_counts(:)     !< Rank i is sending this counts
    class(info_t),      intent(in)    :: recv               !< Information about send buffer
    integer(IP),        intent(in)    :: recv_counts(:)     !< Rank i is recieving this counts
    integer(IP),        intent(in)    :: transpose_id       !< Id of transpose plan. Reserved for future effort flag development.
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI_Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    TYPE_MPI_DATATYPE                 :: temp1              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp2              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp3              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp4              !< Temporary datatype
    integer(IP)                       :: displ              !< Rank i is sending / recieving with this displacement in bytes
    integer(IP)                       :: ierr

    if ( transpose_id == 1 ) then
      call MPI_Type_vector(send%counts(3), 1, send%counts(1) * send%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send%counts(1), temp2, temp3, ierr)
      call MPI_Type_create_hvector(recv_counts(3), 1, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = send%counts(1) * recv_counts(3) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1, temp2, temp3, temp4)

      call MPI_Type_vector(recv%counts(2) * recv%counts(3), send_counts(3), recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + send_counts(3) * base_storage
      call free_datatypes(temp1)
    else
      call MPI_Type_vector(send%counts(3), send%counts(1), send%counts(1) * send%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(3), temp2, self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + send%counts(1) * recv_counts(3) * base_storage
      call free_datatypes(temp1, temp2)

      call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send_counts(3), temp2, temp3, ierr)
      call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      call MPI_Type_create_resized(temp4, LB, int(send_counts(3)* base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)

      ! call MPI_Type_contiguous(send%counts(1) * send_counts(3) * recv%counts(3), base_type, self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + send_counts(3)* base_storage
      call free_datatypes(temp1, temp2, temp3, temp4)
    endif
  end subroutine create_transpose_XZ

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_ZX(self, n_neighbors, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional Z --> X transposition datatypes
!< Can only be used with 3D slab decomposition when slabs are distributed in Z direction
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: n_neighbors        !< Size of 1d comm
    integer(IP),        intent(in)    :: i                  !< Counter
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    integer(IP),        intent(in)    :: send_counts(:)     !< Rank i is sending this counts
    class(info_t),      intent(in)    :: recv               !< Information about send buffer
    integer(IP),        intent(in)    :: recv_counts(:)     !< Rank i is recieving this counts
    integer(IP),        intent(in)    :: transpose_id       !< Id of transpose plan. Reserved for future effort flag development.
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI_Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    TYPE_MPI_DATATYPE                 :: temp1              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp2              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp3              !< Temporary datatype
    TYPE_MPI_DATATYPE                 :: temp4              !< Temporary datatype
    integer(IP)                       :: displ              !< Rank i is sending / recieving with this displacement in bytes
    integer(IP)                       :: ierr

    if ( transpose_id == 1 ) then
      call MPI_Type_vector(send%counts(2) * send%counts(3), 1, send%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(3), temp2, self%send%dtypes(i), ierr)
      displ = recv_counts(3) * base_storage
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1, temp2)

      call MPI_Type_vector(recv%counts(3), recv%counts(1) * send_counts(3), recv%counts(1) * recv%counts(2),  base_type, temp1, ierr)
      displ = recv%counts(1) * send_counts(3) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + displ
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      call free_datatypes(temp1)
    else
      call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(3), send%counts(1), base_type, temp1, ierr)
      displ = recv_counts(3) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if ( i < n_neighbors ) self%send%displs(i + 1) = self%send%displs(i) + displ
      call free_datatypes(temp1)

      call MPI_Type_vector(recv%counts(3), 1, recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv%counts(1), temp2, temp3, ierr)
      call MPI_Type_create_hvector(send_counts(3), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = recv%counts(1) * send_counts(3) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)

      ! call MPI_Type_contiguous(recv%counts(1) * send_counts(3) * recv_counts(3), base_type, self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if ( i < n_neighbors ) self%recv%displs(i + 1) = self%recv%displs(i) + displ
      call free_datatypes(temp1, temp2, temp3, temp4)
    endif
  end subroutine create_transpose_ZX

!------------------------------------------------------------------------------------------------
  subroutine free_datatypes(t1, t2, t3, t4)
!------------------------------------------------------------------------------------------------
!< Frees temporary datatypes
!------------------------------------------------------------------------------------------------
    TYPE_MPI_DATATYPE,  intent(inout), optional :: t1      !< Temporary datatype
    TYPE_MPI_DATATYPE,  intent(inout), optional :: t2      !< Temporary datatype
    TYPE_MPI_DATATYPE,  intent(inout), optional :: t3      !< Temporary datatype
    TYPE_MPI_DATATYPE,  intent(inout), optional :: t4      !< Temporary datatype
    integer(IP)                                 :: ierr

    if ( present(t1) ) call MPI_Type_free(t1, ierr)
    if ( present(t2) ) call MPI_Type_free(t2, ierr)
    if ( present(t3) ) call MPI_Type_free(t3, ierr)
    if ( present(t4) ) call MPI_Type_free(t4, ierr)
  end subroutine free_datatypes
end module dtfft_transpose_m