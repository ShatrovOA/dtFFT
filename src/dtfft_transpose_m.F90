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
module dtfft_transpose_m
!------------------------------------------------------------------------------------------------
!< This module describes [[transpose_t]] class
!------------------------------------------------------------------------------------------------
use dtfft_info_m
use dtfft_precisions
#include "dtfft.i90"
implicit none
private
public :: transpose_t

  integer(MPI_ADDRESS_KIND), parameter :: LB = 0_MPI_ADDRESS_KIND
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
    ! integer :: sendcount, recvcount
  contains
  private
    procedure, pass(self),  public  :: init                   !< Initializes class
    procedure, pass(self),  public  :: transpose              !< Performs MPI_Alltoall(w)
    procedure, pass(self),  public  :: destroy                !< Destroys class
    procedure, pass(self)           :: create_transpose_2d    !< Creates two-dimensional transposition datatypes
    procedure, pass(self)           :: create_transpose_XY    !< Creates three-dimensional X --> Y, Y --> X transposition datatypes
    procedure, pass(self)           :: create_transpose_YZ    !< Creates three-dimensional Y --> Z, Z --> Y transposition datatypes
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

    if(allocated(self%dtypes)) then
      do i = 1, size(self%dtypes)
        call MPI_Type_free(self%dtypes(i), ierr)
      enddo
      deallocate(self%dtypes)
    endif
    if(allocated(self%displs)) deallocate(self%displs)
    if(allocated(self%counts)) deallocate(self%counts)
  end subroutine destroy_handle

!------------------------------------------------------------------------------------------------
  subroutine init(self, comm, send, recv, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates [[transpose_t]] class
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    TYPE_MPI_COMM,      intent(in)    :: comm               !< 1d communicator
    class(info_t),      intent(in)    :: send               !< Information about send buffer
    class(info_t),      intent(in)    :: recv               !< Information about recv buffer
    TYPE_MPI_DATATYPE,  intent(in)    :: base_type          !< Base MPI Datatype
    integer(IP),        intent(in)    :: base_storage       !< Number of bytes needed to store single element
    integer(IP)                       :: comm_size          !< Size of 1d communicator
    integer(IP)                       :: loop_size          !< Number of datatypes to be created
    integer(IP),        allocatable   :: recv_counts(:,:)   !< Each processor should know how much data each processor recieves
    integer(IP),        allocatable   :: send_counts(:,:)   !< Each processor should know how much data each processor sends
    integer(IP)                       :: i                  !< Counter
    integer(IP)                       :: ierr

    call MPI_Comm_dup(comm, self%comm, ierr)
    call MPI_Comm_size(self%comm, comm_size, ierr)
    self%is_even = send%is_even .and. recv%is_even
    loop_size = comm_size;  if ( self%is_even ) loop_size = 1

    call self%send%init(loop_size)
    call self%recv%init(loop_size)

    allocate(recv_counts(recv%rank, comm_size), source = 0_IP)
    allocate(send_counts, source = recv_counts)
    call MPI_Allgather(recv%counts, recv%rank, MPI_INTEGER, recv_counts, recv%rank, MPI_INTEGER, self%comm, ierr)
    call MPI_Allgather(send%counts, send%rank, MPI_INTEGER, send_counts, send%rank, MPI_INTEGER, self%comm, ierr)
    if(send%rank == 2) then
      do i = 1, loop_size
        call self%create_transpose_2d(loop_size, i, send, send_counts(:,i), recv, recv_counts(:,i), 1, base_type, base_storage)
      enddo
    else
      if(send%aligned_dim == 1 .or. recv%aligned_dim == 1) then
        ! self%sendcount = recv%counts(2) * recv%counts(3) * send%counts(2)
        ! self%recvcount = recv%counts(2) * recv%counts(3) * send%counts(2)
        do i = 1, loop_size
          call self%create_transpose_XY(loop_size, i, send, send_counts(:,i), recv, recv_counts(:,i), 1, base_type, base_storage)
        enddo
      elseif(send%aligned_dim == 3 .or. recv%aligned_dim == 3) then
        ! self%sendcount = recv%counts(2) * recv%counts(3) * send%counts(3)
        ! self%recvcount = recv%counts(2) * recv%counts(3) * send%counts(3)
        do i = 1, loop_size
          call self%create_transpose_YZ(loop_size, i, send, send_counts(:,i), recv, recv_counts(:,i), 1, base_type, base_storage)
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
    class(transpose_t), intent(in)    :: self       !< Transposition class
    type(*),            intent(in)    :: send(..)   !< Send buffer
    type(*),            intent(inout) :: recv(..)   !< Recv buffer
    integer(IP)                       :: ierr

! $acc data present(send, recv)
! $acc host_data use_device(send, recv)
    if(self%is_even) then
      call MPI_Alltoall(send, 1, self%send%dtypes(1), recv, 1, self%recv%dtypes(1), self%comm, ierr)
      ! call MPI_Alltoall(send, self%sendcount, MPI_DOUBLE_COMPLEX, recv, self%recvcount, MPI_DOUBLE_COMPLEX, self%comm)
    else
      call MPI_Alltoallw(send, self%send%counts, self%send%displs, self%send%dtypes,          &
                         recv, self%recv%counts, self%recv%displs, self%recv%dtypes, self%comm, ierr)
    endif
! $acc end host_data
! $acc end data
  end subroutine transpose

!------------------------------------------------------------------------------------------------
  subroutine destroy(self)
!------------------------------------------------------------------------------------------------
!< Destroys [[transpose_t]] class
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self       !< Transposition class
    integer(IP)                       :: ierr

    call MPI_Comm_free(self%comm, ierr)
    call self%send%destroy()
    call self%recv%destroy()
  end subroutine destroy

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_2d(self, loop_size, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates two-dimensional transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: loop_size          !< Size of 1d comm
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


    if(transpose_id == 1) then
      call MPI_Type_vector(send%counts(2), recv_counts(2), send%counts(1), base_type, temp1, ierr)
      displ = recv_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if(i < loop_size) self%send%displs(i + 1) = self%send%displs(i) +  displ
      call free_datatypes(temp1)

      call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send_counts(2), temp2, self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if(i < loop_size) self%recv%displs(i + 1) = self%recv%displs(i) +  send_counts(2) * base_storage
      call free_datatypes(temp1, temp2)
    else
      ! call MPI_Type_contiguous(recv_counts(2), base_type, temp1, ierr)
      ! call MPI_Type_vector(send%counts(2), 1, recv%)
!TODO Find another ways to describe 2d XY Datatype
    endif
  end subroutine create_transpose_2d

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_XY(self, loop_size, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional X --> Y, Y --> X transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: loop_size          !< Size of 1d comm
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

    if(transpose_id == 1) then
    ! This transpose_id has "contiguous" send and strided recieve datatype
      call MPI_Type_vector(send%counts(2) * send%counts(3), recv_counts(2), send%counts(1), base_type, temp1, ierr)
      displ = recv_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if(i < loop_size) self%send%displs(i + 1) = self%send%displs(i) +  displ
      call free_datatypes(temp1)

      call MPI_Type_vector(recv%counts(2), 1, recv%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(send_counts(2), temp2, temp3, ierr)
      call MPI_Type_create_hvector(recv%counts(3), 1, int(recv%counts(1) * recv%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = send_counts(2) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if(i < loop_size) self%recv%displs(i + 1) = self%recv%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3, temp4)
    elseif(transpose_id == 2) then
    ! This transpose_id has strided send and "contiguous" recieve datatypes
      call MPI_Type_vector(send%counts(2) * send%counts(3), 1, send%counts(1), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(2), temp2, self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if(i < loop_size) self%send%displs(i + 1) = self%send%displs(i) +  recv_counts(2) * base_storage
      call free_datatypes(temp1, temp2)

      call MPI_Type_vector(recv%counts(3), send_counts(2), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      displ = send_counts(2) * base_storage
      call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_create_hvector(recv%counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      call MPI_Type_create_resized(temp3, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if(i < loop_size) self%recv%displs(i + 1) = self%recv%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3)
    elseif(transpose_id == 3) then
      ! call MPI_Type_vector(send%counts(2), 1, send%counts(1), base_type, temp1, ierr)
      ! call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      ! call MPI_Type_contiguous(recv_counts(2), temp2, temp3, ierr)
      ! call MPI_Type_create_hvector(send%counts(3), 1, int(send%counts(1) * send%counts(2) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      ! call MPI_Type_create_resized(temp4, LB, int(recv_counts(2), MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      ! call MPI_Type_commit(self%send%dtypes(i), ierr)
      ! if(i < loop_size) self%send%displs(i + 1) = self%send%displs(i) +  recv_counts(2) * base_storage
      ! call free_datatypes(temp1, temp2, temp3, temp4)

      ! call MPI_Type_vector(recv%counts(3), send_counts(2), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      ! displ = send_counts(2) * base_storage
      ! call MPI_Type_create_resized(temp1, LB, int(displ, MPI_ADDRESS_KIND), temp2, ierr)
      ! call MPI_Type_create_hvector(recv%counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      ! call MPI_Type_create_resized(temp3, LB, int(displ, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      ! call MPI_Type_commit(self%recv%dtypes(i), ierr)
      ! if(i < loop_size) self%recv%displs(i + 1) = self%recv%displs(i) +  displ
      ! call free_datatypes(temp1, temp2, temp3)
    endif
  end subroutine create_transpose_XY

!------------------------------------------------------------------------------------------------
  subroutine create_transpose_YZ(self, loop_size, i, send, send_counts, recv, recv_counts, transpose_id, base_type, base_storage)
!------------------------------------------------------------------------------------------------
!< Creates three-dimensional Y --> Z, Z --> Y transposition datatypes
!------------------------------------------------------------------------------------------------
    class(transpose_t), intent(inout) :: self               !< Transposition class
    integer(IP),        intent(in)    :: loop_size          !< Size of 1d comm
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

    if(transpose_id == 1) then
      call MPI_Type_vector(send%counts(3), 1, send%counts(1) * send%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_contiguous(recv_counts(3), temp2, temp3, ierr)
      call MPI_Type_create_hvector(send%counts(2), 1, int(send%counts(1) * base_storage, MPI_ADDRESS_KIND), temp3, temp4, ierr)
      displ = recv_counts(3) * base_storage
      call MPI_Type_create_resized(temp4, LB, int(displ, MPI_ADDRESS_KIND), self%send%dtypes(i), ierr)
      call MPI_Type_commit(self%send%dtypes(i), ierr)
      if(i < loop_size) self%send%displs(i + 1) = self%send%displs(i) +  displ
      call free_datatypes(temp1, temp2, temp3, temp4)

      call MPI_Type_vector(recv%counts(3), send_counts(3), recv%counts(1) * recv%counts(2), base_type, temp1, ierr)
      call MPI_Type_create_resized(temp1, LB, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), temp2, ierr)
      call MPI_Type_create_hvector(send_counts(2), 1, int(recv%counts(1) * base_storage, MPI_ADDRESS_KIND), temp2, temp3, ierr)
      call MPI_Type_create_resized(temp3, LB, int(send_counts(3) * base_storage, MPI_ADDRESS_KIND), self%recv%dtypes(i), ierr)
      call MPI_Type_commit(self%recv%dtypes(i), ierr)
      if(i < loop_size) self%recv%displs(i + 1) = self%recv%displs(i) +  send_counts(3) * base_storage
      call free_datatypes(temp1, temp2)
    else
!TODO Find another ways to describe YZ Datatype
    endif
    end subroutine create_transpose_YZ

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

    if(present(t1)) call MPI_Type_free(t1, ierr)
    if(present(t2)) call MPI_Type_free(t2, ierr)
    if(present(t3)) call MPI_Type_free(t3, ierr)
    if(present(t4)) call MPI_Type_free(t4, ierr)
  end subroutine free_datatypes
end module dtfft_transpose_m