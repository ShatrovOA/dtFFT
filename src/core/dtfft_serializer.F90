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
module dtfft_serializer
use iso_fortran_env
use dtfft_errors
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: serializer

    integer(int64), parameter, public :: BYTES_I8  = 1_int64
      !! Number of bytes in `integer(int8)`
    integer(int64), parameter, public :: BYTES_I16 = 2_int64
      !! Number of bytes in `integer(int16)`
    integer(int64), parameter, public :: BYTES_I32 = 4_int64
      !! Number of bytes in `integer(int32)`
    integer(int64), parameter, public :: BYTES_I64 = 8_int64
      !! Number of bytes in `integer(int64)`

    type :: serializer
    private
        TYPE_MPI_FILE  :: fh
        TYPE_MPI_COMM  :: comm
        integer(MPI_OFFSET_KIND) :: pos = 0
        logical :: is_open = .false.
    contains
    private
        procedure, pass(self),  public  :: create
        procedure, pass(self),  public  :: destroy
        procedure, pass(self),  public  :: get_open
        procedure, pass(self)           :: save_i8, save_i16, save_i32, save_i64
        procedure, pass(self)           :: save_i32_arr, save_char, save_logical, save_double
        generic,   public               :: save => save_i8, save_i16, save_i32, save_i64,   &
                                                    save_i32_arr, save_char, save_logical,  &
                                                    save_double
        procedure, pass(self)           :: save_private
        generic,   public               :: load => load_i8, load_i16, load_i32, load_i64,  &
                                           load_i32_arr, load_char, load_logical, load_double
        procedure, pass(self)           :: load_private
        procedure, pass(self)           :: load_i8, load_i16, load_i32, load_i64
        procedure, pass(self)           :: load_i32_arr, load_char, load_logical, load_double
    end type serializer

contains

    ! subroutine report_mpi_io_error(operation, mpi_error, rank, offset)
    !     character(len=*), intent(in) :: operation
    !     integer(int32), intent(in) :: mpi_error, rank
    !     integer(MPI_OFFSET_KIND), intent(in) :: offset
    !     character(len=MPI_MAX_ERROR_STRING) :: error_string
    !     character(len=MPI_MAX_ERROR_STRING + 256) :: message
    !     integer(int32) :: error_length, ierr

    !     call MPI_Error_string(mpi_error, error_string, error_length, ierr)
    !     if ( ierr == MPI_SUCCESS ) then
    !         message = "MPI-IO "//operation//" failed on rank "//to_str(rank)// &
    !             " at offset "//to_str(int(offset, int64))//": "//error_string(:error_length)
    !     else
    !         message = "MPI-IO "//operation//" failed on rank "//to_str(rank)// &
    !             " at offset "//to_str(int(offset, int64))//" with error code "//to_str(mpi_error)
    !     endif
    !     WRITE_ERROR(trim(message))
    ! end subroutine report_mpi_io_error

    ! subroutine report_mpi_io_count_error(operation, rank, offset, expected, actual)
    !     character(len=*), intent(in) :: operation
    !     integer(int32), intent(in) :: rank, expected, actual
    !     integer(MPI_OFFSET_KIND), intent(in) :: offset
    !     character(len=256) :: message

    !     message = "MPI-IO "//operation//" transferred "//to_str(actual)//" of "// &
    !         to_str(expected)//" bytes on rank "//to_str(rank)//" at offset "// &
    !         to_str(int(offset, int64))
    !     WRITE_ERROR(trim(message))
    ! end subroutine report_mpi_io_count_error

    integer(int32) function create(self, comm, filename, mode)
        class(serializer),  intent(inout)   :: self     !! Cursor
        TYPE_MPI_COMM,      intent(in)      :: comm
        character(len=*),   intent(in)      :: filename
        integer(int32),     intent(in)      :: mode
        integer(int32) :: ierr!, mpi_error, comm_rank
        logical :: exists

        create = DTFFT_SUCCESS
        if ( mode == MPI_MODE_RDONLY ) then
            inquire(file=filename, exist=exists)
            if ( .not. exists ) then
                create = DTFFT_ERROR_FILE_NOT_FOUND
                return
            endif
        endif

        call MPI_File_open(comm, filename, mode, MPI_INFO_NULL, self%fh, ierr)
        if ( ierr /= MPI_SUCCESS ) then
            ! mpi_error = ierr
            ! call MPI_Comm_rank(comm, comm_rank, ierr)
            ! if ( mode /= MPI_MODE_RDONLY .or. mpi_error /= MPI_ERR_NO_SUCH_FILE ) then
            !     call report_mpi_io_error("MPI_File_open", mpi_error, comm_rank, 0_MPI_OFFSET_KIND)
            ! endif
            if ( mode == MPI_MODE_RDONLY ) then
                create = DTFFT_ERROR_FILE_NOT_FOUND
            else
                create = DTFFT_ERROR_FILE_WRITE_FAILED
            endif
            return
        endif
        self%comm = comm
        ! call MPI_Comm_rank(comm, comm_rank, ierr)
        ! call MPI_File_set_errhandler(self%fh, MPI_ERRORS_RETURN, ierr)
        if ( ierr /= MPI_SUCCESS ) then
            ! call report_mpi_io_error("MPI_File_set_errhandler", ierr, comm_rank, 0_MPI_OFFSET_KIND)
            call MPI_File_close(self%fh, ierr)
            create = DTFFT_ERROR_FILE_WRITE_FAILED
            return
        endif
        if ( mode /= MPI_MODE_RDONLY ) then
            call MPI_File_set_size(self%fh, 0_MPI_OFFSET_KIND, ierr)
            if ( ierr /= MPI_SUCCESS ) then
                ! call report_mpi_io_error("MPI_File_set_size", ierr, comm_rank, 0_MPI_OFFSET_KIND)
                call MPI_File_close(self%fh, ierr)
                create = DTFFT_ERROR_FILE_WRITE_FAILED
                return
            endif
        endif
        self%pos = 0
        self%is_open = .true.
    end function create

    subroutine destroy(self)
        class(serializer),  intent(inout)   :: self     !! Cursor
        integer(int32) :: ierr, comm_rank
        if ( .not. self%is_open ) then
            INTERNAL_ERROR("serializer: .not. self%is_open")
        endif
        call MPI_Comm_rank(self%comm, comm_rank, ierr)
        call MPI_File_close(self%fh, ierr)
        ! if ( ierr /= MPI_SUCCESS ) then
        !     call report_mpi_io_error("MPI_File_close", ierr, comm_rank, self%pos)
        ! endif
        self%pos = 0
        self%is_open = .false.
    end subroutine destroy

    pure logical function get_open(self)
        class(serializer),  intent(in)   :: self     !! Cursor
        get_open = self%is_open
    end function get_open

!------------------------------------------------------------------------------------------------
! Writers
!------------------------------------------------------------------------------------------------
    subroutine save_private(self, buf, is_collective)
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int8),      intent(in)    :: buf(:)   !! Value
        logical,  optional, intent(in)    :: is_collective
        logical :: is_collective_
        integer(int32) :: ierr, size_buf
        integer(MPI_OFFSET_KIND) :: offset, osize_buf, inc
        integer(int32) :: comm_rank, comm_size
#ifdef DTFFT_USE_MPI
        integer(int32) :: status(MPI_STATUS_SIZE)
#else
        type(MPI_Status) :: status
#endif
        ! integer(int32) :: transferred

        OPTVAL(is_collective_, .false., is_collective)
        size_buf = size(buf)

        call MPI_Comm_rank(self%comm, comm_rank, ierr)
        if ( is_collective_ ) then
            osize_buf = size(buf, kind=MPI_OFFSET_KIND)
            offset = self%pos + int(comm_rank, MPI_OFFSET_KIND) * osize_buf
            call MPI_File_write_at_all(self%fh, offset, buf, size_buf, MPI_INTEGER1, status, ierr)
            ! if ( ierr /= MPI_SUCCESS ) then
            !     call report_mpi_io_error("MPI_File_write_at_all", ierr, comm_rank, offset)
            ! else
            !     call MPI_Get_count(status, MPI_INTEGER1, transferred, ierr)
            !     if ( ierr /= MPI_SUCCESS ) then
            !         call report_mpi_io_error("MPI_Get_count after MPI_File_write_at_all", &
            !             ierr, comm_rank, offset)
            !     elseif ( transferred /= size_buf ) then
            !         call report_mpi_io_count_error("MPI_File_write_at_all", comm_rank, &
            !             offset, size_buf, transferred)
            !     endif
            ! endif

            call MPI_Comm_size(self%comm, comm_size, ierr)
            inc = int(comm_size, MPI_OFFSET_KIND) * osize_buf
        else
            if ( comm_rank == 0 ) then
                call MPI_File_write_at(self%fh, self%pos, buf, size_buf, MPI_INTEGER1, status, ierr)
                ! if ( ierr /= MPI_SUCCESS ) then
                !     call report_mpi_io_error("MPI_File_write_at", ierr, comm_rank, self%pos)
                ! else
                !     call MPI_Get_count(status, MPI_INTEGER1, transferred, ierr)
                !     if ( ierr /= MPI_SUCCESS ) then
                !         call report_mpi_io_error("MPI_Get_count after MPI_File_write_at", &
                !             ierr, comm_rank, self%pos)
                !     elseif ( transferred /= size_buf ) then
                !         call report_mpi_io_count_error("MPI_File_write_at", comm_rank, &
                !             self%pos, size_buf, transferred)
                !     endif
                ! endif
            endif
            inc = size_buf
        endif
        self%pos = self%pos + inc
    end subroutine save_private

    subroutine save_i8(self, val, is_collective)
    !! Appends `integer(int8)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int8),      intent(in)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective

        call self%save_private([val], is_collective)
    end subroutine save_i8

    subroutine save_i16(self, val, is_collective)
    !! Appends `integer(int16)`
        class(serializer), intent(inout)  :: self     !! Cursor
        integer(int16),     intent(in)    :: val     !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I16)

        buf(:) = transfer(val, 0_int8, int(BYTES_I16))
        call self%save_private(buf, is_collective)
    end subroutine save_i16

    subroutine save_i32(self, val, is_collective)
    !! Appends `integer(int32)`
        class(serializer), intent(inout) :: self     !! Cursor
        integer(int32),     intent(in)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I32)

        buf(:) = transfer(val, 0_int8, int(BYTES_I32))
        call self%save_private(buf, is_collective)
    end subroutine save_i32

    subroutine save_i64(self, val, is_collective)
    !! Appends `integer(int64)`
        class(serializer), intent(inout) :: self     !! Cursor
        integer(int64),     intent(in)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I64)

        buf(:) = transfer(val, 0_int8, int(BYTES_I64))
        call self%save_private(buf, is_collective)
    end subroutine save_i64

    subroutine save_i32_arr(self, vals, is_collective)
    !! Appends an array of `integer(int32)`
        class(serializer), intent(inout) :: self     !! Cursor
        integer(int32),     intent(in)    :: vals(:)  !! Values
        logical,  optional, intent(in)    :: is_collective
        integer(int64) :: n_bytes
        integer(int8), allocatable :: buf(:)

        n_bytes = BYTES_I32 * size(vals, kind=int64)
        allocate(buf(n_bytes))
        buf(:) = transfer(vals, 0_int8, int(n_bytes))
        call self%save_private(buf, is_collective)
        deallocate(buf)
    end subroutine save_i32_arr

    subroutine save_char(self, val, is_collective)
    !! Appends a character string, byte for byte
        class(serializer),  intent(inout)  :: self     !! Cursor
        character(len=*),   intent(in)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int64) :: n_bytes, i
        integer(int8), allocatable :: buf(:)

        n_bytes = len(val, kind=int64)
        allocate(buf(n_bytes))
        do i = 1, n_bytes
            buf(i) = int(iachar(val(i:i)), int8)
        enddo
        call self%save_private(buf, is_collective)
        deallocate(buf)
    end subroutine save_char

    subroutine save_logical(self, val, is_collective)
    !! Appends a logical as a single byte
        class(serializer),  intent(inout) :: self     !! Cursor
        logical,            intent(in)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective

        call self%save(merge(1_int8, 0_int8, val), is_collective)
    end subroutine save_logical

    subroutine save_double(self, val, is_collective)
        class(serializer),  intent(inout)   :: self
        real(real64),       intent(in)      :: val
        logical,  optional, intent(in)      :: is_collective

        call self%save(transfer(val, 1_int64), is_collective)
    end subroutine save_double

    subroutine load_private(self, buf, is_collective)
    !! Appends `integer(int8)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int8),      intent(out)   :: buf(:)   !! Value
        logical,  optional, intent(in)    :: is_collective
        logical :: is_collective_
        integer(int32) :: ierr, size_buf
        integer(MPI_OFFSET_KIND) :: offset, osize_buf, inc
        integer(int32) :: comm_rank, comm_size
#ifdef DTFFT_USE_MPI
        integer(int32) :: status(MPI_STATUS_SIZE)
#else
        type(MPI_Status) :: status
#endif
        ! integer(int32) :: transferred

        OPTVAL(is_collective_, .false., is_collective)
        size_buf = size(buf)
        buf = 0_int8

        call MPI_Comm_rank(self%comm, comm_rank, ierr)
        if ( is_collective_ ) then
            osize_buf = size(buf, kind=MPI_OFFSET_KIND)
            offset = self%pos + int(comm_rank, MPI_OFFSET_KIND) * osize_buf
            call MPI_File_read_at_all(self%fh, offset, buf, size_buf, MPI_INTEGER1, status, ierr)
            ! if ( ierr /= MPI_SUCCESS ) then
            !     call report_mpi_io_error("MPI_File_read_at_all", ierr, comm_rank, offset)
            ! else
            !     call MPI_Get_count(status, MPI_INTEGER1, transferred, ierr)
            !     if ( ierr /= MPI_SUCCESS ) then
            !         call report_mpi_io_error("MPI_Get_count after MPI_File_read_at_all", &
            !             ierr, comm_rank, offset)
            !     elseif ( transferred /= size_buf ) then
            !         call report_mpi_io_count_error("MPI_File_read_at_all", comm_rank, &
            !             offset, size_buf, transferred)
            !     endif
            ! endif

            call MPI_Comm_size(self%comm, comm_size, ierr)
            inc = int(comm_size, MPI_OFFSET_KIND) * osize_buf
        else
            call MPI_File_read_at(self%fh, self%pos, buf, size_buf, MPI_INTEGER1, status, ierr)
            ! if ( ierr /= MPI_SUCCESS ) then
            !     call report_mpi_io_error("MPI_File_read_at", ierr, comm_rank, self%pos)
            ! else
            !     call MPI_Get_count(status, MPI_INTEGER1, transferred, ierr)
            !     if ( ierr /= MPI_SUCCESS ) then
            !         call report_mpi_io_error("MPI_Get_count after MPI_File_read_at", &
            !             ierr, comm_rank, self%pos)
            !     elseif ( transferred /= size_buf ) then
            !         call report_mpi_io_count_error("MPI_File_read_at", comm_rank, &
            !             self%pos, size_buf, transferred)
            !     endif
            ! endif
            inc = size_buf
        endif
        self%pos = self%pos + inc
    end subroutine load_private

    subroutine load_i8(self, val, is_collective)
    !! Appends `integer(int8)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int8),      intent(out)   :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: v(1)

        call self%load_private(v, is_collective)
        val = v(1)
    end subroutine load_i8

    subroutine load_i16(self, val, is_collective)
    !! Appends `integer(int16)`
        class(serializer),  intent(inout)  :: self     !! Cursor
        integer(int16),     intent(out)   :: val     !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I16)

        call self%load_private(buf, is_collective)
        val = transfer(buf, 0_int16)
    end subroutine load_i16

    subroutine load_i32(self, val, is_collective)
    !! Appends `integer(int32)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int32),     intent(out)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I32)

        call self%load_private(buf, is_collective)
        val = transfer(buf, 0_int32)
    end subroutine load_i32

    subroutine load_i64(self, val, is_collective)
    !! Appends `integer(int64)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int64),     intent(out)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf(BYTES_I64)

        call self%load_private(buf, is_collective)
        val = transfer(buf, 0_int64)
    end subroutine load_i64

    subroutine load_i32_arr(self, vals, is_collective)
    !! Appends an array of `integer(int32)`
        class(serializer),  intent(inout) :: self     !! Cursor
        integer(int32),     intent(inout)    :: vals(:)  !! Values
        logical,  optional, intent(in)    :: is_collective
        integer(int64) :: n_bytes
        integer(int8), allocatable :: buf(:)

        n_bytes = BYTES_I32 * size(vals, kind=int64)
        allocate(buf(n_bytes))
        call self%load_private(buf, is_collective)
        vals(:) = transfer(buf, 0_int32, size(vals))
        deallocate(buf)
    end subroutine load_i32_arr

    subroutine load_char(self, val, is_collective)
    !! Appends a character string, byte for byte
        class(serializer),  intent(inout)  :: self     !! Cursor
        character(len=*),   intent(out)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int64) :: n_bytes, i
        integer(int8), allocatable :: buf(:)

        n_bytes = len(val, kind=int64)
        allocate(buf(n_bytes))
        call self%load_private(buf, is_collective)
        do i = 1, n_bytes
            val(i:i) = achar(iand(int(buf(i), int32), 255))
        enddo
        deallocate(buf)
    end subroutine load_char

    subroutine load_logical(self, val, is_collective)
    !! Appends a logical as a single byte
        class(serializer),  intent(inout) :: self     !! Cursor
        logical,            intent(out)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int8) :: buf

        call self%load(buf, is_collective)
        val = buf > 0
    end subroutine load_logical

    subroutine load_double(self, val, is_collective)
        class(serializer),  intent(inout) :: self     !! Cursor
        real(real64),       intent(out)    :: val      !! Value
        logical,  optional, intent(in)    :: is_collective
        integer(int64) :: val_

        call self%load(val_, is_collective)
        val = transfer(val_, 1._real64)
    end subroutine load_double
end module dtfft_serializer