!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

#include "dtfft_config.h"
program test_serializer
use iso_fortran_env
use dtfft_errors
use dtfft_serializer, only: serializer
use test_suite, only: test_suite_t
#include "_dtfft_mpi.h"
implicit none

    character(len=*), parameter :: filename = "test_serializer.dat"
    integer(int32) :: ierr
    type(test_suite_t) :: test

    call test%create("Serializer")

    call test_mixed_roundtrip()

    call test%finalize()

contains

    subroutine test_mixed_roundtrip()
        type(serializer) :: writer, reader
        integer(int8) :: i8, loaded_i8
        integer(int16) :: i16, loaded_i16
        integer(int32) :: i32, loaded_i32
        integer(int64) :: i64, loaded_i64
        integer(int32) :: values(3), loaded_values(3)
        character(len=11) :: text, loaded_text
        logical :: flag, loaded_flag
        integer(int32) :: comm_rank, loaded_comm_rank
        integer(int32) :: collective_values(2), loaded_collective_values(2)
        character(len=8) :: collective_text, loaded_collective_text
        logical :: collective_flag, loaded_collective_flag
        integer(int8) :: tail_i8, loaded_tail_i8
        integer(int32) :: unit, io_stat

        i8 = -101_int8
        i16 = -12345_int16
        i32 = 123456789_int32
        i64 = -987654321012345678_int64
        values = [11_int32, -22_int32, 33_int32]
        text = "hello MPI!"
        flag = .true.

        ierr = writer%create(MPI_COMM_WORLD, filename, MPI_MODE_CREATE + MPI_MODE_WRONLY)
        call test%check("local create for write", ierr, DTFFT_SUCCESS)
        call writer%save(i8)
        call writer%save(i16)
        call writer%save(i32)
        call writer%save(i64)
        call writer%save(values)
        call writer%save(text)
        call writer%save(flag)

        call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
        collective_values = [comm_rank, -comm_rank]
        collective_text = "        "
        write(collective_text, "(a,i0)") "rank", comm_rank
        collective_flag = mod(comm_rank, 2) == 0
        call writer%save(comm_rank, .true.)
        call writer%save(collective_values, .true.)
        call writer%save(collective_text, .true.)
        call writer%save(collective_flag, .true.)

        tail_i8 = 77_int8
        call writer%save(tail_i8)
        call writer%destroy()

        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        ierr = reader%create(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY)
        call test%check("local create for read", ierr, DTFFT_SUCCESS)
        call reader%load(loaded_i8)
        call reader%load(loaded_i16)
        call reader%load(loaded_i32)
        call reader%load(loaded_i64)
        call reader%load(loaded_values)
        call reader%load(loaded_text)
        call reader%load(loaded_flag)
        call reader%load(loaded_comm_rank, .true.)
        call reader%load(loaded_collective_values, .true.)
        call reader%load(loaded_collective_text, .true.)
        call reader%load(loaded_collective_flag, .true.)
        call reader%load(loaded_tail_i8)
        call reader%destroy()

        call test%check("local integer(int8)", loaded_i8 == i8)
        call test%check("local integer(int16)", loaded_i16 == i16)
        call test%check("local integer(int32)", loaded_i32 == i32)
        call test%check("local integer(int64)", loaded_i64 == i64)
        call test%check("local integer(int32) array", all(loaded_values == values))
        call test%check("local character", loaded_text == text)
        call test%check("local logical", loaded_flag .eqv. flag)
        call test%check("collective integer", loaded_comm_rank == comm_rank)
        call test%check("collective array", all(loaded_collective_values == collective_values))
        call test%check("collective character", loaded_collective_text == collective_text)
        call test%check("collective logical", loaded_collective_flag .eqv. collective_flag)
        call test%check("local value after collective", loaded_tail_i8 == tail_i8)

        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        if ( comm_rank == 0 ) then
            open(newunit=unit, file=filename, status="old", iostat=io_stat)
            if ( io_stat == 0 ) close(unit, status="delete")
        endif
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
    end subroutine test_mixed_roundtrip

end program test_serializer
