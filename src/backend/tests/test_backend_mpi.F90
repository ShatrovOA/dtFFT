#include "dtfft_config.h"
program test_backend_mpi
use iso_c_binding
use iso_fortran_env
use dtfft_backend_mpi
use dtfft_abstract_backend
! use dtfft_abstract_compressor
use dtfft_abstract_kernel
use dtfft_kernel_host
! #ifdef DTFFT_WITH_ZFP
! use dtfft_compressor_zfp
! #elif defined(DTFFT_WITH_SZ3)
! use dtfft_compressor_sz3
! #endif
#ifdef DTFFT_WITH_CUDA
use dtfft_kernel_device
#endif
use dtfft_errors,          only: DTFFT_SUCCESS
use dtfft_parameters
use dtfft_pencil,          only: pencil
use dtfft_utils
#include "_dtfft_mpi.h"
implicit none
    integer(int32) :: ierr, comm_rank, comm_size
    integer(int32) :: n_backends, n_platforms, global_sizes(4)
    type(dtfft_backend_t) :: backends(10)
    type(dtfft_platform_t) :: platforms(2)
    type(pencil), target :: pencils(1)
    TYPE_MPI_COMM :: comms(1)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)

    n_backends = 0
    call add_backend(DTFFT_BACKEND_MPI_P2P)
    call add_backend(DTFFT_BACKEND_MPI_A2A)
    call add_backend(DTFFT_BACKEND_MPI_P2P_PIPELINED)
    call add_backend(DTFFT_BACKEND_MPI_P2P_SCHEDULED)
    call add_backend(DTFFT_BACKEND_MPI_P2P_FUSED)
#ifdef DTFFT_WITH_RMA
    call add_backend(DTFFT_BACKEND_MPI_RMA)
    call add_backend(DTFFT_BACKEND_MPI_RMA_PIPELINED)
    call add_backend(DTFFT_BACKEND_MPI_RMA_FUSED)
#endif
! #ifdef DTFFT_WITH_COMPRESSION
!     call add_backend(DTFFT_BACKEND_MPI_P2P_COMPRESSED)
! # ifdef DTFFT_WITH_RMA
!     call add_backend(DTFFT_BACKEND_MPI_RMA_COMPRESSED)
! # endif
! #endif

    platforms(1) = DTFFT_PLATFORM_HOST
    n_platforms = 1
#ifdef DTFFT_WITH_CUDA
    platforms(2) = DTFFT_PLATFORM_CUDA
    n_platforms = 2
#endif


    if (comm_rank == 0) then
        write(output_unit, '(a)')        "=========================================="
        write(output_unit, '(a)')        "  test_backend_mpi"
        write(output_unit, '(a)')        "=========================================="
        ! write(output_unit, '(a,i0)')     "  N_GLOBAL: ", N_GLOBAL
        write(output_unit, '(a,i0)')     "  Ranks: ", comm_size
        ! write(output_unit, '(a,i0)')     "  Backends: ", n_backends
        write(output_unit, '(a)')        ""
    end if

    global_sizes(1) = 5
    global_sizes(2) = 13
    global_sizes(3) = 75
    global_sizes(4) = 115

    block
        integer(int32) :: i, j, k
        do k = 1, n_platforms
            do j = 1, size(global_sizes)
                if (comm_rank == 0) then
                    write(output_unit, '(a,i0)') "Testing platform "//get_platform_string(platforms(k))//" with global size ", global_sizes(j)
                end if
                do i = 1, n_backends
                    call run_backend(backends(i), global_sizes(j), platforms(k))
                end do
            enddo
        enddo
    end block

    call MPI_Finalize(ierr)

contains

    subroutine add_backend(b)
        type(dtfft_backend_t), intent(in) :: b
        n_backends = n_backends + 1
        backends(n_backends) = b
    end subroutine add_backend

  !------------------------------------------------------------
  ! Compute the local slice for a given rank.
  ! Mirrors dtfft_pencil get_local_size: the last `res` ranks get one extra.
  !------------------------------------------------------------
    pure subroutine local_size(n_global, nprocs, rank, lstart, lcount)
        integer(int32), intent(in)  :: n_global, nprocs, rank
        integer(int32), intent(out) :: lstart, lcount
        integer(int32) :: res, base, i

        base   = n_global / nprocs
        res    = mod(n_global, nprocs)
        lcount = base + merge(1, 0, rank >= nprocs - res)
        lstart = 0
        do i = 0, rank - 1
            lstart = lstart + base + merge(1, 0, i >= nprocs - res)
        end do
    end subroutine local_size

  !------------------------------------------------------------
  ! Main per-backend test routine.
  !
  ! 1D buffer redistribution:
  !   Each rank r owns local_size(r) = block_dist(N_GLOBAL, comm_size, r) floats.
  !   Rank r splits its local_size(r) into comm_size chunks and sends chunk s
  !   to rank s.  Counts are in floats (base_storage = FLOAT_STORAGE_SIZE,
  !   scaler = 1, so element == float).
  !
  !   send_count(r→s) = block_dist(local_size(r), comm_size, s)
  !   recv_count(r←s) = block_dist(local_size(s), comm_size, r)
  !------------------------------------------------------------
  subroutine run_backend(b, global_size, platform)
    type(dtfft_backend_t),  intent(in)  :: b
    integer(int32),         intent(in)  :: global_size
    type(dtfft_platform_t), intent(in)  :: platform

    integer(int32), allocatable :: send_counts(:), recv_counts(:)
    integer(int32), allocatable :: send_displs(:), recv_displs(:)
    ! integer(int32), allocatable :: all_send_counts(:), all_recv_counts(:)
    real(real32),   allocatable, target :: in_buf(:), out_buf(:), aux_buf(:)
    type(c_ptr) :: aux_ptr
    class(abstract_kernel), allocatable :: pack, unpack
    type(backend_mpi)      :: backend
    type(backend_helper)   :: helper
    integer(int32), allocatable :: pack_locals(:,:), unpack_locals(:,:)
    integer(int32) :: r, err_local, aux_size, ierr
    integer(int32) :: my_local_size, my_start    ! this rank's slice of N_GLOBAL
    integer(int32) :: rs, rc                     ! remote start/count (temp)
    integer(int64) :: aux_bytes
    integer(int32) :: total_recv
    real(real32)   :: expected, got
    logical :: with_compression


    allocate(send_counts(comm_size), recv_counts(comm_size))
    allocate(send_displs(comm_size), recv_displs(comm_size))
    ! allocate(all_send_counts(comm_size), all_recv_counts(comm_size))

    comms(1) = MPI_COMM_WORLD
    call helper%create(platform, MPI_COMM_WORLD, comms, .false., pencils)

    if ( platform == DTFFT_PLATFORM_HOST ) then
        allocate(kernel_host :: pack)
        allocate(kernel_host :: unpack)
#ifdef DTFFT_WITH_CUDA
    else
        allocate(kernel_device :: pack)
        allocate(kernel_device :: unpack)
#endif
    endif

    call local_size(global_size, comm_size, comm_rank, my_start, my_local_size)

    allocate(pack_locals(5, comm_size))
    send_displs(1) = 0
    ! ---- how many of THIS rank's elements go to each rank? ----
    do r = 0, comm_size - 1
      call local_size(my_local_size, comm_size, r, rs, rc)
      send_counts(r + 1) = rc
      if ( r > 0 ) then
        send_displs(r + 1) = send_displs(r) + send_counts(r)
      endif

      pack_locals(1:2, r + 1) = 1
      pack_locals(3, r + 1) = rc
      pack_locals(4:5, r + 1) = send_displs(r + 1)
    end do

    call MPI_Alltoall(send_counts, 1, MPI_INTEGER, recv_counts, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)

    ! ---- send_count(this→s): split my local buffer into comm_size chunks ----
    
    ! total_send = 0
    ! do r = 1, comm_size
    !   call local_slice(my_local_size, comm_size, r - 1, rs, rc)
    !   send_counts(r) = rc
    !   send_displs(r) = total_send
    !   total_send = total_send + rc

    ! end do

    ! ---- recv_count(this←s): chunk of rank s's buffer destined for this rank ----
    allocate(unpack_locals(5, comm_size))
    total_recv = 0
    recv_displs(1) = 0
    do r = 1, comm_size
      ! call local_slice(all_local_sizes(r), comm_size, comm_rank, rs, rc)
      ! recv_counts(r) = rc
      ! recv_displs(r) = total_recv
      if ( r > 1) recv_displs(r) = recv_displs(r - 1) + recv_counts(r - 1)
      ! total_recv = total_recv + rc
      unpack_locals(1:2, r) = 1
      unpack_locals(3, r) = recv_counts(r)
      unpack_locals(4:5, r) = recv_displs(r)
    end do
    total_recv = sum(recv_counts)

    ! ---- create backend (base_storage = FLOAT_STORAGE_SIZE, scaler = 1) ----
    call backend%create(b, helper, platform, 1_int8, &
                        send_displs, send_counts, recv_displs, recv_counts, &
                        FLOAT_STORAGE_SIZE)

    with_compression = .false.
    ! if (is_backend_compressed(b)) then
    !     with_compression = .true.
    ! endif

    call pack%create([max(my_local_size, 1)], DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, KERNEL_COPY_PIPELINED, pack_locals, with_compression=with_compression)
    call unpack%create([max(total_recv, 1)], DTFFT_ESTIMATE, FLOAT_STORAGE_SIZE, KERNEL_COPY_PIPELINED, unpack_locals, with_decompression=with_compression)
    if (backend%is_pipelined .or. backend%is_fused) then
        call backend%set_pack_kernel(pack)
        call backend%set_unpack_kernel(unpack)
    end if

!     if (with_compression) then
! #ifdef DTFFT_WITH_COMPRESSION
! # ifdef DTFFT_WITH_ZFP
!         allocate( compressor_zfp :: compressor )
! # elif defined(DTFFT_WITH_SZ3)
!         allocate( compressor_sz3 :: compressor )
! # endif

!         ! call compressor%create()
! #else
!         INTERNAL_ERROR(".not. DTFFT_WITH_COMPRESSION")
! #endif
!     endif

    ! ---- allocate buffers ----
    ! buf_size = int(backend%buffer_size, int32)
    ! if (buf_size < 1) buf_size = max(total_send, total_recv, 1)

    allocate(in_buf(max(my_local_size, 1)), out_buf(max(total_recv, 1)))
    in_buf  = 0.0_real32
    out_buf = -3.0_real32

    ! fill in_buf: each rank's local slice with its rank+1 value
    in_buf(1:my_local_size) = real(comm_rank + 1, real32)

    aux_bytes = backend%get_aux_bytes()
    ALL_REDUCE(aux_bytes, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, ierr)
    ! if (comm_rank == 0) 
    ! print*,'aux_bytes = ',aux_bytes
    if (aux_bytes > 0) then
        aux_size = int(aux_bytes / FLOAT_STORAGE_SIZE, int32)
        if (aux_size < 1) aux_size = 1
        allocate(aux_buf(aux_size))
        aux_buf = -2.0_real32
        aux_ptr = c_loc(aux_buf)
    else
        allocate(aux_buf(1))
        aux_buf = -2.0_real32
        aux_ptr = c_null_ptr
    end if

    ! ---- execute ----
    call backend%execute(c_loc(in_buf), c_loc(out_buf), NULL_STREAM, aux_ptr, EXEC_BLOCKING, err_local)
    if (err_local /= DTFFT_SUCCESS) then
        write(error_unit, '(a,a,a,i0)') "FAIL [", dtfft_get_backend_string(b), "] execute error=", err_local
        call MPI_Abort(MPI_COMM_WORLD, err_local, ierr)
    end if

    do r = 1, comm_size
        if (recv_counts(r) > 0) then
            expected = real(r, real32)   ! sender rank+1
            got = out_buf(recv_displs(r) + 1)
            if (abs(got - expected) > 1.0e-6_real32) then
                write(error_unit, '(a,a,a,i0,a,f6.1,a,f6.1)') &
                    "FAIL [", dtfft_get_backend_string(b), "] from rank ", r - 1, &
                    " expected=", expected, " got=", got
                call MPI_Abort(MPI_COMM_WORLD, 5, ierr)
            end if
        end if
    end do

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    ! ---- report ----
    if (comm_rank == 0) then
        write(output_unit, '(a)') "  PASSED ["//dtfft_get_backend_string(b)//"]"
    end if

    call pack%destroy()
    call unpack%destroy()
    deallocate(pack, unpack)
    call backend%destroy()
    call helper%destroy()
    deallocate(in_buf, out_buf, aux_buf)
    deallocate(send_counts, recv_counts, send_displs, recv_displs)
    ! deallocate(all_local_sizes)
  end subroutine run_backend

end program test_backend_mpi
