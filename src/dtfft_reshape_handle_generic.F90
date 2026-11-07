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
module dtfft_reshape_handle_generic
!! This module describes [[reshape_handle_generic]] class
!! It is responsible for managing both Host and CUDA-based transposition operations
!! It executes transpose kernels, memory transfers between GPUs/Hosts, and data unpacking if required
use iso_c_binding,                        only: c_ptr, c_f_pointer, c_loc, c_null_ptr
use iso_fortran_env
use dtfft_abstract_backend,               only: abstract_backend, backend_helper
use dtfft_abstract_kernel
use dtfft_abstract_reshape_handle,        only: abstract_reshape_handle, create_args, execute_args
#ifdef DTFFT_WITH_NVSHMEM
use dtfft_backend_cufftmp_m,              only: backend_cufftmp
#endif
#ifdef DTFFT_WITH_NCCL
use dtfft_backend_nccl_m,                 only: backend_nccl
#endif
use dtfft_backend_mpi,                    only: backend_mpi
#ifdef DTFFT_WITH_COMPRESSION
use dtfft_abstract_compressor
use dtfft_compressor
#endif
use dtfft_config
use dtfft_errors
#ifdef DTFFT_WITH_CUDA
use dtfft_kernel_device,                  only: kernel_device
#endif
use dtfft_kernel_host,                    only: kernel_host, host_kernel_t
use dtfft_pencil,                         only: pencil
use dtfft_parameters
use dtfft_serializer,                     only: serializer
use dtfft_utils
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
implicit none
private
public :: reshape_handle_generic

    type :: data_handle
    !! Helper class used to obtain displacements and
    !! counts needed to send to other processes
        integer(int32),        allocatable  :: ls(:,:)                  !! Starts of my data that I should send or recv
                                                                        !! while communicating with other processes
        integer(int32),        allocatable  :: ln(:,:)                  !! Counts of my data that I should send or recv
                                                                        !! while communicating with other processes
        integer(int32),        allocatable  :: sizes(:,:)               !! Counts of every rank in a comm
        integer(int32),        allocatable  :: starts(:,:)              !! Starts of every rank in a comm
        integer(int32),        allocatable  :: displs(:)                !! Local buffer displacement
        integer(int32),        allocatable  :: counts(:)                !! Number of elements to send or recv
    contains
        procedure,  pass(self)  :: create => create_data_handle         !! Creates handle
        procedure,  pass(self)  :: destroy => destroy_data_handle       !! Destroys handle
    end type data_handle

    type, extends(abstract_reshape_handle) :: reshape_handle_generic
    !! Generic Transpose Handle
    !! Executes transposition in 3 steps:
    !!
    !! For ``DTFFT_TRANSPOSE_MODE_PACK``:
    !! - Single transpose kernel execution
    !! - Data exchange between processes
    !! - `P` unpack kernels execution
    !!
    !! For ``DTFFT_TRANSPOSE_MODE_UNPACK``
    !! - `P` pack kernels execution
    !! - Data exchange between processes
    !! - `P` transpose-unpack kernels execution
    !!
    !! For fused backends:
    !! -  For `i` in `P`:
    !!    * find process according to schedule = `k`
    !!    * pack with optional transpose to rank `k`. Compress data (optional)
    !!    * send to rank `k`
    !!
    !! -  Do while true:
    !!    * Check if data arrived from anybody
    !!    * For i in `P_1`:
    !!        Decompress data (optional), then unpack with optional transpose
    !!    * If done for `P` ranks exit
    private
        ! logical                                   :: is_created = .false.
        logical                                   :: has_exchange = .false.   !! If current handle has exchanges
        logical                                   :: is_pipelined = .false.   !! If underlying exchanges are pipelined
        logical                                   :: is_fused = .false.       !! If underlying exchanges are fused
        logical                                   :: is_async_supported = .false. !! If underlying backend support async execution(execute/execute_end)
        logical                                   :: is_pack_free = .false.   !! Are we using pack free reshape or not
        logical                                   :: is_unpack_free = .false. !! Are we using unpack free reshape or not
        integer(int64)                            :: aux_bytes = 0            !! Number of workspace bytes required
        class(abstract_kernel),     allocatable   :: pack_kernel              !! Kernel for data transposition
        class(abstract_kernel),     allocatable   :: unpack_kernel            !! Kernel for unpacking data
        class(abstract_backend),    allocatable   :: comm_handle              !! Communication handle
        type(dtfft_backend_t)                     :: loaded_backend           !! Backend used for communication
        type(kernel_options)                      :: pack_kernel_options       !! Options for pack kernel
        type(kernel_options)                      :: unpack_kernel_options     !! Options for unpack kernel
#ifdef DTFFT_WITH_COMPRESSION
        class(abstract_compressor), allocatable   :: compressor               !! Compressor
#endif
    contains
        procedure, pass(self) :: create_private => create           !! Creates Generic Transpose Handle
        procedure, pass(self) :: execute          !! Executes transpose - exchange - unpack
        procedure, pass(self) :: execute_end      !! Finalizes async transpose
        procedure, pass(self) :: get_async_active !! Returns if async transpose is active
        procedure, pass(self) :: destroy          !! Destroys Generic Transpose Handle
        procedure, pass(self) :: get_aux_bytes    !! Returns number of bytes required by aux buffer
        procedure, pass(self) :: get_backend_private => get_backend
        ! procedure, pass(self) :: get_kernel_decisions  !! Returns tuning results of local kernels
        procedure, pass(self) :: save_private
        procedure, pass(self) :: load_private
#ifdef DTFFT_WITH_COMPRESSION
        procedure, pass(self) :: report_compression
#endif
    end type reshape_handle_generic

contains

    ! subroutine get_kernel_decisions(self, pack_decision, unpack_decision)
    ! !! Returns the tuning results of both local kernels, ready to be written to a plan file
    !     class(reshape_handle_generic),  intent(in)  :: self             !! Generic reshape handle
    !     type(kernel_decision),          intent(out) :: pack_decision    !! Pack kernel tuning result
    !     type(kernel_decision),          intent(out) :: unpack_decision  !! Unpack kernel tuning result

    !     pack_decision = kernel_decision()
    !     unpack_decision = kernel_decision()

    !     ! A slot can be allocated but never created -- a handle without an exchange only
    !     ! builds its pack kernel. Reading tuning results out of such a kernel would store
    !     ! whatever its members happen to hold, which is not a decision at all.
    !     if ( allocated(self%pack_kernel) ) then
    !         if ( self%pack_kernel%is_created ) pack_decision = read_kernel_decision(self%pack_kernel)
    !     endif
    !     if ( allocated(self%unpack_kernel) ) then
    !         if ( self%unpack_kernel%is_created ) unpack_decision = read_kernel_decision(self%unpack_kernel)
    !     endif
    ! end subroutine get_kernel_decisions

! #ifdef DTFFT_WITH_CUDA
!     type(c_ptr) function build_kernel_options(decision, host_opts, device_opts)
! #else
!     type(c_ptr) function build_kernel_options(decision, host_opts)
! #endif
!     !! Turns a stored tuning result back into the options struct the kernel expects.
!     !!
!     !! Returns `c_null_ptr` for an absent slot, which is how a kernel is told to work its
!     !! own parameters out. `host_opts` and `device_opts` must outlive the kernel's `create`,
!     !! which is why they are owned by the caller rather than by this function.
!         type(kernel_decision),               intent(in)    :: decision     !! Stored tuning result
!         type(kernel_host_options),   target, intent(inout) :: host_opts    !! Storage for host options
! #ifdef DTFFT_WITH_CUDA
!         type(kernel_device_options), target, intent(inout) :: device_opts  !! Storage for device options
! #endif

!         build_kernel_options = c_null_ptr
!         select case ( decision%slot_kind )
!         case ( KERNEL_SLOT_HOST )
!             host_opts = kernel_host_options(host_kernel_t(decision%host_kernel),                &
!                                             dtfft_access_mode_t(int(decision%access_mode, int32)), &
!                                             decision%co_threshold)
!             build_kernel_options = c_loc(host_opts)
! #ifdef DTFFT_WITH_CUDA
!         case ( KERNEL_SLOT_DEVICE )
!             device_opts = kernel_device_options(int(decision%tile_size, int32), int(decision%block_rows, int32))
!             build_kernel_options = c_loc(device_opts)
! #endif
!         endselect
!     end function build_kernel_options

!     type(kernel_decision) function read_kernel_decision(kernel)
!     !! Reads the tuning result out of a kernel.
!     !!
!     !! Host and device kernels are tuned on different parameters, so the decision records
!     !! which of the two it describes.
!         class(abstract_kernel), intent(in) :: kernel   !! Kernel to interrogate

!         type(host_kernel_t)       :: host_kernel
!         type(dtfft_access_mode_t) :: access_mode
!         type(kernel_type_t)       :: internal_kernel_type
!         integer(int32)            :: threshold, tile_size, block_rows

!         read_kernel_decision = kernel_decision()
!         select type ( kernel )
!         type is ( kernel_host )
!             call kernel%get_decision(host_kernel, access_mode, threshold)
!             read_kernel_decision%slot_kind = KERNEL_SLOT_HOST
!             read_kernel_decision%host_kernel = host_kernel%val
!             read_kernel_decision%access_mode = int(access_mode%val, int8)
!             read_kernel_decision%co_threshold = threshold
! #ifdef DTFFT_WITH_CUDA
!         type is ( kernel_device )
!             call kernel%get_decision(tile_size, block_rows, internal_kernel_type)
!             read_kernel_decision%slot_kind = KERNEL_SLOT_DEVICE
!             read_kernel_decision%tile_size = int(tile_size, int16)
!             read_kernel_decision%block_rows = int(block_rows, int16)
!             read_kernel_decision%internal_kernel_type = int(internal_kernel_type%val, int8)
! #endif
!         endselect
!     end function read_kernel_decision

    subroutine create_data_handle(self, info, comm, comm_size)
    !! Creates handle
        class(data_handle),   intent(inout) :: self       !! Helper class
        type(pencil),         intent(in)    :: info       !! Pencil info
        TYPE_MPI_COMM,        intent(in)    :: comm       !! MPI communicator
        integer(int32),       intent(in)    :: comm_size  !! Size of ``comm``
        integer(int32)                      :: ierr       !! MPI error flag

        allocate( self%ls( info%rank, 0:comm_size - 1 ), source=0_int32 )
        allocate( self%ln( info%rank, 0:comm_size - 1 ), source=0_int32 )
        allocate( self%sizes ( info%rank, 0:comm_size - 1 ) )
        allocate( self%starts( info%rank, 0:comm_size - 1 ) )
        allocate( self%displs( 0:comm_size - 1 ), source=0_int32 )
        allocate( self%counts( 0:comm_size - 1 ), source=0_int32 )

        call MPI_Allgather(info%counts, int(info%rank, int32), MPI_INTEGER, self%sizes,  int(info%rank, int32), MPI_INTEGER, comm, ierr)
        call MPI_Allgather(info%starts, int(info%rank, int32), MPI_INTEGER, self%starts, int(info%rank, int32), MPI_INTEGER, comm, ierr)
    end subroutine create_data_handle

    subroutine destroy_data_handle(self)
    !! Destroys handle
        class(data_handle),   intent(inout) :: self       !! Helper class

        if(allocated(self%ls))        deallocate(self%ls)
        if(allocated(self%ln))        deallocate(self%ln)
        if(allocated(self%sizes))     deallocate(self%sizes)
        if(allocated(self%starts))    deallocate(self%starts)
        if(allocated(self%displs))    deallocate(self%displs)
        if(allocated(self%counts))    deallocate(self%counts)
    end subroutine destroy_data_handle

    subroutine check_if_overflow(sizes)
    !! Checks if product of sizes fits into integer(int32)
        integer(int32), intent(in) :: sizes(:)  !! Sizes to check
        integer(int64)  :: sizes_prod !! Product of sizes
        integer(int32)  :: i          !! Counter

        sizes_prod = 1_int64
        do i = 1, size(sizes)
            sizes_prod = sizes_prod * int(sizes(i), int64)
        enddo
        if ( sizes_prod /= product(sizes) ) then
            INTERNAL_ERROR("integer(int64) indexing currently not supported")
        endif
    end subroutine check_if_overflow

    subroutine save_private(self, fh)
        class(reshape_handle_generic),  intent(in)    :: self         !! Abstract reshape Handle
        type(serializer),               intent(inout) :: fh
        type(kernel_options) :: options

        if ( allocated(self%comm_handle) ) then
            call fh%save(int(self%comm_handle%backend%val, int8))
        else
            call fh%save(int(DTFFT_BACKEND_NONE%val, int8))
        endif

        options = self%pack_kernel%get_kernel_options()
        call options%save(fh)

        call fh%save(self%unpack_kernel%is_created)
        if ( self%unpack_kernel%is_created ) then
            options = self%unpack_kernel%get_kernel_options()
            call options%save(fh)
        endif
    end subroutine save_private

    integer(int32) function load_private(self, fh) result(ierr)
        class(reshape_handle_generic),  intent(inout) :: self         !! Abstract reshape Handle
        type(serializer),               intent(inout) :: fh
        integer(int8) :: backend_val
        logical :: is_unpacked_created

        ierr = DTFFT_SUCCESS
        call fh%load(backend_val); self%loaded_backend%val = int(backend_val, int32)
        if ( self%loaded_backend /= DTFFT_BACKEND_NONE .and. .not. is_valid_backend(self%loaded_backend) ) then
            ierr = DTFFT_ERROR_FILE_BACKEND_NOT_SUPPORTED
            return
        endif
        call self%pack_kernel_options%load(fh)
        call fh%load(is_unpacked_created)
        if ( is_unpacked_created ) then
            call self%unpack_kernel_options%load(fh)
        endif
    end function load_private

    subroutine create(self, comm, send, recv, kwargs)
    !! Creates Generic Transpose Handle
        class(reshape_handle_generic),    intent(inout) :: self           !! Generic Transpose Handle
        TYPE_MPI_COMM,                    intent(in)    :: comm           !! MPI Communicator
        type(pencil),                     intent(in)    :: send           !! Send pencil
        type(pencil),                     intent(in)    :: recv           !! Recv pencil
        type(create_args),                intent(in)    :: kwargs         !! Additional arguments
        integer(int8)                     :: ndims              !! Number of dimensions
        integer(int32)                    :: comm_size          !! Size of ``comm``
        integer(int32)                    :: comm_rank          !! Rank in ``comm``
        integer(int32)                    :: ierr               !! MPI error flag
        integer(int32)                    :: sdispl             !! Send displacement
        integer(int32)                    :: rdispl             !! Recv displacement
        integer(int32)                    :: sendsize           !! Number of elements to send
        integer(int32)                    :: recvsize           !! Number of elements to recieve
        integer(int32)                    :: i                  !! Counter
        TYPE_MPI_REQUEST                  :: sr                 !! Send request
        TYPE_MPI_REQUEST                  :: rr                 !! Recv request
        type(kernel_type_t)               :: kernel_type        !! Type of kernel
        integer(int32),       allocatable :: neighbor_data(:,:) !! Aggregated neighbor data
        type(data_handle)                 :: in                 !! Send helper
        type(data_handle)                 :: out                !! Recv helper
        logical                           :: is_two_step_permute!! If transpose is two-step permute
        type(dtfft_transpose_t) :: transpose_type
        type(dtfft_reshape_t)   :: reshape_type
        type(dtfft_transpose_mode_t) :: transpose_mode
        type(dtfft_backend_t) :: true_backend
        integer(int32) :: ssdispl, rrdispl
        logical :: is_reshape_only, ignore_transpose_mode
        logical :: with_compression
        integer(int32) :: dims_permutations
        logical :: zslab, yslab
        integer(int8) :: reshape_strat
        logical :: is_pack_free, is_unpack_free
        integer(int32) :: pack_free_int, unpack_free_int
        ! Kernel launch parameters replayed from a saved plan. Declared here, with `target`,
        ! so that the pointers handed to the kernels stay valid for the whole of `create`.
        ! type(kernel_host_options),   target :: host_pack_opts, host_unpack_opts
!         type(c_ptr)                         :: pack_opts, unpack_opts
! #ifdef DTFFT_WITH_CUDA
!         type(kernel_device_options), target :: device_pack_opts, device_unpack_opts
! #endif

!         pack_opts = c_null_ptr
!         unpack_opts = c_null_ptr
! #ifdef DTFFT_WITH_CUDA
!         pack_opts = build_kernel_options(kwargs%pack_decision, host_pack_opts, device_pack_opts)
!         unpack_opts = build_kernel_options(kwargs%unpack_decision, host_unpack_opts, device_unpack_opts)
! #else
!         pack_opts = build_kernel_options(kwargs%pack_decision, host_pack_opts)
!         unpack_opts = build_kernel_options(kwargs%unpack_decision, host_unpack_opts)
! #endif

        transpose_type = kwargs%helper%transpose_type
        reshape_type   = kwargs%helper%reshape_type
        transpose_mode = self%transpose_mode
        ignore_transpose_mode = kwargs%platform == DTFFT_PLATFORM_CUDA

        call self%destroy()
        call check_if_overflow(send%counts)
        call check_if_overflow(recv%counts)

        if ( kwargs%platform == DTFFT_PLATFORM_HOST ) then
            allocate( kernel_host :: self%pack_kernel )
            allocate( kernel_host :: self%unpack_kernel )
#ifdef DTFFT_WITH_CUDA
        else
            allocate( kernel_device :: self%pack_kernel )
            allocate( kernel_device :: self%unpack_kernel )
#endif
        endif

        call MPI_Comm_size(comm, comm_size, ierr)
        call MPI_Comm_rank(comm, comm_rank, ierr)
        self%has_exchange = comm_size > 1

        if ( self%is_transpose ) then
            if( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_X] ) ) then
                kernel_type = KERNEL_PERMUTE_FORWARD
                dims_permutations = DIMS_PERMUTE_BACKWARD
            else
                kernel_type = KERNEL_PERMUTE_BACKWARD
                dims_permutations = DIMS_PERMUTE_FORWARD
            endif
        else
            kernel_type = KERNEL_COPY
            dims_permutations = DIMS_PERMUTE_NONE
        endif

        is_reshape_only = .false.
        if ( .not. self%has_exchange ) then
            self%is_pipelined = .false.
            self%is_fused = .false.
            self%is_async_supported = .false.
            self%is_created = .true.
            if ( self%is_loaded ) then
                call self%pack_kernel%create(send%counts, kwargs%effort, kwargs%base_storage, kernel_type, force_effort=kwargs%force_effort, &
                    options=self%pack_kernel_options)
            else
                call self%pack_kernel%create(send%counts, kwargs%effort, kwargs%base_storage, kernel_type, force_effort=kwargs%force_effort)
            endif
            return
        endif

        allocate( neighbor_data(5, comm_size), source=0_int32 )

        call in%create(send, comm, comm_size)
        call out%create(recv, comm, comm_size)

        ndims = send%rank
        is_pack_free = (reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS) &
            .and. (all([(send%counts(2) == out%sizes(2, i), i=0,comm_size - 1)]) .or. ndims == 2)
        ! Stupid workaround, something is not working for MPI_LAND op for MPI_LOGICAL
        pack_free_int = 0; if ( is_pack_free ) pack_free_int = 1
        ALL_REDUCE(pack_free_int, MPI_INTEGER, MPI_SUM, comm, ierr)
        self%is_pack_free = pack_free_int == comm_size
        if ( .not. self%is_transpose .and. send%rank == 3 ) then
            zslab = .true.
            yslab = .true.
            if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                do i = 0, comm_size - 1
                    zslab = zslab .and. send%counts(2) == out%sizes(2, i)
                    yslab = yslab .and. send%counts(3) == out%sizes(3, i)
                enddo
            else
                do i = 0, comm_size - 1
                    zslab = zslab .and. in%sizes(2, i) == recv%counts(2)
                    yslab = yslab .and. in%sizes(3, i) == recv%counts(3)
                enddo
            endif
            if( zslab ) then
                reshape_strat = 1
            else if ( yslab ) then
                reshape_strat = 2
            else
                reshape_strat = 3
            endif
        endif
        if ( send%rank == 2 .and. .not.self%is_transpose ) reshape_strat = 1

        sdispl = 0; ssdispl = 0
        do i = 0, comm_size - 1
        ! Finding amount of data to send to rank i

            if ( self%is_transpose ) then
                if ( ndims == 2 ) then
                    in%ln(1, i) = out%sizes(2, i)
                    in%ln(2, i) = in%sizes(2, comm_rank)

                    in%ls(1, i) = out%starts(2, i)
                    in%ls(2, i) = in%starts(2, comm_rank)
                else
                    if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
                        in%ln(1, i) = in%sizes(1, comm_rank)
                        in%ln(2, i) = out%sizes(3, i)
                        in%ln(3, i) = in%sizes(3, comm_rank)

                        in%ls(1, i) = in%starts(1, comm_rank)
                        in%ls(2, i) = out%starts(3, i)
                        in%ls(3, i) = in%starts(3, comm_rank)
                    else if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
                        in%ln(1, i) = out%sizes(3, i)
                        in%ln(2, i) = in%sizes(2, comm_rank)
                        in%ln(3, i) = in%sizes(3, comm_rank)

                        in%ls(1, i) = out%starts(3, i)
                        in%ls(2, i) = in%starts(2, comm_rank)
                        in%ls(3, i) = in%starts(3, comm_rank)
                    else if( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z]) ) then
                        in%ln(1, i) = out%sizes(3, i)
                        in%ln(2, i) = in%sizes(2, comm_rank)
                        in%ln(3, i) = in%sizes(3, comm_rank)

                        in%ls(1, i) = out%starts(3, i)
                        in%ls(2, i) = in%starts(2, comm_rank)
                        in%ls(3, i) = in%starts(3, comm_rank)
                    else
                        in%ln(1, i) = out%sizes(2, i)
                        in%ln(2, i) = in%sizes(2, comm_rank)
                        in%ln(3, i) = in%sizes(3, comm_rank)

                        in%ls(1, i) = out%starts(2, i)
                        in%ls(2, i) = in%starts(2, comm_rank)
                        in%ls(3, i) = in%starts(3, comm_rank)
                    endif
                endif
                if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
                    neighbor_data(4, i + 1) = in%ln(1, i) * in%ls(2, i)
                else
                    neighbor_data(4, i + 1) = in%ls(1, i)
                endif
            else ! .not. self%is_transpose
                if ( ndims == 2 ) then
                    if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                        in%ln(1, i) = in%sizes(1, comm_rank)
                        in%ln(2, i) = out%sizes(2, i)

                        in%ls(1, i) = in%starts(1, comm_rank)
                        in%ls(2, i) = out%starts(2, i)
                        ! self%is_pack_free = .true.
                    else
                        in%ln(1, i) = out%sizes(1, i)
                        in%ln(2, i) = in%sizes(2, comm_rank)

                        in%ls(1, i) = out%starts(1, i)
                        in%ls(2, i) = in%starts(2, comm_rank)
                    endif
                else
                    if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                        in%ln(1, i) = in%sizes(1, comm_rank)
                        in%ln(2, i) = out%sizes(2, i)
                        in%ln(3, i) = out%sizes(3, i)

                        in%ls(1, i) = in%starts(1, comm_rank)
                        in%ls(2, i) = out%starts(2, i)
                        in%ls(3, i) = out%starts(3, i)

                        ! if ( send%counts(2) == out%sizes(2, i) ) self%is_pack_free = .true.
                    else
                        in%ln(1, i) = out%sizes(1, i)
                        in%ln(2, i) = in%sizes(2, comm_rank)
                        in%ln(3, i) = in%sizes(3, comm_rank)

                        in%ls(1, i) = out%starts(1, i)
                        in%ls(2, i) = in%starts(2, comm_rank)
                        in%ls(3, i) = in%starts(3, comm_rank)
                    endif
                endif

                if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                    select case ( reshape_strat )
                    case ( 1_int8 )
                        neighbor_data(4, i + 1) = ssdispl
                        ssdispl = ssdispl + product(in%ln(: ,i))
                    case ( 2_int8 )
                        neighbor_data(4, i + 1) = ssdispl
                        ssdispl = ssdispl + in%ln(1, i) * in%ln(2, i)
                    case default
                        ssdispl = (out%starts(3, i) - send%starts(3)) * send%counts(1) * send%counts(2) + (out%starts(2, i) - send%starts(2)) * send%counts(1)
                        neighbor_data(4, i + 1) = ssdispl
                    endselect
                    
                    ! if ( self%is_pack_free ) then
                    !     neighbor_data(4, i + 1) = ssdispl
                    !     ssdispl = ssdispl + product(in%ln(: ,i))
                    ! else
                    !     ! ssdispl = ssdispl + in%ln(1, i) * in%ln(2, i)
                    !     ssdispl = (out%starts(3, i) - send%starts(3)) * send%counts(1) * send%counts(2) + (out%starts(2, i) - send%starts(2)) * send%counts(1)
                    !     neighbor_data(4, i + 1) = ssdispl
                    ! endif
                else
                    neighbor_data(4, i + 1) = in%ls(1, i)
                endif
            endif

            neighbor_data(1, i + 1) = in%ln(1, i)
            neighbor_data(2, i + 1) = in%ln(2, i)
            if ( ndims == 3 ) then
                neighbor_data(3, i + 1) = in%ln(3, i)
            else
                neighbor_data(3, i + 1) = 1
            endif
            neighbor_data(5, i + 1) = sdispl

            sendsize = product( in%ln(:, i) )
            in%counts(i) = sendsize
            in%displs(i) = sdispl
            sdispl = sdispl + sendsize

            ! Sending with tag = me to rank i
            call MPI_Isend(sendsize, 1, MPI_INTEGER4, i, comm_rank, comm, sr, ierr)
            call MPI_Wait(sr, MPI_STATUS_IGNORE, ierr)
        enddo

        is_two_step_permute = .false.
        true_backend = kwargs%backend; if ( self%is_loaded ) true_backend = self%loaded_backend
        self%is_pipelined = is_backend_pipelined(true_backend)
        self%is_fused = is_backend_fused(true_backend) .or. is_backend_compressed(true_backend)
        if ( self%is_fused ) self%is_pipelined = .true.
        if ( self%is_transpose ) then
            is_two_step_permute = any(transpose_type == [DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Z_TO_Y])           &
                            .and. ndims == 3                                                                          &
                            .and. (.not. is_backend_cufftmp(true_backend))                                          &
                            .and. (.not. self%is_fused .or. true_backend == DTFFT_BACKEND_NCCL_COMPRESSED)

            if ( is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_START
            if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK .and. .not.ignore_transpose_mode) kernel_type = KERNEL_PACK
        else ! reshape
            kernel_type = KERNEL_PACK
            if ( self%is_pack_free ) kernel_type = KERNEL_DUMMY
            if ( is_backend_cufftmp(true_backend) ) then
                kernel_type = KERNEL_DUMMY
                is_reshape_only = .true.
                self%is_fused = .true.
            endif
        endif

        if ( self%is_fused .and. .not. true_backend == DTFFT_BACKEND_NCCL_COMPRESSED) kernel_type = get_fused(kernel_type)
        with_compression = is_backend_compressed(true_backend)
#ifndef DTFFT_WITH_COMPRESSION
        if ( with_compression ) then
            INTERNAL_ERROR("ndef DTFFT_WITH_COMPRESSION")
        endif
#endif
#ifdef DTFFT_WITH_COMPRESSION
        if ( with_compression ) then
            ierr = create_compressor(kwargs%compression_config, ndims, kwargs%platform, kwargs%base_type, dims_permutations, self%compressor)
#ifdef DTFFT_DEBUG
            if ( .not. allocated(self%compressor) ) then
                INTERNAL_ERROR("reshape_handle_generic.create: Failed to create compressor")
            endif
#endif
        endif
#endif
        if ( self%is_loaded ) then
            call self%pack_kernel%create(send%counts, kwargs%effort, kwargs%base_storage, kernel_type, neighbor_data, kwargs%force_effort, &
                        with_compression=with_compression, options=self%pack_kernel_options)
        else
            call self%pack_kernel%create(send%counts, kwargs%effort, kwargs%base_storage, kernel_type, neighbor_data, kwargs%force_effort, &
                with_compression=with_compression)
        endif
#ifdef DTFFT_WITH_COMPRESSION
        if ( with_compression ) call self%pack_kernel%set_compressor(self%compressor)
#endif

        is_unpack_free = (reshape_type == DTFFT_RESHAPE_X_PENCILS_TO_BRICKS .or. reshape_type == DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS) &
            .and. (all([(send%counts(2) == out%sizes(2, i), i=0,comm_size - 1)]) .or. ndims == 2)
    ! Another Stupid workaround, something is not working for MPI_LAND op for MPI_LOGICAL
        unpack_free_int = 0; if ( is_unpack_free ) unpack_free_int = 1
        ALL_REDUCE(unpack_free_int, MPI_INTEGER, MPI_SUM, comm, ierr)
        self%is_unpack_free = unpack_free_int == comm_size
        rdispl = 0; rrdispl = 0
        do i = 0, comm_size - 1
        ! Recieving from i with tag i
            call MPI_Irecv(recvsize, 1, MPI_INTEGER4, i, i, comm, rr, ierr)
            call MPI_Wait(rr, MPI_STATUS_IGNORE, ierr)
            if ( recvsize > 0 ) then
                if ( self%is_transpose ) then
                    if ( ndims == 2 ) then
                        out%ln(1, i) = in%sizes(2, i)
                        out%ln(2, i) = out%sizes(2, comm_rank)

                        out%ls(1, i) = in%starts(2, i)
                        out%ls(2, i) = out%starts(2, comm_rank)
                    else
                        if ( transpose_type == DTFFT_TRANSPOSE_X_TO_Z ) then
                            out%ln(1, i) = in%sizes(3, i)
                            out%ln(2, i) = out%sizes(2, comm_rank)
                            out%ln(3, i) = out%sizes(3, comm_rank)

                            out%ls(1, i) = in%starts(3, i)
                            out%ls(2, i) = out%starts(2, comm_rank)
                            out%ls(3, i) = out%starts(3, comm_rank)
                        else if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
                            out%ln(1, i) = out%sizes(1, comm_rank)
                            out%ln(2, i) = in%sizes(3, i)
                            out%ln(3, i) = out%sizes(3, comm_rank)

                            out%ls(1, i) = out%starts(1, comm_rank)
                            out%ls(2, i) = in%starts(3, i)
                            out%ls(3, i) = out%starts(3, comm_rank)
                        else if ( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z]) ) then
                            out%ln(1, i) = in%sizes(2, i)
                            out%ln(2, i) = out%sizes(2, comm_rank)
                            out%ln(3, i) = out%sizes(3, comm_rank)

                            out%ls(1, i) = in%starts(2, i)
                            out%ls(2, i) = in%starts(2, comm_rank)
                            out%ls(3, i) = out%starts(3, comm_rank)
                        else
                        ! ZXY -> YZX
                        ! YZX -> XYZ
                            out%ln(1, i) = in%sizes(3, i)
                            out%ln(2, i) = out%sizes(2, comm_rank)
                            out%ln(3, i) = out%sizes(3, comm_rank)

                            out%ls(1, i) = in%starts(3, i)
                            out%ls(2, i) = in%starts(2, comm_rank)
                            out%ls(3, i) = out%starts(3, comm_rank)
                        endif
                    endif
                else ! is_transpose
                    if ( ndims == 2 ) then
                        if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                            out%ln(1, i) = in%sizes(1, i)
                            out%ln(2, i) = out%sizes(2, comm_rank)

                            out%ls(1, i) = in%starts(1, i)
                            out%ls(2, i) = out%starts(2, comm_rank)
                        else
                            out%ln(1, i) = out%sizes(1, comm_rank)
                            out%ln(2, i) = in%sizes(2, i)

                            out%ls(1, i) = out%starts(1, comm_rank)
                            out%ls(2, i) = in%starts(2, i)
                            ! self%is_unpack_free = .true.
                        endif
                    else
                        if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                            out%ln(1, i) = in%sizes(1, i)
                            out%ln(2, i) = out%sizes(2, comm_rank)
                            out%ln(3, i) = out%sizes(3, comm_rank)

                            out%ls(1, i) = in%starts(1, i)
                            out%ls(2, i) = out%starts(2, comm_rank)
                            out%ls(3, i) = out%starts(3, comm_rank)
                        else
                            out%ln(1, i) = out%sizes(1, comm_rank)
                            out%ln(2, i) = in%sizes(2, i)
                            out%ln(3, i) = in%sizes(3, i)

                            out%ls(1, i) = out%starts(1, comm_rank)
                            out%ls(2, i) = in%starts(2, i)
                            out%ls(3, i) = in%starts(3, i)

                            ! if ( send%counts(2) == out%sizes(2, i) ) self%is_unpack_free = .true.
                        endif
                    endif
                endif
            endif

            neighbor_data(1, i + 1) = out%ln(1, i)
            neighbor_data(2, i + 1) = out%ln(2, i)
            if ( ndims == 3 ) then
                neighbor_data(3, i + 1) = out%ln(3, i)
            else
                neighbor_data(3, i + 1) = 1
            endif
            neighbor_data(4, i + 1) = rdispl
            if ( self%is_transpose ) then
                if ( transpose_type == DTFFT_TRANSPOSE_Z_TO_X ) then
                    neighbor_data(5, i + 1) = out%ln(1, i) * out%ls(2, i)! out%sizes(1, comm_rank) * out%sizes(2, comm_rank)
                else
                    neighbor_data(5, i + 1) = out%ls(1, i)
                endif
            else
                if ( reshape_type == DTFFT_RESHAPE_X_BRICKS_TO_PENCILS .or. reshape_type == DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS ) then
                    neighbor_data(5, i + 1) = out%ls(1, i)

                else
                    select case ( reshape_strat )
                    case ( 1_int8, 2_int8 )
                        neighbor_data(5, i + 1) = rrdispl
                        rrdispl = rrdispl + out%ln(1, i) * out%ln(2, i)
                    case default
                        rrdispl = abs(recv%starts(3) - in%starts(3, i)) * recv%counts(1) * recv%counts(2) + abs(recv%starts(2) - in%starts(2, i)) * recv%counts(1)
                        neighbor_data(5, i + 1) = rrdispl
                    endselect

                    ! neighbor_data(5, i + 1) = rrdispl
                    ! prev:
                    ! rrdispl = rrdispl + out%ln(1, i) * out%ln(2, i)

                    ! rrdispl = abs(recv%starts(3) - in%starts(3, i)) * recv%counts(1) * recv%counts(2) + abs(recv%starts(2) - in%starts(2, i)) * recv%counts(1)
                    ! neighbor_data(5, i + 1) = rrdispl
                    ! endif
                ! endif
                endif
            endif
            out%counts(i) = recvsize
            out%displs(i) = rdispl
            rdispl = rdispl + recvsize
        enddo

        kernel_type = KERNEL_UNPACK
        if ( self%is_pipelined ) kernel_type = KERNEL_UNPACK_PIPELINED
        if ( is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_END
        if ( self%is_pipelined .and. is_two_step_permute ) kernel_type = KERNEL_PERMUTE_BACKWARD_END_PIPELINED
        if ( true_backend == DTFFT_BACKEND_CUFFTMP ) kernel_type = KERNEL_COPY
        if ( true_backend == DTFFT_BACKEND_CUFFTMP_PIPELINED ) kernel_type = KERNEL_DUMMY

        if ( self%is_unpack_free ) kernel_type = KERNEL_DUMMY
        if ( is_reshape_only ) kernel_type = KERNEL_DUMMY
        if ( transpose_mode == DTFFT_TRANSPOSE_MODE_UNPACK .and. .not.ignore_transpose_mode) then
            if ( any( transpose_type == [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_X]) ) then
                kernel_type = KERNEL_UNPACK_FORWARD
                if ( self%is_pipelined ) kernel_type = KERNEL_UNPACK_FORWARD_PIPELINED
            else
                kernel_type = KERNEL_UNPACK_BACKWARD
                if ( self%is_pipelined ) kernel_type = KERNEL_UNPACK_BACKWARD_PIPELINED
            endif
        endif
        if ( self%is_loaded ) then
            call self%unpack_kernel%create(recv%counts, kwargs%effort, kwargs%base_storage, kernel_type, neighbor_data, kwargs%force_effort, &
                with_decompression=with_compression, options=self%unpack_kernel_options)
        else
            call self%unpack_kernel%create(recv%counts, kwargs%effort, kwargs%base_storage, kernel_type, neighbor_data, kwargs%force_effort, &
                with_decompression=with_compression)
        endif
#ifdef DTFFT_WITH_COMPRESSION
        if ( with_compression ) call self%unpack_kernel%set_compressor(self%compressor)
#endif

        if ( is_backend_mpi(true_backend) ) then
            allocate( backend_mpi :: self%comm_handle )
#ifdef DTFFT_WITH_CUDA
        else if ( is_backend_nccl(true_backend) ) then
# ifdef DTFFT_WITH_NCCL
            allocate( backend_nccl :: self%comm_handle )
# else
            INTERNAL_ERROR("not DTFFT_WITH_NCCL")
# endif
        else if ( is_backend_cufftmp(true_backend) ) then
# ifdef DTFFT_WITH_NVSHMEM
            allocate( backend_cufftmp :: self%comm_handle )
# else
            INTERNAL_ERROR("not DTFFT_WITH_NVSHMEM")
# endif
#endif
        else
            INTERNAL_ERROR("Unknown backend")
        endif

        self%is_async_supported = is_backend_mpi(true_backend)                                                &
                                .and. .not. self%is_pipelined                                                 &
                                .and. kwargs%platform == DTFFT_PLATFORM_HOST                                  &
                                .and. .not.(true_backend == DTFFT_BACKEND_MPI_P2P_SCHEDULED)

        block
            type(dtfft_backend_t) :: fixed_backend

            fixed_backend = true_backend
            if ( is_reshape_only ) then
                self%is_pipelined = .false.
                fixed_backend = DTFFT_BACKEND_CUFFTMP
            endif
            call self%comm_handle%create(fixed_backend, kwargs%helper, kwargs%platform, kwargs%comm_id, in%displs, in%counts, out%displs, out%counts, kwargs%base_storage)
        endblock

        if ( self%is_pipelined ) then
            call self%comm_handle%set_unpack_kernel(self%unpack_kernel)
        endif
        if ( self%is_fused ) then
            call self%comm_handle%set_pack_kernel(self%pack_kernel)
        endif

        if ( self%is_pack_free .or. self%is_unpack_free ) then
            self%aux_bytes = kwargs%base_storage * max( product(send%counts), product(recv%counts) )
        endif

        self%is_created = .true.

        call in%destroy()
        call out%destroy()
        deallocate( neighbor_data )
    end subroutine create

    subroutine execute(self, in, out, kwargs, error_code)
    !! Executes transpose - exchange - unpack
        class(reshape_handle_generic),  intent(inout)   :: self       !! Generic Transpose Handle
        type(c_ptr),                    intent(in)      :: in           !! Send pointer
        type(c_ptr),                    intent(in)      :: out          !! Recv pointer
        type(execute_args),             intent(inout)   :: kwargs     !! Additional arguments
        integer(int32),                 intent(out)     :: error_code !! Error code

        error_code = DTFFT_SUCCESS

        if ( self%is_fused ) then
            call self%comm_handle%execute(in, out, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
            return
        endif

        if ( self%is_pipelined ) then
            if ( self%is_pack_free ) then
                ! Reshape pack-free
                ! packing is skipped, since it is already packed in a case of Z-slab reshaping
                ! in -> aux     exchange
                ! aux -> out    unpack
                call self%comm_handle%execute(in, out, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
            else if ( self%is_unpack_free ) then
                ! Reshape unpack-free
                ! in -> aux     pack
                ! aux -> out    exchange
                call self%pack_kernel%execute(in, kwargs%p1, kwargs%stream)
                call self%comm_handle%execute(kwargs%p1, in, kwargs%stream, out, kwargs%exec_type, error_code)
            else
                ! Transpose and reshape with packing
                ! in -> aux     pack
                ! aux -> in     exchange
                ! in -> out     unpack
                call self%pack_kernel%execute(in, kwargs%p1, kwargs%stream)
                call self%comm_handle%execute(kwargs%p1, out, kwargs%stream, in, kwargs%exec_type, error_code)
            endif
            return
        endif

        if ( self%is_pack_free ) then
            call self%comm_handle%execute(in, kwargs%p1, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
            if ( error_code /= DTFFT_SUCCESS ) return
            if ( self%is_async_supported .and. kwargs%exec_type == EXEC_NONBLOCKING ) return
            call self%unpack_kernel%execute(kwargs%p1, out, kwargs%stream)
            return
        endif

        if ( self%is_unpack_free ) then
            call self%pack_kernel%execute(in, kwargs%p1, kwargs%stream)
            call self%comm_handle%execute(kwargs%p1, out, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
            return
        endif

        ! Transpose and reshape with packing
        ! in -> out       pack
        ! out -> in       exchange
        ! in -> out       unpack

        call self%pack_kernel%execute(in, out, kwargs%stream)
        if ( .not. self%has_exchange ) return
        call self%comm_handle%execute(out, in, kwargs%stream, kwargs%p1, kwargs%exec_type, error_code)
        if ( error_code /= DTFFT_SUCCESS ) return
        if ( self%is_async_supported .and. kwargs%exec_type == EXEC_NONBLOCKING ) return
        call self%unpack_kernel%execute(in, out, kwargs%stream)
    end subroutine execute

    subroutine execute_end(self, kwargs, error_code)
    !! Ends execution of transposition
        class(reshape_handle_generic),  intent(inout) :: self       !! Generic Transpose Handle
        type(execute_args),               intent(inout) :: kwargs     !! Additional arguments
        integer(int32),                   intent(out)   :: error_code !! Error code

        error_code = DTFFT_SUCCESS
        if( .not. self%is_async_supported ) return

        call self%comm_handle%execute_end(error_code)
        if( error_code /= DTFFT_SUCCESS ) return
        if ( self%is_pack_free ) then
            call self%unpack_kernel%execute(kwargs%p3, kwargs%p2, kwargs%stream)
        else
            call self%unpack_kernel%execute(kwargs%p1, kwargs%p2, kwargs%stream)
        endif
    end subroutine execute_end

    elemental logical function get_async_active(self)
        class(reshape_handle_generic),  intent(in)    :: self       !! Generic Transpose Handle

        get_async_active = .false.
        if( .not. self%is_async_supported ) return
        get_async_active = self%comm_handle%get_async_active()
    end function get_async_active

    subroutine destroy(self)
    !! Destroys Generic Transpose Handle
        class(reshape_handle_generic),   intent(inout) :: self      !! Generic Transpose Handle

        if ( allocated( self%pack_kernel ) ) then
            call self%pack_kernel%destroy()
            deallocate( self%pack_kernel )
        endif
        if ( allocated( self%comm_handle ) ) then
            call self%comm_handle%destroy()
            deallocate( self%comm_handle )
        endif
        if ( allocated( self%unpack_kernel ) ) then
            call self%unpack_kernel%destroy()
            deallocate( self%unpack_kernel )
        endif
#ifdef DTFFT_WITH_COMPRESSION
        if ( allocated( self%compressor ) ) then
            call self%compressor%destroy()
            deallocate( self%compressor )
        endif
#endif
        self%aux_bytes = 0
        self%is_pack_free = .false.
        self%is_unpack_free = .false.
        self%is_pipelined = .false.
        self%is_fused = .false.
        self%is_created = .false.
    end subroutine destroy

    pure integer(int64) function get_aux_bytes(self)
    !! Returns number of bytes required by aux buffer
        class(reshape_handle_generic),   intent(in)    :: self      !! Generic Transpose Handle

        get_aux_bytes = self%aux_bytes
        if ( .not. self%has_exchange ) return
        get_aux_bytes = max( get_aux_bytes, self%comm_handle%get_aux_bytes() )
    end function get_aux_bytes

    elemental type(dtfft_backend_t) function get_backend(self)
        class(reshape_handle_generic),   intent(in)    :: self       !! Abstract reshape Handle

        get_backend = DTFFT_BACKEND_NONE
        if ( .not. self%has_exchange ) return
        get_backend = self%comm_handle%backend
    end function get_backend

#ifdef DTFFT_WITH_COMPRESSION
    subroutine report_compression(self, name)
        class(reshape_handle_generic),    intent(in)    :: self       !! Abstract reshape Handle
        character(len=*),                 intent(in)    :: name
        real(real64) :: avg_rate, min_rate, max_rate, mean_rate
        integer(int32) :: comm_size, ierr

        if ( .not. self%has_exchange ) return
        if ( .not. is_backend_compressed(self%comm_handle%backend) ) return

        WRITE_INFO("  "//name)
        if ( self%compressor%execution_count == 0 ) then
            WRITE_INFO("    Never executed...")
            return
        endif
        avg_rate = self%compressor%get_average_rate()

        call MPI_Reduce(avg_rate, min_rate, 1, MPI_REAL8, MPI_MIN, 0, self%comm_handle%comm, ierr)
        call MPI_Reduce(avg_rate, max_rate, 1, MPI_REAL8, MPI_MAX, 0, self%comm_handle%comm, ierr)
        call MPI_Reduce(avg_rate, mean_rate, 1, MPI_REAL8, MPI_SUM, 0, self%comm_handle%comm, ierr)
        call MPI_Comm_size(self%comm_handle%comm, comm_size, ierr)

        WRITE_INFO("    Min rate = "//to_str(real(min_rate, real32)))
        WRITE_INFO("    Max rate = "//to_str(real(max_rate, real32)))
        WRITE_INFO("    Mean rate = "//to_str(real(mean_rate, real32) / real(comm_size, real32)))
    end subroutine report_compression
#endif
end module dtfft_reshape_handle_generic