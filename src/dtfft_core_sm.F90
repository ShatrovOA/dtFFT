submodule (dtfft_core_m) dtfft_core_sm
#if !defined(NO_FFTW3) 
use dtfft_executor_fftw_m
#endif
#if defined(MKL_ENABLED)
use dtfft_executor_mkl_m
#endif
! #if defined(CUFFT_ENABLED)
! use dtfft_executor_cufft_m
! #endif
#if defined(KFR_ENABLED)
use dtfft_executor_kfr_m
#endif
use iso_fortran_env, only: output_unit, error_unit
#include "dtfft.i90"
#include "dtfft_core.i90"
implicit none

contains

  module procedure transpose_private
#define __FUNC__ "transpose"
#ifdef __DEBUG
    integer(IP)                         :: ierr
    real(R8P)                           :: elapsed              !< Mean time of transposition
    real(R8P)                           :: t_sum                !< Total time by all processes
    character(len=100)                  :: debug_msg

    elapsed = -MPI_Wtime()
#endif
    if(transpose_type > 0) then
      call self%transpose_out(transpose_type)%transpose(in, out)
    else
      call self%transpose_in(abs(transpose_type))%transpose(in, out)
    endif

#ifdef __DEBUG
    elapsed = elapsed + MPI_Wtime()
    call MPI_Allreduce(elapsed, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
    elapsed = t_sum / real(self%comm_size, R8P)

    write(debug_msg, '(a, f0.8, a4)') 'Transpose '//TRANSPOSE_NAMES(transpose_type)//' took ', elapsed, ' sec'
    DEBUG(debug_msg)
#endif
#undef __FUNC__
  end procedure transpose_private

  module procedure transpose
#define __FUNC__ "transpose"
    CHECK_PLAN
    CHECK_INPUT("transpose_type", transpose_type, VALID_TRANSPOSES)
    if(self%ndims == 2 .and. abs(transpose_type) > 1) FATAL_ERROR("Trying to transpose 2-dimensional data in 3rd dimension")
    call self%transpose_private(in, out, transpose_type)
#undef __FUNC__
  end procedure transpose

  module procedure execute
#define __FUNC__ "execute"
    CHECK_PLAN
    CHECK_INPUT("transpose_type", transpose_type, VALID_FULL_TRANSPOSES)
    if(self%is_transpose_plan) FATAL_ERROR("Plan's been created without FFT support, use 'transpose' method instead")

    call self%check_aux(aux=aux)
    if(self%ndims == 2) then
      if ( present(aux) ) then
        call self%execute_2d(in, out, transpose_type, aux)
      else
        call self%execute_2d(in, out, transpose_type, self%aux)
      endif
    else
      if ( present(aux) ) then
        call self%execute_3d(in, out, transpose_type, aux)
      else
        call self%execute_3d(in, out, transpose_type, self%aux)
      endif
    endif
#undef __FUNC__
  end procedure execute


  module procedure execute_2d
    select case (transpose_type)
    case (DTFFT_TRANSPOSE_OUT)
      ! 1d FFT X direction
      call self%forw_plans(1)%fft%execute(in)
      ! Transpose X -> Y
      call self%transpose_private(in, out, DTFFT_TRANSPOSE_X_TO_Y)
      ! 1d FFT Y direction
      call self%forw_plans(2)%fft%execute(out)
    case (DTFFT_TRANSPOSE_IN)
      ! 1d iFFT Y direction
      call self%back_plans(2)%fft%execute(in)
      ! Transpose Y -> X
      call self%transpose_private(in, out, DTFFT_TRANSPOSE_Y_TO_X)
      ! 1d iFFT X direction
      call self%back_plans(1)%fft%execute(out)
    endselect
  end procedure execute_2d

  module procedure execute_3d
    select case (transpose_type)
    case (DTFFT_TRANSPOSE_OUT)
      ! 1d FFT X direction
      call self%forw_plans(1)%fft%execute(in)
      call self%execute_transposed_out(in, out, aux)
    case (DTFFT_TRANSPOSE_IN)
      call self%execute_transposed_in(in, out, aux)
      call self%back_plans(1)%fft%execute(out)
    endselect
  end procedure execute_3d

  module procedure destroy
    call self%destroy_internal()
  end procedure destroy

  module procedure destroy_internal
    integer(IP) :: d                  !< Counter

    if ( .not. self%is_created ) return

    if(allocated(self%info)) then
      do d = 1, self%ndims
        call self%info(d)%destroy()
      enddo
      deallocate(self%info)
    endif

    if(allocated(self%forw_plans)) then
      do d = 1, self%ndims
        call self%forw_plans(d)%fft%destroy();  deallocate(self%forw_plans(d)%fft)
        call self%back_plans(d)%fft%destroy();  deallocate(self%back_plans(d)%fft)
      enddo
      deallocate(self%forw_plans, self%back_plans)
    endif
    if(allocated(self%aux))         deallocate(self%aux)
    if(allocated(self%comm_dims))   deallocate(self%comm_dims)
    if(allocated(self%comm_coords)) deallocate(self%comm_coords)

    self%is_created = .false.
    self%is_aux_alloc = .false.

    block
      logical     :: is_finalized
      integer(IP) :: ierr

      call MPI_Finalized(is_finalized, ierr)
      if ( is_finalized ) return

      if(allocated(self%transpose_in) .and. allocated(self%transpose_out)) then
        do d = 1, self%ndims - 1
          call self%transpose_in(d)%destroy()
          call self%transpose_out(d)%destroy()
        enddo
        deallocate(self%transpose_in)
        deallocate(self%transpose_out)
      endif

      if(allocated(self%comms)) then
        do d = 1, self%ndims
          call MPI_Comm_free(self%comms(d), ierr)
        enddo
        deallocate(self%comms)
      endif
      call MPI_Comm_free(self%comm, ierr)
    end block

    self%ndims = -1
  end procedure destroy_internal

  module procedure get_local_sizes
#define __FUNC__ "get_local_sizes"
    CHECK_PLAN
    call self%get_local_sizes_internal(in_starts, in_counts, out_starts, out_counts, alloc_size)
#undef __FUNC__
  end procedure get_local_sizes

  module procedure get_local_sizes_internal
    if(present(in_starts))  in_starts(1:self%ndims)   = self%info(1)%starts
    if(present(in_counts))  in_counts(1:self%ndims)   = self%info(1)%counts
    if(present(out_starts)) out_starts(1:self%ndims)  = self%info(self%ndims)%starts
    if(present(out_counts)) out_counts(1:self%ndims)  = self%info(self%ndims)%counts
    if(present(alloc_size)) alloc_size  = max(product(self%info(1)%counts), product(self%info(self%ndims)%counts))
  end procedure get_local_sizes_internal

  module procedure get_aux_size
#define __FUNC__ "get_aux_size"
    CHECK_PLAN
    get_aux_size = 0_SP
    if(self%is_aux_needed) then
      get_aux_size = int(product(self%info(self%aux_pencil)%counts), SP)
    endif
#undef __FUNC__
  end procedure get_aux_size

  module procedure init_core
#define __FUNC__ "init_core"
    integer(IP)                         :: ierr
    integer(IP)                           :: planner_flag         !< DTFFT planner effort flag
    integer(IP)                           :: d                    !< Counter
    integer(IP)                           :: ndims                !< Number of dims in user defined cartesian communicator
    integer(IP)                           :: top_type               !< MPI_Topo_test flag
    integer(IP),            allocatable   :: transposed_dims(:,:)       !< Global counts in transposed coordinates
    integer(IP),            allocatable   :: temp_dims(:)         !< Temporary dims needed by MPI_Cart_get
    integer(IP),            allocatable   :: temp_coords(:)       !< Temporary coordinates needed by MPI_Cart_get
    logical,                allocatable   :: temp_periods(:)      !< Temporary periods needed by MPI_Cart_get
    TYPE_MPI_DATATYPE :: base_dtype
    integer(IP)       :: base_storage

    select case ( self%precision )
    case ( DTFFT_SINGLE )
      base_storage = sngl_storage_size
      base_dtype = sngl_type
    case ( DTFFT_DOUBLE )
      base_storage = dbl_storage_size
      base_dtype = dbl_type
    endselect

    self%storage_size = base_storage

    call self%base_alloc()
    allocate(transposed_dims(self%ndims, self%ndims))
    if(self%ndims == 2) then
      ! Nx x Ny
      transposed_dims(1, :) = dims(:)
      ! Ny x Nx
      transposed_dims(2, 1) = dims(2)
      transposed_dims(2, 2) = dims(1)
    else
      ! Nx x Ny x Nz
      transposed_dims(1, :) = dims(:)
      ! Ny x Nx x Nz
      transposed_dims(2, 1) = dims(2)
      transposed_dims(2, 2) = dims(1)
      transposed_dims(2, 3) = dims(3)
      ! Nz x Nx x Ny
      transposed_dims(3, 1) = dims(3)
      transposed_dims(3, 2) = dims(1)
      transposed_dims(3, 3) = dims(2)
    endif

    call MPI_Topo_test(self%comm, top_type, ierr)

    if(top_type == MPI_UNDEFINED) then
      ! planner_flag = DTFFT_MEASURE
      ! if(present(effort_flag)) planner_flag = effort_flag

!TODO Implement effort_flag algorithms
        self%comm_dims(:) = 0
        self%comm_dims(1) = 1
        call MPI_Dims_create(self%comm_size, self%ndims, self%comm_dims, ierr)
    else
      call MPI_Cartdim_get(self%comm, ndims, ierr)
      if(ndims > self%ndims) FATAL_ERROR("Number of cartesian dims > size of transpose")
      self%comm_dims(:) = 1
      allocate(temp_dims(ndims), temp_periods(ndims), temp_coords(ndims))
      call MPI_Cart_get(self%comm, ndims, temp_dims, temp_periods, temp_coords, ierr)
      if(ndims == self%ndims) then
        if ( temp_dims(1) /= 1 ) FATAL_ERROR("Data distribution in 'fastest' dimension is not supported")
        self%comm_dims(:) = temp_dims
      elseif(ndims == self%ndims - 1) then
        self%comm_dims(2:) = temp_dims
      elseif(ndims == self%ndims - 2) then
        self%comm_dims(3) = temp_dims(1)
      endif
      deallocate(temp_dims, temp_periods, temp_coords)
    endif

    call self%create_cart_comm(self%comm_dims, self%comm, self%comm_coords, self%comms)
    do d = 1, self%ndims
      call self%info(d)%init(self%ndims, d, transposed_dims(d,:), self%comms, self%comm_dims, self%comm_coords)
    enddo

    call self%create_transpose_plans(self%transpose_out, self%transpose_in, self%comms, base_dtype, base_storage)
    call self%alloc_fft_plans()
    deallocate(transposed_dims)

    self%is_aux_needed = .false.
    self%is_aux_alloc = .false.
#undef __FUNC__
  end procedure init_core


  module procedure check_init_args
#define __FUNC__ routine

    integer(IP) :: ierr, top_type, dim
    character(len=10) :: dimstr, dimval

    self%ndims = size(dims)
    CHECK_INPUT("Number of dimensions", self%ndims, VALID_DIMENSIONS)
    do dim = 1, self%ndims
      if ( dims(dim) <= 0 ) then
        write(dimstr, "(i0)") dim
        write(dimval, "(i0)") dims(dim)
        FATAL_ERROR("dims("//trim(dimstr)//") = "//trim(dimval)//" <= 0")
      endif
    enddo

    self%comm = MPI_COMM_WORLD
    if(present(comm)) then
      call MPI_Topo_test(comm, top_type, ierr)

      if(.not.any(top_type == [MPI_UNDEFINED, MPI_CART])) FATAL_ERROR("Wrong `comm` parameter passed...")
      call MPI_Comm_dup(comm, self%comm, ierr)
    endif
    call MPI_Comm_size(self%comm, self%comm_size, ierr)
    ! call MPI_Comm_rank(self%comm, self%comm_rank, ierr)

    self%precision = DTFFT_DOUBLE
    if (present(precision)) then
      CHECK_INPUT("precision", precision, VALID_PRECISIONS)
      self%precision = precision
    endif

    self%effort_flag = DTFFT_MEASURE
    ! if(present(effort_flag)) then
    !   CHECK_INPUT("effort_flag", effort_flag, VALID_EFFORTS)
    !   self%effort_flag = effort_flag
    ! endif

    self%is_transpose_plan = .false.
    self%executor_type = DTFFT_EXECUTOR_FFTW3
    if(present(executor_type)) then
      CHECK_INPUT("executor_type", executor_type, VALID_EXECUTORS)
      self%executor_type = executor_type
    endif
    if ( self%executor_type == DTFFT_EXECUTOR_NONE ) self%is_transpose_plan = .true.
#undef __FUNC__
  end procedure check_init_args

  module procedure create_transpose_plans
    integer(IP) :: dim  !< Counter

    do dim = 1, self%ndims - 1
      call tout(dim)%init(comms(dim + 1), self%info(dim), self%info(dim + 1), base_type, base_storage)
      call tin(dim)%init(comms(dim + 1), self%info(dim + 1), self%info(dim), base_type, base_storage)
    enddo
  end procedure create_transpose_plans

  module procedure create_cart_comm
    logical,              allocatable   :: periods(:)           !< Grid is not periodic
    logical,              allocatable   :: remain_dims(:)       !< Needed by MPI_Cart_sub
    integer(IP)                         :: d                    !< Counter
    integer(IP)                         :: comm_rank            !< Rank of current process in cartesian communcator
    integer(IP)                         :: ierr

    allocate(periods(self%ndims), source = .false.)
    call MPI_Cart_create(MPI_COMM_WORLD, self%ndims, comm_dims, periods, .true., comm, ierr)
    call MPI_Comm_rank(comm, comm_rank, ierr)
    call MPI_Cart_coords(comm, comm_rank, self%ndims, comm_coords, ierr)

    allocate(remain_dims(self%ndims), source = .false.)
    do d = 1, self%ndims
      remain_dims(d) = .true.
      call MPI_Cart_sub(comm, remain_dims, local_comms(d), ierr)
      remain_dims(d) = .false.
    enddo
    deallocate(remain_dims, periods)
  end procedure create_cart_comm

  module procedure base_alloc

    allocate(self%info(self%ndims))
    allocate(self%transpose_in(self%ndims - 1))
    allocate(self%transpose_out(self%ndims - 1))
    allocate(self%comm_dims(self%ndims))
    allocate(self%comms(self%ndims))
    allocate(self%comm_coords(self%ndims))
  end procedure base_alloc

  module procedure execute_transposed_out
    ! Transpose X -> Y
    call self%transpose_private(in, aux, DTFFT_TRANSPOSE_X_TO_Y)
    ! 1d direct FFT Y direction
    call self%forw_plans(2)%fft%execute(aux)
    ! Transpose Y -> Z
    call self%transpose_private(aux, out, DTFFT_TRANSPOSE_Y_TO_Z)
    ! 1d direct FFT Z direction
    call self%forw_plans(3)%fft%execute(out)
  end procedure execute_transposed_out

  module procedure execute_transposed_in
    ! 1d direct FFT Z direction
    call self%back_plans(3)%fft%execute(in)
    ! Transpose Z -> Y
    call self%transpose_private(in, aux, DTFFT_TRANSPOSE_Z_TO_Y)
    ! 1d inverse FFT Y direction
    call self%back_plans(2)%fft%execute(aux)
    ! Transpose Y -> X
    call self%transpose_private(aux, out, DTFFT_TRANSPOSE_Y_TO_X)
  end procedure execute_transposed_in

  module procedure alloc_fft_plans
#define __FUNC__ "alloc_fft_plans"
    ! integer(IP) :: executor         !< External FFT executor
    integer(IP) :: dim              !< Counter

    if ( self%is_transpose_plan ) return

    allocate(self%forw_plans(self%ndims))
    allocate(self%back_plans(self%ndims))

    do dim = 1, self%ndims
      select case(self%executor_type)
      case (DTFFT_EXECUTOR_FFTW3)
#if !defined(NO_FFTW3)
        if ( dim == 1 ) DEBUG("Using FFTW3 executor")
        allocate(fftw_executor :: self%forw_plans(dim)%fft)
        allocate(fftw_executor :: self%back_plans(dim)%fft)
#else
        FATAL_ERROR("FFTW3 is disabled in this build")
#endif
      case (DTFFT_EXECUTOR_MKL)
#if defined(MKL_ENABLED)
        if ( dim == 1 ) DEBUG("Using MKL executor")
        allocate(mkl_executor :: self%forw_plans(dim)%fft)
        allocate(mkl_executor :: self%back_plans(dim)%fft)
#else
        FATAL_ERROR("MKL is disabled in this build")
#endif
      case (DTFFT_EXECUTOR_CUFFT)
! #if defined(CUFFT_ENABLED)
!         if ( dim == 1 ) DEBUG("Using CUFFT executor")
!         ! allocate(cuff :: self%forw_plans(dim)%fft)
!         ! allocate(mkl_executor :: self%back_plans(dim)%fft)
! #else
        FATAL_ERROR("CUFFT is disabled in this build")
! #endif
      case (DTFFT_EXECUTOR_KFR)
#if defined(KFR_ENABLED)
        if ( dim == 1 ) DEBUG("Using KFR executor")
        allocate(kfr_executor :: self%forw_plans(dim)%fft)
        allocate(kfr_executor :: self%back_plans(dim)%fft)
#else
        FATAL_ERROR("KFR is disabled in this build")
#endif
      case default
        FATAL_ERROR("Unrecognized `executor` provided")
      endselect
    enddo
#undef __FUNC__
  end procedure alloc_fft_plans

  module procedure check_aux
#define __FUNC__ "check_aux"
    integer(SP) :: alloc_size
    character(len=100) :: debug_msg

    if(.not. present(aux)) then
      if(.not. self%is_aux_alloc .and. self%is_aux_needed) then
        alloc_size = self%get_aux_size() * self%storage_size / FLOAT_STORAGE_SIZE
        write(debug_msg, '(a, i0, a)') "Allocating auxiliary buffer of ",alloc_size * FLOAT_STORAGE_SIZE, " bytes";    DEBUG(debug_msg)
        allocate( self%aux(alloc_size) )
        self%is_aux_alloc = .true.
      endif
    endif
#undef __FUNC__
  end procedure check_aux

  module subroutine create_c2c_fft(self, start, precision)
    class(dtfft_core),    intent(inout) :: self             !< Core
    integer(IP),          intent(in)    :: start            !< Number of plans to create
    integer(IP),          intent(in)    :: precision
    integer(IP)                         :: d

    do d = start, self%ndims
      call self%forw_plans(d)%fft%create(FFT_C2C, precision, complex_info=self%info(d), sign_or_kind=DTFFT_FORWARD)
      call self%back_plans(d)%fft%create(FFT_C2C, precision, complex_info=self%info(d), sign_or_kind=DTFFT_BACKWARD)
    enddo
  end subroutine create_c2c_fft

  module subroutine DTFFT_FATAL_ERROR(msg, fun)
    character(len=*), intent(in)  :: msg
    character(len=*), intent(in)  :: fun

    write(error_unit, '(a)') 'DTFFT Error -- "' // fun // '": ' // msg
    error stop
  end subroutine DTFFT_FATAL_ERROR

  module procedure DTFFT_DEBUG
#ifdef __DEBUG
    integer(IP) :: comm_rank, ierr

    call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
    if ( comm_rank == 0 ) write(output_unit, '(a)') 'DTFFT Debug -- '//fun//': '//trim(msg)
#endif
  end procedure DTFFT_DEBUG

  ! module subroutine test_plan(self, comm_dims)
  !   class(dtfft_core),    intent(inout) :: self             !< Core
  !   integer(IP),  intent(in)  :: comm_dims(:)
  !   real(R4P), allocatable :: in(:), out(:)
  !   type(transpose_t), allocatable :: tout(:), tin(:)
  !   type(info_t), allocatable :: info(:)
  !   integer(IP) :: alloc_size
  !   integer(IP) :: i
  !   real(R8P) :: time

  !   alloc_size = 0
  !   do i = 1, self%ndims
  !     alloc_size = max(alloc_size, product(info(i)%counts) * self%storage_size / FLOAT_STORAGE_SIZE)
  !   enddo
  !   allocate(in(alloc_size), out(alloc_size))

  !   time = -MPI_Wtime()
  !   do i = 1, self%ndims - 1
  !     call tout(i)%transpose(in, out)
  !     call tin(i)%transpose(out, in)
  !   enddo
  !   time = time + MPI_Wtime()

  ! end subroutine test_plan
end submodule dtfft_core_sm