submodule (dtfft_core_m) dtfft_r2c_sm
#include "dtfft.i90"
#include "dtfft_core.i90"
implicit none

contains

  module procedure create_r2c
#define __FUNC__ "create_r2c"
    integer(IP),  allocatable   :: fixed_dims(:)

    allocate(fixed_dims, source=dims)
    fixed_dims(1) = int(dims(1) / 2, IP) + 1
    call self%create_c2c_internal(__FUNC__, 2, fixed_dims, comm, precision, effort_flag, executor_type)
    if ( self%is_transpose_plan ) FATAL_ERROR("Transpose plan is not supported in R2C, use C2C plan instead")

    call self%real_info%init(self%ndims, 1, dims, self%comms, self%comm_dims, self%comm_coords)
    call self%forw_plans(1)%fft%create(FFT_R2C, self%precision, real_info=self%real_info, complex_info=self%info(1))
    call self%back_plans(1)%fft%create(FFT_C2R, self%precision, real_info=self%real_info, complex_info=self%info(1))

    self%is_aux_needed = .true.
    if(self%ndims == 2) then
      self%aux_pencil = X_PENCIL
    else
      self%aux_pencil = Y_PENCIL
    endif
    deallocate(fixed_dims)
    self%is_created = .true.
#undef __FUNC__
  end procedure create_r2c

  module procedure destroy_r2c
    call self%real_info%destroy()
    call self%destroy_internal()
  end procedure destroy_r2c

  module procedure execute_3d_r2c
    select case (transpose_type)
    case (DTFFT_TRANSPOSE_OUT)
      ! 1d r2c FFT
      call self%forw_plans(1)%fft%execute(in, out)
      call self%execute_transposed_out(out, out, aux)
    case (DTFFT_TRANSPOSE_IN)
      call self%execute_transposed_in(in, in, aux)
      ! 1d c2r FFT
      call self%back_plans(1)%fft%execute(in, out)
    endselect
  end procedure execute_3d_r2c

  module procedure execute_2d_r2c
    select case (transpose_type)
    case (DTFFT_TRANSPOSE_OUT)
      ! 1d r2c FFT
      call self%forw_plans(1)%fft%execute(in, aux)
      ! Transpose X -> Y
      call self%transpose_private(aux, out, DTFFT_TRANSPOSE_X_TO_Y)
      ! 1D Forward c2c FFT
      call self%forw_plans(2)%fft%execute(out)
    case (DTFFT_TRANSPOSE_IN)
      ! 1D Backward c2c FFT
      call self%back_plans(2)%fft%execute(in)
      ! Transpose Y -> X
      call self%transpose_private(in, aux, DTFFT_TRANSPOSE_Y_TO_X)
      ! 1d c2r FFT
      call self%back_plans(1)%fft%execute(aux, out)
    endselect
  end procedure execute_2d_r2c

  module procedure get_local_sizes_r2c
#define __FUNC__ "get_local_sizes_r2c"
    CHECK_PLAN
    if(present(in_starts))    in_starts(1:self%ndims)   = self%real_info%starts
    if(present(in_counts))    in_counts(1:self%ndims)   = self%real_info%counts
    call self%get_local_sizes_internal(out_starts=out_starts, out_counts=out_counts, alloc_size=alloc_size)
    if(present(alloc_size))   then
      alloc_size = max(int(product(self%real_info%counts), SP), 2 * alloc_size)
    endif
#undef __FUNC__
  end procedure get_local_sizes_r2c
end submodule dtfft_r2c_sm