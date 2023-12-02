submodule (dtfft_core_m) dtfft_r2r_sm
#include "dtfft.i90"
#include "dtfft_core.i90"
implicit none

contains

  module procedure create_r2r
#define __FUNC__ "dtfft_plan_r2r.create"

    call self%check_init_args(__FUNC__, dims, comm, precision, effort_flag, executor_type)
    call self%init(dims, MPI_REAL, FLOAT_STORAGE_SIZE, MPI_REAL8, DOUBLE_STORAGE_SIZE)

    if(.not. self%is_transpose_plan) then
      if ( .not. present( in_kinds ) .or. .not. present( out_kinds ) ) FATAL_ERROR("Both `in_kinds` and `out_kinds` must be passed for a non-transpose plan")
      block
        integer(IP)         :: dim                !< Counter
        character(len=100)  :: param_name

        do dim = 1, self%ndims
          write(param_name, '(a, i0, a)') "in_kinds[", dim, "]";    CHECK_INPUT(param_name, in_kinds(dim), VALID_R2R_FFTS)
          call self%forw_plans(dim)%fft%create(FFT_R2R, self%precision, real_info=self%info(dim), sign_or_kind=in_kinds(dim))

          write(param_name, '(a, i0, a)') "out_kinds[", dim, "]";   CHECK_INPUT(param_name, out_kinds(dim), VALID_R2R_FFTS)
          call self%back_plans(dim)%fft%create(FFT_R2R, self%precision, real_info=self%info(dim), sign_or_kind=out_kinds(dim))
        enddo

      endblock

      if(self%ndims == 3) then
        self%aux_pencil = Y_PENCIL
        self%is_aux_needed = .true.
      endif
    endif
    self%is_created = .true.
#undef __FUNC__
  end procedure create_r2r
end submodule dtfft_r2r_sm