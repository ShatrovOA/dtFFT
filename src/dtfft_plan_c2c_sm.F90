submodule (dtfft_core_m) dtfft_c2c_sm
#include "dtfft.i90"
implicit none

contains

  module procedure create_c2c
    call self%create_c2c_internal("dtfft_plan_c2c.create", 1, dims, comm, precision, effort_flag, executor_type)

    ! call self%check_init_args("dtfft_plan_c2c.create", dims, comm, precision, effort_flag, executor_type)
    ! call self%init(dims, MPI_COMPLEX, COMPLEX_STORAGE_SIZE, MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE)
    if(.not. self%is_transpose_plan .and. self%ndims == 3) then
      ! call self%create_c2c_fft(1, self%precision)
      ! if(self%ndims == 3) then
        self%aux_pencil = Y_PENCIL
        self%is_aux_needed = .true.
      ! endif
    endif
    self%is_created = .true.
  end procedure create_c2c

  module procedure create_c2c_internal
    call self%check_init_args(fun, dims, comm, precision, effort_flag, executor_type)
    call self%init(dims, MPI_COMPLEX, COMPLEX_STORAGE_SIZE, MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE)
    if(.not. self%is_transpose_plan) then
      call self%create_c2c_fft(fft_start, self%precision)
    endif
  end procedure create_c2c_internal
end submodule dtfft_c2c_sm