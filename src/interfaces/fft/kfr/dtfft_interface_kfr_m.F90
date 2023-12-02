module dtfft_interface_kfr_m
#if defined(_BUILD_DOCS)
#define KFR_ENABLED
#endif
#if defined(MKL_ENABLED)
use dtfft_precisions
use iso_c_binding
implicit none
private
public :: KFR_PACK_PERM, KFR_PACK_CCS,      &
          kfr_dft_create_plan_f32,          &
          kfr_dft_create_plan_f64,          &
          kfr_dft_get_temp_size_f32,        &
          kfr_dft_get_temp_size_f64,        &
          kfr_dft_real_create_plan_f32,     &
          kfr_dft_real_create_plan_f64,     &
          kfr_dft_real_get_temp_size_f32,   &
          kfr_dft_real_get_temp_size_f64,   &
          kfr_execute_c2c_32,               &
          kfr_execute_c2c_64,               &
          kfr_execute_r2c_32,               &
          kfr_execute_r2c_64

  integer(IP), parameter :: KFR_PACK_PERM = 0
  integer(IP), parameter :: KFR_PACK_CCS = 1

  interface
    type(c_ptr) function kfr_dft_create_plan_f32(size) bind(C, name="kfr_dft_create_plan_f32")
    import :: SP, c_ptr
      integer(SP), value :: size
    end function kfr_dft_create_plan_f32

    type(c_ptr) function kfr_dft_create_plan_f64(size) bind(C, name="kfr_dft_create_plan_f64")
    import :: SP, c_ptr
      integer(SP), value :: size
    end function kfr_dft_create_plan_f64

    integer(SP) function kfr_dft_get_temp_size_f32(plan) bind(C, name="kfr_dft_get_temp_size_f32")
    import :: SP, c_ptr
      type(c_ptr), value :: plan
    end function kfr_dft_get_temp_size_f32

    integer(SP) function kfr_dft_get_temp_size_f64(plan) bind(C, name="kfr_dft_get_temp_size_f64")
    import :: SP, c_ptr
      type(c_ptr), value :: plan
    end function kfr_dft_get_temp_size_f64

    type(c_ptr) function kfr_dft_real_create_plan_f32(size, pack_format) bind(C, name="kfr_dft_real_create_plan_f32")
    import :: SP, IP, c_ptr
      integer(SP),  value :: size
      integer(IP),        value :: pack_format
    end function kfr_dft_real_create_plan_f32

    type(c_ptr) function kfr_dft_real_create_plan_f64(size, pack_format) bind(C, name="kfr_dft_real_create_plan_f64")
    import :: SP, IP, c_ptr
      integer(SP),  value :: size
      integer(IP),        value :: pack_format
    end function kfr_dft_real_create_plan_f64

    integer(SP) function kfr_dft_real_get_temp_size_f32(plan) bind(C, name="kfr_dft_real_get_temp_size_f32")
    import :: SP, c_ptr
      type(c_ptr), value :: plan
    end function kfr_dft_real_get_temp_size_f32

    integer(SP) function kfr_dft_real_get_temp_size_f64(plan) bind(C, name="kfr_dft_real_get_temp_size_f64")
    import :: SP, IP, c_ptr
      type(c_ptr), value :: plan
    end function kfr_dft_real_get_temp_size_f64


    subroutine kfr_execute_c2c_32(plan, in, out, temp, sign, how_many, stride) bind(C)
    import :: SP, IP, c_ptr
      type(c_ptr),  value :: plan
      type(c_ptr),  value :: in
      type(c_ptr),  value :: out
      type(c_ptr),  value :: temp
      integer(IP),  value :: sign
      integer(IP),  value :: how_many
      integer(IP),  value :: stride
    endsubroutine kfr_execute_c2c_32

    subroutine kfr_execute_c2c_64(plan, in, out, temp, sign, how_many, stride) bind(C)
    import :: SP, IP, c_ptr
      type(c_ptr),  value :: plan
      type(c_ptr),  value :: in
      type(c_ptr),  value :: out
      type(c_ptr),  value :: temp
      integer(IP),  value :: sign
      integer(IP),  value :: how_many
      integer(IP),  value :: stride
    endsubroutine kfr_execute_c2c_64

    subroutine kfr_execute_r2c_32(plan, in, out, temp, sign, how_many, stride) bind(C)
    import :: SP, IP, c_ptr
      type(c_ptr),  value :: plan
      type(c_ptr),  value :: in
      type(c_ptr),  value :: out
      type(c_ptr),  value :: temp
      integer(IP),  value :: sign
      integer(IP),  value :: how_many
      integer(IP),  value :: stride
    endsubroutine kfr_execute_r2c_32

    subroutine kfr_execute_r2c_64(plan, in, out, temp, sign, how_many, stride) bind(C)
    import :: SP, IP, c_ptr
      type(c_ptr),  value :: plan
      type(c_ptr),  value :: in
      type(c_ptr),  value :: out
      type(c_ptr),  value :: temp
      integer(IP),  value :: sign
      integer(IP),  value :: how_many
      integer(IP),  value :: stride
    endsubroutine kfr_execute_r2c_64
  endinterface
end module dtfft_interface_kfr_m