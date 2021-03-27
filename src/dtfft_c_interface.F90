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
module dtfft_c_interface
!------------------------------------------------------------------------------------------------
!< This module is a Fortran part of C interface
!------------------------------------------------------------------------------------------------
use iso_c_binding
use dtfft_precisions
use dtfft_plan_c2c_m
use dtfft_plan_r2r_m
use dtfft_plan_r2c_m
use mpi_f08, only: MPI_Comm
implicit none

contains

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2r_3d_c(plan_ptr, comm, nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in)              :: in_kinds(*)      !< FFT R2R kinds for DTFFT_TRANSPOSE_OUT transform
    integer(IP),  intent(in)              :: out_kinds(*)     !< FFT R2R kinds for DTFFT_TRANSPOSE_IN transform
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2r_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_r2r_3d_c(plan_ptr, comm, nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional r2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in)              :: in_kinds(*)      !< FFT R2R kinds for DTFFT_TRANSPOSE_OUT transform
    integer(IP),  intent(in)              :: out_kinds(*)     !< FFT R2R kinds for DTFFT_TRANSPOSE_IN transform
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2r_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2r_2d_c(plan_ptr, comm, ny, nx, in_kinds, out_kinds, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in)              :: in_kinds(*)      !< FFT R2R kinds for DTFFT_TRANSPOSE_OUT transform
    integer(IP),  intent(in)              :: out_kinds(*)     !< FFT R2R kinds for DTFFT_TRANSPOSE_IN transform
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2r_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, ny, nx, in_kinds, out_kinds, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_r2r_2d_c(plan_ptr, comm, ny, nx, in_kinds, out_kinds, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional r2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in)              :: in_kinds(*)      !< FFT R2R kinds for DTFFT_TRANSPOSE_OUT transform
    integer(IP),  intent(in)              :: out_kinds(*)     !< FFT R2R kinds for DTFFT_TRANSPOSE_IN transform
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2r_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, ny, nx, in_kinds, out_kinds, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_c2c_3d_c(plan_ptr, comm, nz, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_c2c_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, nz, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_c2c_3d_c(plan_ptr, comm, nz, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional c2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_c2c_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, nz, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_c2c_2d_c(plan_ptr, comm, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_c2c_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_c2c_2d_c(plan_ptr, comm, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional c2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_c2c_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2c_2d_c(plan_ptr, comm, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional r2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2c_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_r2c_2d_c(plan_ptr, comm, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for two-dimensional r2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2c_2d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_r2c_3d_c(plan_ptr, comm, nz, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional r2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2c_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create(fcomm, nz, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_create_plan_f_r2c_3d_c(plan_ptr, comm, nz, ny, nx, effort_flag, executor_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of creating plan for three-dimensional r2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),  intent(out)             :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  intent(in),   value     :: comm             !< Communicator
    integer(IP),  intent(in),   value     :: nz               !< Number of points in Z direction
    integer(IP),  intent(in),   value     :: ny               !< Number of points in Y direction
    integer(IP),  intent(in),   value     :: nx               !< Number of points in X direction
    integer(IP),  intent(in),   value     :: effort_flag      !< DTFFT planner effort flag
    integer(IP),  intent(in),   value     :: executor_type    !< Type of External FFT Executor
    type(dtfft_plan_r2c_3d),    pointer   :: plan             !< Fortran plan class
    type(MPI_Comm)                        :: fcomm            !< Fortran communicator

    allocate(plan)
    fcomm%MPI_VAL = comm
    call plan%create_f(fcomm, nz, ny, nx, effort_flag, executor_type)
    plan_ptr = c_loc(plan)
  end subroutine dtfft_create_plan_f_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_r2r_3d_c(plan_ptr, in, out, transpose_type, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2r_3d),    pointer :: fplan              !< Fortran plan class
    real(R8P),                  pointer :: fin(:)             !< Fortran incoming buffer
    real(R8P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    real(R8P),                  pointer :: fwork(:)           !< Fortran working buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute(fin, fout, transpose_type, fwork)
    else
      call fplan%execute(fin, fout, transpose_type)
    endif
  end subroutine dtfft_execute_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_r2r_3d_c(plan_ptr, in, out, transpose_type, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional r2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2r_3d),    pointer :: fplan              !< Fortran plan class
    real(R4P),                  pointer :: fin(:)             !< Fortran incoming buffer
    real(R4P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    real(R4P),                  pointer :: fwork(:)           !< Fortran working buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f(fin, fout, transpose_type, fwork)
    else
      call fplan%execute_f(fin, fout, transpose_type)
    endif
  end subroutine dtfft_execute_f_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_r2r_2d_c(plan_ptr, in, out, transpose_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional r2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN
    type(dtfft_plan_r2r_2d),    pointer :: fplan              !< Fortran plan class
    real(R8P),                  pointer :: fin(:)             !< Fortran incoming buffer
    real(R8P),                  pointer :: fout(:)            !< Fortran outgoing buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    call fplan%execute(fin, fout, transpose_type)
  end subroutine dtfft_execute_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_r2r_2d_c(plan_ptr, in, out, transpose_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional r2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN
    type(dtfft_plan_r2r_2d),    pointer :: fplan              !< Fortran plan class
    real(R4P),                  pointer :: fin(:)             !< Fortran incoming buffer
    real(R4P),                  pointer :: fout(:)            !< Fortran outgoing buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    call fplan%execute_f(fin, fout, transpose_type)
  end subroutine dtfft_execute_f_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_c2c_3d_c(plan_ptr, in, out, transpose_type, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN    
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_c2c_3d),    pointer :: fplan              !< Fortran plan class
    complex(C8P),               pointer :: fin(:)             !< Fortran incoming buffer
    complex(C8P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C8P),               pointer :: fwork(:)           !< Fortran working buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute(fin, fout, transpose_type, fwork)
    else
      call fplan%execute(fin, fout, transpose_type)
    endif
  end subroutine dtfft_execute_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_c2c_3d_c(plan_ptr, in, out, transpose_type, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional c2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN    
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_c2c_3d),    pointer :: fplan              !< Fortran plan class
    complex(C4P),               pointer :: fin(:)             !< Fortran incoming buffer
    complex(C4P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C4P),               pointer :: fwork(:)           !< Fortran working buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f(fin, fout, transpose_type, fwork)
    else
      call fplan%execute_f(fin, fout, transpose_type)
    endif
  end subroutine dtfft_execute_f_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_c2c_2d_c(plan_ptr, in, out, transpose_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional c2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN    
    type(dtfft_plan_c2c_2d),    pointer :: fplan              !< Fortran plan class
    complex(C8P),               pointer :: fin(:)             !< Fortran incoming buffer
    complex(C8P),               pointer :: fout(:)            !< Fortran outgoing buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    call fplan%execute(fin, fout, transpose_type)
  end subroutine dtfft_execute_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_c2c_2d_c(plan_ptr, in, out, transpose_type) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional c2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    integer(IP),  intent(in),   value   :: transpose_type     !< Type of transpose: DTFFT_TRANSPOSE_OUT or DTFFT_TRANSPOSE_IN    
    type(dtfft_plan_c2c_2d),    pointer :: fplan              !< Fortran plan class
    complex(C4P),               pointer :: fin(:)             !< Fortran incoming buffer
    complex(C4P),               pointer :: fout(:)            !< Fortran outgoing buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    call fplan%execute_f(fin, fout, transpose_type)
  end subroutine dtfft_execute_f_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_r2c_2d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional r2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class
    real(R8P),                  pointer :: fin(:)             !< Fortran incoming buffer
    complex(C8P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C8P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_r2c(fin, fout, fwork)
    else
      call fplan%execute_r2c(fin, fout)
    endif
  end subroutine dtfft_execute_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_r2c_2d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional r2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class
    real(R4P),                  pointer :: fin(:)             !< Fortran incoming buffer
    complex(C4P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C4P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f_r2c(fin, fout, fwork)
    else
      call fplan%execute_f_r2c(fin, fout)
    endif
  end subroutine dtfft_execute_f_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_c2r_2d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional c2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class
    complex(C8P),               pointer :: fin(:)             !< Fortran incoming buffer
    real(C8P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C8P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_c2r(fin, fout, fwork)
    else
      call fplan%execute_c2r(fin, fout)
    endif
  end subroutine dtfft_execute_c2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_c2r_2d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for two-dimensional c2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class
    complex(C4P),               pointer :: fin(:)             !< Fortran incoming buffer
    real(C4P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C4P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f_c2r(fin, fout, fwork)
    else
      call fplan%execute_f_c2r(fin, fout)
    endif
  end subroutine dtfft_execute_f_c2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_r2c_3d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional r2c transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_3d),    pointer :: fplan              !< Fortran plan class
    real(R8P),                  pointer :: fin(:)             !< Fortran incoming buffer
    complex(C8P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C8P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_r2c(fin, fout, fwork)
    else
      call fplan%execute_r2c(fin, fout)
    endif
  end subroutine dtfft_execute_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_r2c_3d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional r2c transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_3d),    pointer :: fplan              !< Fortran plan class
    real(R4P),                  pointer :: fin(:)             !< Fortran incoming buffer
    complex(C4P),               pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C4P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f_r2c(fin, fout, fwork)
    else
      call fplan%execute_f_r2c(fin, fout)
    endif
  end subroutine dtfft_execute_f_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_c2r_3d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional c2r transform, double precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_3d),    pointer :: fplan              !< Fortran plan class
    complex(C8P),               pointer :: fin(:)             !< Fortran incoming buffer
    real(C8P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C8P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_c2r(fin, fout, fwork)
    else
      call fplan%execute_c2r(fin, fout)
    endif
  end subroutine dtfft_execute_c2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_execute_f_c2r_3d_c(plan_ptr, in, out, work) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of executing plan for three-dimensional c2r transform, single precision
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(C_PTR),                value   :: in                 !< C incoming pointer
    type(C_PTR),                value   :: out                !< C outgoing pointer
    type(C_PTR),                value   :: work               !< C working pointer
    type(dtfft_plan_r2c_3d),    pointer :: fplan              !< Fortran plan class
    complex(C4P),               pointer :: fin(:)             !< Fortran incoming buffer
    real(C4P),                  pointer :: fout(:)            !< Fortran outgoing buffer
    complex(C4P),               pointer :: fwork(:)           !< Fortran work buffer

    call c_f_pointer(plan_ptr, fplan)
    call c_f_pointer(in, fin, [1])
    call c_f_pointer(out, fout, [1])

    if(c_associated(work)) then 
      call c_f_pointer(work, fwork, [1])
      call fplan%execute_f_c2r(fin, fout, fwork)
    else
      call fplan%execute_f_c2r(fin, fout)
    endif
  end subroutine dtfft_execute_f_c2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_r2r_3d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for three-dimensional r2r transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_r2r_3d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_r2r_2d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for two-dimensional r2r transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_r2r_2d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_c2c_3d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for three-dimensional c2c transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_c2c_3d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_c2c_2d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for two-dimensional c2c transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_c2c_2d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_r2c_2d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for two-dimensional r2c transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_destroy_r2c_3d_c(plan_ptr) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface of destroying plan for three-dimensional r2c transform
!------------------------------------------------------------------------------------------------
    type(C_PTR),                value   :: plan_ptr           !< C pointer to Fortran plan
    type(dtfft_plan_r2c_2d),    pointer :: fplan              !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%destroy()
  end subroutine dtfft_destroy_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_r2r_3d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, r2r 3d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(3)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(3)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(3)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(3)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_r2r_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_r2r_2d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, r2r 2d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(2)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(2)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(2)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(2)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_r2r_2d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_r2r_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_c2c_3d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, c2c 3d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(3)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(3)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(3)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(3)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_c2c_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_c2c_2d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, c2c 2d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(2)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(2)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(2)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(2)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_c2c_2d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_c2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_r2c_3d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, r2c 3d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(3)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(3)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(3)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(3)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_r2c_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_local_sizes_r2c_2d_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts in real and Fourier spaces, r2c 2d
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: in_starts(2)     !< Starts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: in_counts(2)     !< Counts of local portion of data in 'real' space
    integer(IP),  optional,   intent(out) :: out_starts(2)    !< Starts of local portion of data in 'fourier' space
    integer(IP),  optional,   intent(out) :: out_counts(2)    !< Counts of local portion of data in 'fourier' space
    integer(IP),              intent(out) :: alloc_size       !< Maximum data needs to be allocated in case of in-place transform
    type(dtfft_plan_r2c_2d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size)
  end subroutine dtfft_get_local_sizes_r2c_2d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_worker_size_r2r_3d_c(plan_ptr, starts, counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts of optional work buffer, r2r
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: starts(3)        !< Starts of local portion of data of worker buffer
    integer(IP),  optional,   intent(out) :: counts(3)        !< Counts of local portion of data of worker buffer
    integer(IP),              intent(out) :: alloc_size       !< Number of elements to be allocated
    type(dtfft_plan_r2r_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_worker_size(starts, counts, alloc_size)
  end subroutine dtfft_get_worker_size_r2r_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_worker_size_c2c_3d_c(plan_ptr, starts, counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts of optional work buffer, r2r
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: starts(3)        !< Starts of local portion of data of worker buffer
    integer(IP),  optional,   intent(out) :: counts(3)        !< Counts of local portion of data of worker buffer
    integer(IP),              intent(out) :: alloc_size       !< Number of elements to be allocated
    type(dtfft_plan_c2c_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_worker_size(starts, counts, alloc_size)
  end subroutine dtfft_get_worker_size_c2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_worker_size_r2c_3d_c(plan_ptr, starts, counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts of optional work buffer, r2c
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: starts(3)        !< Starts of local portion of data of worker buffer
    integer(IP),  optional,   intent(out) :: counts(3)        !< Counts of local portion of data of worker buffer
    integer(IP),              intent(out) :: alloc_size       !< Number of elements to be allocated
    type(dtfft_plan_r2c_3d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_worker_size(starts, counts, alloc_size)
  end subroutine dtfft_get_worker_size_r2c_3d_c

!------------------------------------------------------------------------------------------------
  subroutine dtfft_get_worker_size_r2c_2d_c(plan_ptr, starts, counts, alloc_size) bind(C)
!------------------------------------------------------------------------------------------------
!< C interface to function that computes local sizes and counts of optional work buffer, r2c
!------------------------------------------------------------------------------------------------
    type(C_PTR),              value       :: plan_ptr         !< C pointer to Fortran plan
    integer(IP),  optional,   intent(out) :: starts(2)        !< Starts of local portion of data of worker buffer
    integer(IP),  optional,   intent(out) :: counts(2)        !< Counts of local portion of data of worker buffer
    integer(IP),              intent(out) :: alloc_size       !< Number of elements to be allocated
    type(dtfft_plan_r2c_2d),  pointer     :: fplan            !< Fortran plan class

    call c_f_pointer(plan_ptr, fplan)
    call fplan%get_worker_size(starts, counts, alloc_size)
  end subroutine dtfft_get_worker_size_r2c_2d_c
end module dtfft_c_interface