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
module dtfft_executor_fftw_m
!! This module describes FFTW3 based FFT Executor: [[fftw_executor]]
!!
!! http://www.fftw.org
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env
#ifdef DTFFT_WITH_OPENMP
use omp_lib
#endif
use dtfft_abstract_executor
use dtfft_parameters
use dtfft_errors
use dtfft_interface_fftw
use dtfft_utils,                only: get_inverse_kind, is_null_ptr
implicit none
private
public :: fftw_executor


    abstract interface
        subroutine apply_interface(plan, in, out) bind(C)
        !! Executes FFTW3 Plan
#include "fftw_args_execute.i90"
        end subroutine apply_interface

        subroutine free_interface(plan) bind(C)
        !! Destroys FFTW3 Plan
        import
            type(c_ptr), value :: plan
        end subroutine free_interface
    end interface

    type, extends(abstract_executor) :: fftw_executor
    !! FFTW3 FFT Executor
    private
        procedure(apply_interface), nopass, pointer :: apply => NULL()          !! Pointer to FFTW3 function that executes FFT plan
        procedure(free_interface),  nopass, pointer :: free => NULL()           !! Pointer to FFTW3 function that destroys FFT plan
        procedure(apply_interface), nopass, pointer :: apply_inverse => NULL()  !! Pointer to FFTW3 function that executes inverse FFT plan
                                                                                !! Used in R2C only
    contains
        procedure :: create_private => create       !! Creates FFT plan via FFTW3 Interface
        procedure :: execute_private => execute     !! Executes FFTW3 plan
        procedure :: destroy_private => destroy     !! Destroys FFTW3 plan
        procedure, nopass :: mem_alloc              !! Allocates FFTW3 memory
        procedure, nopass :: mem_free               !! Frees FFTW3 aligned memory
    end type fftw_executor

#ifdef DTFFT_WITH_OPENMP
    logical, save :: CREATED_WITH_THREADS = .false.
#endif

    integer(int32), parameter :: FFTW3_FLAGS = FFTW_MEASURE + FFTW_DESTROY_INPUT
    !! FFTW3 planner flags

contains
    subroutine create(self, fft_rank, fft_type, precision, idist, odist, how_many, fft_sizes, inembed, onembed, error_code, r2r_kinds)
    !! Creates FFT plan via FFTW3 Interface
        class(fftw_executor),             intent(inout) :: self           !! FFTW FFT Executor
        type(fft_rank_t),                 intent(in)    :: fft_rank       !! Rank of fft: 1 or 2
        type(fft_type_t),                 intent(in)    :: fft_type       !! Type of fft: r2r, r2c, c2c
        type(dtfft_precision_t),          intent(in)    :: precision      !! Precision of fft: DTFFT_SINGLE or DTFFT_DOUBLE
        integer(int32),                   intent(in)    :: idist          !! Distance between the first element of two consecutive signals in a batch of the input data.
        integer(int32),                   intent(in)    :: odist          !! Distance between the first element of two consecutive signals in a batch of the output data.
        integer(int32),                   intent(in)    :: how_many       !! Number of transforms to create
        integer(int32),                   intent(in)    :: fft_sizes(:)   !! Dimensions of transform
        integer(int32),                   intent(in)    :: inembed(:)     !! Storage dimensions of the input data in memory.
        integer(int32),                   intent(in)    :: onembed(:)     !! Storage dimensions of the output data in memory.
        integer(int32),                   intent(inout) :: error_code     !! Error code to be returned to user
        type(dtfft_r2r_kind_t), optional, intent(in)    :: r2r_kinds(:)   !! Kinds of r2r transform
        integer(int32)  :: n_elements     !! Number of elements in `buf`
        type(c_ptr)     :: ptr            !! C pointer to `buf`
        integer(c_int) :: fft_rnk

#ifdef DTFFT_WITH_OPENMP
        if ( .not. CREATED_WITH_THREADS ) then
            error_code = fftw_init_threads()
            if ( error_code == 0 ) then
                ! It has been called by user
                ! Check if number of threads is configured
                block
                    integer(c_int) :: n_threads
# ifdef FFTW_HAS_PLANNER_NTHREADS
                    n_threads = fftw_planner_nthreads()
#else
                    n_threads = 1_c_int
#endif
                    if ( n_threads == 1 ) then ! 1 is default
                        call fftw_plan_with_nthreads(omp_get_max_threads())
                    endif
                endblock
            else
                call fftw_plan_with_nthreads(omp_get_max_threads())
            endif
            CREATED_WITH_THREADS = .true.
        endif
#endif
        error_code = DTFFT_SUCCESS

        n_elements = product(fft_sizes) * how_many
        if ( fft_type == FFT_C2C .or. fft_type == FFT_R2C ) then
            n_elements = n_elements * 2
        endif
        if ( precision == DTFFT_DOUBLE ) then
            n_elements = n_elements * 2
        endif

        call self%mem_alloc(FLOAT_STORAGE_SIZE * n_elements, ptr)

        if ( precision == DTFFT_SINGLE ) then
            self%free => fftwf_destroy_plan
        else
            self%free => fftw_destroy_plan
        endif

        fft_rnk = int(fft_rank%val, c_int)

        select case (fft_type%val)
        case (FFT_C2C%val)
            if ( precision == DTFFT_SINGLE ) then
                self%plan_forward = fftwf_plan_many_dft(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_FORWARD, int32), FFTW3_FLAGS)
                self%plan_backward = fftwf_plan_many_dft(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_BACKWARD, int32), FFTW3_FLAGS)
                self%apply => fftwf_execute_dft
            else
                self%plan_forward = fftw_plan_many_dft(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_FORWARD, int32), FFTW3_FLAGS)
                self%plan_backward = fftw_plan_many_dft(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, int(FFT_BACKWARD, int32), FFTW3_FLAGS)
                self%apply => fftw_execute_dft
            endif
            self%apply_inverse => NULL()
        case (FFT_R2C%val)
            block
                type(c_ptr) :: r_ptr

                call self%mem_alloc(FLOAT_STORAGE_SIZE * n_elements / 2, r_ptr)
                if ( precision == DTFFT_SINGLE ) then
                    self%plan_forward = fftwf_plan_many_dft_r2c(fft_rnk, fft_sizes, how_many, r_ptr, inembed, 1, idist, ptr, onembed, 1, odist, FFTW3_FLAGS)
                    self%plan_backward = fftwf_plan_many_dft_c2r(fft_rnk, fft_sizes, how_many, ptr, onembed, 1, odist, r_ptr, inembed, 1, idist, FFTW3_FLAGS)
                    self%apply => fftwf_execute_dft_r2c
                    self%apply_inverse => fftwf_execute_dft_c2r
                else
                    self%plan_forward = fftw_plan_many_dft_r2c(fft_rnk, fft_sizes, how_many, r_ptr, inembed, 1, idist, ptr, onembed, 1, odist, FFTW3_FLAGS)
                    self%plan_backward = fftw_plan_many_dft_c2r(fft_rnk, fft_sizes, how_many, ptr, onembed, 1, odist, r_ptr, inembed, 1, idist, FFTW3_FLAGS)
                    self%apply => fftw_execute_dft_r2c
                    self%apply_inverse => fftw_execute_dft_c2r
                endif

                call self%mem_free(r_ptr)
            endblock
        case (FFT_R2R%val)
            block
                type(dtfft_r2r_kind_t),   allocatable :: inverse_kinds(:)
                integer(C_FFTW_R2R_KIND), allocatable :: knds(:), iknds(:)
                logical :: need_inverse

                allocate( inverse_kinds(fft_rnk), knds(fft_rnk), iknds(fft_rnk) )
                knds(:) = int(r2r_kinds(:)%val, C_FFTW_R2R_KIND)
                inverse_kinds(:) = get_inverse_kind(r2r_kinds)
                need_inverse = .not. all( inverse_kinds == r2r_kinds )
                iknds(:) = int(inverse_kinds(:)%val, C_FFTW_R2R_KIND)

                if ( precision == DTFFT_SINGLE ) then
                    self%plan_forward = fftwf_plan_many_r2r(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, knds, FFTW3_FLAGS)
                    if ( need_inverse ) then
                        self%plan_backward = fftwf_plan_many_r2r(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, iknds, FFTW3_FLAGS)
                    endif
                    self%apply => fftwf_execute_r2r
                else
                    self%plan_forward = fftw_plan_many_r2r(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, knds, FFTW3_FLAGS)
                    if ( need_inverse ) then
                        self%plan_backward = fftw_plan_many_r2r(fft_rnk, fft_sizes, how_many, ptr, inembed, 1, idist, ptr, onembed, 1, odist, iknds, FFTW3_FLAGS)
                    endif
                    self%apply => fftw_execute_r2r
                endif
                self%apply_inverse => NULL()
                if ( .not. need_inverse ) then
                    self%plan_backward = self%plan_forward
                    self%is_inverse_copied = .true.
                endif
            endblock
        endselect

        call self%mem_free(ptr)
    end subroutine create

    subroutine execute(self, a, b, sign)
    !! Executes FFTW3 plan
        class(fftw_executor), intent(in)  :: self                 !! FFTW FFT Executor
        type(c_ptr),          intent(in)  :: a                    !! Source pointer
        type(c_ptr),          intent(in)  :: b                    !! Target pointer
        integer(int8),        intent(in)  :: sign                 !! Sign of transform

        if ( sign == FFT_FORWARD ) then
            call self%apply(self%plan_forward, a, b)
        else
            if ( associated( self%apply_inverse ) ) then
                call self%apply_inverse(self%plan_backward, a, b)
            else
                call self%apply(self%plan_backward, a, b)
            endif
        endif
    end subroutine execute

    subroutine destroy(self)
    !! Destroys FFTW3 plan
        class(fftw_executor), intent(inout) :: self               !! FFTW FFT Executor

        call self%free(self%plan_forward)
        if( .not. self%is_inverse_copied ) call self%free(self%plan_backward)
        self%apply => NULL()
        self%apply_inverse => NULL()
        self%free  => NULL()
    end subroutine destroy

    subroutine mem_alloc(alloc_bytes, ptr)
    !! Allocates FFTW3 memory
        integer(int64),           intent(in)  :: alloc_bytes      !! Number of bytes to allocate
        type(c_ptr),              intent(out) :: ptr              !! Allocated pointer

        ptr = fftw_malloc(alloc_bytes)
    end subroutine mem_alloc

    subroutine mem_free(ptr)
    !! Frees FFTW3 aligned memory
        type(c_ptr),               intent(in)   :: ptr            !! Pointer to free

        call fftw_free(ptr)
    end subroutine mem_free
end module dtfft_executor_fftw_m