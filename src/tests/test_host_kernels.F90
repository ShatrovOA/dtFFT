program test_host_kernels
use iso_fortran_env
use dtfft_kernel_host
implicit none
    real(real32), allocatable :: in(:), out(:), gold(:)
    integer(int32) :: dims3(3), dims2(2), i

    dims3 = [33, 77, 21]
    dims2 = [90, 57]


    call run_permute_forward(dims3)
    call run_permute_forward(dims2)

    call run_permute_backward(dims3)

    call run_permute_backward_start(dims3)

    call run_permute_backward_end(dims3)

    call run_unpack(dims3)
    call run_unpack(dims2)

contains

subroutine run_permute_forward(dims)
    integer(int32), intent(in) :: dims(:)

    print*,'Testing permute_forward kernels: ndims = ',size(dims)

    allocate( in(product(dims)), out(product(dims)), gold(product(dims)) )

    do i = 1, size(in)
        in(i) = real(i, real32)
    enddo

    call permute_forward_write_f32(in, gold, dims)

    call permute_forward_read_f32(in, out, dims); call compare
    call permute_forward_write_f32_block_16(in, out, dims); call compare
    call permute_forward_write_f32_block_32(in, out, dims); call compare
    call permute_forward_write_f32_block_64(in, out, dims); call compare
    call permute_forward_read_f32_block_16(in, out, dims); call compare
    call permute_forward_read_f32_block_32(in, out, dims); call compare
    call permute_forward_read_f32_block_64(in, out, dims); call compare
!     print*,'Testing transpose_3d_cache_oblivious kernel'
!     call cpu_time(start_time)
!     call transpose_3d_cache_oblivious(in, out, dims)!;call compare
! call cpu_time(end_time)
! print*,'Time for transpose_3d_cache_oblivious: ', end_time - start_time
    deallocate(in, out, gold)

    print*,'SUCCESS'
end subroutine run_permute_forward

subroutine run_permute_backward(dims)
    integer(int32), intent(in) :: dims(:)

    print*,'Testing permute_backward kernels'

    allocate( in(product(dims)), out(product(dims)), gold(product(dims)) )

    do i = 1, size(in)
        in(i) = real(i, real32)
    enddo

    call permute_backward_write_f32(in, gold, dims)

    call permute_backward_read_f32(in, out, dims); call compare
    call permute_backward_write_f32_block_16(in, out, dims); call compare
    call permute_backward_write_f32_block_32(in, out, dims); call compare
    call permute_backward_write_f32_block_64(in, out, dims); call compare
    call permute_backward_read_f32_block_16(in, out, dims); call compare
    call permute_backward_read_f32_block_32(in, out, dims); call compare
    call permute_backward_read_f32_block_64(in, out, dims); call compare

    deallocate(in, out, gold)

    print*,'SUCCESS'
end subroutine run_permute_backward

subroutine run_permute_backward_start(dims)
    integer(int32), intent(in) :: dims(:)

    print*,'Testing permute_backward_start kernels'

    allocate( in(product(dims)), out(product(dims)), gold(product(dims)) )

    do i = 1, size(in)
        in(i) = real(i, real32)
    enddo

    call permute_backward_start_write_f32(in, gold, dims)

    call permute_backward_start_read_f32(in, out, dims); call compare
    call permute_backward_start_write_f32_block_16(in, out, dims); call compare
    call permute_backward_start_write_f32_block_32(in, out, dims); call compare
    call permute_backward_start_write_f32_block_64(in, out, dims); call compare
    call permute_backward_start_read_f32_block_16(in, out, dims); call compare
    call permute_backward_start_read_f32_block_32(in, out, dims); call compare
    call permute_backward_start_read_f32_block_64(in, out, dims); call compare

    deallocate(in, out, gold)

    print*,'SUCCESS'
end subroutine run_permute_backward_start

subroutine run_permute_backward_end(dims)
    integer(int32), intent(in) :: dims(:)
    integer(int32) :: ndata(5)

    print*,'Testing permute_backward_end kernels'

    allocate( in(product(dims)), out(product(dims)), gold(product(dims)) )

    do i = 1, size(in)
        in(i) = real(i, real32)
    enddo
    gold(:) = -1._real32
    out(:) = -1._real32
    ndata = [dims(1) / 2, dims(2) / 2, dims(3) / 2, 1, 1]

    call permute_backward_end_pipelined_write_f32(in, gold, dims, ndata)


    call permute_backward_end_pipelined_write_f32_block_16(in, out, dims, ndata); call compare
    call permute_backward_end_pipelined_write_f32_block_32(in, out, dims, ndata); call compare
    call permute_backward_end_pipelined_write_f32_block_64(in, out, dims, ndata); call compare
    call permute_backward_end_pipelined_read_f32_block_16(in, out, dims, ndata); call compare
    call permute_backward_end_pipelined_read_f32_block_32(in, out, dims, ndata); call compare
    call permute_backward_end_pipelined_read_f32_block_64(in, out, dims, ndata); call compare

    deallocate(in, out, gold)

    print*,'SUCCESS'
end subroutine run_permute_backward_end

subroutine run_unpack(dims)
    integer(int32), intent(in) :: dims(:)
    integer(int32) :: ndata(5)

    print*,'Testing run_unpack kernels, ndims = ',size(dims)

    allocate( in(product(dims)), out(product(dims)), gold(product(dims)) )

    do i = 1, size(in)
        in(i) = real(i, real32)
    enddo
    gold(:) = -1._real32
    out(:) = -1._real32
    if( size(dims) == 3) then
        ndata = [dims(1) / 2, dims(2) / 2, dims(3) / 2, 1, 1]
    else
        ndata = [dims(1) / 2, dims(2) / 2, 0, 1, 1]
    endif

    call unpack_pipelined_f32(in, gold, dims, ndata)

    call unpack_pipelined_f32_block_16(in, out, dims, ndata); call compare
    call unpack_pipelined_f32_block_32(in, out, dims, ndata); call compare
    call unpack_pipelined_f32_block_64(in, out, dims, ndata); call compare

    deallocate(in, out, gold)

    print*,'SUCCESS'
end subroutine run_unpack


subroutine compare
    ! print*, 'Comparing results...'
    ! print*, gold(1:10)
    ! print*, out(1:10)
    do i = 1, size(gold)
        if ( abs(out(i) - gold(i)) > 1e-6 ) then
            print*, i, out(i), gold(i)
            error stop "Test failed"
        endif
        out(i) = -1._real32
    enddo
end subroutine compare


!    recursive subroutine transpose_3d_recursive(in, out, &
!                                               in_start, in_dims, in_strides, &
!                                               out_start, out_dims, out_strides, &
!                                               base_case_size)
!         real(real32), intent(in) :: in(*)
!         real(real32), intent(out) :: out(*)
!         integer(int32), intent(in) :: in_start(3), in_dims(3), in_strides(3)
!         integer(int32), intent(in) :: out_start(3), out_dims(3), out_strides(3)
!         integer(int32), intent(in) :: base_case_size

!         integer(int32) :: new_start(3), new_dims(3)
!         integer(int32) :: split_dim


!         if (all(in_dims < base_case_size)) then
!             call transpose_3d_base_case(in, out, &
!                                       in_start, in_dims, in_strides, &
!                                       out_start, out_dims, out_strides)
!             return
!         end if

!         split_dim = maxloc(in_dims, dim=1)


!         new_dims = in_dims
!         new_dims(split_dim) = in_dims(split_dim) / 2

!         new_start = in_start
!         call transpose_3d_recursive(in, out, &
!                                   new_start, new_dims, in_strides, &
!                                   out_start, out_dims, out_strides, &
!                                   base_case_size)

!         new_start(split_dim) = in_start(split_dim) + new_dims(split_dim)
!         new_dims(split_dim) = in_dims(split_dim) - new_dims(split_dim)
!         call transpose_3d_recursive(in, out, &
!                                   new_start, new_dims, in_strides, &
!                                   out_start, out_dims, out_strides, &
!                                   base_case_size)
!     end subroutine transpose_3d_recursive

!     subroutine transpose_3d_base_case(in, out, &
!                                     in_start, in_dims, in_strides, &
!                                     out_start, out_dims, out_strides)
!         real(real32), intent(in) :: in(*)
!         real(real32), intent(out) :: out(*)
!         integer(int32), intent(in) :: in_start(3), in_dims(3), in_strides(3)
!         integer(int32), intent(in) :: out_start(3), out_dims(3), out_strides(3)

!         integer(int32) :: x, y, z
!         integer(int32) :: in_idx, out_idx
!         integer(int32) :: nx, ny, nz

!         nx = in_dims(1)
!         ny = in_dims(2)
!         nz = in_dims(3)

!         do x = in_start(1), in_start(1) + nx - 1
!             do z = in_start(3), in_start(3) + nz - 1
!                 do y = in_start(2), in_start(2) + ny - 1
!                     in_idx = x * in_strides(1) + &
!                              y * in_strides(2) + &
!                              z * in_strides(3) + 1

!                     out_idx = y * out_strides(1) + &
!                               z * out_strides(2) + &
!                               x * out_strides(3) + 1

!                     out(out_idx) = in(in_idx)
!                 end do
!             end do
!         end do
!     end subroutine transpose_3d_base_case

    ! subroutine transpose_3d_cache_oblivious(in, out, dims)
    !     real(real32), intent(in) :: in(*)
    !     real(real32), intent(out) :: out(*)
    !     integer(int32), intent(in) :: dims(3)

    !     integer(int32) :: in_start(3), in_dims(3), in_strides(3)
    !     integer(int32) :: out_start(3), out_dims(3), out_strides(3)
    !     integer(int32) :: base_case_size

    !     in_start = [0, 0, 0]
    !     in_dims = dims
    !     in_strides = [1, dims(1), dims(1) * dims(2)]

    !     out_start = [0, 0, 0]
    !     out_dims = [dims(2), dims(3), dims(1)]  ! (y, z, x)
    !     out_strides = [1, dims(2), dims(2) * dims(3)]

    !     base_case_size = 32

    !     call transpose_3d_recursive(in, out, &
    !                               in_start, in_dims, in_strides, &
    !                               out_start, out_dims, out_strides, &
    !                               base_case_size)
    ! end subroutine transpose_3d_cache_oblivious

end program test_host_kernels