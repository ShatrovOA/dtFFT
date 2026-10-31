#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif
#ifndef BUFFER_TYPE
#error "BUFFER_TYPE not defined"
#endif
#ifndef PREC
#error "PREC not defined"
#endif

module procedure GET_SUB_NAME(execute)
    BUFFER_TYPE, pointer, contiguous :: inbuf(:), outbuf(:)

    CONVERT_POINTER(in, inbuf)
    CONVERT_POINTER(out, outbuf)

    if (  any(self%kernel_type == [KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED, KERNEL_PACK, KERNEL_PACK_PIPELINED, KERNEL_COPY, KERNEL_COPY_PIPELINED]) ) then
        select case( self%kernel_type%val )
        case( KERNEL_UNPACK%val )
            call GET_SUB_NAME(unpack)(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_UNPACK_PIPELINED%val )
            call GET_SUB_NAME(unpack_pipelined)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK%val )
            call GET_SUB_NAME(pack)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PACK_PIPELINED%val )
            call GET_SUB_NAME(pack_pipelined)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_COPY%val )
            call GET_SUB_NAME(copy)(inbuf, outbuf, self%dims)
        case ( KERNEL_COPY_PIPELINED%val )
            call GET_SUB_NAME(copy)(inbuf, outbuf, self%neighbor_data(1:3, neighbor), self%neighbor_data(4:5, neighbor))
        endselect
        return
    endif

    if ( self%access_mode == DTFFT_ACCESS_MODE_WRITE ) then
        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
            call GET_SUB_NAME(permute_forward_write)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
            call GET_SUB_NAME(permute_backward_write)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
            call GET_SUB_NAME(permute_backward_start_write)(inbuf, outbuf, self%dims)
        case ( KERNEL_PERMUTE_BACKWARD_END%val )
            call GET_SUB_NAME(permute_backward_end_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            call GET_SUB_NAME(permute_backward_end_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_FORWARD%val )
            call GET_SUB_NAME(unpack_forward_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
            call GET_SUB_NAME(unpack_forward_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_BACKWARD%val )
            call GET_SUB_NAME(unpack_backward_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
            call GET_SUB_NAME(unpack_backward_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_FORWARD%val )
            call GET_SUB_NAME(pack_forward_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_BACKWARD%val )
            call GET_SUB_NAME(pack_backward_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
    else if ( self%access_mode == DTFFT_ACCESS_MODE_READ ) then
        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
            call GET_SUB_NAME(permute_forward_read)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
            call GET_SUB_NAME(permute_backward_read)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
            call GET_SUB_NAME(permute_backward_start_read)(inbuf, outbuf, self%dims)
        case ( KERNEL_PERMUTE_BACKWARD_END%val )
            call GET_SUB_NAME(permute_backward_end_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            call GET_SUB_NAME(permute_backward_end_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_FORWARD%val )
            call GET_SUB_NAME(unpack_forward_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
            call GET_SUB_NAME(unpack_forward_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_BACKWARD%val )
            call GET_SUB_NAME(unpack_backward_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
            call GET_SUB_NAME(unpack_backward_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_FORWARD%val )
            call GET_SUB_NAME(pack_forward_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_BACKWARD%val )
            call GET_SUB_NAME(pack_backward_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
    else
        INTERNAL_ERROR("basic execute invalid access_mode")
    !     select case( self%kernel_type%val )
    !     case( KERNEL_PERMUTE_FORWARD%val )
    !         call GET_SUB_NAME(permute_forward_co)(inbuf, outbuf, self%dims, self%co_threshold)
    !     case( KERNEL_PERMUTE_BACKWARD%val )
    !         call GET_SUB_NAME(permute_backward_co)(inbuf, outbuf, self%dims, self%co_threshold)
    !     case( KERNEL_PERMUTE_BACKWARD_START%val )
    !         call GET_SUB_NAME(permute_backward_start_co)(inbuf, outbuf, self%dims, self%co_threshold)
    !     case ( KERNEL_PERMUTE_BACKWARD_END%val )
    !         call GET_SUB_NAME(permute_backward_end_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    !     case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
    !         call GET_SUB_NAME(permute_backward_end_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    !     case ( KERNEL_UNPACK_FORWARD%val )
    !         call GET_SUB_NAME(unpack_forward_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    !     case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
    !         call GET_SUB_NAME(unpack_forward_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    !     case ( KERNEL_UNPACK_BACKWARD%val )
    !         call GET_SUB_NAME(unpack_backward_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    !     case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
    !         call GET_SUB_NAME(unpack_backward_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    !     case ( KERNEL_PACK_FORWARD%val )
    !         call GET_SUB_NAME(pack_forward_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    !     case ( KERNEL_PACK_BACKWARD%val )
    !         call GET_SUB_NAME(pack_backward_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    !     endselect
    endif
end procedure GET_SUB_NAME(execute)

module procedure GET_SUB_NAME(select_access_mode)
    real(real64) :: write_time, read_time

    REGION_BEGIN("WRITE", COLOR_LAVENDER)
    self%access_mode = DTFFT_ACCESS_MODE_WRITE
    call execute_benchmark(self, in, out, n_warmup_iters, n_iters, write_time)
    REGION_END("WRITE")

    REGION_BEGIN("READ", COLOR_PEACH)
    self%access_mode = DTFFT_ACCESS_MODE_READ
    call execute_benchmark(self, in, out, n_warmup_iters, n_iters, read_time)
    REGION_END("READ")

    if ( write_time < read_time ) then
        self%access_mode = DTFFT_ACCESS_MODE_WRITE
        execution_time = write_time
        if ( write_time > 0._real64 ) then
            WRITE_INFO("      Selected WRITE access mode, which is "//to_str(real(read_time / write_time, real32))//"x times faster")
        else
            WRITE_INFO("      Selected WRITE access mode")
        endif
    else
        self%access_mode = DTFFT_ACCESS_MODE_READ
        execution_time = read_time
        if ( read_time > 0._real64 ) then
            WRITE_INFO("      Selected READ access mode, which is "//to_str(real(write_time / read_time, real32))//"x times faster")
        else
            WRITE_INFO("      Selected READ access mode")
        endif
    endif
end procedure GET_SUB_NAME(select_access_mode)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_forward_write)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, ny_nz
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do x = 0, nx - 1
            do y = 0, ny - 1
                iidx = y * nx + x + 1
                oidx = x * ny + y + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nx_ny = nx * ny
    ny_nz = ny * nz
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do x = 0, nx - 1
        do z = 0, nz - 1
            do y = 0, ny - 1
                iidx = z * nx_ny + 1 + x + y * nx
                oidx = x * ny_nz + 1 + z * ny + y
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do x = 0, nx - 1
        obase = x * ny_nz + 1
        do z = 0, nz - 1
            ibase = z * nx_ny + 1 + x
            oidx = obase + z * ny
            do y = 0, ny - 1
                out(oidx + y) = in(ibase + y * nx)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_forward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_forward_read)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, ny_nz
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do y = 0, ny - 1
            do x = 0, nx - 1
                iidx = y * nx + x + 1
                oidx = x * ny + y + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nx_ny = nx * ny
    ny_nz = ny * nz
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nz - 1
        do y = 0, ny - 1
            do x = 0, nx - 1
                iidx = z * nx_ny + 1 + x + y * nx
                oidx = z * ny + 1 + x * ny_nz + y
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nz - 1
        ibase = z * nx_ny + 1
        obase = z * ny + 1
        do y = 0, ny - 1
            iidx = ibase + y * nx
            do x = 0, nx - 1
                out(obase + x * ny_nz + y) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_forward_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_write)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, nz_nx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do y = 0, ny - 1
        do x = 0, nx - 1
            do z = 0, nz - 1
                iidx = y * nx + 1 + x + z * nx_ny
                oidx = y * nz_nx + 1 + x * nz + z
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do y = 0, ny - 1
        obase = y * nz_nx + 1
        ibase = y * nx + 1
        do x = 0, nx - 1
            oidx = obase + x * nz
            iidx = ibase + x
            do z = 0, nz - 1
                out(oidx + z) = in(iidx + z * nx_ny)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_read)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx
    integer(int32) :: nx_ny, nz_nx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#else
    integer(int32) :: oidx
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nz - 1
        do y = 0, ny - 1
            do x = 0, nx - 1
                iidx = z * nx_ny + 1 + y * nx + x
                oidx = y * nz_nx + z + 1 + x * nz
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nz - 1
        ibase = z * nx_ny + 1
        do y = 0, ny - 1
            obase = y * nz_nx + z + 1
            iidx = ibase + y * nx
            do x = 0, nx - 1
                out(obase + x * nz) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_start_write)(in, out, dims)
!! Backward permutation start of a 3D array, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,oidx
    integer(int32) :: nx_ny, nz_ny
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#else
    integer(int32) :: iidx
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do x = 0, nx - 1
        do y = 0, ny - 1
            do z = 0, nz - 1
                iidx = y * nx + 1 + x + z * nx_ny
                oidx = x * nz_ny + 1 + y * nz + z
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do x = 0, nx - 1
        obase = x * nz_ny + 1
        do y = 0, ny - 1
            ibase = y * nx + 1 + x
            oidx = obase + y * nz
            do z = 0, nz - 1
                out(oidx + z) = in(ibase + z * nx_ny)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_start_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_start_read)(in, out, dims)
!! Backward permutation start of a 3D array, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx
    integer(int32) :: nx_ny, nz_ny
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#else
    integer(int32) :: oidx
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nz - 1
        do y = 0, ny - 1
            do x = 0, nx - 1
                iidx = z * nx_ny + 1 + y * nx + x
                oidx = y * nz + z + 1 + x * nz_ny
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nz - 1
        ibase = z * nx_ny + 1
        do y = 0, ny - 1
            obase = y * nz + z + 1
            iidx = ibase + y * nx
            do x = 0, nx - 1
                out(obase + x * nz_ny) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_start_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nzz_nxx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + z * nxx + 1 + y * nzz_nxx + x
                oidx = dout + z * nx_ny + 1 + y * nx + x
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z * nxx + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nzz_nxx
            oidx = obase + y * nx
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_end_pipelined_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_end_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nzz_nxx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do y = 0, nyy - 1
        do z = 0, nzz - 1
            do x = 0, nxx - 1
                iidx = din + y * nzz_nxx + 1 + z * nxx + x
                oidx = dout + y * nx + 1 + z * nx_ny + x
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do y = 0, nyy - 1
        ibase = din + y * nzz_nxx + 1
        obase = dout + y * nx + 1
        do z = 0, nzz - 1
            iidx = ibase + z * nxx
            oidx = obase + z * nx_ny
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(permute_backward_end_pipelined_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(permute_backward_end_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_pipelined)(in, out, dims, locals)
!! Unpacks part of contiguous buffer recieved from a single rank, BUFFER_TYPE version.
!!
!! For 3D arrays:
!!
!!  out(x + out_displacement, y, z) = in(in_displacement + z * nxx * nyy + y * nxx + x)
!!
!! where `out` buffer x,y,z are full slab indices, while `in` buffer x,y,z are indices that were sent from neighbor rank.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nxx_nyy
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + y * nxx + 1 + x
                oidx = dout + y * nx + 1 + x
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif
    nzz = locals(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(2) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            iidx = din + z * nxx_nyy + 1 + y * nxx
            oidx = dout + z * nx_ny + 1 + y * nx
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z * nxx_nyy + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nxx
            oidx = obase + y * nx
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(unpack_pipelined)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack)(in, out, dims, locals)
!! Unpacks contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_pipelined)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack_pipelined)(in, out, dims, locals)
!! Pack part `in` buffer into contiguous buffer in order to send to a single rank, BUFFER_TYPE version.
!!
!! For 3D arrays:
!!
!!  out(out_displacement + z * nxx * nyy + y * nxx + x) = in(x + in_displacement, y, z)
!!
!! where `in` buffer x,y,z are full slab indices, while `out` buffer x,y,z are indices that are to be send to neighbor rank.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nxx_nyy
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
    !$omp parallel do private(x,y,iidx,oidx) schedule(static)
        do y = 0, nyy - 1
            iidx = din + y * nx + 1
            oidx = dout + y * nxx + 1
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
    !$omp end parallel do
        return
    endif

    nzz = locals(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(2) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            iidx = din + z * nx_ny + 1 + y * nx
            oidx = dout + z * nxx_nyy + 1 + y * nxx
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z * nx_ny + 1
        obase = dout + z * nxx_nyy + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nx
            oidx = obase + y * nxx
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(pack_pipelined)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack)(in, out, dims, locals)
!! Packs local buffer to contiguous memory, making it ready to send to all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(pack_pipelined)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(pack)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(copy)(in, out, dims, starts)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in), optional :: starts(2)
    integer(int32) :: n_elements, starts_(2), i

    starts_(:) = 1; if ( present(starts) ) starts_(:) = starts_(:) + starts(:)
    n_elements = product(dims)
!$omp parallel do schedule(static)
    do i = 1, n_elements
        out(starts_(2) + i - 1) = in(starts_(1) + i - 1)
    enddo
!$omp end parallel do
end subroutine GET_SUB_NAME(copy)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_forward_pipelined_write)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: ny_nx,nzz_nxx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + x * nyy + y + 1
                oidx = dout + y * nx + x + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nzz_nxx = nzz * nxx
    ny_nx = ny * nx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + z + 1 + y * nzz_nxx + x * nzz
                oidx = dout + z * ny_nx + 1 + y * nx + x
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z + 1
        obase = dout + z * ny_nx + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nzz_nxx
            oidx = obase + y * nx
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x * nzz)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(unpack_forward_pipelined_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_forward_pipelined_read)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nxx_nzz
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do x = 0, nxx - 1
            do y = 0, nyy - 1
                iidx = din + x * nyy + y + 1
                oidx = dout + y * nx + x + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nxx_nzz = nxx * nzz
    nx_ny = nx * ny
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do y = 0, nyy - 1
        do x = 0, nxx - 1
            do z = 0, nzz - 1
                iidx = din + 1 + y * nxx_nzz + x * nzz + z
                oidx = dout + 1 + y * nx + x + z * nx_ny
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do y = 0, nyy - 1
        ibase = din + 1 + y * nxx_nzz
        obase = dout + 1 + y * nx
        do x = 0, nxx - 1
            iidx = ibase + x * nzz
            oidx = obase + x
            do z = 0, nzz - 1
                out(oidx + z * nx_ny) = in(iidx + z)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(unpack_forward_pipelined_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_forward_write)(in, out, dims, locals)
!! Unpacks contiguous buffer recieved from all ranks and performs forward permutation using contiguous write access, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_forward_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack_forward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_forward_read)(in, out, dims, locals)
!! Unpacks contiguous buffer recieved from all ranks and performs forward permutation using contiguous read access, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_forward_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack_forward_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_backward_pipelined_write)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: ny_nx, nzz_nyy
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)

    nzz_nyy = nzz * nyy
    ny_nx = ny * nx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + 1 + z * nyy + y + x * nzz_nyy
                oidx = dout + 1 + z * ny_nx + y * nx + x
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + 1 + z * nyy
        obase = dout + 1 + z * ny_nx
        do y = 0, nyy - 1
            iidx = ibase + y
            oidx = obase + y * nx
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x * nzz_nyy)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(unpack_backward_pipelined_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_backward_pipelined_read)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nzz_nyy
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nzz_nyy = nzz * nyy
    nx_ny = nx * ny
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do x = 0, nxx - 1
        do z = 0, nzz - 1
            do y = 0, nyy - 1
                iidx = din + 1 + x * nzz_nyy + z * nyy + y
                oidx = dout + 1 + x + z * nx_ny + y * nx
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do x = 0, nxx - 1
        ibase = din + 1 + x * nzz_nyy
        obase = dout + 1 + x
        do z = 0, nzz - 1
            iidx = ibase + z * nyy
            oidx = obase + z * nx_ny
            do y = 0, nyy - 1
                out(oidx + y * nx) = in(iidx + y)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(unpack_backward_pipelined_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_backward_write)(in, out, dims, locals)
!! Unpacks contiguous buffer recieved from all ranks and performs backward permutation using contiguous write access, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_backward_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack_backward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(unpack_backward_read)(in, out, dims, locals)
!! Unpacks contiguous buffer recieved from all ranks and performs backward permutation using contiguous read access, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)                !! Neighbor data
    integer(int32)  :: neighbor                                 !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_backward_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack_backward_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack_forward_write)(in, out, dims, locals)
!! Forward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,din,dout,nxx,nyy,nzz
    integer(int32) :: nx_ny, nyy_nzz
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do x = 0, nxx - 1
            do y = 0, nyy - 1
                iidx = din + y * nx + x + 1
                oidx = dout + x * nyy + y + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nx_ny = nx * ny
    nyy_nzz = nyy * nzz
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do x = 0, nxx - 1
        do z = 0, nzz - 1
            do y = 0, nyy - 1
                iidx = din + z * nx_ny + 1 + x + y * nx
                oidx = dout + x * nyy_nzz + 1 + z * nyy + y
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do x = 0, nxx - 1
        obase = dout + x * nyy_nzz + 1
        do z = 0, nzz - 1
            ibase = din + z * nx_ny + 1 + x
            oidx = obase + z * nyy
            do y = 0, nyy - 1
                out(oidx + y) = in(ibase + y * nx)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(pack_forward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack_forward_read)(in, out, dims, locals)
!! Forward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,din,dout,nxx,nyy,nzz
    integer(int32) :: nx_ny, nyy_nzz
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)
    if ( size(dims) == 2 ) then
    !$omp parallel do collapse(2) private(x,y,iidx,oidx) schedule(static)
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + y * nx + x + 1
                oidx = dout + x * nyy + y + 1
                out(oidx) = in(iidx)
            enddo
        enddo
    !$omp end parallel do
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nx_ny = nx * ny
    nyy_nzz = nyy * nzz
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + z * nx_ny + 1 + y * nx + x
                oidx = dout + z * nyy + 1 + x * nyy_nzz + y
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z * nx_ny + 1
        obase = dout + z * nyy + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nx
            do x = 0, nxx - 1
                out(obase + x * nyy_nzz + y) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(pack_forward_read)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack_backward_write)(in, out, dims, locals)
!! Backward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,din,dout,nxx,nyy,nzz
    integer(int32) :: nx_ny, nzz_nxx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do y = 0, nyy - 1
        do x = 0, nxx - 1
            do z = 0, nzz - 1
                iidx = din + y * nx + 1 + x + z * nx_ny
                oidx = dout + y * nzz_nxx + 1 + x * nzz + z
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do y = 0, nyy - 1
        obase = dout + y * nzz_nxx + 1
        ibase = din + y * nx + 1
        do x = 0, nxx - 1
            oidx = obase + x * nzz
            iidx = ibase + x
            do z = 0, nzz - 1
                out(oidx + z) = in(iidx + z * nx_ny)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(pack_backward_write)

#ifndef DTFFT_WITH_OPENMP
pure &
#endif
subroutine GET_SUB_NAME(pack_backward_read)(in, out, dims, locals)
!! Backward permutation of 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,iidx,din,dout,nxx,nyy,nzz
    integer(int32) :: nx_ny, nzz_nxx
#ifndef DTFFT_WITH_OPENMP
    integer(int32) :: obase, ibase
#else
    integer(int32) :: oidx
#endif

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx
#ifdef DTFFT_WITH_OPENMP
!$omp parallel do collapse(3) private(x,y,z,iidx,oidx) schedule(static)
    do z = 0, nzz - 1
        do y = 0, nyy - 1
            do x = 0, nxx - 1
                iidx = din + z * nx_ny + 1 + y * nx + x
                oidx = dout + y * nzz_nxx + z + 1 + x * nzz
                out(oidx) = in(iidx)
            enddo
        enddo
    enddo
!$omp end parallel do
#else
    do z = 0, nzz - 1
        ibase = din + z * nx_ny + 1
        do y = 0, nyy - 1
            obase = dout + y * nzz_nxx + z + 1
            iidx = ibase + y * nx
            do x = 0, nxx - 1
                out(obase + x * nzz) = in(iidx + x)
            enddo
        enddo
    enddo
#endif
end subroutine GET_SUB_NAME(pack_backward_read)

! #include "_dtfft_kernel_host_co_routines.inc"
! #define BLOCK_SIZE 4
! #include "_dtfft_kernel_host_block_routines.inc"
! #define BLOCK_SIZE 8
! #include "_dtfft_kernel_host_block_routines.inc"
! #define BLOCK_SIZE 16
! #include "_dtfft_kernel_host_block_routines.inc"
! #define BLOCK_SIZE 32
! #include "_dtfft_kernel_host_block_routines.inc"
! #define BLOCK_SIZE 64
! #include "_dtfft_kernel_host_block_routines.inc"

#undef STORAGE_BYTES
#undef BUFFER_TYPE
#undef PREC
