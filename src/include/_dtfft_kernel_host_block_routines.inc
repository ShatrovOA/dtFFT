#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

#define GET_BLOCK_SUB_NAME(name) CONCAT(GET_SUB_NAME(name),_block_)
#define GET_BLOCK_SUB_NAME_EXPANDED(name) GET_BLOCK_SUB_NAME(name)
#define GET_BLOCK_KERNEL_NAME(name) CONCAT(GET_BLOCK_SUB_NAME_EXPANDED(name),BLOCK_SIZE)


subroutine GET_BLOCK_KERNEL_NAME(execute)(self, in, out, neighbor)
!! Executes the given kernel on host
    class(kernel_host),         intent(in)  :: self       !! Host kernel class
    type(c_ptr),                intent(in)  :: in       !! Source host-allocated buffer
    type(c_ptr),                intent(in)  :: out      !! Target host-allocated buffer
    integer(int32),   optional, intent(in)  :: neighbor   !! Source rank for pipelined unpacking
    BUFFER_TYPE, pointer, contiguous :: inbuf(:), outbuf(:)

    CONVERT_POINTER(in, inbuf)
    CONVERT_POINTER(out, outbuf)

    if (  any(self%kernel_type == [KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED, KERNEL_PACK, KERNEL_PACK_PIPELINED]) ) then
        select case( self%kernel_type%val )
        case( KERNEL_UNPACK%val )
            call GET_BLOCK_KERNEL_NAME(unpack)(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_UNPACK_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(unpack_pipelined)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK%val )
            call GET_BLOCK_KERNEL_NAME(pack)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PACK_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(pack_pipelined)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
        return
    endif

    if ( self%access_mode == DTFFT_ACCESS_MODE_WRITE ) then
        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(permute_forward_write)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_write)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_start_write)(inbuf, outbuf, self%dims)
        case ( KERNEL_PERMUTE_BACKWARD_END%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_end_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(unpack_forward_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(unpack_backward_write)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(pack_forward_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(pack_backward_write)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
    else
        select case( self%kernel_type%val )
        case( KERNEL_PERMUTE_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(permute_forward_read)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_read)(inbuf, outbuf, self%dims)
        case( KERNEL_PERMUTE_BACKWARD_START%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_start_read)(inbuf, outbuf, self%dims)
        case ( KERNEL_PERMUTE_BACKWARD_END%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_end_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(unpack_forward_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_UNPACK_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(unpack_backward_read)(inbuf, outbuf, self%dims, self%neighbor_data)
        case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
            call GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_FORWARD%val )
            call GET_BLOCK_KERNEL_NAME(pack_forward_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        case ( KERNEL_PACK_BACKWARD%val )
            call GET_BLOCK_KERNEL_NAME(pack_backward_read)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
    endif
end subroutine GET_BLOCK_KERNEL_NAME(execute)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_forward_write)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, ny_nz

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
        do xb = 0, nx - 1, BLOCK_SIZE
            do yb = 0, ny - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                        iidx = y * nx + x + 1
                        oidx = x * ny + y + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nx_ny = nx * ny
    ny_nz = ny * nz
    do xb = 0, nx - 1, BLOCK_SIZE
        do zb = 0, nz - 1, BLOCK_SIZE
            do yb = 0, ny - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                    do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                        do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = x * ny_nz + z * ny + y + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_forward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_forward_read)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, ny_nz

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
        do yb = 0, ny - 1, BLOCK_SIZE
            do xb = 0, nx - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                        iidx = y * nx + x + 1
                        oidx = x * ny + y + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nx_ny = nx * ny
    ny_nz = ny * nz
    do zb = 0, nz - 1, BLOCK_SIZE
        do yb = 0, ny - 1, BLOCK_SIZE
            do xb = 0, nx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = x * ny_nz + z * ny + y + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_forward_read)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_write)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, nz_nx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
    do yb = 0, ny - 1, BLOCK_SIZE
        do xb = 0, nx - 1, BLOCK_SIZE
            do zb = 0, nz - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                        do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = y * nz_nx + x * nz + z + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_read)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, nz_nx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
    do zb = 0, nz - 1, BLOCK_SIZE
        do yb = 0, ny - 1, BLOCK_SIZE
            do xb = 0, nx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = y * nz_nx + x * nz + z + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_read)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_start_write)(in, out, dims)
!! Backward permutation start of a 3D array
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, nz_ny


    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
    do xb = 0, nx - 1, BLOCK_SIZE
        do yb = 0, ny - 1, BLOCK_SIZE
            do zb = 0, nz - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                        do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = x * nz_ny + y * nz + z + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_start_write)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_start_read)(in, out, dims)
!! Backward permutation start of a 3D array
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,xb,yb,zb
    integer(int32) :: nx_ny, nz_ny


    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
    do zb = 0, nz - 1, BLOCK_SIZE
        do yb = 0, ny - 1, BLOCK_SIZE
            do xb = 0, nx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, ny - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nx - 1)
                            iidx = z * nx_ny + y * nx + x + 1
                            oidx = x * nz_ny + y * nz + z + 1
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_start_read)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout,yb,zb
    integer(int32) :: nx_ny, nzz_nxx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    iidx = din + y * nzz_nxx + z * nxx + 1
                    oidx = dout + y * nx + z * nx_ny + 1
                    do x = 0, nxx - 1
                        out(oidx + x) = in(iidx + x)
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_write)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_write)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout,yb,zb
    integer(int32) :: nx_ny, nzz_nxx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do yb = 0, nyy - 1, BLOCK_SIZE
        do zb = 0, nzz - 1, BLOCK_SIZE
            do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                    iidx = din + y * nzz_nxx + z * nxx + 1
                    oidx = dout + y * nx + z * nx_ny + 1
                    do x = 0, nxx - 1
                        out(oidx + x) = in(iidx + x)
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_read)

pure subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(permute_backward_end_read)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_pipelined)(in, out, dims, locals)
!! Unpacks part of contiguous buffer recieved from a single rank.
!!
!! For 3D arrays:
!!
!!  out(x + in_displacement, y, z) = in(out_displacement + z * nxx * nyy + y * nxx + x)
!!
!! where `out` buffer x,y,z are full slab indices, while `in` buffer x,y,z are indices that were sent from neighbor rank.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout,yb,zb
    integer(int32) :: nx_ny, nxx_nyy

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do yb = 0, nyy - 1, BLOCK_SIZE
            do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                iidx = din  + y * nxx + 1
                oidx = dout + y * nx  + 1
                do x = 0, nxx - 1
                    out(oidx + x) = in(iidx + x)
                enddo
            enddo
        enddo
        return
    endif
    nzz = locals(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    iidx = din  + z * nxx_nyy + y * nxx + 1
                    oidx = dout + z * nx_ny   + y * nx  + 1
                    do x = 0, nxx - 1
                        out(oidx + x) = in(iidx + x)
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_pipelined)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(unpack_pipelined)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack)

pure subroutine GET_BLOCK_KERNEL_NAME(pack_pipelined)(in, out, dims, locals)
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
    integer(int32) :: yb,zb,x,y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nxx_nyy

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do yb = 0, nyy - 1, BLOCK_SIZE
            do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                iidx = din + y * nx + 1
                oidx = dout + y * nxx + 1
                do x = 0, nxx - 1
                    out(oidx + x) = in(iidx + x)
                enddo
            enddo
        enddo
        return
    endif

    nzz = locals(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    iidx = din + z * nx_ny + y * nx + 1
                    oidx = dout + z * nxx_nyy + y * nxx + 1
                    do x = 0, nxx - 1
                        out(oidx + x) = in(iidx + x)
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack_pipelined)

pure subroutine GET_BLOCK_KERNEL_NAME(pack)(in, out, dims, locals)
!! Packs data into contiguous buffer ready to send to all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(pack_pipelined)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_write)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: ny_nx,nzz_nxx,xb,yb,zb

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                        iidx = din + x * nyy + y + 1
                        oidx = dout + y * nx + x + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nzz_nxx = nzz * nxx
    ny_nx = ny * nx

    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                            iidx = din + z + 1 + y * nzz_nxx + x * nzz
                            oidx = dout + z * ny_nx + 1 + y * nx + x
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_write)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_read)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny,nxx_nzz,xb,yb,zb

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do xb = 0, nxx - 1, BLOCK_SIZE
            do yb = 0, nyy - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        iidx = din + x * nyy + y + 1
                        oidx = dout + y * nx + x + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nxx_nzz = nxx * nzz
    nx_ny = nx * ny

    do yb = 0, nyy - 1, BLOCK_SIZE
        do xb = 0, nxx - 1, BLOCK_SIZE
            do zb = 0, nzz - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                        do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                            iidx = din + 1 + y * nxx_nzz + x * nzz + z
                            oidx = dout + 1 + y * nx + x + z * nx_ny
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_read)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_write)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_read)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(unpack_forward_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_forward_read)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_write)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: ny_nx, nzz_nyy,xb,yb,zb

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)

    nzz_nyy = nzz * nyy
    ny_nx = ny * nx

    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                            iidx = din + 1 + z * nyy + y + x * nzz_nyy
                            oidx = dout + 1 + z * ny_nx + y * nx + x
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_write)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_read)(in, out, dims, locals)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)            !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)           !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                    !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)        !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, nzz_nyy,xb,yb,zb

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nzz_nyy = nzz * nyy
    nx_ny = nx * ny

    do xb = 0, nxx - 1, BLOCK_SIZE
        do zb = 0, nzz - 1, BLOCK_SIZE
            do yb = 0, nyy - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                    do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                        do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                            iidx = din + 1 + x * nzz_nyy + z * nyy + y
                            oidx = dout + 1 + x + z * nx_ny + y * nx
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_read)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_write)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_read)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor                         !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_BLOCK_KERNEL_NAME(unpack_backward_pipelined_read)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(unpack_backward_read)

pure subroutine GET_BLOCK_KERNEL_NAME(pack_forward_write)(in, out, dims, locals)
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
    integer(int32) :: nx_ny, nyy_nzz,xb,yb,zb

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do xb = 0, nxx - 1, BLOCK_SIZE
            do yb = 0, nyy - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        iidx = din + y * nx + x + 1
                        oidx = dout + x * nyy + y + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nx_ny = nx * ny
    nyy_nzz = nyy * nzz
    do xb = 0, nxx - 1, BLOCK_SIZE
        do zb = 0, nzz - 1, BLOCK_SIZE
            do yb = 0, nyy - 1, BLOCK_SIZE
                do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                    do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                        do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                            iidx = din + z * nx_ny + 1 + x + y * nx
                            oidx = dout + x * nyy_nzz + 1 + z * nyy + y
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack_forward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(pack_forward_read)(in, out, dims, locals)
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
    integer(int32) :: nx_ny, nyy_nzz,xb,yb,zb

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)
    if ( size(dims) == 2 ) then
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                        iidx = din + y * nx + x + 1
                        oidx = dout + x * nyy + y + 1
                        out(oidx) = in(iidx)
                    enddo
                enddo
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nzz = locals(3)
    nx_ny = nx * ny
    nyy_nzz = nyy * nzz
    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                            iidx = din + z * nx_ny + 1 + y * nx + x
                            oidx = dout + z * nyy + 1 + x * nyy_nzz + y
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack_forward_read)

pure subroutine GET_BLOCK_KERNEL_NAME(pack_backward_write)(in, out, dims, locals)
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
    integer(int32) :: nx_ny, nzz_nxx,xb,yb,zb

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do yb = 0, nyy - 1, BLOCK_SIZE
        do xb = 0, nxx - 1, BLOCK_SIZE
            do zb = 0, nzz - 1, BLOCK_SIZE
                do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                    do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                        do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                            iidx = din + y * nx + 1 + x + z * nx_ny
                            oidx = dout + y * nzz_nxx + 1 + x * nzz + z
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack_backward_write)

pure subroutine GET_BLOCK_KERNEL_NAME(pack_backward_read)(in, out, dims, locals)
!! Backward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)        !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)       !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)                !! Dimensions of the array
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)    !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx,din,dout,nxx,nyy,nzz
    integer(int32) :: nx_ny, nzz_nxx,xb,yb,zb

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx
    do zb = 0, nzz - 1, BLOCK_SIZE
        do yb = 0, nyy - 1, BLOCK_SIZE
            do xb = 0, nxx - 1, BLOCK_SIZE
                do z = zb, min(zb + BLOCK_SIZE - 1, nzz - 1)
                    do y = yb, min(yb + BLOCK_SIZE - 1, nyy - 1)
                        do x = xb, min(xb + BLOCK_SIZE - 1, nxx - 1)
                            iidx = din + z * nx_ny + 1 + y * nx + x
                            oidx = dout + y * nzz_nxx + z + 1 + x * nzz
                            out(oidx) = in(iidx)
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
end subroutine GET_BLOCK_KERNEL_NAME(pack_backward_read)

#undef GET_BLOCK_SUB_NAME
#undef GET_BLOCK_SUB_NAME_EXPANDED
#undef GET_BLOCK_KERNEL_NAME
#undef BLOCK_SIZE