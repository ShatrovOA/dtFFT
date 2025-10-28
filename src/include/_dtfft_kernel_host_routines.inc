#ifndef BUFFER_TYPE
#error "BUFFER_TYPE not defined"
#endif
#ifndef PREC
#error "PREC not defined"
#endif

subroutine CONCAT(permute_forward,PREC)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(:)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, ny_nz, obase, ibase

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
        do x = 0, nx - 1
            do y = 0, ny - 1
                iidx = y * nx + x + 1
                oidx = x * ny + y + 1
                out(oidx) = in(iidx)
            enddo
        enddo
        return
    endif

    nz = dims(3)
    nx_ny = nx * ny
    ny_nz = ny * nz
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
end subroutine CONCAT(permute_forward,PREC)

subroutine CONCAT(permute_backward,PREC)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(:)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, obase, ibase, nz_nx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
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
end subroutine CONCAT(permute_backward,PREC)

subroutine CONCAT(permute_backward_start,PREC)(in, out, dims)
!! Backward permutation start of a 3D array
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(:)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,oidx
    integer(int32) :: nx_ny, nz_ny, obase, ibase


    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
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
end subroutine CONCAT(permute_backward_start,PREC)

subroutine CONCAT(permute_backward_end_pipelined,PREC)(in, out, dims, neighbor_data)
!! Backward permutation end of a 3D array for a single neighbor
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(:)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: neighbor_data(:)   !! Neighbor data
    integer(int32) :: x,y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, obase, ibase, nzz_nxx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = neighbor_data(1); nyy = neighbor_data(2); nzz = neighbor_data(3)
    din = neighbor_data(4); dout = neighbor_data(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do z = 0, nzz - 1
        ibase = din + z * nxx + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nzz_nxx
            oidx = obase + y * nx
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
    enddo
end subroutine CONCAT(permute_backward_end_pipelined,PREC)

subroutine CONCAT(permute_backward_end,PREC)(in, out, dims, neighbor_data)
!! Backward permutation end of a 3D array for all neighbors
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(:)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: neighbor_data(:,:) !! Neighbor data
    integer(int32)  :: neighbor !! Neighbor index

    do neighbor = 1, size(neighbor_data, dim=2)
        call CONCAT(permute_backward_end_pipelined,PREC)(in, out, dims, neighbor_data(:, neighbor))
    enddo
end subroutine CONCAT(permute_backward_end,PREC)

subroutine CONCAT(unpack_pipelined,PREC)(in, out, dims, neighbor_data)
!! Unpacks part of contiguous buffer recieved from a single rank.
!!
!! For 3D arrays:
!!
!!  out(x + in_displacement, y, z) = in(out_displacement + z * nxx * nyy + y * nxx + x)
!!
!! where `out` buffer x,y,z are full slab indices, while `in` buffer x,y,z are indices that were sent from neighbor rank.
    BUFFER_TYPE,    intent(in)    :: in(:)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: neighbor_data(:)   !! Neighbor data
    integer(int32) :: x,y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, obase, ibase, nxx_nyy

    nx = dims(1); ny = dims(2)
    nxx = neighbor_data(1); nyy = neighbor_data(2)
    din = neighbor_data(4); dout = neighbor_data(5)

    if ( size(dims) == 2 ) then
        do y = 0, nyy - 1
            iidx = din + y * nxx + 1
            oidx = dout + y * nx + 1
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
        return
    endif
    nzz = neighbor_data(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
    do z = 0, nzz - 1
        ibase = din + z * nxx_nyy + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nxx
            oidx = obase + y * nx
        !$omp simd
            do x = 0, nxx - 1
                out(oidx + x) = in(iidx + x)
            enddo
        !$omp end simd
        enddo
    enddo
end subroutine CONCAT(unpack_pipelined,PREC)

subroutine CONCAT(unpack,PREC)(in, out, dims, neighbor_data)
!! Unpacks pack of contiguous buffer recieved from all ranks.
    BUFFER_TYPE,    intent(in)    :: in(:)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(:)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: neighbor_data(:,:) !! Neighbor data
    integer(int32)  :: neighbor !! Neighbor index

    do neighbor = 1, size(neighbor_data, dim=2)
        call CONCAT(unpack_pipelined,PREC)(in, out, dims, neighbor_data(:, neighbor))
    enddo
end subroutine CONCAT(unpack,PREC)

#undef BUFFER_TYPE
#undef PREC
