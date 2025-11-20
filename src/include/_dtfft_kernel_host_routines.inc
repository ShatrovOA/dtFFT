#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif
#ifndef BUFFER_TYPE
#error "BUFFER_TYPE not defined"
#endif
#ifndef PREC
#error "PREC not defined"
#endif

#define GET_SUB_NAME(name) CONCAT(name,PREC)

#define CONVERT_POINTER(in, out)                                                                                \
    block;                                                                                                      \
        type(c_ptr) :: p;                                                                                       \
        p = c_loc(in); call c_f_pointer(p, out, [size(in) / (int(STORAGE_BYTES / FLOAT_STORAGE_SIZE, int32))]); \
    endblock


subroutine GET_SUB_NAME(execute)(self, in, out, neighbor)
!! Executes kernel based on its type and access mode, BUFFER_TYPE version
    class(kernel_host),         intent(in)    :: self       !! Host kernel class
    real(real32),     target,   intent(in)    :: in(:)      !! Source host-allocated buffer
    real(real32),     target,   intent(inout) :: out(:)     !! Target host-allocated buffer
    integer(int32),   optional, intent(in)    :: neighbor   !! Source rank for pipelined unpacking
    BUFFER_TYPE, pointer, contiguous :: inbuf(:), outbuf(:)

    CONVERT_POINTER(in, inbuf)
    CONVERT_POINTER(out, outbuf)

    if ( self%kernel_type == KERNEL_UNPACK .or. self%kernel_type == KERNEL_UNPACK_PIPELINED ) then
        select case( self%kernel_type%val )
        case( KERNEL_UNPACK%val )
            call GET_SUB_NAME(unpack)(inbuf, outbuf, self%dims, self%neighbor_data)
        case( KERNEL_UNPACK_PIPELINED%val )
            call GET_SUB_NAME(unpack_pipelined)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor))
        endselect
        return
    endif

    if ( self%access_mode == ACCESS_MODE_WRITE ) then
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
        endselect
    else
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
        endselect
    endif
end subroutine GET_SUB_NAME(execute)

subroutine GET_SUB_NAME(select_access_mode)(self, in, out, n_warmup_iters, n_iters, execution_time)
!! Selects the best access mode for host kernels, BUFFER_TYPE version
    class(kernel_host), intent(inout)   :: self           !! Host kernel class
    real(real32),       intent(in)      :: in(:)          !! Source host-allocated buffer
    real(real32),       intent(inout)   :: out(:)         !! Target host-allocated buffer
    integer(int32),     intent(in)      :: n_warmup_iters !! Number of warmup iterations to perform before testing kernel
    integer(int32),     intent(in)      :: n_iters        !! Number of iterations to perform when testing kernel
    real(real64),       intent(out)     :: execution_time !! Execution time of the selected access
    real(real64) :: write_time, read_time

    REGION_BEGIN("WRITE", COLOR_TRANSPOSE_XY)
    self%access_mode = ACCESS_MODE_WRITE
    call execute_benchmark(self, in, out, n_warmup_iters, n_iters, write_time)
    REGION_END("WRITE")

    REGION_BEGIN("READ", COLOR_TRANSPOSE_YX)
    self%access_mode = ACCESS_MODE_READ
    call execute_benchmark(self, in, out, n_warmup_iters, n_iters, read_time)
    REGION_END("READ")

    if ( write_time < read_time ) then
        self%access_mode = ACCESS_MODE_WRITE
        execution_time = write_time
        WRITE_INFO("      Selected WRITE access mode, which is "//to_str(real(read_time / write_time, real32))//"x times faster")
    else
        self%access_mode = ACCESS_MODE_READ
        execution_time = read_time
        WRITE_INFO("      Selected READ access mode, which is "//to_str(real(write_time / read_time, real32))//"x times faster")
    endif
end subroutine GET_SUB_NAME(select_access_mode)

pure subroutine GET_SUB_NAME(permute_forward_write)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
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
end subroutine GET_SUB_NAME(permute_forward_write)

pure subroutine GET_SUB_NAME(permute_forward_read)(in, out, dims)
!! Forward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Forward permutation is defined as:
!!
!!   out(x,y,z) = in(z,x,y) for 3D arrays
!!
!!   out(x,y)   = in(y,x)   for 2D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx,oidx
    integer(int32) :: nx_ny, ny_nz, obase, ibase

    nx = dims(1); ny = dims(2)

    if ( size(dims) == 2 ) then
        do y = 0, ny - 1
            do x = 0, nx - 1
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
end subroutine GET_SUB_NAME(permute_forward_read)

pure subroutine GET_SUB_NAME(permute_backward_write)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
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
end subroutine GET_SUB_NAME(permute_backward_write)

pure subroutine GET_SUB_NAME(permute_backward_read)(in, out, dims)
!! Backward permutation of a 2D and 3D arrays, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation is defined as:
!!
!!   out(x,y,z) = in(y,z,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx
    integer(int32) :: nx_ny, obase, ibase, nz_nx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_nx = nz * nx
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
end subroutine GET_SUB_NAME(permute_backward_read)

pure subroutine GET_SUB_NAME(permute_backward_start_write)(in, out, dims)
!! Backward permutation start of a 3D array, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
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
end subroutine GET_SUB_NAME(permute_backward_start_write)

pure subroutine GET_SUB_NAME(permute_backward_start_read)(in, out, dims)
!! Backward permutation start of a 3D array, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation start is defined as:
!!
!!   out(x,y,z) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)    !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)   !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)  !! Dimensions of the array
    integer(int32) :: x,y,z,nx,ny,nz,iidx
    integer(int32) :: nx_ny, nz_ny, obase, ibase


    nx = dims(1); ny = dims(2); nz = dims(3)
    nx_ny = nx * ny
    nz_ny = nz * ny
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
end subroutine GET_SUB_NAME(permute_backward_start_read)

pure subroutine GET_SUB_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(*)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(*)   !! Neighbor data
    integer(int32) :: y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, obase, ibase, nzz_nxx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do z = 0, nzz - 1
        ibase = din + z * nxx + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nzz_nxx
            oidx = obase + y * nx
            out(oidx:oidx + nxx - 1) = in(iidx:iidx + nxx - 1)
        enddo
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_pipelined_write)

pure subroutine GET_SUB_NAME(permute_backward_end_write)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors, contiguous writing, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(*)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)        !! Neighbor data
    integer(int32)  :: neighbor !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(permute_backward_end_pipelined_write)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_write)

pure subroutine GET_SUB_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for a single neighbor, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(x,y,z) = in(x,z,y) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(*)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(*)   !! Neighbor data
    integer(int32) :: y,z,nx,ny,nz,nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, obase, ibase, nzz_nxx

    nx = dims(1); ny = dims(2); nz = dims(3)
    nxx = locals(1); nyy = locals(2); nzz = locals(3)
    din = locals(4); dout = locals(5)
    nx_ny = nx * ny
    nzz_nxx = nzz * nxx

    do y = 0, nyy - 1
        ibase = din + y * nzz_nxx + 1
        obase = dout + y * nx + 1
        do z = 0, nzz - 1
            iidx = ibase + z * nxx
            oidx = obase + z * nx_ny
            out(oidx:oidx + nxx - 1) = in(iidx:iidx + nxx - 1)
        enddo
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_pipelined_read)

pure subroutine GET_SUB_NAME(permute_backward_end_read)(in, out, dims, locals)
!! Backward permutation end of a 3D array for all neighbors, contiguous reading, BUFFER_TYPE version
!!
!! Backward permutation end is defined as:
!!
!!   out(z,x,y) = in(z,y,x) for 3D arrays
    BUFFER_TYPE,    intent(in)    :: in(*)          !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)         !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(*)        !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:)    !! Neighbor data
    integer(int32)                :: local          !! Neighbor index

    do local = 1, size(locals, dim=2)
        call GET_SUB_NAME(permute_backward_end_pipelined_read)(in, out, dims, locals(:, local))
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_read)

pure subroutine GET_SUB_NAME(unpack_pipelined)(in, out, dims, locals)
!! Unpacks part of contiguous buffer recieved from a single rank, BUFFER_TYPE version.
!!
!! For 3D arrays:
!!
!!  out(x + in_displacement, y, z) = in(out_displacement + z * nxx * nyy + y * nxx + x)
!!
!! where `out` buffer x,y,z are full slab indices, while `in` buffer x,y,z are indices that were sent from neighbor rank.
    BUFFER_TYPE,    intent(in)    :: in(*)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(*)          !! Neighbor data
    integer(int32) :: y,z,nx,ny, nxx,nyy,nzz,iidx,oidx,din,dout
    integer(int32) :: nx_ny, obase, ibase, nxx_nyy

    nx = dims(1); ny = dims(2)
    nxx = locals(1); nyy = locals(2)
    din = locals(4); dout = locals(5)

    if ( size(dims) == 2 ) then
        do y = 0, nyy - 1
            iidx = din + y * nxx + 1
            oidx = dout + y * nx + 1
            out(oidx:oidx + nxx - 1) = in(iidx:iidx + nxx - 1)
        enddo
        return
    endif
    nzz = locals(3)
    nx_ny = nx * ny
    nxx_nyy = nxx * nyy
    do z = 0, nzz - 1
        ibase = din + z * nxx_nyy + 1
        obase = dout + z * nx_ny + 1
        do y = 0, nyy - 1
            iidx = ibase + y * nxx
            oidx = obase + y * nx
            out(oidx:oidx + nxx - 1) = in(iidx:iidx + nxx - 1)
        enddo
    enddo
end subroutine GET_SUB_NAME(unpack_pipelined)

pure subroutine GET_SUB_NAME(unpack)(in, out, dims, locals)
!! Unpacks pack of contiguous buffer recieved from all ranks, BUFFER_TYPE version.
    BUFFER_TYPE,    intent(in)    :: in(*)              !! Source host-allocated buffer
    BUFFER_TYPE,    intent(inout) :: out(*)             !! Target host-allocated buffer
    integer(int32), intent(in)    :: dims(:)            !! Dimensions of the array
    integer(int32), intent(in)    :: locals(:,:) !! Neighbor data
    integer(int32)  :: neighbor !! Neighbor index

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_pipelined)(in, out, dims, locals(:, neighbor))
    enddo
end subroutine GET_SUB_NAME(unpack)

#define BLOCK_SIZE 16
#include "_dtfft_kernel_host_block_routines.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 32
#include "_dtfft_kernel_host_block_routines.inc"
#undef BLOCK_SIZE
#define BLOCK_SIZE 64
#include "_dtfft_kernel_host_block_routines.inc"
#undef BLOCK_SIZE

#undef STORAGE_BYTES
#undef BUFFER_TYPE
#undef PREC
#undef CONVERT_POINTER
#undef GET_SUB_NAME
