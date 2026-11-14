#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif
#ifndef BUFFER_TYPE
#error "BUFFER_TYPE not defined"
#endif
#ifndef PREC
#error "PREC not defined"
#endif

module procedure GET_CO_SUB_NAME(execute)
    BUFFER_TYPE, pointer, contiguous :: inbuf(:), outbuf(:)

    CONVERT_POINTER(in, inbuf)
    CONVERT_POINTER(out, outbuf)

#ifdef DTFFT_DEBUG
    if (  any(self%kernel_type == [KERNEL_UNPACK, KERNEL_UNPACK_PIPELINED, KERNEL_PACK, KERNEL_PACK_PIPELINED, KERNEL_COPY, KERNEL_COPY_PIPELINED]) ) then
        INTERNAL_ERROR("CO_ROUTINE invalid kernel type")
    endif
#endif

    select case( self%kernel_type%val )
    case( KERNEL_PERMUTE_FORWARD%val )
        call GET_SUB_NAME(permute_forward_co)(inbuf, outbuf, self%dims, self%co_threshold)
    case( KERNEL_PERMUTE_BACKWARD%val )
        call GET_SUB_NAME(permute_backward_co)(inbuf, outbuf, self%dims, self%co_threshold)
    case( KERNEL_PERMUTE_BACKWARD_START%val )
        call GET_SUB_NAME(permute_backward_start_co)(inbuf, outbuf, self%dims, self%co_threshold)
    case ( KERNEL_PERMUTE_BACKWARD_END%val )
        call GET_SUB_NAME(permute_backward_end_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    case ( KERNEL_PERMUTE_BACKWARD_END_PIPELINED%val )
        call GET_SUB_NAME(permute_backward_end_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    case ( KERNEL_UNPACK_FORWARD%val )
        call GET_SUB_NAME(unpack_forward_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    case ( KERNEL_UNPACK_FORWARD_PIPELINED%val )
        call GET_SUB_NAME(unpack_forward_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    case ( KERNEL_UNPACK_BACKWARD%val )
        call GET_SUB_NAME(unpack_backward_co)(inbuf, outbuf, self%dims, self%neighbor_data, self%co_threshold)
    case ( KERNEL_UNPACK_BACKWARD_PIPELINED%val )
        call GET_SUB_NAME(unpack_backward_pipelined_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    case ( KERNEL_PACK_FORWARD%val )
        call GET_SUB_NAME(pack_forward_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    case ( KERNEL_PACK_BACKWARD%val )
        call GET_SUB_NAME(pack_backward_co)(inbuf, outbuf, self%dims, self%neighbor_data(:, neighbor), self%co_threshold)
    endselect
end procedure GET_CO_SUB_NAME(execute)

subroutine GET_SUB_NAME(permute_backward_co)(in, out, dims, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product(dims) / MAX_TASKS)
!$omp parallel
!$omp single
    call GET_SUB_NAME(co_permute_backward)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, dims(1), dims(2), dims(3), threshold, gran)
!$omp end single
!$omp end parallel
end subroutine GET_SUB_NAME(permute_backward_co)

recursive subroutine GET_SUB_NAME(co_permute_backward)(in, out, nx, ny, nz, x0, y0, z0, fnx, fny, fnz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz   !! current block extents
    integer(int32), intent(in)    :: x0, y0, z0   !! block offsets (0-based)
    integer(int32), intent(in)    :: fnx, fny, fnz !! full array dimensions
    integer(int32), intent(in)    :: threshold     !! stop recursion when nx*ny*nz <= threshold
    integer(int32), intent(in)    :: gran          !! stop spawning tasks when nx*ny*nz <= gran
    integer(int32) :: x, y, z, half, ibase, obase, iidx, fnx_fny, fnz_fnx

    if (nx * ny * nz <= threshold) then
        fnx_fny = fnx * fny
        fnz_fnx = fnz * fnx
        do z = z0, z0 + nz - 1
            ibase = z * fnx_fny + 1
            do y = y0, y0 + ny - 1
                obase = y * fnz_fnx + z + 1
                iidx = ibase + y * fnx
                do x = x0, x0 + nx - 1
                    out(obase + x * fnz) = in(iidx + x)
                end do
            end do
        end do
    else if (nx >= ny .and. nx >= nz) then
        half = nx / 2
        !$omp task shared(in, out) final(half * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, half,      ny, nz, x0,        y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nx - half) * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, nx - half, ny, nz, x0 + half, y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (ny >= nz) then
        half = ny / 2
        !$omp task shared(in, out) final(nx * half * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, nx, half,      nz, x0, y0,        z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * (ny - half) * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, nx, ny - half, nz, x0, y0 + half, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nz / 2
        !$omp task shared(in, out) final(nx * ny * half <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, nx, ny, half,      x0, y0, z0,        fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * ny * (nz - half) <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward)(in, out, nx, ny, nz - half, x0, y0, z0 + half, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_permute_backward)

subroutine GET_SUB_NAME(permute_forward_co)(in, out, dims, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product(dims) / MAX_TASKS)
    if ( size(dims) == 2 ) then
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_permute_forward_2d)(in, out, dims(1), dims(2), 0, 0, dims(1), dims(2), threshold, gran)
    !$omp end single
    !$omp end parallel
    else
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_permute_forward)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, dims(1), dims(2), dims(3), threshold, gran)
    !$omp end single
    !$omp end parallel
    endif
end subroutine GET_SUB_NAME(permute_forward_co)

recursive subroutine GET_SUB_NAME(co_permute_forward_2d)(in, out, nx, ny, x0, y0, fnx, fny, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny
    integer(int32), intent(in)    :: x0, y0
    integer(int32), intent(in)    :: fnx, fny
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, half, iidx, oidx

    if (nx * ny <= threshold) then
        do y = y0, y0 + ny - 1
            do x = x0, x0 + nx - 1
                iidx = y * fnx + x + 1
                oidx = x * fny + y + 1
                out(oidx) = in(iidx)
            end do
        end do
    else if (nx >= ny) then
        half = nx / 2
        !$omp task shared(in, out) final(half * ny <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward_2d)(in, out, half,      ny, x0,        y0, fnx, fny, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nx - half) * ny <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward_2d)(in, out, nx - half, ny, x0 + half, y0, fnx, fny, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = ny / 2
        !$omp task shared(in, out) final(nx * half <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward_2d)(in, out, nx, half,      x0, y0,        fnx, fny, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * (ny - half) <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward_2d)(in, out, nx, ny - half, x0, y0 + half, fnx, fny, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_permute_forward_2d)

recursive subroutine GET_SUB_NAME(co_permute_forward)(in, out, nx, ny, nz, x0, y0, z0, fnx, fny, fnz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: fnx, fny, fnz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, oidx, nx_ny, ny_nz

    if (nx * ny * nz <= threshold) then
        nx_ny = fnx * fny
        ny_nz = fny * fnz
        do x = x0, x0 + nx - 1
            obase = x * ny_nz + 1
            do z = z0, z0 + nz - 1
                ibase = z * nx_ny + 1 + x
                oidx = obase + z * fny
                do y = y0, y0 + ny - 1
                    out(oidx + y) = in(ibase + y * fnx)
                end do
            end do
        end do
    else if (nx >= ny .and. nx >= nz) then
        half = nx / 2
        !$omp task shared(in, out) final(half * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, half,      ny, nz, x0,        y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nx - half) * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, nx - half, ny, nz, x0 + half, y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (ny >= nz) then
        half = ny / 2
        !$omp task shared(in, out) final(nx * half * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, nx, half,      nz, x0, y0,        z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * (ny - half) * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, nx, ny - half, nz, x0, y0 + half, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nz / 2
        !$omp task shared(in, out) final(nx * ny * half <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, nx, ny, half,      x0, y0, z0,        fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * ny * (nz - half) <= gran) mergeable
        call GET_SUB_NAME(co_permute_forward)(in, out, nx, ny, nz - half, x0, y0, z0 + half, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_permute_forward)

subroutine GET_SUB_NAME(permute_backward_start_co)(in, out, dims, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product(dims) / MAX_TASKS)

!$omp parallel
!$omp single
    call GET_SUB_NAME(co_permute_backward_start)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, dims(1), dims(2), dims(3), threshold, gran)
!$omp end single
!$omp end parallel
end subroutine GET_SUB_NAME(permute_backward_start_co)

recursive subroutine GET_SUB_NAME(co_permute_backward_start)(in, out, nx, ny, nz, x0, y0, z0, fnx, fny, fnz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: fnx, fny, fnz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, in_base, out_base, fnx_fny, fnz_fny

    if (nx * ny * nz <= threshold) then
        fnx_fny = fnx * fny
        fnz_fny = fnz * fny
        do x = x0, x0 + nx - 1
            do y = y0, y0 + ny - 1
                in_base  = 1 + y * fnx + x
                out_base = 1 + x * fnz_fny + y * fnz
                do z = z0, z0 + nz - 1
                    out(out_base + z) = in(in_base + z * fnx_fny)
                end do
            end do
        end do
    else if (nx >= ny .and. nx >= nz) then
        half = nx / 2
        !$omp task shared(in, out) final(half * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, half,      ny, nz, x0,        y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nx - half) * ny * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, nx - half, ny, nz, x0 + half, y0, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (ny >= nz) then
        half = ny / 2
        !$omp task shared(in, out) final(nx * half * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, nx, half,      nz, x0, y0,        z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * (ny - half) * nz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, nx, ny - half, nz, x0, y0 + half, z0, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nz / 2
        !$omp task shared(in, out) final(nx * ny * half <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, nx, ny, half,      x0, y0, z0,        fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nx * ny * (nz - half) <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_start)(in, out, nx, ny, nz - half, x0, y0, z0 + half, fnx, fny, fnz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_permute_backward_start)

subroutine GET_SUB_NAME(permute_backward_end_pipelined_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product([locals(1), locals(2), locals(3)]) / MAX_TASKS)
!$omp parallel
!$omp single
    call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, &
        locals(1), locals(2), locals(3), locals(4), locals(5), locals(1), locals(3), threshold, gran)
!$omp end single
!$omp end parallel
end subroutine GET_SUB_NAME(permute_backward_end_pipelined_co)

recursive subroutine GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0, y0, z0, &
    nxx, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: nxx, nyy, nzz, din, dout, fnxx, fnzz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, nx_ny, nzz_nxx, iidx, oidx

    if (nxx * nyy * nzz <= threshold) then
        nx_ny = nx * ny
        nzz_nxx = fnzz * fnxx
        do z = z0, z0 + nzz - 1
            ibase = din + z * fnxx + 1
            obase = dout + z * nx_ny + 1
            do y = y0, y0 + nyy - 1
                iidx = ibase + y * nzz_nxx
                oidx = obase + y * nx
                do x = x0, x0 + nxx - 1
                    out(oidx + x) = in(iidx + x)
                end do
            end do
        end do
    else if (nxx >= nyy .and. nxx >= nzz) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0,        y0, z0, &
            half,       nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0 + half, y0, z0, &
            nxx - half, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (nyy >= nzz) then
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half * nzz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0, y0,        z0, &
            nxx, half,       nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) * nzz <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0, y0 + half, z0, &
            nxx, nyy - half, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nzz / 2
        !$omp task shared(in, out) final(nxx * nyy * half <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0, y0, z0,        &
            nxx, nyy, half,       din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * nyy * (nzz - half) <= gran) mergeable
        call GET_SUB_NAME(co_permute_backward_end_pipelined)(in, out, nx, ny, nz, x0, y0, z0 + half, &
            nxx, nyy, nzz - half, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_permute_backward_end_pipelined)

subroutine GET_SUB_NAME(permute_backward_end_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(:,:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: neighbor

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(permute_backward_end_pipelined_co)(in, out, dims, locals(:, neighbor), threshold)
    enddo
end subroutine GET_SUB_NAME(permute_backward_end_co)

subroutine GET_SUB_NAME(unpack_forward_pipelined_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    if ( size(dims) == 2 ) then
        gran = max(threshold, product([locals(1), locals(2)]) / MAX_TASKS)
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, dims(1), dims(2), 0, 0, &
            locals(1), locals(2), locals(4), locals(5), locals(2), threshold, gran)
    !$omp end single
    !$omp end parallel
    else
        gran = max(threshold, product([locals(1), locals(2), locals(3)]) / MAX_TASKS)
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, &
            locals(1), locals(2), locals(3), locals(4), locals(5), locals(1), locals(3), threshold, gran)
    !$omp end single
    !$omp end parallel
    endif
end subroutine GET_SUB_NAME(unpack_forward_pipelined_co)

recursive subroutine GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, nx, ny, x0, y0, &
    nxx, nyy, din, dout, fnyy, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny
    integer(int32), intent(in)    :: x0, y0
    integer(int32), intent(in)    :: nxx, nyy, din, dout, fnyy
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, half, iidx, oidx

    if (nxx * nyy <= threshold) then
        do y = y0, y0 + nyy - 1
            do x = x0, x0 + nxx - 1
                iidx = din + x * fnyy + y + 1
                oidx = dout + y * nx + x + 1
                out(oidx) = in(iidx)
            end do
        end do
    else if (nxx >= nyy) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, nx, ny, x0,        y0, half,       nyy, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, nx, ny, x0 + half, y0, nxx - half, nyy, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, nx, ny, x0, y0,        nxx, half,       din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined_2d)(in, out, nx, ny, x0, y0 + half, nxx, nyy - half, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_unpack_forward_pipelined_2d)

recursive subroutine GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0, y0, z0, &
    nxx, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: nxx, nyy, nzz, din, dout, fnxx, fnzz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, iidx, oidx, ny_nx, nzz_nxx

    if (nxx * nyy * nzz <= threshold) then
        ny_nx = ny * nx
        nzz_nxx = fnzz * fnxx
        do z = z0, z0 + nzz - 1
            ibase = din + z + 1
            obase = dout + z * ny_nx + 1
            do y = y0, y0 + nyy - 1
                iidx = ibase + y * nzz_nxx
                oidx = obase + y * nx
                do x = x0, x0 + nxx - 1
                    out(oidx + x) = in(iidx + x * fnzz)
                end do
            end do
        end do
    else if (nxx >= nyy .and. nxx >= nzz) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0,        y0, z0, &
            half,       nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0 + half, y0, z0, &
            nxx - half, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (nyy >= nzz) then
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0, y0,        z0, &
            nxx, half,       nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0, y0 + half, z0, &
            nxx, nyy - half, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nzz / 2
        !$omp task shared(in, out) final(nxx * nyy * half <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0, y0, z0,        &
            nxx, nyy, half,       din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * nyy * (nzz - half) <= gran) mergeable
        call GET_SUB_NAME(co_unpack_forward_pipelined)(in, out, nx, ny, nz, x0, y0, z0 + half, &
            nxx, nyy, nzz - half, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_unpack_forward_pipelined)

subroutine GET_SUB_NAME(unpack_forward_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(:,:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: neighbor

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_forward_pipelined_co)(in, out, dims, locals(:, neighbor), threshold)
    enddo
end subroutine GET_SUB_NAME(unpack_forward_co)

subroutine GET_SUB_NAME(unpack_backward_pipelined_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product([locals(1), locals(2), locals(3)]) / MAX_TASKS)
!$omp parallel
!$omp single
    call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, &
        locals(1), locals(2), locals(3), locals(4), locals(5), locals(2), locals(3), threshold, gran)
!$omp end single
!$omp end parallel
end subroutine GET_SUB_NAME(unpack_backward_pipelined_co)

recursive subroutine GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0, y0, z0, &
    nxx, nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: nxx, nyy, nzz, din, dout, fnyy, fnzz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, iidx, oidx, ny_nx, nzz_nyy

    if (nxx * nyy * nzz <= threshold) then
        ny_nx = ny * nx
        nzz_nyy = fnzz * fnyy
        do z = z0, z0 + nzz - 1
            ibase = din + 1 + z * fnyy
            obase = dout + 1 + z * ny_nx
            do y = y0, y0 + nyy - 1
                iidx = ibase + y
                oidx = obase + y * nx
                do x = x0, x0 + nxx - 1
                    out(oidx + x) = in(iidx + x * nzz_nyy)
                end do
            end do
        end do
    else if (nxx >= nyy .and. nxx >= nzz) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0,        y0, z0, &
            half,       nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0 + half, y0, z0, &
            nxx - half, nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (nyy >= nzz) then
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0, y0,        z0, &
            nxx, half,       nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) * nzz <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0, y0 + half, z0, &
            nxx, nyy - half, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nzz / 2
        !$omp task shared(in, out) final(nxx * nyy * half <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0, y0, z0,        &
            nxx, nyy, half,       din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * nyy * (nzz - half) <= gran) mergeable
        call GET_SUB_NAME(co_unpack_backward_pipelined)(in, out, nx, ny, nz, x0, y0, z0 + half, &
            nxx, nyy, nzz - half, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_unpack_backward_pipelined)

subroutine GET_SUB_NAME(unpack_backward_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(:,:)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: neighbor

    do neighbor = 1, size(locals, dim=2)
        call GET_SUB_NAME(unpack_backward_pipelined_co)(in, out, dims, locals(:, neighbor), threshold)
    enddo
end subroutine GET_SUB_NAME(unpack_backward_co)

subroutine GET_SUB_NAME(pack_forward_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    if ( size(dims) == 2 ) then
        gran = max(threshold, product([locals(1), locals(2)]) / MAX_TASKS)
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_pack_forward_2d)(in, out, dims(1), dims(2), 0, 0, &
            locals(1), locals(2), locals(4), locals(5), locals(2), threshold, gran)
    !$omp end single
    !$omp end parallel
    else
        gran = max(threshold, product([locals(1), locals(2), locals(3)]) / MAX_TASKS)
    !$omp parallel
    !$omp single
        call GET_SUB_NAME(co_pack_forward)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, &
            locals(1), locals(2), locals(3), locals(4), locals(5), locals(2), locals(3), threshold, gran)
    !$omp end single
    !$omp end parallel
    endif
end subroutine GET_SUB_NAME(pack_forward_co)

recursive subroutine GET_SUB_NAME(co_pack_forward_2d)(in, out, nx, ny, x0, y0, &
    nxx, nyy, din, dout, fnyy, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny
    integer(int32), intent(in)    :: x0, y0
    integer(int32), intent(in)    :: nxx, nyy, din, dout, fnyy
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, half, iidx, oidx

    if (nxx * nyy <= threshold) then
        do x = x0, x0 + nxx - 1
            do y = y0, y0 + nyy - 1
                iidx = din + y * nx + x + 1
                oidx = dout + x * fnyy + y + 1
                out(oidx) = in(iidx)
            end do
        end do
    else if (nxx >= nyy) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward_2d)(in, out, nx, ny, x0,        y0, half,       nyy, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward_2d)(in, out, nx, ny, x0 + half, y0, nxx - half, nyy, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward_2d)(in, out, nx, ny, x0, y0,        nxx, half,       din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward_2d)(in, out, nx, ny, x0, y0 + half, nxx, nyy - half, din, dout, fnyy, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_pack_forward_2d)

recursive subroutine GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0, y0, z0, &
    nxx, nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: nxx, nyy, nzz, din, dout, fnyy, fnzz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, oidx, nx_ny, nyy_nzz

    if (nxx * nyy * nzz <= threshold) then
        ! Base case: pack forward with write access
        nx_ny = nx * ny
        nyy_nzz = fnyy * fnzz
        do x = x0, x0 + nxx - 1
            obase = dout + x * nyy_nzz + 1
            do z = z0, z0 + nzz - 1
                ibase = din + z * nx_ny + 1 + x
                oidx = obase + z * fnyy
                do y = y0, y0 + nyy - 1
                    out(oidx + y) = in(ibase + y * nx)
                end do
            end do
        end do
    else if (nxx >= nyy .and. nxx >= nzz) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0,        y0, z0, &
            half,       nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0 + half, y0, z0, &
            nxx - half, nyy, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (nyy >= nzz) then
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0, y0,        z0, &
            nxx, half,       nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0, y0 + half, z0, &
            nxx, nyy - half, nzz, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nzz / 2
        !$omp task shared(in, out) final(nxx * nyy * half <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0, y0, z0,        &
            nxx, nyy, half,       din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * nyy * (nzz - half) <= gran) mergeable
        call GET_SUB_NAME(co_pack_forward)(in, out, nx, ny, nz, x0, y0, z0 + half, &
            nxx, nyy, nzz - half, din, dout, fnyy, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_pack_forward)

subroutine GET_SUB_NAME(pack_backward_co)(in, out, dims, locals, threshold)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: dims(:)
    integer(int32), intent(in)    :: locals(BUFFER_SPEC)
    integer(int32), intent(in)    :: threshold
    integer(int32) :: gran

    gran = max(threshold, product([locals(1), locals(2), locals(3)]) / MAX_TASKS)
!$omp parallel
!$omp single
    call GET_SUB_NAME(co_pack_backward)(in, out, dims(1), dims(2), dims(3), 0, 0, 0, &
        locals(1), locals(2), locals(3), locals(4), locals(5), locals(1), locals(3), threshold, gran)
!$omp end single
!$omp end parallel
end subroutine GET_SUB_NAME(pack_backward_co)

recursive subroutine GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0, y0, z0, &
    nxx, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
    BUFFER_TYPE,    intent(in)    :: in(BUFFER_SPEC)
    BUFFER_TYPE,    intent(inout) :: out(BUFFER_SPEC)
    integer(int32), intent(in)    :: nx, ny, nz
    integer(int32), intent(in)    :: x0, y0, z0
    integer(int32), intent(in)    :: nxx, nyy, nzz, din, dout, fnxx, fnzz
    integer(int32), intent(in)    :: threshold, gran
    integer(int32) :: x, y, z, half, ibase, obase, nx_ny, nzz_nxx, iidx, oidx

    if (nxx * nyy * nzz <= threshold) then
        ! Base case: pack backward with write access
        nx_ny = nx * ny
        nzz_nxx = fnzz * fnxx
        do y = y0, y0 + nyy - 1
            obase = dout + y * nzz_nxx + 1
            ibase = din + y * nx + 1
            do x = x0, x0 + nxx - 1
                oidx = obase + x * fnzz
                iidx = ibase + x
                do z = z0, z0 + nzz - 1
                    out(oidx + z) = in(iidx + z * nx_ny)
                end do
            end do
        end do
    else if (nxx >= nyy .and. nxx >= nzz) then
        half = nxx / 2
        !$omp task shared(in, out) final(half * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0,        y0, z0, &
            half,      nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final((nxx - half) * nyy * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0 + half, y0, z0, &
            nxx - half, nyy, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else if (nyy >= nzz) then
        half = nyy / 2
        !$omp task shared(in, out) final(nxx * half * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0, y0,        z0, &
            nxx, half,      nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * (nyy - half) * nzz <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0, y0 + half, z0, &
            nxx, nyy - half, nzz, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    else
        half = nzz / 2
        !$omp task shared(in, out) final(nxx * nyy * half <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0, y0, z0,        &
            nxx, nyy, half,      din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp task shared(in, out) final(nxx * nyy * (nzz - half) <= gran) mergeable
        call GET_SUB_NAME(co_pack_backward)(in, out, nx, ny, nz, x0, y0, z0 + half, &
            nxx, nyy, nzz - half, din, dout, fnxx, fnzz, threshold, gran)
        !$omp end task
        !$omp taskwait
    end if
end subroutine GET_SUB_NAME(co_pack_backward)


#undef STORAGE_BYTES
#undef BUFFER_TYPE
#undef PREC