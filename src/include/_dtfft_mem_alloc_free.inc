#ifndef STORAGE_BYTES
#error "STORAGE_BYTES not defined"
#endif
#ifndef BUFFER_TYPE
#error "BUFFER_TYPE not defined"
#endif
#ifndef ALLOC_1D
#error "ALLOC_1D not defined"
#endif
#ifndef ALLOC_2D
#error "ALLOC_2D not defined"
#endif
#ifndef ALLOC_3D
#error "ALLOC_3D not defined"
#endif
#ifndef FREE_1D
#error "FREE_1D not defined"
#endif
#ifndef FREE_2D
#error "FREE_2D not defined"
#endif
#ifndef FREE_3D
#error "FREE_3D not defined"
#endif

subroutine ALLOC_1D(self, alloc_size, ptr, lbound, error_code)
!! Allocates pointer of rank 1
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    integer(int64), intent(in) :: alloc_size
    !! Number of elements to allocate
    BUFFER_TYPE, pointer, intent(out) :: ptr(:)
    !! Allocated pointer
    integer(int32), optional, intent(in) :: lbound
    !! Lower boundary of pointer, default is 1
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    type(c_ptr) :: ptr_
    !! Temporary pointer
    BUFFER_TYPE, pointer :: ptr2(:)
    !! Temporary pointer
    integer(int32) :: ierr
    integer(int32) :: lbound_

    ptr => null()
    ptr_ = self%mem_alloc_ptr(alloc_size * STORAGE_BYTES, ierr)
    lbound_ = 1; if (present(lbound)) lbound_ = lbound
    if (ierr == DTFFT_SUCCESS) then
        call c_f_pointer(ptr_, ptr2, [alloc_size])
        ptr(lbound_:) => ptr2
    end if
    if (present(error_code)) error_code = ierr
end subroutine ALLOC_1D

subroutine ALLOC_2D(self, alloc_size, ptr, sizes, lbounds, error_code)
!! Allocates pointer of rank 2
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    integer(int64), intent(in) :: alloc_size
    !! Number of elements to allocate
    BUFFER_TYPE, pointer, intent(out) :: ptr(:, :)
    !! Allocated pointer
    integer(int32), intent(in) :: sizes(2)
    !! Number of elements in each direction
    integer(int32), optional, intent(in) :: lbounds(2)
    !! Lower boundaries of pointer, default is (1, 1)
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    type(c_ptr) :: ptr_
    !! Temporary pointer
    BUFFER_TYPE, pointer :: ptr2(:, :)
    !! Temporary pointer
    integer(int32) :: ierr
    integer(int32) :: lbounds_(2)

    ptr => null()
    ptr_ = self%mem_alloc_ptr(alloc_size * STORAGE_BYTES, error_code=ierr)
    lbounds_ = 1; if (present(lbounds)) lbounds_ = lbounds
    if (ierr == DTFFT_SUCCESS) then
        call c_f_pointer(ptr_, ptr2, sizes)
        ptr(lbounds_(1):, lbounds_(2):) => ptr2
    end if
    if (present(error_code)) error_code = ierr
end subroutine ALLOC_2D

subroutine ALLOC_3D(self, alloc_size, ptr, sizes, lbounds, error_code)
!! Allocates pointer of rank 3
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    integer(int64), intent(in) :: alloc_size
    !! Number of elements to allocate
    BUFFER_TYPE, pointer, intent(out) :: ptr(:, :, :)
    !! Allocated pointer
    integer(int32), intent(in) :: sizes(3)
    !! Number of elements in each direction
    integer(int32), optional, intent(in) :: lbounds(3)
    !! Lower boundaries of pointer, default is (1, 1)
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    type(c_ptr) :: ptr_
    !! Temporary pointer
    BUFFER_TYPE, pointer :: ptr2(:, :, :)
    !! Temporary pointer
    integer(int32) :: ierr
    integer(int32) :: lbounds_(3)

    ptr => null()
    ptr_ = self%mem_alloc_ptr(alloc_size * STORAGE_BYTES, error_code=ierr)
    lbounds_ = 1; if (present(lbounds)) lbounds_ = lbounds
    if (ierr == DTFFT_SUCCESS) then
        call c_f_pointer(ptr_, ptr2, sizes)
        ptr(lbounds_(1):, lbounds_(2):, lbounds_(3):) => ptr2
    end if
    if (present(error_code)) error_code = ierr
end subroutine ALLOC_3D

subroutine FREE_1D(self, ptr, error_code)
!! Frees previously allocated memory specific for this plan
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    BUFFER_TYPE, target, intent(inout) :: ptr(:)
    !! Pointer allocated with [[dtfft_plan_t(type):mem_alloc]]
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    call self%mem_free_ptr(c_loc(ptr), error_code)
end subroutine FREE_1D

subroutine FREE_2D(self, ptr, error_code)
!! Frees previously allocated memory specific for this plan
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    BUFFER_TYPE, target, intent(inout) :: ptr(:, :)
    !! Pointer allocated with [[dtfft_plan_t(type):mem_alloc]]
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    call self%mem_free_ptr(c_loc(ptr), error_code)
end subroutine FREE_2D

subroutine FREE_3D(self, ptr, error_code)
!! Frees previously allocated memory specific for this plan
    class(dtfft_plan_t), intent(inout) :: self
    !! Abstract plan
    BUFFER_TYPE, target, intent(inout) :: ptr(:, :, :)
    !! Pointer allocated with [[dtfft_plan_t(type):mem_alloc]]
    integer(int32), optional, intent(out) :: error_code
    !! Optional error code returned to user
    call self%mem_free_ptr(c_loc(ptr), error_code)
end subroutine FREE_3D

#undef STORAGE_BYTES
#undef BUFFER_TYPE
#undef ALLOC_1D
#undef ALLOC_2D
#undef ALLOC_3D
#undef FREE_1D
#undef FREE_2D
#undef FREE_3D