#ifndef MACRO_NAME
#error "MACRO_NAME not defined"
#endif

module subroutine MACRO_NAME(execute)(self, in, out, neighbor)
!! Executes kernel based on its type and access mode, real32 version
    class(kernel_host),         intent(in)  :: self         !! Host kernel class
    type(c_ptr),                intent(in)  :: in           !! Source host-allocated buffer
    type(c_ptr),                intent(in)  :: out          !! Target host-allocated buffer
    integer(int32),   optional, intent(in)  :: neighbor     !! Source rank for pipelined unpacking
end subroutine MACRO_NAME(execute)
