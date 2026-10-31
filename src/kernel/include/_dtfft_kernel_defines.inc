#define GET_SUB_NAME(name) CONCAT(name,PREC)
#define GET_BLOCK_SUB_NAME(name) CONCAT(GET_SUB_NAME(name),_block_)
#define GET_BLOCK_SUB_NAME_EXPANDED(name) GET_BLOCK_SUB_NAME(name)
#define GET_BLOCK_KERNEL_NAME(name) CONCAT(GET_BLOCK_SUB_NAME_EXPANDED(name),BLOCK_SIZE)
#define GET_CO_SUB_NAME(name) CONCAT(GET_SUB_NAME(name),_co)

#define CONVERT_POINTER(in, out) call c_f_pointer(in, out, [product(self%dims)])