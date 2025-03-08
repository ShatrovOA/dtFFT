include(CheckFortranSourceCompiles)

function(check_assumed_rank_and_type)
# Check if compiler supports assumed-rank and assumed-type (Fortran 2018)
  check_fortran_source_compiles(
    "program test
    contains
    subroutine test_sub(a)
    type(*), intent(inout) :: a(..)
    end subroutine
    end program"
    HAVE_ASSUMED_RANK_AND_TYPE
    SRC_EXT .F90
  )
  # Export variable to parent scope
  set(HAVE_ASSUMED_RANK_AND_TYPE ${HAVE_ASSUMED_RANK_AND_TYPE} PARENT_SCOPE)
endfunction()


function(check_block_statement)
# Check if compiler supports block statements (Fortran 2008)
  check_fortran_source_compiles(
    "program test
    integer :: i = 2
    print *, mod(i,i)
    block
      integer :: j
      j = i + 1
      print *, mod(j,i)
    endblock
    end program"
    HAVE_BLOCK_STATEMENT
    SRC_EXT .F90
  )
  # Export variable to parent scope
  set(HAVE_BLOCK_STATEMENT ${HAVE_BLOCK_STATEMENT} PARENT_SCOPE)
endfunction()