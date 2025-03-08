include(CheckFortranSourceCompiles)

# Function to check support for MPI persistent collectives
function(check_persistent_collectives MPI_INCLUDES MPI_LIBS)
  # Set required includes and libraries for MPI
  set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDES})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBS})

  # Check if compiler supports MPI_Alltoall_init (persistent collectives)
  check_fortran_source_compiles(
    "program test
    use mpi
    implicit none
    integer :: send, recv, ierr, request
    call MPI_Alltoall_init(send, 1, MPI_INTEGER, recv, 1, MPI_INTEGER, MPI_COMM_WORLD, MPI_INFO_NULL, request, ierr)
    end program"
    HAVE_PERSISTENT_COLLECTIVES
    SRC_EXT .F90
  )
  set(HAVE_PERSISTENT_COLLECTIVES ${HAVE_PERSISTENT_COLLECTIVES} PARENT_SCOPE)

  # Reset required includes and libraries to avoid affecting other checks
  set(CMAKE_REQUIRED_INCLUDES "" PARENT_SCOPE)
  set(CMAKE_REQUIRED_LIBRARIES "" PARENT_SCOPE)
endfunction()

# Function to check support for mpi_f08 module integer64 interfaces
function(check_int64_supported MPI_INCLUDES MPI_LIBS)
  # Set required includes and libraries for MPI
  set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDES})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBS})
  check_fortran_source_compiles("
  program test
  use mpi_f08
  implicit none
    integer(MPI_COUNT_KIND) :: send_count
    real, allocatable :: send_buf(:)
    integer :: ierr
    type(MPI_Request) :: request

    call MPI_Isend(send_buf, send_count, MPI_REAL, 0, 0, MPI_COMM_WORLD, request, ierr)"
  HAVE_MPI_INT64
  SRC_EXT .F90)
  set(HAVE_MPI_INT64 ${HAVE_MPI_INT64} PARENT_SCOPE)

  # Reset required includes and libraries to avoid affecting other checks
  set(CMAKE_REQUIRED_INCLUDES "" PARENT_SCOPE)
  set(CMAKE_REQUIRED_LIBRARIES "" PARENT_SCOPE)
endfunction()