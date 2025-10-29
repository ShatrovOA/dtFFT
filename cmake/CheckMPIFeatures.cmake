include(CheckSourceCompiles)
include(CheckFortranSourceCompiles)

# Function to check support for MPI persistent collectives
function(check_persistent_collectives MPI_INCLUDES MPI_LIBS MPI_MOD)
  # Set required includes and libraries for MPI
  set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDES})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBS})

  # Check if compiler supports MPI_Alltoall_init (persistent collectives)
  if( MPI_MOD STREQUAL "mpi")
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
  else()
  check_fortran_source_compiles(
    "program test
    use mpi_f08
    implicit none
    integer :: send, recv, ierr
    type(MPI_Request) :: request
    call MPI_Alltoall_init(send, 1, MPI_INTEGER, recv, 1, MPI_INTEGER, MPI_COMM_WORLD, MPI_INFO_NULL, request, ierr)
    end program"
    HAVE_PERSISTENT_COLLECTIVES
    SRC_EXT .F90
  )
  endif()
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

    call MPI_Isend(send_buf, send_count, MPI_REAL, 0, 0, MPI_COMM_WORLD, request, ierr)
  end program"
  HAVE_MPI_INT64
  SRC_EXT .F90)
  set(HAVE_MPI_INT64 ${HAVE_MPI_INT64} PARENT_SCOPE)

  # Reset required includes and libraries to avoid affecting other checks
  set(CMAKE_REQUIRED_INCLUDES "" PARENT_SCOPE)
  set(CMAKE_REQUIRED_LIBRARIES "" PARENT_SCOPE)
endfunction()

function(check_ompi_fix_required MPI_INCLUDES)
  # Set required includes and libraries for MPI
  set(file "${PROJECT_BINARY_DIR}/detect_ompi_version.c")
  file(WRITE ${file} "
      #include <stdio.h>
      #include <mpi.h>
      int main() {
        printf(\"%d\\n\", OMPI_MAJOR_VERSION);
        return OMPI_MAJOR_VERSION < 5 ? 1 : 0;
      }
    ")
  try_run(OMPI_LESS_5 compile_result ${PROJECT_BINARY_DIR} ${file}
          CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MPI_INCLUDES}"
          COMPILE_OUTPUT_VARIABLE compile_output
          RUN_OUTPUT_VARIABLE run_output)
  if(NOT compile_result)
    SET(OMPI_FIX_REQUIRED FALSE PARENT_SCOPE)
  else()
    if (OMPI_LESS_5)
      SET(OMPI_FIX_REQUIRED TRUE PARENT_SCOPE)
    else()
      SET(OMPI_FIX_REQUIRED FALSE PARENT_SCOPE)
    endif()
  endif()
endfunction(check_ompi_fix_required)

function(check_mpich_fix_required MPI_INCLUDES)
  # Set required includes and libraries for MPI
  set(file "${PROJECT_BINARY_DIR}/detect_mpich_version.c")
  file(WRITE ${file} "
      #include <stdio.h>
      #include <mpi.h>
      int main() {
        const int correct_version = MPICH_CALC_VERSION(4, 1, 0, 0, 0);
        printf(\"Valid version = %d\\n\", correct_version);
        printf(\"Current version = %d\\n\", MPICH_NUMVERSION);
        return correct_version >= MPICH_NUMVERSION ? 1 : 0;
      }
    ")
  try_run(MPICH_LESS_41 compile_result ${PROJECT_BINARY_DIR} ${file}
          CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MPI_INCLUDES}"
          COMPILE_OUTPUT_VARIABLE compile_output
          RUN_OUTPUT_VARIABLE run_output)
  if(NOT compile_result)
    SET(MPICH_FIX_REQUIRED FALSE PARENT_SCOPE)
  else()
    if (MPICH_LESS_41)
      SET(MPICH_FIX_REQUIRED TRUE PARENT_SCOPE)
    else()
      SET(MPICH_FIX_REQUIRED FALSE PARENT_SCOPE)
    endif()
  endif()
endfunction(check_mpich_fix_required)