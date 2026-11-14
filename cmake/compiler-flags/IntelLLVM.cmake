set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -free")
set(CMAKE_Fortran_FLAGS_DEBUG
  "${CMAKE_Fortran_FLAGS_DEBUG} -ftrapuv -debug all -gen-interfaces -traceback -warn all"
)
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast")
