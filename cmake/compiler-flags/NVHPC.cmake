set(CMAKE_Fortran_FLAGS
  "${CMAKE_Fortran_FLAGS} -Mbackslash -traceback -Mcontiguous -Mnodepchk"
)
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -C -Mchkstk -Mbounds")
set(CMAKE_Fortran_FLAGS_RELEASE
  "${CMAKE_Fortran_FLAGS_RELEASE} -Ofast -Mnobounds"
)
