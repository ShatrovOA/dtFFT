set(_dtfft_gnu_fortran_debug_flags
  -Og -g -Waliasing -Wampersand -Wconversion -Wsurprising
  -Wintrinsics-std -Wtabs -Wintrinsic-shadow -Wline-truncation -Wtarget-lifetime
  -Winteger-division -Wreal-q-constant -Wunused -Wrealloc-lhs-all -Wdo-subscript
  -Wundefined-do-loop -Wextra -ggdb -fopt-info -fopt-info-optall-optimized
  -fbacktrace -fdump-fortran-optimized -ftree-vectorize -Wimplicit-interface
  -Wunused-parameter -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow
  -ffpe-summary=none -Warray-bounds -Wimplicit-procedure -Wunderflow
  -Wuninitialized -fimplicit-none -fdump-core -finit-real=nan -fsanitize=address
)
string(JOIN " " _dtfft_gnu_fortran_debug_flags ${_dtfft_gnu_fortran_debug_flags})

set(CMAKE_Fortran_FLAGS
  "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -std=f2018 -pedantic-errors -fbacktrace -Wall -fsanitize=address"
)
set(CMAKE_Fortran_FLAGS_DEBUG
  "${CMAKE_Fortran_FLAGS_DEBUG} ${_dtfft_gnu_fortran_debug_flags}"
)
