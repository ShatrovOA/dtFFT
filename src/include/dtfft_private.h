/*
  Copyright (c) 2021, Oleg Shatrov
  All rights reserved.
  This file is part of dtFFT library.

  dtFFT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  dtFFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef DTFFT_PRIVATE_H
#define DTFFT_PRIVATE_H

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_FORWARD (-1)
#define FFT_BACKWARD (+1)

#define CHECK_CALL( func, ierr ) \
  ierr = func;  if ( ierr /= DTFFT_SUCCESS ) return

#define CHECK_INTERNAL_CALL( func ) CHECK_CALL(func, __FUNC__)

#ifdef __cplusplus
}
#endif

#endif