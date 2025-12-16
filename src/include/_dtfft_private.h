/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
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

#define CONF_FFT_FORWARD (-1)
#define CONF_FFT_BACKWARD (+1)

#define CHECK_CALL( func, ierr ) \
  ierr = func;  if ( ierr /= DTFFT_SUCCESS ) return

#define CHECK_INTERNAL_CALL( func ) CHECK_CALL(func, __FUNC__)

#ifdef DTFFT_DEBUG
# define WRITE_DEBUG(msg) if(get_conf_log_enabled()) call write_message(output_unit, msg, "dtFFT DEBUG: ")
#else
# define WRITE_DEBUG(msg)
#endif

#define WRITE_REPORT(msg) call write_message(output_unit, msg, "dtFFT: ")
#define WRITE_INFO(msg) if(get_conf_log_enabled()) WRITE_REPORT(msg)
#define WRITE_WARN(msg) if(get_conf_log_enabled()) call write_message(error_unit, msg, "dtFFT WARNING: ")
#define WRITE_ERROR(msg) call write_message(error_unit, msg, "dtFFT ERROR: ")

#define ALLOC_ALIGNMENT 16

#ifdef __GFORTRAN__
#define PASTE(a) a
#define CONCAT(a,b) PASTE(a)b
#else
#define PASTE(a,b) a ## b
#define CONCAT(a,b) PASTE(a,b)
#endif

#define INTERNAL_ERROR(message) error stop "dtFFT Internal Error: "//message

#define CHECK_INPUT_PARAMETER(param, check_func, code)            \
  if ( .not.check_func(param)) then;                              \
    __FUNC__ = code;                                              \
    return;                                                       \
  endif

#define CHECK_ERROR                                 \
  if ( ierr /= DTFFT_SUCCESS ) then;                \
    if ( present( error_code ) ) error_code = ierr; \
    return;                                         \
  endif


#define CHECK_ERROR_AND_RETURN                      \
  if ( ierr /= DTFFT_SUCCESS ) then;                \
    if ( present( error_code ) ) error_code = ierr; \
    WRITE_ERROR( dtfft_get_error_string(ierr) );    \
    return;                                         \
  endif

#define CHECK_ERROR_AND_RETURN_AGG(comm)                          \
  block;                                                          \
    integer(int32) :: mpi_ierr;                                   \
    ALL_REDUCE(error_code, MPI_INTEGER, MPI_MAX, comm, mpi_ierr); \
    if ( error_code /= DTFFT_SUCCESS ) then;                      \
      WRITE_ERROR( dtfft_get_error_string(error_code) );          \
      return;                                                     \
    endif;                                                        \
  end block

#define CHECK_ERROR_AND_RETURN_NO_MSG               \
  if ( ierr /= DTFFT_SUCCESS ) then;                \
    if ( present( error_code ) ) error_code = ierr; \
    return;                                         \
  endif

#define CHECK_OPTIONAL_CALL( func )                 \
  ierr = func;                                      \
  CHECK_ERROR_AND_RETURN

#define MAKE_EQ_FUN(datatype, name)                         \
  pure elemental function name(left, right) result(res);    \
    type(datatype), intent(in) :: left;                     \
    type(datatype), intent(in) :: right;                    \
    logical :: res;                                         \
    res = left%val == right%val;                            \
  end function name

#define MAKE_NE_FUN(datatype, name)                         \
  pure elemental function name(left, right) result(res);    \
    type(datatype), intent(in) :: left;                     \
    type(datatype), intent(in) :: right;                    \
    logical :: res;                                         \
    res = left%val /= right%val;                            \
  end function name

#define MAKE_VALID_FUN(type, name, valid_values)            \
  pure elemental function name(param) result(res);          \
    type, intent(in)  :: param;                             \
    logical :: res;                                         \
    res = any(param == valid_values);                       \
  end function name

#define MAKE_VALID_FUN_DTYPE(datatype, name, valid_values)  \
  MAKE_VALID_FUN(type(datatype), name, valid_values)

#ifdef DTFFT_DEBUG
#define BUFFER_SPEC :
#else
#define BUFFER_SPEC *
#endif

#ifdef __cplusplus
}
#endif

#endif