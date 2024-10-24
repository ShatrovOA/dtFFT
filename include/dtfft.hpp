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

/**
 * @file dtfft.hpp
 * @author Oleg Shatrov
 * @date 2024
 * @brief File containing C++ API functions of dtFFT Library
 */

#ifndef DTFFT_HPP
#define DTFFT_HPP

#include <mpi.h>
#include <vector>
#include <dtfft.h>
#include <stdexcept>

namespace dtfft
{
  namespace core
  {
    class dtfft_core
    {
      protected:
        dtfft_plan _plan;

      public:

/** \brief Plan execution without optional auxiliary vector
  *
  * \param[inout]   in              Incoming vector
  * \param[out]     out             Result vector
  * \param[in]      transpose_type  Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  *
  * \return Status code of method execution
*/
        template<typename T1, typename T2>
        int
        execute(std::vector<T1> &in, std::vector<T2> &out, const int transpose_type)
        {return execute(in.data(), out.data(), transpose_type, NULL);}



/** \brief Plan execution with optional auxiliary vector
  *
  * \param[inout]   in              Incoming vector
  * \param[out]     out             Result vector
  * \param[in]      transpose_type  Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  * \param[inout]   aux             Optional auxiliary vector
  *
  * \return Status code of method execution
*/
        template<typename T1, typename T2, typename T3>
        int
        execute(std::vector<T1> &in, std::vector<T2> &out, const int transpose_type, std::vector<T3> &aux)
        {return execute(in.data(), out.data(), transpose_type, aux.data());}



/** \brief Plan execution without auxiliary buffer using C-style pointers instead of vectors
  *
  * \param[inout]   in              Incoming buffer
  * \param[out]     out             Result buffer
  * \param[in]      transpose_type  Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  *
  * \return Status code of method execution
*/
        int
        execute(void *in, void *out, const int transpose_type)
        {return execute(in, out, transpose_type, NULL);}



/** \brief Plan execution with auxiliary buffer using C-style pointers instead of vectors
  *
  * \param[inout]   in              Incoming buffer
  * \param[out]     out             Result buffer
  * \param[in]      transpose_type  Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  * \param[inout]   aux             Optional auxiliary buffer
  *
  * \return Status code of method execution
*/
        int
        execute(void *in, void *out, const int transpose_type, void *aux)
        {return dtfft_execute(_plan, in, out, transpose_type, aux);}



/** \brief Transpose data in single dimension using vector, e.g. X align -> Y align
  * \attention `in` and `out` cannot be the same vectors
  *
  * \param[inout]   in              Incoming vector
  * \param[out]     out             Transposed vector
  * \param[in]      transpose_type  Type of transpose:
  *                                   - `DTFFT_TRANSPOSE_X_TO_Y`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_X`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_Y` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_X_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_X` (3d plan only)
  *
  * \return Status code of method execution
*/
        template<typename T1, typename T2>
        int
        transpose(std::vector<T1>const &in, std::vector<T2> &out, const int transpose_type)
        {return transpose(in.data(), out.data(), transpose_type);}



/** \brief Transpose data in single dimension using C-style pointers, e.g. X align -> Y align
  * \attention `in` and `out` cannot be the same pointers
  *
  * \param[inout]   in              Incoming vector
  * \param[out]     out             Transposed vector
  * \param[in]      transpose_type  Type of transpose:
  *                                   - `DTFFT_TRANSPOSE_X_TO_Y`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_X`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_Y` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_X_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_X` (3d plan only)
  *
  * \return Status code of method execution
*/
        int
        transpose(const void *in, void *out, const int transpose_type)
        {return dtfft_transpose(_plan, in, out, transpose_type);}



/** \brief Wrapper around `get_local_sizes`
  *
  * \param[out]     alloc_size      Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers:
  *                                   - C2C plan: 2 * `alloc_size` * sizeof(double/float) or `alloc_size` * sizeof(dtfft_complex/dtfftf_complex)
  *                                   - R2R plan: `alloc_size` * sizeof(double/float)
  *                                   - R2C plan: `alloc_size` * sizeof(double/float)
  *
  * \return Status code of method execution
*/
        int
        get_alloc_size(size_t *alloc_size)
        {return dtfft_get_alloc_size(_plan, alloc_size);}



/** \brief Get grid decomposition information. Results may differ on different MPI processes
  * 
  * \param[out]   in_starts             Starts of local portion of data in 'real' space in reversed order
  * \param[out]   in_counts             Sizes  of local portion of data in 'real' space in reversed order
  * \param[out]   out_starts            Starts of local portion of data in 'fourier' space in reversed order
  * \param[out]   out_counts            Sizes  of local portion of data in 'fourier' space in reversed order
  * \param[out]   alloc_size            Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers:
  *                                       - C2C plan: 2 * `alloc_size` * sizeof(double/float) or `alloc_size` * sizeof(dtfft_complex/dtfftf_complex)
  *                                       - R2R plan: `alloc_size` * sizeof(double/float)
  *                                       - R2C plan: `alloc_size` * sizeof(double/float)
  *
  * \return Status code of method execution
*/
        int
        get_local_sizes(std::vector<int>&in_starts, std::vector<int>&in_counts, std::vector<int>&out_starts, std::vector<int>&out_counts, size_t *alloc_size)
        {return get_local_sizes(in_starts.data(), in_counts.data(), out_starts.data(), out_counts.data(), alloc_size);}



/** \brief Get grid decomposition information. Results may differ on different MPI processes
  *
  * \param[out]   in_starts             Starts of local portion of data in 'real' space in reversed order
  * \param[out]   in_counts             Sizes  of local portion of data in 'real' space in reversed order
  * \param[out]   out_starts            Starts of local portion of data in 'fourier' space in reversed order
  * \param[out]   out_counts            Sizes  of local portion of data in 'fourier' space in reversed order
  * \param[out]   alloc_size            Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers:
  *                                       - C2C plan: 2 * `alloc_size` * sizeof(double/float) or `alloc_size` * sizeof(dtfft_complex/dtfftf_complex)
  *                                       - R2R plan: `alloc_size` * sizeof(double/float)
  *                                       - R2C plan: `alloc_size` * sizeof(double/float)
  *
  * \return Status code of method execution
*/
        int
        get_local_sizes(int *in_starts=NULL, int *in_counts=NULL, int *out_starts=NULL, int *out_counts=NULL, size_t *alloc_size=NULL)
        {return dtfft_get_local_sizes(_plan, in_starts, in_counts, out_starts, out_counts, alloc_size);}



/** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
*/
        int destroy()
        {return dtfft_destroy(&_plan);}



/** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
*/
        ~dtfft_core() {destroy();}
    };
  }

  class PlanC2C final: public core::dtfft_core
  {
    public:
/** \brief Complex-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    dims                  Vector with global dimensions in reversed order.
  *                                     `dims.size()` must be 2 or 3
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
*/
      PlanC2C(
        const std::vector<int> &dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ):PlanC2C(dims.size(), dims.data(), comm, precision, effort_flag, executor_type) {}



/** \brief Complex-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    dims                  Vector with global dimensions in reversed order.
  *                                     `dims.size()` must be 2 or 3
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
*/
      PlanC2C(
        const std::vector<int> &dims,
        const int precision,
        const int executor_type
      ): PlanC2C(dims.size(), dims.data(), MPI_COMM_WORLD, precision, DTFFT_ESTIMATE, executor_type) {}



/** \brief Complex-to-Complex Plan constructor using C-style arguments. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
*/
      PlanC2C(
        const int ndims,
        const int *dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ){
        int error_code = dtfft_create_plan_c2c(ndims, dims, comm, precision, effort_flag, executor_type, &_plan);
        if ( error_code != DTFFT_SUCCESS)
          throw std::runtime_error(dtfft_get_error_string(error_code));
      }
  };

  class PlanR2C final: public core::dtfft_core
  {
    public:
/** \brief Real-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    dims                  Vector with global dimensions in reversed order.
  *                                     `dims.size()` must be 2 or 3
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
  *
  * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C or R2R plans instead
  *
  * \throws std::runtime_error In case error occurs during plan creation
*/
      PlanR2C(
        const std::vector<int> &dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ):PlanR2C(dims.size(), dims.data(), comm, precision, effort_flag, executor_type) {}

/** \brief Real-to-Complex Plan constructor using C-style arguments. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
  *
  * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C or R2R plans instead
  *
  * \throws std::runtime_error In case error occurs during plan creation
*/
      PlanR2C(
        const int ndims,
        const int *dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ){
        int error_code = dtfft_create_plan_r2c(ndims, dims, comm, precision, effort_flag, executor_type, &_plan);
        if ( error_code != DTFFT_SUCCESS)
          throw std::runtime_error(dtfft_get_error_string(error_code));
      }
  };

  class PlanR2R final: public core::dtfft_core
  {
    public:
/** \brief Real-to-Real Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    dims                  Vector with global dimensions in reversed order.
  *                                     `dims.size()` must be 2 or 3
  * \param[in]    in_kinds              Real FFT kinds in reversed order, forward transform.
  *                                     `in_kinds.size()` must be equal to `dims.size()` for non transpose plans.
  *                                     Can be empty vector if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    out_kinds             Real FFT kinds in reversed order, backward transform
  *                                     `out_kinds.size()` must be equal to `dims.size()` for non transpose plans.
  *                                     Can be empty vector if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *
  * \throws std::runtime_error In case error occurs during plan creation
*/
      PlanR2R(
        const std::vector<int> &dims,
        const std::vector<int> &kinds=std::vector<int>(),
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ):PlanR2R(dims.size(), dims.data(), kinds.data(), comm, precision, effort_flag, executor_type) {}



/** \brief Real-to-Real Transpose only Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    dims                  Vector with global dimensions in reversed order.
  *                                     `dims.size()` must be 2 or 3
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  *
  * \throws std::runtime_error In case error occurs during plan creation
*/
      PlanR2R(
        const std::vector<int> &dims,
        const int precision
      ):PlanR2R(dims.size(), dims.data(), NULL, MPI_COMM_WORLD, precision, DTFFT_ESTIMATE, DTFFT_EXECUTOR_NONE) {}

/** \brief Real-to-Real Plan constructor using C-style arguments. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    kinds                 Buffer of size `ndims` with Real FFT kinds in reversed order.
  *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *
  * \throws std::runtime_error In case error occurs during plan creation
*/
      PlanR2R(
        const int ndims,
        const int *dims,
        const int *kinds=NULL,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_NONE
      ) {
        int error_code = dtfft_create_plan_r2r(ndims, dims, kinds, comm, precision, effort_flag, executor_type, &_plan);
        if ( error_code != DTFFT_SUCCESS)
          throw std::runtime_error(dtfft_get_error_string(error_code));
      }
  };
}
// DTFFT_HPP
#endif