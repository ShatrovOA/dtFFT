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

#ifndef DTFFT_HPP
#define DTFFT_HPP

#include <mpi.h>
#include <vector>
#include <dtfft.h>


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
        * \attention Inplace execution is possible only in these plans
        *     - R2R 3d
        *     - C2C 3d
        *     - R2C 2d
        * 
        * \param[inout]   in              Incoming vector
        * \param[out]     out             Result vector
        * \param[in]      transpose_type  Type of transform: `DTFFT_TRANSPOSE_OUT` or `DTFFT_TRANSPOSE_IN`
      */
        template<typename T1, typename T2>
        void
        execute(
          std::vector<T1> &in,
          std::vector<T2> &out,
          const int transpose_type
        )
        {execute(in.data(), out.data(), transpose_type, NULL);}
      /** \brief Plan execution with optional auxiliary vector
        * \attention Inplace execution is possible only in these plans:
        *     - R2R 3d
        *     - C2C 3d
        *     - R2C 2d
        * 
        * \param[inout]   in              Incoming vector
        * \param[out]     out             Result vector
        * \param[in]      transpose_type  Type of transform: `DTFFT_TRANSPOSE_OUT` or `DTFFT_TRANSPOSE_IN`
        * \param[inout]   aux             Optional auxiliary vector
      */
        template<typename T1, typename T2, typename T3>
        void
        execute(
          std::vector<T1> &in,
          std::vector<T2> &out,
          const int transpose_type,
          std::vector<T3> &aux
        )
        {execute(in.data(), out.data(), transpose_type, aux.data());}
      /** \brief Plan execution without auxiliary buffer using C-style pointers instead of vectors
        * \attention Inplace execution is possible only in these plans:
        *     - R2R 3d
        *     - C2C 3d
        *     - R2C 2d
        * 
        * \param[inout]   in              Incoming buffer
        * \param[out]     out             Result buffer
        * \param[in]      transpose_type  Type of transform: `DTFFT_TRANSPOSE_OUT` or `DTFFT_TRANSPOSE_IN`
      */
        void
        execute(
          void *in,
          void *out,
          const int transpose_type
        )
        {execute(in, out, transpose_type, NULL);}
      /** \brief Plan execution with auxiliary buffer using C-style pointers instead of vectors
        * \attention Inplace execution is possible only in these plans:
        *     - R2R 3d
        *     - C2C 3d
        *     - R2C 2d
        * 
        * \param[inout]   in              Incoming buffer
        * \param[out]     out             Result buffer
        * \param[in]      transpose_type  Type of transform: `DTFFT_TRANSPOSE_OUT` or `DTFFT_TRANSPOSE_IN`
        * \param[inout]   aux             Optional auxiliary buffer
      */
        void
        execute(
          void *in,
          void *out,
          const int transpose_type,
          void *aux
        )
        {dtfft_execute(_plan, in, out, transpose_type, aux);}

      /** \brief Transpose data in single dimension using vector, e.g. X align -> Y align 
        * \attention `in` and `out` cannot be the same vectors
        * 
        * \param[inout]   in              Incoming vector
        * \param[out]     out             Transposed vector
        * \param[in]      transpose_type  Type of transpose: `DTFFT_TRANSPOSE_X_TO_Y`, `DTFFT_TRANSPOSE_Y_TO_X`
        *                                 `DTFFT_TRANSPOSE_Y_TO_Z` or `DTFFT_TRANSPOSE_Z_TO_Y`
      */
        template<typename T1, typename T2>
        void
        transpose(
          std::vector<T1>const &in,
          std::vector<T2> &out,
          const int transpose_type
        )
        {transpose(in.data(), out.data(), transpose_type);}
      /** \brief Transpose data in single dimension using C-style pointers, e.g. X align -> Y align 
        * \attention `in` and `out` cannot be the same pointers
        * 
        * \param[inout]   in              Incoming vector
        * \param[out]     out             Transposed vector
        * \param[in]      transpose_type  Type of transpose: `DTFFT_TRANSPOSE_X_TO_Y`, `DTFFT_TRANSPOSE_Y_TO_X`
        *                                 `DTFFT_TRANSPOSE_Y_TO_Z` or `DTFFT_TRANSPOSE_Z_TO_Y`
      */
        void
        transpose(
          void *in,
          void *out,
          const int transpose_type
        )
        {dtfft_transpose(_plan, in, out, transpose_type);}

      /** \brief Wrapper around `get_local_sizes`
        * \return Minimum number of elements needs to be allocated
        *
        * Minimum number of bytes that needs allocation:
        * - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(complex<double/float>)
        * - R2R plan: alloc_size * sizeof(double/float)
        * - R2C plan: alloc_size * sizeof(double/float)
      */
        size_t
        get_alloc_size()
        {return dtfft_get_alloc_size(_plan);}

      /** \brief Get grid decomposition information. Results may differ on different MPI processes
        *
        * Minimum number of bytes that needs allocation:
        * - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(complex<double/float>)
        * - R2R plan: alloc_size * sizeof(double/float)
        * - R2C plan: alloc_size * sizeof(double/float)
        * \return Minimum number of elements needs to be allocated
        * 
        * 
        * \param[out]   in_starts             Starts of local portion of data in 'real' space in reversed order
        * \param[out]   in_counts             Sizes  of local portion of data in 'real' space in reversed order
        * \param[out]   out_starts            Starts of local portion of data in 'fourier' space in reversed order
        * \param[out]   out_counts            Sizes  of local portion of data in 'fourier' space in reversed order
        *
      */
        size_t
        get_local_sizes(
          std::vector<int>&in_starts,
          std::vector<int>&in_counts,
          std::vector<int>&out_starts,
          std::vector<int>&out_counts
        ){return get_local_sizes(in_starts.data(), in_counts.data(), out_starts.data(), out_counts.data());}

      /** \brief Get grid decomposition information. Results may differ on different MPI processes
        *
        * Minimum number of bytes that needs allocation:
        * - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(complex<double/float>)
        * - R2R plan: alloc_size * sizeof(double/float)
        * - R2C plan: alloc_size * sizeof(double/float)
        * \return Minimum number of elements needs to be allocated
        * 
        * 
        * \param[out]   in_starts             Starts of local portion of data in 'real' space in reversed order
        * \param[out]   in_counts             Sizes  of local portion of data in 'real' space in reversed order
        * \param[out]   out_starts            Starts of local portion of data in 'fourier' space in reversed order
        * \param[out]   out_counts            Sizes  of local portion of data in 'fourier' space in reversed order
        *
      */
        size_t
        get_local_sizes(
          int *in_starts=NULL,
          int *in_counts=NULL,
          int *out_starts=NULL,
          int *out_counts=NULL
        )
        {return dtfft_get_local_sizes(_plan, in_starts, in_counts, out_starts, out_counts);}

      /** \brief Get minimal size needed for optional aux buffer. Results may differ on different MPI processes
        *
        * Minimum number of bytes that needs allocation:
        * - C2C plan: 2 * aux_size * sizeof(double/float) or aux_size * sizeof(complex<double/float>)
        * - R2R plan: aux_size * sizeof(double/float)
        * - R2C plan: 2 * aux_size * sizeof(double/float) or aux_size * sizeof(complex<double/float>)
        * \return Minimum number of elements needs to be allocated
        *
      */
        size_t get_aux_size()
        {return dtfft_get_aux_size(_plan);}

      /** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
      */
        void destroy()
        {dtfft_destroy(_plan);}
      /** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
      */
        ~dtfft_core() {destroy();}
    };
  }

  class PlanC2C: public core::dtfft_core
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
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
      */
      PlanC2C(
        const std::vector<int> &dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
      ):PlanC2C(dims.size(), dims.data(), comm, precision, effort_flag, executor_type) {}

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
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
      */
      PlanC2C(
        const int ndims,
        const int *dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
      ){_plan = dtfft_create_plan_c2c(ndims, dims, comm, precision, effort_flag, executor_type);}
  };

  class PlanR2C: public core::dtfft_core
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
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
        * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C plan instead
      */
      PlanR2C(
        const std::vector<int> &dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
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
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
        * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C plan instead
      */
      PlanR2C(
        const int ndims,
        const int *dims,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
      ){_plan = dtfft_create_plan_r2c(ndims, dims, comm, precision, effort_flag, executor_type);}
  };

  class PlanR2R: public core::dtfft_core
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
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
      */
      PlanR2R(
        const std::vector<int> &dims,
        const std::vector<int> &in_kinds=std::vector<int>(),
        const std::vector<int> &out_kinds=std::vector<int>(),
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
      ):PlanR2R(dims.size(), dims.data(), in_kinds.data(), out_kinds.data(), comm, precision, effort_flag, executor_type) {}

      /** \brief Real-to-Real Plan constructor using C-style arguments. Must be called after MPI_Init
        * 
        * \param[in]    ndims                 Number of dimensions: 2 or 3
        * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
        * \param[in]    in_kinds              Buffer of size `ndims` with Real FFT kinds in reversed order, forward transform.
        *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
        * \param[in]    out_kinds             Buffer of size `ndims` with Real FFT kinds in reversed order, backward transform
        *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
        * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
        * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
        * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
        * \param[in]    executor_type         Type of external FFT executor. One of the
        *                                     - `DTFFT_EXECUTOR_NONE`
        *                                     - `DTFFT_EXECUTOR_FFTW3`
        *                                     - `DTFFT_EXECUTOR_KFR`
        *
        * \note Parameter `effort_flag` is not yet used and reserved for future.
      */
      PlanR2R(
        const int ndims,
        const int *dims,
        const int *in_kinds=NULL,
        const int *out_kinds=NULL,
        MPI_Comm comm=MPI_COMM_WORLD,
        const int precision=DTFFT_DOUBLE,
        const int effort_flag=DTFFT_ESTIMATE,
        const int executor_type=DTFFT_EXECUTOR_FFTW3
      ){_plan = dtfft_create_plan_r2r(ndims, dims, in_kinds, out_kinds, comm, precision, effort_flag, executor_type);}
  };
}
// DTFFT_HPP
#endif 