#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/native_enum.h>
#include <dtfft.hpp>
#include <mpi.h>
#include <iostream>

namespace py = pybind11;

#define DTFFT_PYTHON_CALL(call)                                            \
    do                                                                     \
    {                                                                      \
        dtfft::Error ierr = call;                                          \
        if (ierr != dtfft::Error::SUCCESS)                                 \
        {                                                                  \
            py::set_error(                                                 \
                dtfft_exception,                                           \
                dtfft_get_error_string(static_cast<dtfft_error_t>(ierr))); \
            throw py::error_already_set();                                 \
        }                                                                  \
    } while (false);


// Trampoline class for Plan to allow inheritance in Python
class PyPlan : public dtfft::Plan
{
public:
    using dtfft::Plan::Plan; // Inherit constructors
};

static inline void *convert_pointer(const py::object &array)
{
    uintptr_t ptr;

#ifdef DTFFT_WITH_CUDA
    if (py::hasattr(array, "__cuda_array_interface__"))
    {
        auto interface = array.attr("__cuda_array_interface__").cast<py::dict>();
        auto data_tuple = interface["data"].cast<py::tuple>();
        ptr = data_tuple[0].cast<uintptr_t>();
    }
    else
#endif
        if (py::hasattr(array, "__array_interface__"))
    {
        auto interface = array.attr("__array_interface__").cast<py::dict>();
        auto data_tuple = interface["data"].cast<py::tuple>();
        ptr = data_tuple[0].cast<uintptr_t>();
    }
    else
    {
        throw std::runtime_error("Unsupported array type");
    }
    return reinterpret_cast<void *>(ptr);
}

struct PlanPointers
{
    void *in;
    void *out;
    void *aux;
};

static inline PlanPointers convert_plan_pointers(const py::object &in_obj, const py::object &out_obj, const py::object &aux_obj = py::none())
{
    PlanPointers ptrs;
    ptrs.in = convert_pointer(in_obj);
    ptrs.out = convert_pointer(out_obj);
    ptrs.aux = aux_obj.is_none() ? nullptr : convert_pointer(aux_obj);
    return ptrs;
}

PYBIND11_MODULE(_dtfft, m)
{
    m.doc() = "Python bindings for the dtFFT library";

#ifdef DTFFT_WITH_FFTW
    m.def("is_fftw_enabled", []()
          { return py::bool_(true); }, "Check if FFTW3 is available.");
#else
    m.def("is_fftw_enabled", []()
          { return py::bool_(false); }, "Check if FFTW3 is available.");
#endif

#ifdef DTFFT_WITH_MKL
    m.def("is_mkl_enabled", []()
          { return py::bool_(true); }, "Check if MKL is available.");
#else
    m.def("is_mkl_enabled", []()
          { return py::bool_(false); }, "Check if MKL is available.");
#endif

#ifdef DTFFT_WITH_CUFFT
    m.def("is_cufft_enabled", []()
          { return py::bool_(true); }, "Check if CUFFT is available.");
#else
    m.def("is_cufft_enabled", []()
          { return py::bool_(false); }, "Check if CUFFT is available.");
#endif

#ifdef DTFFT_WITH_VKFFT
    m.def("is_vkfft_enabled", []()
          { return py::bool_(true); }, "Check if VkFFT is available.");
#else
    m.def("is_vkfft_enabled", []()
          { return py::bool_(false); }, "Check if VkFFT is available.");
#endif

#ifdef DTFFT_TRANSPOSE_ONLY
    m.def("is_transpose_only_enabled", []()
          { return py::bool_(true); }, "Check if dtFFT was built in transpose-only mode.");
#else
    m.def("is_transpose_only_enabled", []()
          { return py::bool_(false); }, "Check if dtFFT was built in transpose-only mode.");
#endif

#ifdef DTFFT_WITH_NCCL
    m.def("is_nccl_enabled", []()
          { return py::bool_(true); }, "Check if NCCL is available.");
#else
    m.def("is_nccl_enabled", []()
          { return py::bool_(false); }, "Check if NCCL is available.");
#endif

#ifdef DTFFT_WITH_NVSHMEM
    m.def("is_nvshmem_enabled", []()
          { return py::bool_(true); }, "Check if NVSHMEM is available.");
#else
    m.def("is_nvshmem_enabled", []()
          { return py::bool_(false); }, "Check if NVSHMEM is available.");
#endif

#ifdef DTFFT_WITH_COMPRESSION
    m.def("is_compression_enabled", []()
          { return py::bool_(true); }, "Check if compression backends are available.");
#else
    m.def("is_compression_enabled", []()
          { return py::bool_(false); }, "Check if compression backends are available.");
#endif

#ifdef DTFFT_WITH_CUDA
    m.def("is_cuda_enabled", []()
          { return py::bool_(true); }, "Check if CUDA is available.");

    py::native_enum<dtfft::Platform>(m, "Platform", "enum.Enum", "Class that specifies runtime platform, e.g. Host, CUDA, HIP")
        .value("HOST", dtfft::Platform::HOST, "Host platform")
        .value("CUDA", dtfft::Platform::CUDA, "CUDA platform")
        .finalize();
#else
    m.def("is_cuda_enabled", []()
          { return py::bool_(false); }, "Check if CUDA is available.");
#endif

    py::native_enum<dtfft::Backend>(m, "Backend", "enum.Enum", "Various backends available in dtFFT.")
        .value("MPI_DATATYPE", dtfft::Backend::MPI_DATATYPE, "Backend that uses MPI datatypes.")
        .value("MPI_P2P", dtfft::Backend::MPI_P2P, "MPI peer-to-peer algorithm")
        .value("MPI_P2P_PIPELINED", dtfft::Backend::MPI_P2P_PIPELINED, "MPI peer-to-peer algorithm with overlapping data copying and unpacking")
        .value("MPI_A2A", dtfft::Backend::MPI_A2A, "MPI backend using MPI_Alltoall[v]")
        .value("MPI_RMA", dtfft::Backend::MPI_RMA, "MPI backend using one-sided communications")
        .value("MPI_RMA_PIPELINED", dtfft::Backend::MPI_RMA_PIPELINED, "MPI backend using pipelined one-sided communications")
        .value("MPI_P2P_SCHEDULED", dtfft::Backend::MPI_P2P_SCHEDULED, "MPI peer-to-peer algorithm with scheduled communication")
        .value("MPI_P2P_FUSED", dtfft::Backend::MPI_P2P_FUSED, "MPI peer-to-peer pipelined algorithm with overlapping packing, exchange and unpacking with scheduled communication")
        .value("MPI_RMA_FUSED", dtfft::Backend::MPI_RMA_FUSED, "MPI RMA pipelined algorithm with overlapping packing, exchange and unpacking with scheduled communication")
        .value("MPI_P2P_COMPRESSED", dtfft::Backend::MPI_P2P_COMPRESSED, "MPI peer-to-peer fused backend with compression")
        .value("MPI_RMA_COMPRESSED", dtfft::Backend::MPI_RMA_COMPRESSED, "MPI RMA fused backend with compression")
        .value("NCCL", dtfft::Backend::NCCL, "NCCL backend")
        .value("NCCL_PIPELINED", dtfft::Backend::NCCL_PIPELINED, "NCCL backend with overlapping data copying and unpacking")
        .value("NCCL_COMPRESSED", dtfft::Backend::NCCL_COMPRESSED, "NCCL backend with compression")
        .value("CUFFTMP", dtfft::Backend::CUFFTMP, "cuFFTMp backend")
        .value("CUFFTMP_PIPELINED", dtfft::Backend::CUFFTMP_PIPELINED, "cuFFTMp backend with an additional internal buffer")
        .value("ADAPTIVE", dtfft::Backend::ADAPTIVE, "Adaptive backend selection")
        .value("NONE", dtfft::Backend::NONE, "Backend is not defined")
        .finalize();

    m.def("get_backend_string", &dtfft::get_backend_string, "Returns string with name of backend provided as argument.");

    //=========================================================================
    // Classes (former Enums)
    //=========================================================================
    py::native_enum<dtfft::Execute>(m, "Execute", "enum.Enum", "This class lists valid `execute_type` parameters that can be passed to Plan.execute.")
        .value("FORWARD", dtfft::Execute::FORWARD, "Perform XYZ --> YZX --> ZXY plan execution (Forward)")
        .value("BACKWARD", dtfft::Execute::BACKWARD, "Perform ZXY --> YZX --> XYZ plan execution (Backward)")
        .finalize();

    py::native_enum<dtfft::Transpose>(m, "Transpose", "enum.Enum", "This class lists valid ``transpose_type`` parameters that can be passed to Plan.transpose.")
        .value("X_TO_Y", dtfft::Transpose::X_TO_Y, "Transpose from Fortran X aligned to Fortran Y aligned")
        .value("Y_TO_X", dtfft::Transpose::Y_TO_X, "Transpose from Fortran Y aligned to Fortran X aligned")
        .value("Y_TO_Z", dtfft::Transpose::Y_TO_Z, "Transpose from Fortran Y aligned to Fortran Z aligned")
        .value("Z_TO_Y", dtfft::Transpose::Z_TO_Y, "Transpose from Fortran Z aligned to Fortran Y aligned")
        .value("X_TO_Z", dtfft::Transpose::X_TO_Z, "Transpose from Fortran X aligned to Fortran Z aligned.\nNote: This value is valid to pass only in 3D Plan and value returned by Plan.get_z_slab_enabled must be `true`")
        .value("Z_TO_X", dtfft::Transpose::Z_TO_X, "Transpose from Fortran Z aligned to Fortran X aligned.\nNote: This value is valid to pass only in 3D Plan and value returned by Plan.get_z_slab_enabled must be `true`")
        .finalize();

    py::native_enum<dtfft::Layout>(m, "Layout", "enum.Enum", "This class lists valid layout parameters that can be passed to Plan.get_pencil.")
        .value("X_BRICKS", dtfft::Layout::X_BRICKS, "X-brick layout")
        .value("X_PENCILS", dtfft::Layout::X_PENCILS, "X-pencil layout")
        .value("X_PENCILS_FOURIER", dtfft::Layout::X_PENCILS_FOURIER, "X-pencil Fourier layout for R2C plans")
        .value("Y_PENCILS", dtfft::Layout::Y_PENCILS, "Y-pencil layout")
        .value("Z_PENCILS", dtfft::Layout::Z_PENCILS, "Z-pencil layout")
        .value("Z_BRICKS", dtfft::Layout::Z_BRICKS, "Z-brick layout")
        .finalize();

    py::native_enum<dtfft::Reshape>(m, "Reshape", "enum.Enum", "This class lists valid reshape types that can be passed to Plan.reshape.")
        .value("X_BRICKS_TO_PENCILS", dtfft::Reshape::X_BRICKS_TO_PENCILS, "Reshape from X bricks to X pencils")
        .value("X_PENCILS_TO_BRICKS", dtfft::Reshape::X_PENCILS_TO_BRICKS, "Reshape from X pencils to X bricks")
        .value("Z_BRICKS_TO_PENCILS", dtfft::Reshape::Z_BRICKS_TO_PENCILS, "Reshape from Z bricks to Z pencils")
        .value("Z_PENCILS_TO_BRICKS", dtfft::Reshape::Z_PENCILS_TO_BRICKS, "Reshape from Z pencils to Z bricks")
        .value("Y_BRICKS_TO_PENCILS", dtfft::Reshape::Y_BRICKS_TO_PENCILS, "Reshape from Y bricks to Y pencils")
        .value("Y_PENCILS_TO_BRICKS", dtfft::Reshape::Y_PENCILS_TO_BRICKS, "Reshape from Y pencils to Y bricks")
        .finalize();

    py::native_enum<dtfft::TransposeMode>(m, "TransposeMode", "enum.Enum", "This class lists possible local transposition modes for generic backends.")
        .value("PACK", dtfft::TransposeMode::PACK, "Perform transposition during packing")
        .value("UNPACK", dtfft::TransposeMode::UNPACK, "Perform transposition during unpacking")
        .finalize();

    py::native_enum<dtfft::AccessMode>(m, "AccessMode", "enum.Enum", "This class lists memory access modes for local transposition in generic backends.")
        .value("WRITE", dtfft::AccessMode::WRITE, "Write-aligned access")
        .value("READ", dtfft::AccessMode::READ, "Read-aligned access")
        .finalize();

    py::native_enum<dtfft::Precision>(m, "Precision", "enum.Enum", "This class lists valid `precision` parameters that can be passed to Plan constructors.")
        .value("SINGLE", dtfft::Precision::SINGLE, "Use Single precision")
        .value("DOUBLE", dtfft::Precision::DOUBLE, "Use Double precision")
        .finalize();

    py::native_enum<dtfft::Effort>(m, "Effort", "enum.Enum", "This class lists valid `effort` parameters that can be passed to Plan constructors.")
        .value("ESTIMATE", dtfft::Effort::ESTIMATE, "Create plan as fast as possible")
        .value("MEASURE", dtfft::Effort::MEASURE, "Will attempt to find best MPI Grid decomposition.\nPassing this flag and MPI Communicator with cartesian topology to any Plan Constructor is same as Effort::ESTIMATE.")
        .value("PATIENT", dtfft::Effort::PATIENT, "Same as Effort::MEASURE plus cycle through various send and receive MPI_Datatypes.\nFor GPU Build this flag will run autotune procedure to find best backend")
        .value("EXHAUSTIVE", dtfft::Effort::EXHAUSTIVE, "Same as Effort::PATIENT plus autotune all possible kernels and reshape backends.")
        .finalize();

    py::native_enum<dtfft::Executor>(m, "Executor", "enum.Enum", "This class lists available FFT executors.")
        .value("NONE", dtfft::Executor::NONE, "Do not create any FFT plans. Creates transpose only plan.")
        .value("FFTW3", dtfft::Executor::FFTW3, "FFTW3 Executor (Host only)")
        .value("MKL", dtfft::Executor::MKL, "MKL DFTI Executor (Host only)")
        .value("CUFFT", dtfft::Executor::CUFFT, "CUFFT Executor (GPU Only)")
        .value("VKFFT", dtfft::Executor::VKFFT, "VkFFT Executor (GPU Only)")
        .finalize();

    py::native_enum<dtfft::R2RKind>(m, "R2RKind", "enum.Enum", "Real-to-Real FFT kinds available in dtFFT")
        .value("DCT_1", dtfft::R2RKind::DCT_1, "DCT-I (Logical N=2*(n-1), inverse is R2RKind::DCT_1)")
        .value("DCT_2", dtfft::R2RKind::DCT_2, "DCT-II (Logical N=2*n, inverse is R2RKind::DCT_3)")
        .value("DCT_3", dtfft::R2RKind::DCT_3, "DCT-III (Logical N=2*n, inverse is R2RKind::DCT_2)")
        .value("DCT_4", dtfft::R2RKind::DCT_4, "DCT-IV (Logical N=2*n, inverse is R2RKind::DCT_4)")
        .value("DST_1", dtfft::R2RKind::DST_1, "DST-I (Logical N=2*(n+1), inverse is R2RKind::DST_1)")
        .value("DST_2", dtfft::R2RKind::DST_2, "DST-II (Logical N=2*n, inverse is R2RKind::DST_3)")
        .value("DST_3", dtfft::R2RKind::DST_3, "DST-III (Logical N=2*n, inverse is R2RKind::DST_2)")
        .value("DST_4", dtfft::R2RKind::DST_4, "DST-IV (Logical N=2*n, inverse is R2RKind::DST_4)")
        .finalize();

#ifdef DTFFT_WITH_COMPRESSION
    py::native_enum<dtfft::CompressionMode>(m, "CompressionMode", "enum.Enum", "Compression mode")
        .value("LOSSLESS", dtfft::CompressionMode::LOSSLESS, "Lossless compression mode")
        .value("FIXED_RATE", dtfft::CompressionMode::FIXED_RATE, "Fixed rate compression mode")
        .value("FIXED_PRECISION", dtfft::CompressionMode::FIXED_PRECISION, "Fixed precision compression mode")
        .value("FIXED_ACCURACY", dtfft::CompressionMode::FIXED_ACCURACY, "Fixed accuracy compression mode")
        .finalize();

    py::native_enum<dtfft::CompressionLib>(m, "CompressionLib", "enum.Enum", "Compression library")
        .value("ZFP", dtfft::CompressionLib::ZFP, "ZFP compression library")
        .finalize();

    py::class_<dtfft::CompressionConfig>(m, "CompressionConfig", "Compression configuration")
        .def(py::init([&](const dtfft::CompressionLib compression_lib,
                          const dtfft::CompressionMode compression_mode,
                          const double rate, const int precision, const double tolerance)
                      { return std::make_unique<dtfft::CompressionConfig>(compression_lib, compression_mode, rate, precision, tolerance); }),
             py::arg("compression_lib") = dtfft::CompressionLib::ZFP,
             py::arg("compression_mode") = dtfft::CompressionMode::LOSSLESS,
             py::arg("rate") = 0.0, py::arg("precision") = 0, py::arg("tolerance") = 0.0,
             "Creates CompressionConfig with specified parameters.")
        .def("__repr__", [](const dtfft::CompressionConfig &config)
            {
                std::string repr = "CompressionConfig(";
                switch (config.compression_lib)
                {
                case dtfft::CompressionLib::ZFP:
                    repr += "lib=ZFP";
                    break;
                
                default:
                    break;
                }
                repr += ", ";
                switch (config.compression_mode)
                {
                case dtfft::CompressionMode::LOSSLESS:
                    repr += "mode=LOSSLESS";
                    break;
                case dtfft::CompressionMode::FIXED_RATE:
                    repr += "mode=FIXED_RATE, rate=" + std::to_string(config.rate);
                    break;
                case dtfft::CompressionMode::FIXED_PRECISION:
                    repr += "mode=FIXED_PRECISION, precision=" + std::to_string(config.precision);
                    break;
                case dtfft::CompressionMode::FIXED_ACCURACY:
                    repr += "mode=FIXED_ACCURACY, tolerance=" + std::to_string(config.tolerance);
                    break;
                default:
                    break;
                }
                return repr + ")";
            });
#endif

    static py::exception<dtfft::Exception> dtfft_exception(m, "Exception", PyExc_RuntimeError);

    //=========================================================================
    // Free Functions
    //=========================================================================
    m.def("set_config", [&](const dtfft::Config &config)
          { DTFFT_PYTHON_CALL(dtfft::set_config(config)) }, "Sets configuration values to dtFFT. Must be called before plan creation to take effect.");

    //=========================================================================
    // Classes
    //=========================================================================
    py::class_<dtfft::Version>(m, "Version", "dtFFT version information")
        .def_property_readonly_static("MAJOR", [](py::object)
                                      { return dtfft::Version::MAJOR; }, "dtFFT Major Version")
        .def_property_readonly_static("MINOR", [](py::object)
                                      { return dtfft::Version::MINOR; }, "dtFFT Minor Version")
        .def_property_readonly_static("PATCH", [](py::object)
                                      { return dtfft::Version::PATCH; }, "dtFFT Patch Version")
        .def_property_readonly_static("CODE", [](py::object)
                                      { return dtfft::Version::CODE; }, "dtFFT Version Code. Can be used for version comparison")
        .def_static("get", static_cast<int32_t (*)() noexcept>(&dtfft::Version::get), "Get Version Code defined during compilation")
        .def_static("get", static_cast<int32_t (*)(int32_t, int32_t, int32_t) noexcept>(&dtfft::Version::get), "Get Version Code based on input parameters", py::arg("major"), py::arg("minor"), py::arg("patch"));

    py::class_<dtfft::Pencil>(m, "Pencil", "Class to handle Pencils")
        .def(py::init([&](const std::vector<int32_t> &starts, const std::vector<int32_t> &counts)
                      { return std::make_unique<dtfft::Pencil>(starts, counts); }),
             py::arg("starts"), py::arg("counts"), "Creates Pencil from local starts and counts in natural Fortran order.")
        .def("get_ndims", [&](const dtfft::Pencil &p) {
            try {
                return p.get_ndims();
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            }
        }, "Number of dimensions in a pencil")
        .def("get_dim", [&](const dtfft::Pencil &p) {
            try {
                return p.get_dim();
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            }
        }, "Aligned dimension id starting from 1")
        .def("get_starts", [&](const dtfft::Pencil &p) {
            try {
                return p.get_starts();
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            }
        }, "Local starts in natural Fortran order")
        .def("get_counts", [&](const dtfft::Pencil &p) {
            try {
                return p.get_counts();
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            }
        }, "Local counts in natural Fortran order")
        .def("get_size", [&](const dtfft::Pencil &p) {
            try {
                return p.get_size();
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            }
        }, "Total number of elements in a pencil");

    py::class_<dtfft::Config> config(m, "Config", "Class to set additional configuration parameters");
    config.def(py::init<>(), "Creates and sets default configuration values")
        .def("set_enable_log", &dtfft::Config::set_enable_log, "Sets whether dtFFT should print additional information or not (default: false).", py::arg("enable_log"))
        .def("set_enable_z_slab", &dtfft::Config::set_enable_z_slab, "Sets whether dtFFT use Z-slab optimization or not (default: true).\nOne should consider disabling Z-slab optimization in order to resolve `Error::VKFFT_R2R_2D_PLAN` error OR when underlying FFT implementation of 2D plan is too slow.", py::arg("enable_z_slab"))
        .def("set_enable_y_slab", &dtfft::Config::set_enable_y_slab, "Sets whether dtFFT should use Y-slab optimization or not (default: false).", py::arg("enable_y_slab"))
        .def("set_measure_warmup_iters", &dtfft::Config::set_measure_warmup_iters, "Sets number of warmup iterations used during autotuning.", py::arg("n_measure_warmup_iters"))
        .def("set_measure_iters", &dtfft::Config::set_measure_iters, "Sets number of iterations used during autotuning.", py::arg("n_measure_iters"))
        .def("set_backend", &dtfft::Config::set_backend, "Sets Backend that will be used by dtFFT when `effort` is Effort::ESTIMATE or Effort::MEASURE.", py::arg("backend"))
        .def("set_reshape_backend", &dtfft::Config::set_reshape_backend, "Sets backend used by dtFFT for reshape operations when `effort` is Effort::ESTIMATE or Effort::MEASURE.", py::arg("backend"))
        .def("set_enable_datatype_backend", &dtfft::Config::set_enable_datatype_backend, "Sets whether MPI datatype backend is enabled for autotuning (default: true).", py::arg("enable_datatype_backend"))
        .def("set_enable_mpi_backends", &dtfft::Config::set_enable_mpi_backends, "Sets whether MPI backends be enabled when `effort` is Effort::PATIENT or not (default: false).", py::arg("enable_mpi_backends"))
        .def("set_enable_pipelined_backends", &dtfft::Config::set_enable_pipelined_backends, "Sets whether pipelined backends be enabled when `effort` is Effort::PATIENT or not (default: true).\nPipelined backends require an additional internal buffer.", py::arg("enable_pipelined_backends"))
        .def("set_enable_rma_backends", &dtfft::Config::set_enable_rma_backends, "Sets whether RMA backends be enabled when `effort` is Effort::PATIENT or not (default: true).", py::arg("enable_rma_backends"))
        .def("set_enable_fused_backends", &dtfft::Config::set_enable_fused_backends, "Sets whether fused backends be enabled when `effort` is Effort::PATIENT or not (default: true).", py::arg("enable_fused_backends"))
        .def("set_enable_kernel_autotune", &dtfft::Config::set_enable_kernel_autotune, "Sets whether kernel launch parameter autotuning is enabled for effort levels below EXHAUSTIVE (default: false).", py::arg("enable_kernel_autotune"))
        .def("set_enable_fourier_reshape", &dtfft::Config::set_enable_fourier_reshape, "Sets whether reshapes between pencils and bricks are also executed in Fourier space during Plan.execute (default: false).", py::arg("enable_fourier_reshape"))
        .def("set_transpose_mode", &dtfft::Config::set_transpose_mode, "Sets at which stage local transposition is performed for generic backends.", py::arg("transpose_mode"))
        .def("set_access_mode", &dtfft::Config::set_access_mode, "Sets memory access mode for local transposition in generic backends.", py::arg("access_mode"));
#ifdef DTFFT_WITH_COMPRESSION
    config.def("set_enable_compressed_backends", &dtfft::Config::set_enable_compressed_backends, "Sets whether compressed backends are enabled for autotuning.", py::arg("enable_compressed_backends"))
        .def("set_compression_config_transpose", &dtfft::Config::set_compression_config_transpose, "Sets compression configuration for transpose operations.", py::arg("compression_config"))
        .def("set_compression_config_reshape", &dtfft::Config::set_compression_config_reshape, "Sets compression configuration for reshape operations.", py::arg("compression_config"));
#endif
#ifdef DTFFT_WITH_CUDA
    config.def("set_platform", &dtfft::Config::set_platform, "Sets platform to execute plan (default: Platform::HOST).", py::arg("platform"))
        .def("set_stream", [](dtfft::Config &c, uintptr_t stream_ptr)
             { return c.set_stream(reinterpret_cast<dtfft_stream_t>(stream_ptr)); }, "Sets Main CUDA stream that will be used in dtFFT. Stream must not be destroyed before plan destruction.", py::arg("stream_ptr"))
        .def("set_enable_nccl_backends", &dtfft::Config::set_enable_nccl_backends, "Sets whether NCCL Backends be enabled when `effort` is Effort::PATIENT or not (default: true).", py::arg("enable_nccl_backends"))
        .def("set_enable_nvshmem_backends", &dtfft::Config::set_enable_nvshmem_backends, "Sets whether NVSHMEM Backends be enabled when `effort` is Effort::PATIENT or not (default: true).", py::arg("enable_nvshmem_backends"));
#endif

    py::class_<dtfft::Plan, PyPlan> plan(m, "Plan", "Abstract base class for all dtFFT plans. This class does not have any public constructors.");
    plan.def("get_z_slab_enabled", [&](const dtfft::Plan &p)
             {
            bool is_z_slab_enabled;
            DTFFT_PYTHON_CALL(p.get_z_slab_enabled(&is_z_slab_enabled));
            return is_z_slab_enabled; }, "Checks if plan is using Z-slab optimization. If true, Transpose.X_TO_Z and Transpose.Z_TO_X are valid.")
            .def("get_y_slab_enabled", [&](const dtfft::Plan &p)
             {
            bool is_y_slab_enabled;
            DTFFT_PYTHON_CALL(p.get_y_slab_enabled(&is_y_slab_enabled));
            return is_y_slab_enabled; }, "Checks if plan is using Y-slab optimization.")
        .def("report", [&](dtfft::Plan &p)
             { DTFFT_PYTHON_CALL(p.report()); }, "Prints plan-related information to stdout")
#ifdef DTFFT_WITH_COMPRESSION
        .def("report_compression", [&](dtfft::Plan &p)
             { DTFFT_PYTHON_CALL(p.report_compression()); }, "Prints compression-related information to stdout")
#endif
        .def("get_pencil", [&](const dtfft::Plan &p, dtfft::Layout layout)
             {
            dtfft::Pencil pencil;
            DTFFT_PYTHON_CALL(p.get_pencil(layout, pencil));
            return pencil; }, "Obtains pencil information for a given layout.", py::arg("layout"))
        .def("execute", [&](dtfft::Plan &p, py::object in_obj, py::object out_obj, dtfft::Execute exec_type, py::object aux_obj)
             {
            auto ptrs = convert_plan_pointers(in_obj, out_obj, aux_obj);
            DTFFT_PYTHON_CALL(p.execute(ptrs.in, ptrs.out, exec_type, ptrs.aux)); }, py::arg("in"), py::arg("out"), py::arg("execute_type"), py::arg("aux") = py::none(), "Plan execution with numpy arrays or device pointers.")
        .def("transpose", [&](dtfft::Plan &p, py::object in_obj, py::object out_obj, dtfft::Transpose transpose_type, py::object aux_obj)
             {
            auto ptrs = convert_plan_pointers(in_obj, out_obj, aux_obj);
            DTFFT_PYTHON_CALL(p.transpose(ptrs.in, ptrs.out, transpose_type, ptrs.aux)); }, py::arg("in"), py::arg("out"), py::arg("transpose_type"), py::arg("aux") = py::none(), "Transpose data in single dimension. `in` and `out` cannot be the same.")
        .def("transpose_start", [&](dtfft::Plan &p, py::object in_obj, py::object out_obj, dtfft::Transpose transpose_type, py::object aux_obj)
             {
            auto ptrs = convert_plan_pointers(in_obj, out_obj, aux_obj);
            dtfft_request_t request;
            DTFFT_PYTHON_CALL(p.transpose_start(ptrs.in, ptrs.out, transpose_type, ptrs.aux, &request));
            return reinterpret_cast<uintptr_t>(request); }, py::arg("in"), py::arg("out"), py::arg("transpose_type"), py::arg("aux") = py::none(), "Starts an asynchronous transpose operation and returns request handle.")
        .def("transpose_end", [&](dtfft::Plan &p, uintptr_t request)
             { DTFFT_PYTHON_CALL(p.transpose_end(reinterpret_cast<dtfft_request_t>(request))); }, py::arg("request"), "Ends an asynchronous transpose operation.")
        .def("reshape", [&](dtfft::Plan &p, py::object in_obj, py::object out_obj, dtfft::Reshape reshape_type, py::object aux_obj)
             {
            auto ptrs = convert_plan_pointers(in_obj, out_obj, aux_obj);
            DTFFT_PYTHON_CALL(p.reshape(ptrs.in, ptrs.out, reshape_type, ptrs.aux)); }, py::arg("in"), py::arg("out"), py::arg("reshape_type"), py::arg("aux") = py::none(), "Reshape data from bricks to pencils and vice versa.")
        .def("reshape_start", [&](dtfft::Plan &p, py::object in_obj, py::object out_obj, dtfft::Reshape reshape_type, py::object aux_obj)
             {
            auto ptrs = convert_plan_pointers(in_obj, out_obj, aux_obj);
            dtfft_request_t request;
            DTFFT_PYTHON_CALL(p.reshape_start(ptrs.in, ptrs.out, reshape_type, ptrs.aux, &request));
            return reinterpret_cast<uintptr_t>(request); }, py::arg("in"), py::arg("out"), py::arg("reshape_type"), py::arg("aux") = py::none(), "Starts an asynchronous reshape operation and returns request handle.")
        .def("reshape_end", [&](dtfft::Plan &p, uintptr_t request)
             { DTFFT_PYTHON_CALL(p.reshape_end(reinterpret_cast<dtfft_request_t>(request))); }, py::arg("request"), "Ends an asynchronous reshape operation.")
        .def("get_alloc_size", [&](const dtfft::Plan &p)
             {
            size_t alloc_size;
            DTFFT_PYTHON_CALL(p.get_alloc_size(&alloc_size));
            return alloc_size; }, "Minimum number of elements to be allocated for `in`, `out` or `aux` buffers.")
        .def("get_aux_size", [&](const dtfft::Plan &p)
             {
            size_t aux_size;
            DTFFT_PYTHON_CALL(p.get_aux_size(&aux_size));
            return aux_size; }, "Number of elements required for auxiliary buffer during execute operations.")
        .def("get_aux_bytes", [&](const dtfft::Plan &p)
             {
            size_t aux_bytes;
            DTFFT_PYTHON_CALL(p.get_aux_bytes(&aux_bytes));
            return aux_bytes; }, "Number of bytes required for auxiliary buffer during execute operations.")
        .def("get_aux_size_reshape", [&](const dtfft::Plan &p)
             {
            size_t aux_size;
            DTFFT_PYTHON_CALL(p.get_aux_size_reshape(&aux_size));
            return aux_size; }, "Number of elements required for auxiliary buffer during reshape operations.")
        .def("get_aux_bytes_reshape", [&](const dtfft::Plan &p)
             {
            size_t aux_bytes;
            DTFFT_PYTHON_CALL(p.get_aux_bytes_reshape(&aux_bytes));
            return aux_bytes; }, "Number of bytes required for auxiliary buffer during reshape operations.")
        .def("get_aux_size_transpose", [&](const dtfft::Plan &p)
             {
            size_t aux_size;
            DTFFT_PYTHON_CALL(p.get_aux_size_transpose(&aux_size));
            return aux_size; }, "Number of elements required for auxiliary buffer during transpose operations.")
        .def("get_aux_bytes_transpose", [&](const dtfft::Plan &p)
             {
            size_t aux_bytes;
            DTFFT_PYTHON_CALL(p.get_aux_bytes_transpose(&aux_bytes));
            return aux_bytes; }, "Number of bytes required for auxiliary buffer during transpose operations.")
        .def("get_local_sizes", [&](const dtfft::Plan &p)
             {
            int8_t ndims;
            const int32_t* dims_ptr;
            DTFFT_PYTHON_CALL(p.get_dims(&ndims, &dims_ptr));
            std::vector<int32_t> in_starts(ndims), in_counts(ndims), out_starts(ndims), out_counts(ndims);
            size_t alloc_size;
            DTFFT_PYTHON_CALL(p.get_local_sizes(in_starts, in_counts, out_starts, out_counts, &alloc_size));
            return py::make_tuple(in_starts, in_counts, out_starts, out_counts, alloc_size); }, "Get grid decomposition information. Returns (in_starts, in_counts, out_starts, out_counts, alloc_size).")
        .def("get_dims", [&](const dtfft::Plan &p)
             {
            int8_t ndims;
            const int32_t* dims_ptr;
            DTFFT_PYTHON_CALL(p.get_dims(&ndims, &dims_ptr));
            return std::vector<int32_t>(dims_ptr, dims_ptr + ndims); }, "Returns global dimensions in natural Fortran order.")
        .def("get_grid_dims", [&](const dtfft::Plan &p)
             {
            int8_t ndims;
            const int32_t* dims_ptr;
            DTFFT_PYTHON_CALL(p.get_grid_dims(&ndims, &dims_ptr));
            return std::vector<int32_t>(dims_ptr, dims_ptr + ndims); }, "Returns grid decomposition dimensions in natural Fortran order.")
        .def("get_element_size", [&](const dtfft::Plan &p)
             {
            size_t element_size;
            DTFFT_PYTHON_CALL(p.get_element_size(&element_size));
            return element_size; }, "Obtains number of bytes required to store a single element.")
        .def("get_alloc_bytes", [&](const dtfft::Plan &p)
             {
            size_t alloc_bytes;
            DTFFT_PYTHON_CALL(p.get_alloc_bytes(&alloc_bytes));
            return alloc_bytes; }, "Returns minimum number of bytes required to execute plan (alloc_size * element_size).")
        .def("get_executor", [&](const dtfft::Plan &p)
             {
            dtfft::Executor executor;
            DTFFT_PYTHON_CALL(p.get_executor(&executor));
            return executor; }, "Returns executor used by this plan.")
        .def("get_precision", [&](const dtfft::Plan &p)
             {
            dtfft::Precision precision;
            DTFFT_PYTHON_CALL(p.get_precision(&precision));
            return precision; }, "Returns precision of this plan.")
        .def("get_backend", [&](const dtfft::Plan &p)
             {
            dtfft::Backend backend;
            DTFFT_PYTHON_CALL(p.get_backend(backend));
            return backend; }, "Returns selected backend.")
        .def("get_reshape_backend", [&](const dtfft::Plan &p)
             {
            dtfft::Backend backend;
            DTFFT_PYTHON_CALL(p.get_reshape_backend(backend));
            return backend; }, "Returns backend used for reshape operations.")
        .def("alloc_ptr", [&](const dtfft::Plan &p, size_t alloc_bytes)
             {
            void* ptr;
            DTFFT_PYTHON_CALL(p.mem_alloc(alloc_bytes, &ptr));
            return reinterpret_cast<uintptr_t>(ptr); }, py::arg("alloc_bytes"), "Allocates plan-specific memory and returns raw pointer as integer.")
        .def("free_ptr", [&](const dtfft::Plan &p, uintptr_t ptr)
             { DTFFT_PYTHON_CALL(p.mem_free(reinterpret_cast<void *>(ptr))); }, py::arg("ptr"), "Frees plan-specific memory by raw pointer (integer).")
        .def("destroy", [&](dtfft::Plan &p)
             { DTFFT_PYTHON_CALL(p.destroy()); }, "Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize.");
#ifdef DTFFT_WITH_CUDA
    plan.def("get_stream", [&](const dtfft::Plan &p)
             {
            dtfft_stream_t stream;
            DTFFT_PYTHON_CALL(p.get_stream(&stream));
            return reinterpret_cast<uintptr_t>(stream); }, "Returns the CUDA stream associated with the plan as a pointer (integer).")
        .def("get_platform", [&](const dtfft::Plan &p)
             {
            dtfft::Platform platform;
            DTFFT_PYTHON_CALL(p.get_platform(platform));
            return platform; }, "Returns plan execution platform.");
#endif

    py::class_<dtfft::PlanC2C, dtfft::Plan>(m, "PlanC2C", "Complex-to-Complex Plan")
        .def(py::init([&](const std::vector<int32_t> &dims, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanC2C>(dims, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("dims"), py::arg("comm_handle"), py::arg("precision"),
             py::arg("effort"), py::arg("executor"),
             "Complex-to-Complex Plan constructor.")
        .def(py::init([&](const dtfft::Pencil &pencil, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanC2C>(pencil, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("pencil"), py::arg("comm_handle"), py::arg("precision"),
             py::arg("effort"), py::arg("executor"),
             "Complex-to-Complex Plan constructor from Pencil decomposition.");

    py::class_<dtfft::PlanR2C, dtfft::Plan>(m, "PlanR2C", "Real-to-Complex Plan")
        .def(py::init([&](const std::vector<int32_t> &dims, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanR2C>(dims, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("dims"), py::arg("comm_handle"), py::arg("precision"),
             py::arg("effort"), py::arg("executor"),
             "Real-to-Complex Plan constructor. `executor` cannot be Executor::NONE.")
        .def(py::init([&](const dtfft::Pencil &pencil, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanR2C>(pencil, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("pencil"), py::arg("comm_handle"), py::arg("precision"),
             py::arg("effort"), py::arg("executor"),
             "Real-to-Complex Plan constructor from Pencil decomposition. `executor` cannot be Executor::NONE.");

    py::class_<dtfft::PlanR2R, dtfft::Plan>(m, "PlanR2R", "Real-to-Real Plan")
        .def(py::init([&](const std::vector<int32_t> &dims, const std::vector<dtfft::R2RKind> &kinds, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanR2R>(dims, kinds, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("dims"), py::arg("kinds"), py::arg("comm_handle"),
             py::arg("precision"), py::arg("effort"),
             py::arg("executor"),
             "Real-to-Real Plan constructor.")
        .def(py::init([&](const dtfft::Pencil &pencil, const std::vector<dtfft::R2RKind> &kinds, intptr_t comm, dtfft::Precision precision, dtfft::Effort effort, dtfft::Executor executor)
                      {
            try {
                return std::make_unique<dtfft::PlanR2R>(pencil, kinds, (MPI_Comm)comm, precision, effort, executor);
            } catch (const dtfft::Exception& ex) {
                py::set_error(dtfft_exception, dtfft_get_error_string(static_cast<dtfft_error_t>(ex.get_error_code())));
                throw py::error_already_set();
            } }),
             py::arg("pencil"), py::arg("kinds"), py::arg("comm_handle"),
             py::arg("precision"), py::arg("effort"),
             py::arg("executor"),
             "Real-to-Real Plan constructor from Pencil decomposition.");
}
