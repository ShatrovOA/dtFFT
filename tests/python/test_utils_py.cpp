#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include <dtfft.hpp>
#include "../c/test_utils.h"

namespace py = pybind11;

static inline void* convert_pointer(const py::object& array) {
    uintptr_t ptr = 0;

#ifdef DTFFT_WITH_CUDA
    if (py::hasattr(array, "__cuda_array_interface__")) {
        auto interface = array.attr("__cuda_array_interface__").cast<py::dict>();
        auto data_tuple = interface["data"].cast<py::tuple>();
        ptr = data_tuple[0].cast<uintptr_t>();
    } else
#endif
    if (py::hasattr(array, "__array_interface__")) {
        auto interface = array.attr("__array_interface__").cast<py::dict>();
        auto data_tuple = interface["data"].cast<py::tuple>();
        ptr = data_tuple[0].cast<uintptr_t>();
    } else {
        throw std::runtime_error("Unsupported array type");
    }
    return reinterpret_cast<void*>(ptr);
}

static inline size_t get_array_nitems(const py::object& array) {
    return array.attr("size").cast<size_t>();
}

PYBIND11_MODULE(_dtfft_test_utils, m) {
    m.doc() = "Python bindings for dtFFT test utilities";

    m.def("createGridDims", [](const std::vector<int32_t>& dims) {
            const int32_t ndims = static_cast<int32_t>(dims.size());
            std::vector<int32_t> dims_copy = dims;
            std::vector<int32_t> grid(ndims, 0), starts(ndims, 0), counts(ndims, 0);
            createGridDims(ndims, dims_copy.data(), grid.data(), starts.data(), counts.data());
            return py::make_tuple(grid, starts, counts);
    }, py::arg("dims"), "Returns tuple (grid, starts, counts) for provided global dimensions.");

    m.def("attach_gpu_to_process", []() {
            attach_gpu_to_process();
    }, "Binds process rank to GPU device for test execution.");

    m.def("scaleFloat", [](int32_t executor, py::object buffer, size_t scale, int32_t platform, uintptr_t stream) {
#if defined(DTFFT_WITH_CUDA)
        scaleFloat(executor, convert_pointer(buffer), get_array_nitems(buffer), scale, platform, reinterpret_cast<dtfft_stream_t>(stream));
#else
        (void)platform; (void)stream;
        scaleFloat(executor, convert_pointer(buffer), get_array_nitems(buffer), scale);
#endif
    }, py::arg("executor"), py::arg("buffer"), py::arg("scale"), py::arg("platform") = 0, py::arg("stream") = 0);

    m.def("scaleDouble", [](int32_t executor, py::object buffer, size_t scale, int32_t platform, uintptr_t stream) {
#if defined(DTFFT_WITH_CUDA)
        scaleDouble(executor, convert_pointer(buffer), get_array_nitems(buffer), scale, platform, reinterpret_cast<dtfft_stream_t>(stream));
#else
        (void)platform; (void)stream;
        scaleDouble(executor, convert_pointer(buffer), get_array_nitems(buffer), scale);
#endif
    }, py::arg("executor"), py::arg("buffer"), py::arg("scale"), py::arg("platform") = 0, py::arg("stream") = 0);

    m.def("scaleComplexFloat", [](int32_t executor, py::object buffer, size_t scale, int32_t platform, uintptr_t stream) {
#if defined(DTFFT_WITH_CUDA)
        scaleComplexFloat(executor, convert_pointer(buffer), get_array_nitems(buffer), scale, platform, reinterpret_cast<dtfft_stream_t>(stream));
#else
        (void)platform; (void)stream;
        scaleComplexFloat(executor, convert_pointer(buffer), get_array_nitems(buffer), scale);
#endif
    }, py::arg("executor"), py::arg("buffer"), py::arg("scale"), py::arg("platform") = 0, py::arg("stream") = 0);

    m.def("scaleComplexDouble", [](int32_t executor, py::object buffer, size_t scale, int32_t platform, uintptr_t stream) {
#if defined(DTFFT_WITH_CUDA)
        scaleComplexDouble(executor, convert_pointer(buffer), get_array_nitems(buffer), scale, platform, reinterpret_cast<dtfft_stream_t>(stream));
#else
        (void)platform; (void)stream;
        scaleComplexDouble(executor, convert_pointer(buffer), get_array_nitems(buffer), scale);
#endif
    }, py::arg("executor"), py::arg("buffer"), py::arg("scale"), py::arg("platform") = 0, py::arg("stream") = 0);

    m.def("complexDoubleH2D", [](py::object src, py::object dst, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        complexDoubleH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src), platform);
#else
        (void)platform;
        complexDoubleH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src));
#endif
    }, py::arg("src"), py::arg("dst"), py::arg("platform") = 0);

    m.def("complexFloatH2D", [](py::object src, py::object dst, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        complexFloatH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src), platform);
#else
        (void)platform;
        complexFloatH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src));
#endif
    }, py::arg("src"), py::arg("dst"), py::arg("platform") = 0);

    m.def("doubleH2D", [](py::object src, py::object dst, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        doubleH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src), platform);
#else
        (void)platform;
        doubleH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src));
#endif
    }, py::arg("src"), py::arg("dst"), py::arg("platform") = 0);

    m.def("floatH2D", [](py::object src, py::object dst, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        floatH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src), platform);
#else
        (void)platform;
        floatH2D(convert_pointer(src), convert_pointer(dst), get_array_nitems(src));
#endif
    }, py::arg("src"), py::arg("dst"), py::arg("platform") = 0);

    m.def("checkAndReportComplexDouble", [](size_t n_global, double tf, double tb, py::object buffer, py::object check, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        checkAndReportComplexDouble(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check), platform);
#else
        (void)platform;
        checkAndReportComplexDouble(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check));
#endif
    }, py::arg("n_global"), py::arg("tf"), py::arg("tb"), py::arg("buffer"), py::arg("check"), py::arg("platform") = 0);

    m.def("checkAndReportComplexFloat", [](size_t n_global, double tf, double tb, py::object buffer, py::object check, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        checkAndReportComplexFloat(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check), platform);
#else
        (void)platform;
        checkAndReportComplexFloat(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check));
#endif
    }, py::arg("n_global"), py::arg("tf"), py::arg("tb"), py::arg("buffer"), py::arg("check"), py::arg("platform") = 0);

    m.def("checkAndReportDouble", [](size_t n_global, double tf, double tb, py::object buffer, py::object check, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        checkAndReportDouble(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check), platform);
#else
        (void)platform;
        checkAndReportDouble(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check));
#endif
    }, py::arg("n_global"), py::arg("tf"), py::arg("tb"), py::arg("buffer"), py::arg("check"), py::arg("platform") = 0);

    m.def("checkAndReportFloat", [](size_t n_global, double tf, double tb, py::object buffer, py::object check, int32_t platform) {
#if defined(DTFFT_WITH_CUDA)
        checkAndReportFloat(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check), platform);
#else
        (void)platform;
        checkAndReportFloat(n_global, tf, tb, convert_pointer(buffer), get_array_nitems(check), convert_pointer(check));
#endif
    }, py::arg("n_global"), py::arg("tf"), py::arg("tb"), py::arg("buffer"), py::arg("check"), py::arg("platform") = 0);
}
