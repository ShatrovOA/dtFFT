#include <cassert>
#include <dtfft.hpp>

namespace dtfft {
    Exception::Exception(Error error_code, std::string msg, const char* file, int line)
        : _error_code(error_code)
        , _message(std::move(msg))
        , _file(file)
        , _line(line)
    {
    }

    /** Exception explanation */
    const char* Exception::what() const noexcept
    {
        if (_what_cache.empty()) {
            _what_cache = "dtFFT Exception: '" + _message + "' (code: " + std::to_string(static_cast<int>(_error_code)) + ") at " + _file + ":" + std::to_string(_line);
        }
        return _what_cache.c_str();
    }

    /** Returns error code of exception */
    Error Exception::get_error_code() const noexcept
    {
        return _error_code;
    }

    /** Returns error message of exception */
    const std::string& Exception::get_message() const noexcept
    {
        return _message;
    }

    /** Returns file name where exception occurred */
    const std::string& Exception::get_file() const noexcept
    {
        return _file;
    }

    /** Returns line number where exception occurred */
    int Exception::get_line() const noexcept
    {
        return _line;
    }

    Pencil::Pencil(dtfft_pencil_t& c_pencil)
        : is_created(true)
        , is_obtained(true)
        , pencil(c_pencil)
    {
    }

    Pencil::Pencil()
        : is_created(false)
        , is_obtained(false)
    {
    }

    Pencil::Pencil(const int32_t n_dims, const int32_t* starts, const int32_t* counts)
    {
        assert(n_dims >= 2 && n_dims <= 3);
        pencil.ndims = n_dims;
        for (int32_t i = 0; i < n_dims; ++i) {
            pencil.starts[i] = starts[i];
            pencil.counts[i] = counts[i];
        }
        is_created = true;
        is_obtained = false;
    }

    Pencil::Pencil(
        const std::vector<int32_t>& starts,
        const std::vector<int32_t>& counts)
        : Pencil(static_cast<int8_t>(starts.size()), starts.data(), counts.data())
    {
    }

    uint8_t Pencil::get_ndims() const
    {
        assert(is_created);
        return pencil.ndims;
    }

    uint8_t Pencil::get_dim() const
    {
        assert(is_obtained);
        return pencil.dim;
    }

    std::vector<int32_t> Pencil::get_starts() const
    {
        assert(is_obtained);
        return { pencil.starts, pencil.starts + pencil.ndims };
    }

    std::vector<int32_t> Pencil::get_counts() const
    {
        assert(is_obtained);
        return { pencil.counts, pencil.counts + pencil.ndims };
    }

    size_t Pencil::get_size() const
    {
        assert(is_obtained);
        return pencil.size;
    }

    const dtfft_pencil_t& Pencil::c_struct() const
    {
        assert(is_created);
        return pencil;
    }

    std::string get_error_string(Error error_code) noexcept
    {
        const char* error_str = dtfft_get_error_string(static_cast<dtfft_error_t>(error_code));
        return { error_str };
    }

    std::string get_precision_string(Precision precision) noexcept
    {
        const char* precision_str = dtfft_get_precision_string(static_cast<dtfft_precision_t>(precision));
        return { precision_str };
    }

    std::string get_executor_string(Executor executor) noexcept
    {
        const char* executor_str = dtfft_get_executor_string(static_cast<dtfft_executor_t>(executor));
        return { executor_str };
    }

    std::string get_backend_string(const Backend backend)
    {
        const char* backend_str = dtfft_get_backend_string(static_cast<dtfft_backend_t>(backend));
        return { backend_str };
    }

    bool get_backend_pipelined(const Backend backend)
    {
        bool flag;
        dtfft_get_backend_pipelined(static_cast<dtfft_backend_t>(backend), &flag);
        return flag;
    }

    Error Plan::get_z_slab_enabled(bool* is_z_slab_enabled) const noexcept
    {
        return static_cast<Error>(dtfft_get_z_slab_enabled(_plan, is_z_slab_enabled));
    }

    bool Plan::get_z_slab_enabled() const
    {
        bool is_z_slab_enabled;
        DTFFT_CXX_CALL(get_z_slab_enabled(&is_z_slab_enabled))
        return is_z_slab_enabled;
    }

    Error Plan::get_y_slab_enabled(bool* is_y_slab_enabled) const noexcept
    {
        return static_cast<Error>(dtfft_get_y_slab_enabled(_plan, is_y_slab_enabled));
    }

    bool Plan::get_y_slab_enabled() const
    {
        bool is_y_slab_enabled;
        DTFFT_CXX_CALL(get_y_slab_enabled(&is_y_slab_enabled))
        return is_y_slab_enabled;
    }

    Error Plan::report() const noexcept
    {
        return static_cast<Error>(dtfft_report(_plan));
    }

#ifdef DTFFT_WITH_COMPRESSION
    Error Plan::report_compression() const noexcept
    {
        return static_cast<Error>(dtfft_report_compression(_plan));
    }
#endif

    Error Plan::get_pencil(const Layout layout, Pencil& pencil) const noexcept
    {
        dtfft_pencil_t c_pencil;
        dtfft_error_t error_code = dtfft_get_pencil(_plan, static_cast<dtfft_layout_t>(layout), &c_pencil);
        if (error_code == DTFFT_SUCCESS) {
            pencil = Pencil(c_pencil);
        }
        return static_cast<Error>(error_code);
    }

    Pencil
    Plan::get_pencil(const Layout layout) const
    {
        Pencil pencil;
        DTFFT_CXX_CALL(get_pencil(layout, pencil))
        return pencil;
    }

    Error Plan::execute(void* in, void* out, const Execute execute_type, void* aux) const noexcept
    {
        dtfft_error_t error_code = dtfft_execute(_plan, in, out, static_cast<dtfft_execute_t>(execute_type), aux);
        return static_cast<Error>(error_code);
    }

    Error Plan::forward(void* in, void* out, void* aux) const noexcept
    {
        return execute(in, out, Execute::FORWARD, aux);
    }

    Error Plan::backward(void* in, void* out, void* aux) const noexcept
    {
        return execute(in, out, Execute::BACKWARD, aux);
    }

    Error Plan::transpose(void* in, void* out, const Transpose transpose_type, void* aux) const noexcept
    {
        dtfft_error_t error_code = dtfft_transpose(_plan, in, out, static_cast<dtfft_transpose_t>(transpose_type), aux);
        return static_cast<Error>(error_code);
    }

    Error Plan::transpose_start(void* in, void* out, const Transpose transpose_type, dtfft_request_t* request) const noexcept
    {
        return transpose_start(in, out, transpose_type, nullptr, request);
    }

    Error Plan::transpose_start(void* in, void* out, const Transpose transpose_type, void* aux, dtfft_request_t* request) const noexcept
    {
        dtfft_error_t error_code = dtfft_transpose_start(_plan, in, out, static_cast<dtfft_transpose_t>(transpose_type), aux, request);
        return static_cast<Error>(error_code);
    }

    Error Plan::transpose_end(dtfft_request_t request) const noexcept
    {
        dtfft_error_t error_code = dtfft_transpose_end(_plan, request);
        return static_cast<Error>(error_code);
    }

    dtfft_request_t Plan::transpose_start(void* in, void* out, const Transpose transpose_type, void* aux) const
    {
        dtfft_request_t request;
        DTFFT_CXX_CALL(transpose_start(in, out, transpose_type, aux, &request));
        return request;
    }

    Error Plan::reshape(void* in, void* out, Reshape reshape_type, void* aux) const noexcept
    {
        dtfft_error_t error_code = dtfft_reshape(_plan, in, out, static_cast<dtfft_reshape_t>(reshape_type), aux);
        return static_cast<Error>(error_code);
    }

    Error Plan::reshape_start(void* in, void* out, Reshape reshape_type, void* aux, dtfft_request_t* request) const noexcept
    {
        dtfft_error_t error_code = dtfft_reshape_start(_plan, in, out, static_cast<dtfft_reshape_t>(reshape_type), aux, request);
        return static_cast<Error>(error_code);
    }

    Error Plan::reshape_start(void* in, void* out, Reshape reshape_type, dtfft_request_t* request) const noexcept
    {
        return reshape_start(in, out, reshape_type, nullptr, request);
    }

    Error Plan::reshape_end(dtfft_request_t request) const noexcept
    {
        dtfft_error_t error_code = dtfft_reshape_end(_plan, request);
        return static_cast<Error>(error_code);
    }

    dtfft_request_t Plan::reshape_start(void* in, void* out, Reshape reshape_type, void* aux) const
    {
        dtfft_request_t request;
        DTFFT_CXX_CALL(reshape_start(in, out, reshape_type, aux, &request));
        return request;
    }

    Error Plan::get_alloc_size(size_t* alloc_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_alloc_size(_plan, alloc_size));
    }

    size_t
    Plan::get_alloc_size() const
    {
        size_t alloc_size;
        DTFFT_CXX_CALL(get_alloc_size(&alloc_size))
        return alloc_size;
    }

    Error Plan::get_aux_size(size_t* aux_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_size(_plan, aux_size));
    }

    size_t
    Plan::get_aux_size() const
    {
        size_t aux_size;
        DTFFT_CXX_CALL(get_aux_size(&aux_size))
        return aux_size;
    }

    Error Plan::get_aux_bytes(size_t* aux_bytes) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_bytes(_plan, aux_bytes));
    }

    size_t
    Plan::get_aux_bytes() const
    {
        size_t aux_bytes;
        DTFFT_CXX_CALL(get_aux_bytes(&aux_bytes))
        return aux_bytes;
    }

    Error Plan::get_aux_size_reshape(size_t* aux_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_size_reshape(_plan, aux_size));
    }

    size_t
    Plan::get_aux_size_reshape() const
    {
        size_t aux_size;
        DTFFT_CXX_CALL(get_aux_size_reshape(&aux_size))
        return aux_size;
    }

    Error Plan::get_aux_bytes_reshape(size_t* aux_bytes) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_bytes_reshape(_plan, aux_bytes));
    }

    size_t
    Plan::get_aux_bytes_reshape() const
    {
        size_t aux_bytes;
        DTFFT_CXX_CALL(get_aux_bytes_reshape(&aux_bytes))
        return aux_bytes;
    }

    Error Plan::get_aux_size_transpose(size_t* aux_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_size_transpose(_plan, aux_size));
    }

    size_t
    Plan::get_aux_size_transpose() const
    {
        size_t aux_size;
        DTFFT_CXX_CALL(get_aux_size_transpose(&aux_size))
        return aux_size;
    }

    Error Plan::get_aux_bytes_transpose(size_t* aux_bytes) const noexcept
    {
        return static_cast<Error>(dtfft_get_aux_bytes_transpose(_plan, aux_bytes));
    }

    size_t
    Plan::get_aux_bytes_transpose() const
    {
        size_t aux_bytes;
        DTFFT_CXX_CALL(get_aux_bytes_transpose(&aux_bytes))
        return aux_bytes;
    }


    Error Plan::get_local_sizes(
        std::vector<int32_t>& in_starts,
        std::vector<int32_t>& in_counts,
        std::vector<int32_t>& out_starts,
        std::vector<int32_t>& out_counts,
        size_t* alloc_size) const noexcept
    {
        return get_local_sizes(in_starts.data(), in_counts.data(), out_starts.data(), out_counts.data(), alloc_size);
    }

    Error Plan::get_local_sizes(
        int32_t* in_starts,
        int32_t* in_counts,
        int32_t* out_starts,
        int32_t* out_counts,
        size_t* alloc_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_local_sizes(_plan, in_starts, in_counts, out_starts, out_counts, alloc_size));
    }

    Error Plan::get_element_size(size_t* element_size) const noexcept
    {
        return static_cast<Error>(dtfft_get_element_size(_plan, element_size));
    }

    size_t
    Plan::get_element_size() const
    {
        size_t element_size;
        DTFFT_CXX_CALL(get_element_size(&element_size))
        return element_size;
    }

    Error Plan::get_alloc_bytes(size_t* alloc_bytes) const noexcept
    {
        return static_cast<Error>(dtfft_get_alloc_bytes(_plan, alloc_bytes));
    }

    size_t
    Plan::get_alloc_bytes() const
    {
        size_t alloc_bytes;
        DTFFT_CXX_CALL(get_alloc_bytes(&alloc_bytes))
        return alloc_bytes;
    }

    Error Plan::get_executor(Executor* executor) const noexcept
    {
        dtfft_executor_t executor_;
        const auto error_code = static_cast<Error>(dtfft_get_executor(_plan, &executor_));
        if (error_code == Error::SUCCESS) {
            *executor = static_cast<Executor>(executor_);
        }
        return error_code;
    }

    Executor
    Plan::get_executor() const
    {
        Executor executor;
        DTFFT_CXX_CALL(get_executor(&executor))
        return executor;
    }

    Error Plan::get_precision(Precision* precision) const noexcept
    {
        dtfft_precision_t precision_;
        const auto error_code = static_cast<Error>(dtfft_get_precision(_plan, &precision_));
        if (error_code == Error::SUCCESS) {
            *precision = static_cast<Precision>(precision_);
        }
        return error_code;
    }

    Precision
    Plan::get_precision() const
    {
        Precision precision;
        DTFFT_CXX_CALL(get_precision(&precision))
        return precision;
    }

    Error Plan::get_dims(int8_t* ndims, const int32_t* dims[]) const noexcept
    {
        return static_cast<Error>(dtfft_get_dims(_plan, ndims, dims));
    }

    std::vector<int32_t>
    Plan::get_dims() const
    {
        int8_t ndims;
        const int32_t* dims_ptr;
        DTFFT_CXX_CALL(get_dims(&ndims, &dims_ptr))
        return { dims_ptr, dims_ptr + ndims };
    }

    Error Plan::get_grid_dims(int8_t* ndims, const int32_t* grid_dims[]) const noexcept
    {
        return static_cast<Error>(dtfft_get_grid_dims(_plan, ndims, grid_dims));
    }

    std::vector<int32_t>
    Plan::get_grid_dims() const
    {
        int8_t ndims;
        const int32_t* dims_ptr;
        DTFFT_CXX_CALL(get_grid_dims(&ndims, &dims_ptr))
        return { dims_ptr, dims_ptr + ndims };
    }

    Error Plan::mem_alloc(size_t alloc_bytes, void** ptr) const noexcept
    {
        return static_cast<Error>(dtfft_mem_alloc(_plan, alloc_bytes, ptr));
    }

    void* Plan::mem_alloc(size_t alloc_bytes) const
    {
        void* ptr = nullptr;
        DTFFT_CXX_CALL(mem_alloc(alloc_bytes, &ptr))
        return ptr;
    }

    Error Plan::mem_free(void* ptr) const noexcept
    {
        return static_cast<Error>(dtfft_mem_free(_plan, ptr));
    }

    Error Plan::destroy() noexcept
    {
        return static_cast<Error>(dtfft_destroy(&_plan));
    }

    Error Plan::get_backend(Backend& backend) const noexcept
    {
        dtfft_backend_t backend_;
        dtfft_error_t error_code = dtfft_get_backend(_plan, &backend_);
        backend = static_cast<Backend>(backend_);
        return static_cast<Error>(error_code);
    }

    Backend
    Plan::get_backend() const
    {
        Backend backend;
        DTFFT_CXX_CALL(get_backend(backend))
        return backend;
    }

    Error Plan::get_reshape_backend(Backend& backend) const noexcept
    {
        dtfft_backend_t backend_;
        dtfft_error_t error_code = dtfft_get_reshape_backend(_plan, &backend_);
        backend = static_cast<Backend>(backend_);
        return static_cast<Error>(error_code);
    }

    Backend
    Plan::get_reshape_backend() const
    {
        Backend backend;
        DTFFT_CXX_CALL(get_reshape_backend(backend))
        return backend;
    }

    Error set_config(const Config& config) noexcept
    {
        const auto c_config = config.c_struct(); // Store in variable
        return static_cast<Error>(dtfft_set_config(&c_config));
    }

#ifdef DTFFT_WITH_CUDA
    Error Plan::get_stream(dtfft_stream_t* stream) const noexcept
    {
        return static_cast<Error>(dtfft_get_stream(_plan, stream));
    }

    dtfft_stream_t
    Plan::get_stream() const
    {
        dtfft_stream_t stream = nullptr;
        DTFFT_CXX_CALL(get_stream(&stream))
        return stream;
    }

    Error Plan::get_platform(Platform& platform) const noexcept
    {
        dtfft_platform_t platform_;
        dtfft_error_t error_code = dtfft_get_platform(_plan, &platform_);
        platform = static_cast<Platform>(platform_);
        return static_cast<Error>(error_code);
    }

    Platform
    Plan::get_platform() const
    {
        Platform platform;
        DTFFT_CXX_CALL(get_platform(platform))
        return platform;
    }
#endif

    PlanC2C::PlanC2C(
        const std::vector<int32_t>& dims,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
        : PlanC2C(static_cast<int8_t>(dims.size()), dims.data(), comm, precision, effort, executor)
    {
    }

    PlanC2C::PlanC2C(
        const std::vector<int32_t>& dims,
        const Precision precision,
        const Effort effort)
        : PlanC2C(static_cast<int8_t>(dims.size()), dims.data(), MPI_COMM_WORLD, precision, effort)
    {
    }

    PlanC2C::PlanC2C(
        const int8_t ndims,
        const int32_t* dims,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
    {
        dtfft_error_t error_code = dtfft_create_plan_c2c(ndims, dims, comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }

    PlanC2C::PlanC2C(
        const Pencil& pencil,
        const Precision precision,
        const Effort effort)
        : PlanC2C(pencil, MPI_COMM_WORLD, precision, effort)
    {
    }

    PlanC2C::PlanC2C(
        const Pencil& pencil,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
    {
        dtfft_error_t error_code = dtfft_create_plan_c2c_pencil(&pencil.c_struct(), comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }
    #ifndef DTFFT_TRANSPOSE_ONLY
    PlanR2C::PlanR2C(
        const std::vector<int32_t>& dims,
        const Executor executor,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort)
        : PlanR2C(static_cast<int8_t>(dims.size()), dims.data(), executor, comm, precision, effort)
    {
    }

    PlanR2C::PlanR2C(
        const int8_t ndims,
        const int32_t* dims,
        const Executor executor,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort)
    {
        dtfft_error_t error_code = dtfft_create_plan_r2c(ndims, dims, comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }

    PlanR2C::PlanR2C(
        const Pencil& pencil,
        const Executor executor,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort)
    {
        dtfft_error_t error_code = dtfft_create_plan_r2c_pencil(&pencil.c_struct(), comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }
    #endif

    PlanR2R::PlanR2R(
        const std::vector<int32_t>& dims,
        const std::vector<R2RKind>& kinds,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
        : PlanR2R(static_cast<int8_t>(dims.size()), dims.data(), kinds.data(), comm, precision, effort, executor)
    {
    }

    PlanR2R::PlanR2R(
        const std::vector<int32_t>& dims,
        const Precision precision,
        const Effort effort)
        : PlanR2R(static_cast<int8_t>(dims.size()), dims.data(), nullptr, MPI_COMM_WORLD, precision, effort)
    {
    }

    PlanR2R::PlanR2R(
        const int8_t ndims,
        const int32_t* dims,
        const R2RKind* kinds,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
    {
        dtfft_error_t error_code = dtfft_create_plan_r2r(ndims, dims, reinterpret_cast<const dtfft_r2r_kind_t*>(kinds), comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }

    PlanR2R::PlanR2R(
        const Pencil& pencil,
        const Precision precision,
        const Effort effort)
        : PlanR2R(pencil, nullptr, MPI_COMM_WORLD, precision, effort)
    {
    }

    PlanR2R::PlanR2R(
        const Pencil& pencil,
        const std::vector<R2RKind>& kinds,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
        : PlanR2R(pencil, kinds.data(), comm, precision, effort, executor)
    {
    }

    PlanR2R::PlanR2R(
        const Pencil& pencil,
        const R2RKind* kinds,
        MPI_Comm comm,
        const Precision precision,
        const Effort effort,
        const Executor executor)
    {
        dtfft_error_t error_code = dtfft_create_plan_r2r_pencil(
            &pencil.c_struct(),
            reinterpret_cast<const dtfft_r2r_kind_t*>(kinds),
            comm,
            static_cast<dtfft_precision_t>(precision),
            static_cast<dtfft_effort_t>(effort),
            static_cast<dtfft_executor_t>(executor), &_plan);
        DTFFT_CXX_CALL(static_cast<Error>(error_code))
    }
}
