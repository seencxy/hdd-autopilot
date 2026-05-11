#include "app/ffi.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "argon2-opencl/globalcontext.h"

#include "app/job.hpp"
#include "app/solver.hpp"

namespace {
thread_local std::string g_last_error;

void set_last_error(const std::exception& error) {
    g_last_error = error.what();
}

void clear_last_error() {
    g_last_error.clear();
}

template <std::size_t N>
void copy_cstr(char (&target)[N], const std::string& value) {
    std::fill(std::begin(target), std::end(target), '\0');
    std::strncpy(target, value.c_str(), N - 1);
}

bool is_supported_compute_device(const argon2::opencl::Device& device) {
    try {
        const auto& cl_device = device.getCLDevice();
        const auto device_type = cl_device.getInfo<CL_DEVICE_TYPE>();
        const auto is_gpu_like = (device_type & CL_DEVICE_TYPE_GPU) != 0
            || (device_type & CL_DEVICE_TYPE_ACCELERATOR) != 0;
        return is_gpu_like
            && cl_device.getInfo<CL_DEVICE_AVAILABLE>()
            && cl_device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
    } catch (const std::exception&) {
        return false;
    }
}

std::size_t first_supported_device_index(const std::vector<argon2::opencl::Device>& devices) {
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (is_supported_compute_device(devices[index])) {
            return index;
        }
    }
    throw std::runtime_error("no supported OpenCL GPU devices found");
}

app::Job benchmark_job() {
    return app::Job(app::JobConfig{
        .seed = "benchmark-seed-fixed",
        .round_id = 1,
        .visitor_id = "benchmark-visitor-fixed",
        .challenge_id = 1,
        .session_salt = "benchmark-session-salt-fixed",
        .time_cost = 1,
        .memory_cost_mb = 64,
        .parallelism = 1,
        .difficulty_bits = 255,
    });
}

app::Job make_job(const mining_opencl_job& raw) {
    const auto seed = std::string(
        reinterpret_cast<const char*>(raw.seed_ptr),
        reinterpret_cast<const char*>(raw.seed_ptr) + raw.seed_len);
    const auto pass_prefix = std::string(
        reinterpret_cast<const char*>(raw.pass_prefix_ptr),
        reinterpret_cast<const char*>(raw.pass_prefix_ptr) + raw.pass_prefix_len);
    return app::Job(app::JobConfig{
        .seed = seed,
        .pass_prefix_override = pass_prefix,
        .time_cost = static_cast<int>(raw.time_cost),
        .memory_cost_mb = static_cast<int>(raw.memory_cost_kib / 1024),
        .parallelism = static_cast<int>(raw.parallelism),
        .difficulty_bits = raw.difficulty_bits,
    });
}

app::SolverConfig make_solver_config(const mining_opencl_solver_config& raw) {
    return app::SolverConfig{
        .batch_size = raw.batch_size,
        .by_segment = raw.by_segment,
        .precompute_refs = raw.precompute_refs,
    };
}

void fill_mine_result(const app::SolveResult& mined,
                      const std::atomic<std::int64_t>& attempts,
                      mining_opencl_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = attempts.load();
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (!mined.digest.empty()) {
        std::strncpy(result->digest_hex, mined.digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_device_info(std::size_t device_index, mining_opencl_device_info* result) {
    argon2::opencl::GlobalContext global;
    const auto& devices = global.getAllDevices();
    if (device_index >= devices.size()) {
        throw std::runtime_error("OpenCL device index out of range");
    }

    const auto& device = devices[device_index];
    const auto& cl_device = device.getCLDevice();
    const auto device_type = cl_device.getInfo<CL_DEVICE_TYPE>();
    const auto vendor = cl_device.getInfo<CL_DEVICE_VENDOR>();
    const cl::Platform platform(cl_device.getInfo<CL_DEVICE_PLATFORM>());
    const auto platform_name = platform.getInfo<CL_PLATFORM_NAME>();

    result->device_index = device_index;
    result->device_type = static_cast<std::uint64_t>(device_type);
    result->global_memory_bytes = static_cast<std::uint64_t>(cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    result->max_alloc_bytes = static_cast<std::uint64_t>(cl_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    result->compute_units = static_cast<std::uint32_t>(cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    result->max_work_group_size = static_cast<std::uint32_t>(cl_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    result->local_memory_bytes = static_cast<std::uint64_t>(cl_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    result->host_unified_memory = cl_device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>() != 0;

    const auto device_id = std::string("opencl:") + std::to_string(device_index);
    const auto name = device.getName();
    copy_cstr(result->device_id, device_id);
    copy_cstr(result->name, name);
    copy_cstr(result->vendor, vendor);
    copy_cstr(result->platform, platform_name);
}
} // namespace

struct mining_opencl_session {
    app::Job job;
    app::Solver solver;
    app::SolverSession session;
    std::atomic_bool stop{false};
    std::atomic<std::int64_t> attempts{0};

    mining_opencl_session(std::size_t device_index,
                       app::Job&& native_job,
                       const app::SolverConfig& native_config,
                       std::uint64_t start_nonce)
        : job(std::move(native_job)),
          solver(device_index),
          session(solver.create_session(job, native_config, start_nonce)) {
    }
};

bool mining_opencl_is_available() {
    try {
        clear_last_error();
        argon2::opencl::GlobalContext global;
        const auto& devices = global.getAllDevices();
        return std::any_of(devices.begin(), devices.end(), is_supported_compute_device);
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_opencl_validate() {
    try {
        clear_last_error();
        argon2::opencl::GlobalContext global;
        app::Solver solver(first_supported_device_index(global.getAllDevices()));
        solver.validate_against_reference(benchmark_job(), 1);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

std::size_t mining_opencl_device_count() {
    try {
        clear_last_error();
        argon2::opencl::GlobalContext global;
        return global.getAllDevices().size();
    } catch (const std::exception& error) {
        set_last_error(error);
        return 0;
    }
}

bool mining_opencl_get_device_info(std::size_t device_index, mining_opencl_device_info* result) {
    if (result == nullptr) {
        g_last_error = "device info pointer is null";
        return false;
    }
    try {
        clear_last_error();
        fill_device_info(device_index, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

const char* mining_opencl_last_error_message() {
    return g_last_error.c_str();
}

bool mining_opencl_default_solver_config(
    std::size_t device_index,
    const mining_opencl_job* job,
    mining_opencl_solver_config* result) {
    if (job == nullptr || result == nullptr) {
        g_last_error = "default_solver_config parameter is null";
        return false;
    }
    try {
        clear_last_error();
        app::Solver solver(device_index);
        const auto native_job = make_job(*job);
        const auto config = solver.default_config_for(native_job);
        result->batch_size = config.batch_size;
        result->by_segment = config.by_segment;
        result->precompute_refs = config.precompute_refs;
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_opencl_find_best_benchmark_config(std::size_t device_index, mining_opencl_benchmark_result* result) {
    if (result == nullptr) {
        g_last_error = "benchmark result pointer is null";
        return false;
    }
    try {
        clear_last_error();
        app::Solver solver(device_index);
        const auto best = solver.find_best_benchmark_config();
        result->batch_size = best.config.batch_size;
        result->by_segment = best.config.by_segment;
        result->precompute_refs = best.config.precompute_refs;
        result->attempts = best.attempts;
        result->elapsed_ms = best.elapsed.count();
        result->attempts_per_second = best.attempts_per_second;
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_opencl_mine_batch(
    std::size_t device_index,
    const mining_opencl_job* job,
    const mining_opencl_solver_config* config,
    std::uint64_t start_nonce,
    mining_opencl_mine_result* result) {
    if (job == nullptr || config == nullptr || result == nullptr) {
        g_last_error = "mine_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        app::Solver solver(device_index);
        const auto native_job = make_job(*job);
        const auto native_config = make_solver_config(*config);
        std::atomic_bool stop{false};
        std::atomic<std::int64_t> attempts{0};
        const auto mined = solver.mine_batch(native_job, native_config, start_nonce, stop, attempts);
        fill_mine_result(mined, attempts, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

mining_opencl_session* mining_opencl_session_create(
    std::size_t device_index,
    const mining_opencl_job* job,
    const mining_opencl_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        auto native_job = make_job(*job);
        const auto native_config = make_solver_config(*config);
        return new mining_opencl_session(device_index, std::move(native_job), native_config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_opencl_session_mine_next_batch(
    mining_opencl_session* session,
    mining_opencl_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = session->solver.mine_next_batch(session->job, session->session, session->stop, session->attempts);
        fill_mine_result(mined, session->attempts, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

void mining_opencl_session_destroy(mining_opencl_session* session) {
    delete session;
}
