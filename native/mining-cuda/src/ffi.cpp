#include "app/ffi.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "argon2-cuda/globalcontext.h"

#include <argon2.h>

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

app::JobConfig benchmark_job_config(int difficulty_bits) {
    app::JobConfig config;
    config.seed = "benchmark-seed-fixed";
    config.round_id = 1;
    config.visitor_id = "benchmark-visitor-fixed";
    config.challenge_id = 1;
    config.session_salt = "benchmark-session-salt-fixed";
    config.time_cost = 1;
    config.memory_cost_mb = 64;
    config.parallelism = 1;
    config.difficulty_bits = difficulty_bits;
    return config;
}

app::Job benchmark_job() {
    return app::Job(benchmark_job_config(255));
}

void validate_mining_path(app::Solver& solver) {
    auto job_config = benchmark_job_config(0);
    job_config.memory_cost_mb = 8;
    auto job = app::Job(std::move(job_config));
    app::SolverConfig config;
    config.batch_size = 1;
    config.by_segment = false;
    config.precompute_refs = false;
    config.generate_passwords_on_gpu = true;

    std::atomic_bool stop{false};
    std::atomic<std::int64_t> attempts{0};
    const auto mined = solver.mine_batch(job, config, 1, stop, attempts);
    if (!mined.found || mined.digest.empty()) {
        throw std::runtime_error("CUDA difficulty check validation did not return a hit");
    }

    const auto password = job.password_for_nonce(mined.nonce);
    std::vector<std::uint8_t> digest(32);
    const auto result = argon2id_hash_raw(job.time_cost(),
                                          job.memory_cost_kb(),
                                          job.parallelism(),
                                          password.data(),
                                          password.size(),
                                          job.seed_bytes().data(),
                                          job.seed_bytes().size(),
                                          digest.data(),
                                          digest.size());
    if (result != ARGON2_OK) {
        throw std::runtime_error(argon2_error_message(result));
    }
    const auto expected = app::hex_encode(digest.data(), digest.size());
    if (mined.digest != expected) {
        throw std::runtime_error("CUDA difficulty check digest mismatch");
    }
}

app::Job make_job(const mining_cuda_job& raw) {
    const auto seed = std::string(
        reinterpret_cast<const char*>(raw.seed_ptr),
        reinterpret_cast<const char*>(raw.seed_ptr) + raw.seed_len);
    const auto pass_prefix = std::string(
        reinterpret_cast<const char*>(raw.pass_prefix_ptr),
        reinterpret_cast<const char*>(raw.pass_prefix_ptr) + raw.pass_prefix_len);
    app::JobConfig config;
    config.seed = seed;
    config.pass_prefix_override = pass_prefix;
    config.time_cost = static_cast<int>(raw.time_cost);
    config.memory_cost_mb = static_cast<int>(raw.memory_cost_kib / 1024);
    config.parallelism = static_cast<int>(raw.parallelism);
    config.difficulty_bits = raw.difficulty_bits;
    return app::Job(std::move(config));
}

app::SolverConfig make_solver_config(const mining_cuda_solver_config& raw) {
    app::SolverConfig config;
    config.batch_size = raw.batch_size;
    config.by_segment = raw.by_segment;
    config.precompute_refs = raw.precompute_refs;
    config.generate_passwords_on_gpu = raw.generate_passwords_on_gpu;
    return config;
}

void fill_mine_result(const app::SolveResult& mined,
                      const std::atomic<std::int64_t>& attempts,
                      mining_cuda_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = attempts.load();
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (!mined.digest.empty()) {
        std::strncpy(result->digest_hex, mined.digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_rpow2_mine_result(const app::Rpow2CudaBatchResult& mined,
                            mining_cuda_rpow2_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = mined.attempts;
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (mined.found) {
        const auto digest = app::hex_encode(mined.digest.data(), mined.digest.size());
        std::strncpy(result->digest_hex, digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_dwc_mine_result(const app::DwcCudaBatchResult& mined,
                          mining_cuda_dwc_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = mined.attempts;
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (mined.found) {
        const auto digest = app::hex_encode(mined.digest.data(), mined.digest.size());
        std::strncpy(result->digest_hex, digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_h98hash_mine_result(const app::H98HashCudaBatchResult& mined,
                              mining_cuda_h98hash_mine_result* result) {
    result->found = mined.found;
    result->nonce_tail = mined.nonce_tail;
    result->attempts = mined.attempts;
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (mined.found) {
        const auto digest = app::hex_encode(mined.digest.data(), mined.digest.size());
        std::strncpy(result->digest_hex, digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_h256hash_mine_result(const app::H256HashCudaBatchResult& mined,
                               mining_cuda_h256hash_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = mined.attempts;
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (mined.found) {
        const auto digest = app::hex_encode(mined.digest.data(), mined.digest.size());
        std::strncpy(result->digest_hex, digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_device_info(std::size_t device_index, mining_cuda_device_info* result) {
    argon2::cuda::GlobalContext global;
    const auto& devices = global.getAllDevices();
    if (device_index >= devices.size()) {
        throw std::runtime_error("CUDA device index out of range");
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, static_cast<int>(device_index));

    result->device_index = device_index;
    result->global_memory_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
    result->max_alloc_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
    result->compute_units = static_cast<std::uint32_t>(std::max(prop.multiProcessorCount, 0));
    result->max_threads_per_block = static_cast<std::uint32_t>(std::max(prop.maxThreadsPerBlock, 0));
    result->warp_size = static_cast<std::uint32_t>(std::max(prop.warpSize, 0));
    result->shared_memory_per_block_bytes = static_cast<std::uint64_t>(prop.sharedMemPerBlock);
    std::fill(std::begin(result->device_id), std::end(result->device_id), '\0');
    std::fill(std::begin(result->name), std::end(result->name), '\0');

    const auto device_id = std::string("cuda:") + std::to_string(device_index);
    std::strncpy(result->device_id, device_id.c_str(), sizeof(result->device_id) - 1);
    std::strncpy(result->name, prop.name, sizeof(result->name) - 1);
}
} // namespace

struct mining_cuda_session {
    app::Job job;
    app::Solver solver;
    app::SolverSession session;
    std::atomic_bool stop{false};
    std::atomic<std::int64_t> attempts{0};

    mining_cuda_session(std::size_t device_index,
                     app::Job&& native_job,
                     const app::SolverConfig& native_config,
                     std::uint64_t start_nonce)
        : job(std::move(native_job)),
          solver(device_index),
          session(solver.create_session(job, native_config, start_nonce)) {
    }
};

struct mining_cuda_rpow2_session {
    app::Rpow2CudaSession session;

    mining_cuda_rpow2_session(std::size_t device_index,
                              const mining_cuda_rpow2_job& job,
                              const mining_cuda_rpow2_solver_config& config,
                              std::uint64_t start_nonce)
        : session(device_index,
                  job.nonce_prefix_ptr,
                  job.nonce_prefix_len,
	                  job.difficulty_bits,
	                  config.batch_size,
	                  start_nonce,
	                  config.threads_per_block,
	                  config.nonces_per_thread,
	                  config.max_blocks,
	                  config.early_exit) {
	    }
	};

struct mining_cuda_h98hash_session {
    app::H98HashCudaSession session;

    mining_cuda_h98hash_session(std::size_t device_index,
                                const mining_cuda_h98hash_job& job,
                                const mining_cuda_rpow2_solver_config& config,
                                std::uint64_t start_nonce)
        : session(device_index,
                  job.challenge_ptr,
                  job.challenge_len,
                  job.nonce_prefix_ptr,
                  job.nonce_prefix_len,
                  job.difficulty_bits,
                  config.batch_size,
                  start_nonce,
                  config.threads_per_block,
                  config.nonces_per_thread,
                  config.max_blocks,
                  config.early_exit) {
    }
};

struct mining_cuda_h256hash_session {
    app::H256HashCudaSession session;

    mining_cuda_h256hash_session(std::size_t device_index,
                                 const mining_cuda_h256hash_job& job,
                                 const mining_cuda_rpow2_solver_config& config,
                                 std::uint64_t start_nonce)
        : session(device_index,
                  job.challenge_ptr,
                  job.challenge_len,
                  job.target_ptr,
                  job.target_len,
                  config.batch_size,
                  start_nonce,
                  config.threads_per_block,
                  config.nonces_per_thread,
                  config.max_blocks,
                  config.early_exit) {
    }
};

struct mining_cuda_dwc_session {
    app::DwcCudaSession session;

    mining_cuda_dwc_session(std::size_t device_index,
                            const mining_cuda_dwc_job& job,
                            const mining_cuda_dwc_solver_config& config,
                            std::uint64_t start_nonce)
        : session(device_index,
                  job.prefix_ptr,
                  job.prefix_len,
                  job.difficulty_bits,
                  config.batch_size,
                  start_nonce,
                  config.threads_per_block,
                  config.nonces_per_thread,
                  config.max_blocks,
                  config.early_exit) {
    }
};

bool mining_cuda_is_available() {
    try {
        clear_last_error();
        app::Solver solver(0);
        solver.default_config_for(benchmark_job());
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_cuda_dwc_is_available() {
    return mining_cuda_is_available();
}

mining_cuda_dwc_session* mining_cuda_dwc_session_create(
    std::size_t device_index,
    const mining_cuda_dwc_job* job,
    const mining_cuda_dwc_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "dwc session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        return new mining_cuda_dwc_session(device_index, *job, *config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_cuda_dwc_session_mine_next_batch(
    mining_cuda_dwc_session* session,
    mining_cuda_dwc_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "dwc session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = session->session.mine_next_batch();
        fill_dwc_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

void mining_cuda_dwc_session_destroy(mining_cuda_dwc_session* session) {
    delete session;
}

bool mining_cuda_validate() {
    try {
        clear_last_error();
        app::Solver solver(0);
        solver.validate_against_reference(benchmark_job(), 1);
        validate_mining_path(solver);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

std::size_t mining_cuda_device_count() {
    try {
        clear_last_error();
        argon2::cuda::GlobalContext global;
        return global.getAllDevices().size();
    } catch (const std::exception& error) {
        set_last_error(error);
        return 0;
    }
}

bool mining_cuda_get_device_info(std::size_t device_index, mining_cuda_device_info* result) {
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

const char* mining_cuda_last_error_message() {
    return g_last_error.c_str();
}

bool mining_cuda_default_solver_config(
    std::size_t device_index,
    const mining_cuda_job* job,
    mining_cuda_solver_config* result) {
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
        result->generate_passwords_on_gpu = config.generate_passwords_on_gpu;
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_cuda_find_best_benchmark_config(std::size_t device_index, mining_cuda_benchmark_result* result) {
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
        result->generate_passwords_on_gpu = best.config.generate_passwords_on_gpu;
        result->attempts = best.attempts;
        result->elapsed_ms = best.elapsed.count();
        result->attempts_per_second = best.attempts_per_second;
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_cuda_mine_batch(
    std::size_t device_index,
    const mining_cuda_job* job,
    const mining_cuda_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_mine_result* result) {
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

mining_cuda_session* mining_cuda_session_create(
    std::size_t device_index,
    const mining_cuda_job* job,
    const mining_cuda_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        auto native_job = make_job(*job);
        const auto native_config = make_solver_config(*config);
        return new mining_cuda_session(device_index, std::move(native_job), native_config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_cuda_session_mine_next_batch(
    mining_cuda_session* session,
    mining_cuda_mine_result* result) {
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

void mining_cuda_session_destroy(mining_cuda_session* session) {
    delete session;
}

bool mining_cuda_rpow2_mine_batch(
    std::size_t device_index,
    const mining_cuda_rpow2_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_rpow2_mine_result* result) {
    if (job == nullptr || config == nullptr || result == nullptr) {
        g_last_error = "rpow2 mine_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
	        const auto mined = app::mine_rpow2_cuda_batch(
	            device_index,
	            job->nonce_prefix_ptr,
	            job->nonce_prefix_len,
	            job->difficulty_bits,
	            config->batch_size,
	            start_nonce,
	            config->threads_per_block,
	            config->nonces_per_thread,
	            config->max_blocks,
	            config->early_exit);
	        fill_rpow2_mine_result(mined, result);
	        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

mining_cuda_rpow2_session* mining_cuda_rpow2_session_create(
    std::size_t device_index,
    const mining_cuda_rpow2_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "rpow2 session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        return new mining_cuda_rpow2_session(device_index, *job, *config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_cuda_rpow2_session_mine_next_batch(
    mining_cuda_rpow2_session* session,
    mining_cuda_rpow2_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "rpow2 session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = session->session.mine_next_batch();
        fill_rpow2_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

	void mining_cuda_rpow2_session_destroy(mining_cuda_rpow2_session* session) {
	    delete session;
	}

	bool mining_cuda_rpow2_benchmark(
	    std::size_t device_index,
	    const mining_cuda_rpow2_job* job,
	    const mining_cuda_rpow2_solver_config* config,
	    std::uint32_t duration_ms,
	    mining_cuda_rpow2_benchmark_result* result) {
	    if (job == nullptr || config == nullptr || result == nullptr) {
	        g_last_error = "rpow2 benchmark parameter is null";
	        return false;
	    }
	    try {
	        clear_last_error();
	        const auto benchmark = app::benchmark_rpow2_cuda(
	            device_index,
	            job->nonce_prefix_ptr,
	            job->nonce_prefix_len,
	            job->difficulty_bits,
	            config->batch_size,
	            duration_ms,
	            config->threads_per_block,
	            config->nonces_per_thread,
	            config->max_blocks,
	            config->early_exit);
	        result->attempts = benchmark.attempts;
	        result->batches = benchmark.batches;
	        result->elapsed_ms = benchmark.elapsed_milliseconds;
	        result->kernel_ms = benchmark.kernel_milliseconds;
	        result->empty_launch_us = benchmark.empty_launch_microseconds;
	        result->kernel_hashrate = benchmark.kernel_hashrate;
	        result->effective_hashrate = benchmark.effective_hashrate;
	        return true;
	    } catch (const std::exception& error) {
	        set_last_error(error);
	        return false;
	    }
	}

bool mining_cuda_h98hash_mine_batch(
    std::size_t device_index,
    const mining_cuda_h98hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_h98hash_mine_result* result) {
    if (job == nullptr || config == nullptr || result == nullptr) {
        g_last_error = "h98hash mine_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = app::mine_h98hash_cuda_batch(
            device_index,
            job->challenge_ptr,
            job->challenge_len,
            job->nonce_prefix_ptr,
            job->nonce_prefix_len,
            job->difficulty_bits,
            config->batch_size,
            start_nonce,
            config->threads_per_block,
            config->nonces_per_thread,
            config->max_blocks,
            config->early_exit);
        fill_h98hash_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

mining_cuda_h98hash_session* mining_cuda_h98hash_session_create(
    std::size_t device_index,
    const mining_cuda_h98hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "h98hash session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        return new mining_cuda_h98hash_session(device_index, *job, *config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_cuda_h98hash_session_mine_next_batch(
    mining_cuda_h98hash_session* session,
    mining_cuda_h98hash_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "h98hash session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = session->session.mine_next_batch();
        fill_h98hash_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

void mining_cuda_h98hash_session_destroy(mining_cuda_h98hash_session* session) {
    delete session;
}

bool mining_cuda_h256hash_mine_batch(
    std::size_t device_index,
    const mining_cuda_h256hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_h256hash_mine_result* result) {
    if (job == nullptr || config == nullptr || result == nullptr) {
        g_last_error = "h256hash mine_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = app::mine_h256hash_cuda_batch(
            device_index,
            job->challenge_ptr,
            job->challenge_len,
            job->target_ptr,
            job->target_len,
            config->batch_size,
            start_nonce,
            config->threads_per_block,
            config->nonces_per_thread,
            config->max_blocks,
            config->early_exit);
        fill_h256hash_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

mining_cuda_h256hash_session* mining_cuda_h256hash_session_create(
    std::size_t device_index,
    const mining_cuda_h256hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "h256hash session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        return new mining_cuda_h256hash_session(device_index, *job, *config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_cuda_h256hash_session_mine_next_batch(
    mining_cuda_h256hash_session* session,
    mining_cuda_h256hash_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "h256hash session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto mined = session->session.mine_next_batch();
        fill_h256hash_mine_result(mined, result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

void mining_cuda_h256hash_session_destroy(mining_cuda_h256hash_session* session) {
    delete session;
}

	bool mining_argon2id_hash_raw(
    const std::uint8_t* password_ptr,
    std::size_t password_len,
    const std::uint8_t* salt_ptr,
    std::size_t salt_len,
    std::uint32_t time_cost,
    std::uint32_t memory_cost_kib,
    std::uint32_t parallelism,
    std::uint8_t* digest_ptr,
    std::size_t digest_len) {
    if (password_ptr == nullptr || salt_ptr == nullptr || digest_ptr == nullptr || digest_len == 0) {
        g_last_error = "argon2 parameter is null";
        return false;
    }
    try {
        clear_last_error();
        const auto result = argon2id_hash_raw(time_cost,
                                              memory_cost_kib,
                                              parallelism,
                                              password_ptr,
                                              password_len,
                                              salt_ptr,
                                              salt_len,
                                              digest_ptr,
                                              digest_len);
        if (result != ARGON2_OK) {
            g_last_error = argon2_error_message(result);
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}
