#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "app/job.hpp"

namespace app {

struct SolverSessionState;

struct SolverConfig {
    std::size_t batch_size = 1;
    bool by_segment = false;
    bool precompute_refs = false;
    bool generate_passwords_on_gpu = true;
};

struct SolveResult {
    bool found = false;
    std::uint64_t nonce = 0;
    std::string digest;
};

struct SolverSession {
    SolverConfig config;
    std::uint64_t next_nonce = 1;
    std::unique_ptr<SolverSessionState> state;

    SolverSession();
    SolverSession(SolverSession&&) noexcept;
    SolverSession& operator=(SolverSession&&) noexcept;
    ~SolverSession();

    SolverSession(const SolverSession&) = delete;
    SolverSession& operator=(const SolverSession&) = delete;
};

struct BenchmarkResult {
    SolverConfig config;
    std::int64_t attempts = 0;
    std::chrono::milliseconds elapsed{0};
    double attempts_per_second = 0.0;
};

struct Rpow2CudaBatchResult {
    bool found = false;
    std::uint64_t nonce = 0;
    std::int64_t attempts = 0;
    std::array<std::uint8_t, 32> digest{};
};

struct Rpow2CudaProfiledBatchResult {
    Rpow2CudaBatchResult result;
    double wall_milliseconds = 0.0;
    double kernel_milliseconds = 0.0;
};

struct Rpow2CudaBenchmarkResult {
    std::uint64_t attempts = 0;
    std::uint64_t batches = 0;
    double elapsed_milliseconds = 0.0;
    double kernel_milliseconds = 0.0;
    double empty_launch_microseconds = 0.0;
    double kernel_hashrate = 0.0;
    double effective_hashrate = 0.0;
};

class Rpow2CudaSession {
public:
    Rpow2CudaSession(
        std::size_t device_index,
        const std::uint8_t* nonce_prefix,
        std::size_t nonce_prefix_len,
        std::uint32_t difficulty_bits,
        std::uint64_t batch_size,
        std::uint64_t start_nonce,
        std::uint32_t threads_per_block,
        std::uint32_t nonces_per_thread,
        std::uint32_t max_blocks);
    ~Rpow2CudaSession();

    Rpow2CudaSession(const Rpow2CudaSession&) = delete;
    Rpow2CudaSession& operator=(const Rpow2CudaSession&) = delete;
    Rpow2CudaSession(Rpow2CudaSession&&) = delete;
    Rpow2CudaSession& operator=(Rpow2CudaSession&&) = delete;

    Rpow2CudaBatchResult mine_next_batch();
    Rpow2CudaProfiledBatchResult mine_next_batch_profiled();

private:
    int device_index_ = 0;
    void* device_result_ = nullptr;
    std::array<std::uint32_t, 8> initial_state_{};
    std::array<std::uint32_t, 32> template_words_{};
    std::uint32_t prefix_len_ = 0;
    std::uint32_t nonce_offset_ = 0;
    std::uint32_t difficulty_bits_ = 0;
    std::uint32_t block_count_ = 0;
    std::uint32_t threads_per_block_ = 0;
    std::uint32_t nonces_per_thread_ = 0;
    std::uint32_t max_blocks_ = 0;
    std::uint64_t batch_size_ = 0;
    std::uint64_t next_nonce_ = 0;
    std::uint64_t attempts_ = 0;
};

class Solver {
public:
    explicit Solver(std::size_t device_index);

    SolverConfig default_config_for(const Job& job) const;
    SolveResult mine_batch(
        const Job& job,
        const SolverConfig& config,
        std::uint64_t start_nonce,
        std::atomic_bool& stop,
        std::atomic<std::int64_t>& attempts) const;
    SolverSession create_session(const Job& job, const SolverConfig& config, std::uint64_t start_nonce) const;
    SolveResult mine_next_batch(
        const Job& job,
        SolverSession& session,
        std::atomic_bool& stop,
        std::atomic<std::int64_t>& attempts) const;

    BenchmarkResult run_benchmark_case(
        const Job& job,
        const SolverConfig& config,
        std::chrono::milliseconds duration) const;

    void validate_against_reference(const Job& job, std::uint64_t nonce) const;

    BenchmarkResult find_best_benchmark_config() const;

private:
    std::size_t device_index_ = 0;

    std::size_t estimate_max_batch_size(const Job& job) const;
    static std::vector<SolverConfig> build_benchmark_candidates(std::size_t max_batch_size);
};

Rpow2CudaBatchResult mine_rpow2_cuda_batch(
    std::size_t device_index,
    const std::uint8_t* nonce_prefix,
    std::size_t nonce_prefix_len,
    std::uint32_t difficulty_bits,
    std::uint64_t batch_size,
    std::uint64_t start_nonce);

Rpow2CudaBatchResult mine_rpow2_cuda_batch(
    std::size_t device_index,
    const std::uint8_t* nonce_prefix,
    std::size_t nonce_prefix_len,
    std::uint32_t difficulty_bits,
    std::uint64_t batch_size,
    std::uint64_t start_nonce,
    std::uint32_t threads_per_block,
    std::uint32_t nonces_per_thread,
    std::uint32_t max_blocks);

Rpow2CudaBenchmarkResult benchmark_rpow2_cuda(
    std::size_t device_index,
    const std::uint8_t* nonce_prefix,
    std::size_t nonce_prefix_len,
    std::uint32_t difficulty_bits,
    std::uint64_t batch_size,
    std::uint32_t duration_ms,
    std::uint32_t threads_per_block,
    std::uint32_t nonces_per_thread,
    std::uint32_t max_blocks);

} // namespace app
