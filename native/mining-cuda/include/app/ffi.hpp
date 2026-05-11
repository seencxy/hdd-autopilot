#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define MINING_CUDA_EXPORT extern "C" __declspec(dllexport)
#else
#define MINING_CUDA_EXPORT extern "C"
#endif

struct mining_cuda_solver_config {
    std::size_t batch_size;
    bool by_segment;
    bool precompute_refs;
    bool generate_passwords_on_gpu;
};

struct mining_cuda_job {
    const std::uint8_t* seed_ptr;
    std::size_t seed_len;
    const std::uint8_t* pass_prefix_ptr;
    std::size_t pass_prefix_len;
    std::uint32_t time_cost;
    std::uint32_t memory_cost_kib;
    std::uint32_t parallelism;
    int difficulty_bits;
};

struct mining_cuda_session;
struct mining_cuda_rpow2_session;
struct mining_cuda_h98hash_session;
struct mining_cuda_h256hash_session;

struct mining_cuda_rpow2_solver_config {
    std::uint64_t batch_size;
    std::uint32_t threads_per_block;
    std::uint32_t nonces_per_thread;
    std::uint32_t max_blocks;
    bool early_exit;
};

struct mining_cuda_rpow2_job {
    const std::uint8_t* nonce_prefix_ptr;
    std::size_t nonce_prefix_len;
    std::uint32_t difficulty_bits;
};

struct mining_cuda_rpow2_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

struct mining_cuda_rpow2_benchmark_result {
    std::uint64_t attempts;
    std::uint64_t batches;
    double elapsed_ms;
    double kernel_ms;
    double empty_launch_us;
    double kernel_hashrate;
    double effective_hashrate;
};

struct mining_cuda_h98hash_job {
    const std::uint8_t* challenge_ptr;
    std::size_t challenge_len;
    const std::uint8_t* nonce_prefix_ptr;
    std::size_t nonce_prefix_len;
    std::uint32_t difficulty_bits;
};

struct mining_cuda_h98hash_mine_result {
    bool found;
    std::uint64_t nonce_tail;
    std::int64_t attempts;
    char digest_hex[65];
};

struct mining_cuda_h256hash_job {
    const std::uint8_t* challenge_ptr;
    std::size_t challenge_len;
    const std::uint8_t* target_ptr;
    std::size_t target_len;
};

struct mining_cuda_h256hash_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

struct mining_cuda_benchmark_result {
    std::size_t batch_size;
    bool by_segment;
    bool precompute_refs;
    bool generate_passwords_on_gpu;
    std::int64_t attempts;
    std::int64_t elapsed_ms;
    double attempts_per_second;
};

struct mining_cuda_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

struct mining_cuda_device_info {
    std::size_t device_index;
    std::uint64_t global_memory_bytes;
    std::uint64_t max_alloc_bytes;
    std::uint32_t compute_units;
    std::uint32_t max_threads_per_block;
    std::uint32_t warp_size;
    std::uint64_t shared_memory_per_block_bytes;
    char device_id[32];
    char name[128];
};

MINING_CUDA_EXPORT bool mining_cuda_is_available();
MINING_CUDA_EXPORT bool mining_cuda_validate();
MINING_CUDA_EXPORT std::size_t mining_cuda_device_count();
MINING_CUDA_EXPORT bool mining_cuda_get_device_info(std::size_t device_index, mining_cuda_device_info* result);
MINING_CUDA_EXPORT const char* mining_cuda_last_error_message();
MINING_CUDA_EXPORT bool mining_cuda_default_solver_config(
    std::size_t device_index,
    const mining_cuda_job* job,
    mining_cuda_solver_config* result);
MINING_CUDA_EXPORT bool mining_cuda_find_best_benchmark_config(
    std::size_t device_index,
    mining_cuda_benchmark_result* result);
MINING_CUDA_EXPORT bool mining_cuda_mine_batch(
    std::size_t device_index,
    const mining_cuda_job* job,
    const mining_cuda_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_mine_result* result);
MINING_CUDA_EXPORT mining_cuda_session* mining_cuda_session_create(
    std::size_t device_index,
    const mining_cuda_job* job,
    const mining_cuda_solver_config* config,
    std::uint64_t start_nonce);
MINING_CUDA_EXPORT bool mining_cuda_session_mine_next_batch(
    mining_cuda_session* session,
    mining_cuda_mine_result* result);
MINING_CUDA_EXPORT void mining_cuda_session_destroy(mining_cuda_session* session);
MINING_CUDA_EXPORT bool mining_cuda_rpow2_mine_batch(
    std::size_t device_index,
    const mining_cuda_rpow2_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_rpow2_mine_result* result);
MINING_CUDA_EXPORT mining_cuda_rpow2_session* mining_cuda_rpow2_session_create(
    std::size_t device_index,
    const mining_cuda_rpow2_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce);
MINING_CUDA_EXPORT bool mining_cuda_rpow2_session_mine_next_batch(
    mining_cuda_rpow2_session* session,
    mining_cuda_rpow2_mine_result* result);
MINING_CUDA_EXPORT void mining_cuda_rpow2_session_destroy(mining_cuda_rpow2_session* session);
MINING_CUDA_EXPORT bool mining_cuda_rpow2_benchmark(
    std::size_t device_index,
    const mining_cuda_rpow2_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint32_t duration_ms,
    mining_cuda_rpow2_benchmark_result* result);
MINING_CUDA_EXPORT bool mining_cuda_h98hash_mine_batch(
    std::size_t device_index,
    const mining_cuda_h98hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_h98hash_mine_result* result);
MINING_CUDA_EXPORT mining_cuda_h98hash_session* mining_cuda_h98hash_session_create(
    std::size_t device_index,
    const mining_cuda_h98hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce);
MINING_CUDA_EXPORT bool mining_cuda_h98hash_session_mine_next_batch(
    mining_cuda_h98hash_session* session,
    mining_cuda_h98hash_mine_result* result);
MINING_CUDA_EXPORT void mining_cuda_h98hash_session_destroy(mining_cuda_h98hash_session* session);
MINING_CUDA_EXPORT bool mining_cuda_h256hash_mine_batch(
    std::size_t device_index,
    const mining_cuda_h256hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_cuda_h256hash_mine_result* result);
MINING_CUDA_EXPORT mining_cuda_h256hash_session* mining_cuda_h256hash_session_create(
    std::size_t device_index,
    const mining_cuda_h256hash_job* job,
    const mining_cuda_rpow2_solver_config* config,
    std::uint64_t start_nonce);
MINING_CUDA_EXPORT bool mining_cuda_h256hash_session_mine_next_batch(
    mining_cuda_h256hash_session* session,
    mining_cuda_h256hash_mine_result* result);
MINING_CUDA_EXPORT void mining_cuda_h256hash_session_destroy(mining_cuda_h256hash_session* session);
MINING_CUDA_EXPORT bool mining_argon2id_hash_raw(
    const std::uint8_t* password_ptr,
    std::size_t password_len,
    const std::uint8_t* salt_ptr,
    std::size_t salt_len,
    std::uint32_t time_cost,
    std::uint32_t memory_cost_kib,
    std::uint32_t parallelism,
    std::uint8_t* digest_ptr,
    std::size_t digest_len);
