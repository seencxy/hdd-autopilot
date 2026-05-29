#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define MINING_METAL_EXPORT extern "C" __declspec(dllexport)
#else
#define MINING_METAL_EXPORT extern "C"
#endif

struct mining_metal_solver_config {
    std::size_t batch_size;
    bool by_segment;
    bool precompute_refs;
};

struct mining_metal_job {
    const std::uint8_t* seed_ptr;
    std::size_t seed_len;
    const std::uint8_t* pass_prefix_ptr;
    std::size_t pass_prefix_len;
    std::uint32_t time_cost;
    std::uint32_t memory_cost_kib;
    std::uint32_t parallelism;
    int difficulty_bits;
};

struct mining_metal_session;
struct mining_metal_rpow2_session;
struct mining_metal_dwc_session;

struct mining_metal_benchmark_result {
    std::size_t batch_size;
    bool by_segment;
    bool precompute_refs;
    std::int64_t attempts;
    std::int64_t elapsed_ms;
    double attempts_per_second;
};

struct mining_metal_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

struct mining_metal_device_info {
    std::size_t device_index;
    std::uint64_t recommended_working_set_bytes;
    std::uint64_t max_buffer_bytes;
    std::uint64_t max_threadgroup_memory_bytes;
    std::uint32_t max_threads_per_group;
    bool unified_memory;
    bool low_power;
    bool removable;
    char device_id[64];
    char name[128];
};

struct mining_metal_rpow2_solver_config {
    std::uint64_t batch_size;
};

struct mining_metal_rpow2_job {
    const std::uint8_t* nonce_prefix_ptr;
    std::size_t nonce_prefix_len;
    std::uint32_t difficulty_bits;
};

struct mining_metal_rpow2_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

MINING_METAL_EXPORT bool mining_metal_is_available();
MINING_METAL_EXPORT bool mining_metal_validate();
MINING_METAL_EXPORT bool mining_metal_validate_device(std::size_t device_index);
MINING_METAL_EXPORT std::size_t mining_metal_device_count();
MINING_METAL_EXPORT bool mining_metal_get_device_info(std::size_t device_index, mining_metal_device_info* result);
MINING_METAL_EXPORT const char* mining_metal_last_error_message();
MINING_METAL_EXPORT bool mining_metal_default_solver_config(
    std::size_t device_index,
    const mining_metal_job* job,
    mining_metal_solver_config* result);
MINING_METAL_EXPORT bool mining_metal_find_best_benchmark_config(
    std::size_t device_index,
    mining_metal_benchmark_result* result);
MINING_METAL_EXPORT bool mining_metal_mine_batch(
    std::size_t device_index,
    const mining_metal_job* job,
    const mining_metal_solver_config* config,
    std::uint64_t start_nonce,
    mining_metal_mine_result* result);
MINING_METAL_EXPORT mining_metal_session* mining_metal_session_create(
    std::size_t device_index,
    const mining_metal_job* job,
    const mining_metal_solver_config* config,
    std::uint64_t start_nonce);
MINING_METAL_EXPORT bool mining_metal_session_mine_next_batch(
    mining_metal_session* session,
    mining_metal_mine_result* result);
MINING_METAL_EXPORT void mining_metal_session_destroy(mining_metal_session* session);
MINING_METAL_EXPORT bool mining_metal_rpow2_mine_batch(
    std::size_t device_index,
    const mining_metal_rpow2_job* job,
    const mining_metal_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_metal_rpow2_mine_result* result);
MINING_METAL_EXPORT mining_metal_rpow2_session* mining_metal_rpow2_session_create(
    std::size_t device_index,
    const mining_metal_rpow2_job* job,
    const mining_metal_rpow2_solver_config* config,
    std::uint64_t start_nonce);
MINING_METAL_EXPORT bool mining_metal_rpow2_session_mine_next_batch(
    mining_metal_rpow2_session* session,
    mining_metal_rpow2_mine_result* result);
MINING_METAL_EXPORT void mining_metal_rpow2_session_destroy(mining_metal_rpow2_session* session);

// DigitalWaterCoin (DWC): sha256("{address}|{epoch}|{salt}" + 16-hex counter),
// difficulty measured in *leading* zero bits. `prefix_ptr` holds the constant
// "address|epoch|salt" ASCII bytes; the kernel appends the 16-hex counter.
struct mining_metal_dwc_solver_config {
    std::uint64_t batch_size;
};

struct mining_metal_dwc_job {
    const std::uint8_t* prefix_ptr;
    std::size_t prefix_len;
    std::uint32_t difficulty_bits;
};

struct mining_metal_dwc_mine_result {
    bool found;
    std::uint64_t nonce;
    std::int64_t attempts;
    char digest_hex[65];
};

MINING_METAL_EXPORT bool mining_metal_dwc_is_available();
MINING_METAL_EXPORT mining_metal_dwc_session* mining_metal_dwc_session_create(
    std::size_t device_index,
    const mining_metal_dwc_job* job,
    const mining_metal_dwc_solver_config* config,
    std::uint64_t start_nonce);
MINING_METAL_EXPORT bool mining_metal_dwc_session_mine_next_batch(
    mining_metal_dwc_session* session,
    mining_metal_dwc_mine_result* result);
MINING_METAL_EXPORT void mining_metal_dwc_session_destroy(mining_metal_dwc_session* session);
