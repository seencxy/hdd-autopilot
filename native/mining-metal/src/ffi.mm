#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "app/ffi.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

app::Job make_job(const mining_metal_job& raw) {
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

app::SolverConfig make_solver_config(const mining_metal_solver_config& raw) {
    return app::SolverConfig{
        .batch_size = raw.batch_size,
        .by_segment = raw.by_segment,
        .precompute_refs = raw.precompute_refs,
    };
}

void fill_mine_result(const app::SolveResult& mined,
                      const std::atomic<std::int64_t>& attempts,
                      mining_metal_mine_result* result) {
    result->found = mined.found;
    result->nonce = mined.nonce;
    result->attempts = attempts.load();
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (!mined.digest.empty()) {
        std::strncpy(result->digest_hex, mined.digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}

void fill_device_info(std::size_t device_index, mining_metal_device_info* result) {
    const auto devices = MTLCopyAllDevices();
    if (devices == nil || device_index >= devices.count) {
        throw std::runtime_error("Metal device index out of range");
    }
    const auto device = devices[device_index];
    result->device_index = device_index;
    result->recommended_working_set_bytes = 0;
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        result->recommended_working_set_bytes = static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize);
    }
    result->max_buffer_bytes = static_cast<std::uint64_t>(device.maxBufferLength);
    result->max_threadgroup_memory_bytes = static_cast<std::uint64_t>(device.maxThreadgroupMemoryLength);
    result->max_threads_per_group = device.isLowPower ? 512 : 1024;
    result->unified_memory = device.hasUnifiedMemory;
    result->low_power = device.isLowPower;
    result->removable = device.isRemovable;
    std::fill(std::begin(result->device_id), std::end(result->device_id), '\0');
    std::fill(std::begin(result->name), std::end(result->name), '\0');
    const auto device_id = std::string("metal:") + std::to_string(static_cast<unsigned long long>(device.registryID));
    const auto name = std::string(device.name.UTF8String ?: "");
    std::strncpy(result->device_id, device_id.c_str(), sizeof(result->device_id) - 1);
    std::strncpy(result->name, name.c_str(), sizeof(result->name) - 1);
}

struct Rpow2KernelParams {
    std::uint32_t prefix_len;
    std::uint32_t difficulty_bits;
    std::uint32_t block_count;
    std::uint32_t _padding;
    std::uint64_t start_nonce;
    std::uint64_t batch_size;
};

const char* rpow2_kernel_source() {
    return R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Rpow2KernelParams {
    uint prefix_len;
    uint difficulty_bits;
    uint block_count;
    uint _padding;
    ulong start_nonce;
    ulong batch_size;
};

constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline void rpow2_write_nonce_words(thread uint* w,
                                    uint block,
                                    uint prefix_len,
                                    ulong nonce) {
    for (uint i = 0; i < 8; ++i) {
        const uint absolute_byte = prefix_len + i;
        if ((absolute_byte >> 6) != block) {
            continue;
        }
        const uint block_byte = absolute_byte & 63u;
        const uint word_index = block_byte >> 2;
        const uint byte_index = block_byte & 3u;
        const uint shift = (3u - byte_index) * 8u;
        w[word_index] |= uint((nonce >> (i * 8u)) & 0xfful) << shift;
    }
}

inline uint rpow2_trailing_zero_bits(uint h0,
                                     uint h1,
                                     uint h2,
                                     uint h3,
                                     uint h4,
                                     uint h5,
                                     uint h6,
                                     uint h7) {
    uint words[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint bits = 0;
    for (int i = 7; i >= 0; --i) {
        uint word = words[i];
        if (word == 0) {
            bits += 32;
            continue;
        }
        for (uint bit = 0; bit < 32; ++bit) {
            if ((word & (1u << bit)) != 0) {
                return bits + bit;
            }
        }
    }
    return bits;
}

kernel void rpow2_mine_kernel(constant uint* template_words [[buffer(0)]],
                              constant Rpow2KernelParams& params [[buffer(1)]],
                              device atomic_uint* found [[buffer(2)]],
                              device ulong* found_nonce [[buffer(3)]],
                              device uint* found_digest [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= params.batch_size ||
        atomic_load_explicit(found, memory_order_relaxed) != 0) {
        return;
    }

    const ulong nonce = params.start_nonce + ulong(gid);

    uint h0 = 0x6a09e667;
    uint h1 = 0xbb67ae85;
    uint h2 = 0x3c6ef372;
    uint h3 = 0xa54ff53a;
    uint h4 = 0x510e527f;
    uint h5 = 0x9b05688c;
    uint h6 = 0x1f83d9ab;
    uint h7 = 0x5be0cd19;

    for (uint block = 0; block < params.block_count; ++block) {
        uint w[64];
        for (uint i = 0; i < 16; ++i) {
            w[i] = template_words[block * 16u + i];
        }
        rpow2_write_nonce_words(w, block, params.prefix_len, nonce);
        for (uint i = 16; i < 64; ++i) {
            const uint s0 = rotate(w[i - 15], 25u) ^ rotate(w[i - 15], 14u) ^ (w[i - 15] >> 3);
            const uint s1 = rotate(w[i - 2], 15u) ^ rotate(w[i - 2], 13u) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint a = h0;
        uint b = h1;
        uint c = h2;
        uint d = h3;
        uint e = h4;
        uint f = h5;
        uint g = h6;
        uint h = h7;
        for (uint i = 0; i < 64; ++i) {
            const uint s1 = rotate(e, 26u) ^ rotate(e, 21u) ^ rotate(e, 7u);
            const uint ch = (e & f) ^ ((~e) & g);
            const uint temp1 = h + s1 + ch + K[i] + w[i];
            const uint s0 = rotate(a, 30u) ^ rotate(a, 19u) ^ rotate(a, 10u);
            const uint maj = (a & b) ^ (a & c) ^ (b & c);
            const uint temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    if (rpow2_trailing_zero_bits(h0, h1, h2, h3, h4, h5, h6, h7) >= params.difficulty_bits) {
        if (atomic_exchange_explicit(found, 1u, memory_order_relaxed) == 0u) {
            *found_nonce = nonce;
            found_digest[0] = h0;
            found_digest[1] = h1;
            found_digest[2] = h2;
            found_digest[3] = h3;
            found_digest[4] = h4;
            found_digest[5] = h5;
            found_digest[6] = h6;
            found_digest[7] = h7;
        }
    }
}
)METAL";
}

id<MTLComputePipelineState> rpow2_pipeline_for_device(id<MTLDevice> device) {
    static NSMutableDictionary<NSNumber*, id<MTLComputePipelineState>>* pipelines = nil;
    if (pipelines == nil) {
        pipelines = [NSMutableDictionary dictionary];
    }
    const auto key = @((unsigned long long)device.registryID);
    id<MTLComputePipelineState> cached = pipelines[key];
    if (cached != nil) {
        return cached;
    }

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:rpow2_kernel_source()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        throw std::runtime_error(std::string("Metal RPOW2 kernel compile failed: ")
            + (error.localizedDescription.UTF8String ?: "unknown error"));
    }
    id<MTLFunction> function = [library newFunctionWithName:@"rpow2_mine_kernel"];
    if (function == nil) {
        throw std::runtime_error("Metal RPOW2 kernel function not found");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string("Metal RPOW2 pipeline creation failed: ")
            + (error.localizedDescription.UTF8String ?: "unknown error"));
    }
    pipelines[key] = pipeline;
    return pipeline;
}

std::string digest_words_to_hex(const std::uint32_t words[8]) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < 8; ++index) {
        out << std::setw(2) << ((words[index] >> 24) & 0xff)
            << std::setw(2) << ((words[index] >> 16) & 0xff)
            << std::setw(2) << ((words[index] >> 8) & 0xff)
            << std::setw(2) << (words[index] & 0xff);
    }
    return out.str();
}

std::vector<std::uint32_t> build_rpow2_template_words(const mining_metal_rpow2_job& job) {
    const auto message_len = job.nonce_prefix_len + 8;
    const auto padded_len = ((message_len + 9 + 63) / 64) * 64;
    if (padded_len > 128) {
        throw std::runtime_error("RPOW2 padded message is too long for Metal solver");
    }

    std::array<std::uint8_t, 128> padded{};
    if (job.nonce_prefix_len > 0) {
        std::memcpy(padded.data(), job.nonce_prefix_ptr, job.nonce_prefix_len);
    }
    padded[message_len] = 0x80;
    const auto bit_len = static_cast<std::uint64_t>(message_len) * 8;
    for (std::size_t i = 0; i < 8; ++i) {
        padded[padded_len - 1 - i] = static_cast<std::uint8_t>((bit_len >> (i * 8)) & 0xff);
    }

    std::vector<std::uint32_t> words(padded_len / 4);
    for (std::size_t index = 0; index < words.size(); ++index) {
        const auto offset = index * 4;
        words[index] = (static_cast<std::uint32_t>(padded[offset]) << 24)
            | (static_cast<std::uint32_t>(padded[offset + 1]) << 16)
            | (static_cast<std::uint32_t>(padded[offset + 2]) << 8)
            | static_cast<std::uint32_t>(padded[offset + 3]);
    }
    return words;
}

void fill_rpow2_result(bool found,
                       std::uint64_t nonce,
                       std::int64_t attempts,
                       const std::uint32_t digest_words[8],
                       mining_metal_rpow2_mine_result* result) {
    result->found = found;
    result->nonce = nonce;
    result->attempts = attempts;
    std::fill(std::begin(result->digest_hex), std::end(result->digest_hex), '\0');
    if (found) {
        const auto digest = digest_words_to_hex(digest_words);
        std::strncpy(result->digest_hex, digest.c_str(), sizeof(result->digest_hex) - 1);
    }
}
} // namespace

struct mining_metal_session {
    app::Job job;
    app::Solver solver;
    app::SolverSession session;
    std::atomic_bool stop{false};
    std::atomic<std::int64_t> attempts{0};

    mining_metal_session(std::size_t device_index,
                      app::Job&& native_job,
                      const app::SolverConfig& native_config,
                      std::uint64_t start_nonce)
        : job(std::move(native_job)),
          solver(device_index),
          session(solver.create_session(job, native_config, start_nonce)) {
    }
};

struct mining_metal_rpow2_session {
    id<MTLComputePipelineState> pipeline;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> template_buffer;
    id<MTLBuffer> params_buffer;
    id<MTLBuffer> found_buffer;
    id<MTLBuffer> nonce_buffer;
    id<MTLBuffer> digest_buffer;
    Rpow2KernelParams params{};
    std::uint64_t next_nonce = 0;
    std::int64_t attempts = 0;

    mining_metal_rpow2_session(std::size_t device_index,
                               const mining_metal_rpow2_job& job,
                               const mining_metal_rpow2_solver_config& config,
                               std::uint64_t start_nonce) {
        if (job.nonce_prefix_ptr == nullptr && job.nonce_prefix_len > 0) {
            throw std::runtime_error("RPOW2 nonce prefix pointer is null");
        }
        if (job.nonce_prefix_len > 111) {
            throw std::runtime_error("RPOW2 nonce prefix is too long for Metal solver");
        }
        if (config.batch_size == 0) {
            throw std::runtime_error("RPOW2 Metal batch size must be greater than zero");
        }
        if (config.batch_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("RPOW2 Metal batch size exceeds kernel grid limit");
        }

        const auto devices = MTLCopyAllDevices();
        if (devices == nil || device_index >= devices.count) {
            throw std::runtime_error("Metal device index out of range");
        }
        id<MTLDevice> device = devices[device_index];
        pipeline = rpow2_pipeline_for_device(device);
        queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("failed to create Metal command queue");
        }

        const auto template_words = build_rpow2_template_words(job);
        template_buffer = [device newBufferWithBytes:template_words.data()
                                              length:template_words.size() * sizeof(std::uint32_t)
                                             options:MTLResourceStorageModeShared];
        params = Rpow2KernelParams{
            .prefix_len = static_cast<std::uint32_t>(job.nonce_prefix_len),
            .difficulty_bits = job.difficulty_bits,
            .block_count = static_cast<std::uint32_t>(template_words.size() / 16),
            ._padding = 0,
            .start_nonce = start_nonce,
            .batch_size = config.batch_size,
        };
        params_buffer = [device newBufferWithLength:sizeof(params)
                                            options:MTLResourceStorageModeShared];
        found_buffer = [device newBufferWithLength:sizeof(std::uint32_t)
                                           options:MTLResourceStorageModeShared];
        nonce_buffer = [device newBufferWithLength:sizeof(std::uint64_t)
                                           options:MTLResourceStorageModeShared];
        digest_buffer = [device newBufferWithLength:sizeof(std::uint32_t) * 8
                                            options:MTLResourceStorageModeShared];
        if (template_buffer == nil || params_buffer == nil || found_buffer == nil ||
            nonce_buffer == nil || digest_buffer == nil) {
            throw std::runtime_error("failed to allocate Metal RPOW2 buffers");
        }
        next_nonce = start_nonce;
    }

    void mine_next_batch(mining_metal_rpow2_mine_result* result) {
        if (std::numeric_limits<std::uint64_t>::max() - next_nonce < params.batch_size) {
            throw std::runtime_error("RPOW2 Metal nonce range exhausted");
        }

        params.start_nonce = next_nonce;
        std::memcpy(params_buffer.contents, &params, sizeof(params));
        *static_cast<std::uint32_t*>(found_buffer.contents) = 0;
        *static_cast<std::uint64_t*>(nonce_buffer.contents) = 0;
        std::memset(digest_buffer.contents, 0, sizeof(std::uint32_t) * 8);

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:template_buffer offset:0 atIndex:0];
        [encoder setBuffer:params_buffer offset:0 atIndex:1];
        [encoder setBuffer:found_buffer offset:0 atIndex:2];
        [encoder setBuffer:nonce_buffer offset:0 atIndex:3];
        [encoder setBuffer:digest_buffer offset:0 atIndex:4];

        const auto threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(params.batch_size), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.error != nil) {
            throw std::runtime_error(std::string("Metal RPOW2 command failed: ")
                + (command_buffer.error.localizedDescription.UTF8String ?: "unknown error"));
        }

        const auto found_ptr = static_cast<const std::uint32_t*>(found_buffer.contents);
        const auto nonce_ptr = static_cast<const std::uint64_t*>(nonce_buffer.contents);
        const auto digest_ptr = static_cast<const std::uint32_t*>(digest_buffer.contents);
        attempts = static_cast<std::int64_t>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            static_cast<std::uint64_t>(attempts) + params.batch_size));
        fill_rpow2_result(*found_ptr != 0, *nonce_ptr, attempts, digest_ptr, result);
        next_nonce += params.batch_size;
    }
};

bool mining_metal_is_available() {
    try {
        clear_last_error();
        const auto devices = MTLCopyAllDevices();
        return devices != nil && devices.count > 0;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_metal_validate_device(std::size_t device_index) {
    try {
        clear_last_error();
        const auto devices = MTLCopyAllDevices();
        if (devices == nil || devices.count == 0) {
            throw std::runtime_error("当前没有检测到可用的 Metal 设备。");
        }
        if (device_index >= devices.count) {
            throw std::runtime_error("Metal device index out of range");
        }
        app::Solver solver(device_index);
        solver.validate_against_reference(benchmark_job(), 1);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

bool mining_metal_validate() {
    return mining_metal_validate_device(0);
}

std::size_t mining_metal_device_count() {
    try {
        clear_last_error();
        const auto devices = MTLCopyAllDevices();
        return devices == nil ? 0 : devices.count;
    } catch (const std::exception& error) {
        set_last_error(error);
        return 0;
    }
}

bool mining_metal_get_device_info(std::size_t device_index, mining_metal_device_info* result) {
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

const char* mining_metal_last_error_message() {
    return g_last_error.c_str();
}

bool mining_metal_default_solver_config(
    std::size_t device_index,
    const mining_metal_job* job,
    mining_metal_solver_config* result) {
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

bool mining_metal_find_best_benchmark_config(std::size_t device_index, mining_metal_benchmark_result* result) {
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

bool mining_metal_mine_batch(
    std::size_t device_index,
    const mining_metal_job* job,
    const mining_metal_solver_config* config,
    std::uint64_t start_nonce,
    mining_metal_mine_result* result) {
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

mining_metal_session* mining_metal_session_create(
    std::size_t device_index,
    const mining_metal_job* job,
    const mining_metal_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        auto native_job = make_job(*job);
        const auto native_config = make_solver_config(*config);
        return new mining_metal_session(device_index, std::move(native_job), native_config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_metal_session_mine_next_batch(
    mining_metal_session* session,
    mining_metal_mine_result* result) {
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

void mining_metal_session_destroy(mining_metal_session* session) {
    delete session;
}

bool mining_metal_rpow2_mine_batch(
    std::size_t device_index,
    const mining_metal_rpow2_job* job,
    const mining_metal_rpow2_solver_config* config,
    std::uint64_t start_nonce,
    mining_metal_rpow2_mine_result* result) {
    if (job == nullptr || config == nullptr || result == nullptr) {
        g_last_error = "rpow2 mine_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        mining_metal_rpow2_session session(device_index, *job, *config, start_nonce);
        session.mine_next_batch(result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

mining_metal_rpow2_session* mining_metal_rpow2_session_create(
    std::size_t device_index,
    const mining_metal_rpow2_job* job,
    const mining_metal_rpow2_solver_config* config,
    std::uint64_t start_nonce) {
    if (job == nullptr || config == nullptr) {
        g_last_error = "rpow2 session_create parameter is null";
        return nullptr;
    }
    try {
        clear_last_error();
        return new mining_metal_rpow2_session(device_index, *job, *config, start_nonce);
    } catch (const std::exception& error) {
        set_last_error(error);
        return nullptr;
    }
}

bool mining_metal_rpow2_session_mine_next_batch(
    mining_metal_rpow2_session* session,
    mining_metal_rpow2_mine_result* result) {
    if (session == nullptr || result == nullptr) {
        g_last_error = "rpow2 session_mine_next_batch parameter is null";
        return false;
    }
    try {
        clear_last_error();
        session->mine_next_batch(result);
        return true;
    } catch (const std::exception& error) {
        set_last_error(error);
        return false;
    }
}

void mining_metal_rpow2_session_destroy(mining_metal_rpow2_session* session) {
    delete session;
}
