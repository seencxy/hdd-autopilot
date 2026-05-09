#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "app/solver.hpp"
#include "app/embedded_kernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "argon2-gpu-common/argon2-common.h"
#include "argon2-gpu-common/argon2params.h"

#include <argon2.h>
#ifdef ARGON2_SYNC_POINTS
#undef ARGON2_SYNC_POINTS
#endif

namespace app {
namespace {

constexpr std::size_t kDigestSize = 32;
constexpr std::size_t kDefaultRunBatchCap = 32;
constexpr std::chrono::milliseconds kBenchmarkCaseDuration{5000};
constexpr std::uint32_t kThreadsPerLane = 32;
constexpr std::size_t kThreadgroupShuffleBytesPerWarp = kThreadsPerLane * sizeof(std::uint32_t) * 2;

const char* localized_bool(bool value) noexcept {
    return value ? "\xE6\x98\xAF" : "\xE5\x90\xA6";
}

struct BatchResult {
    bool found = false;
    std::uint64_t nonce = 0;
    std::array<std::uint8_t, kDigestSize> digest{};
};

struct PrecomputeArgs {
    std::uint32_t passes;
    std::uint32_t lanes;
    std::uint32_t segment_blocks;
};

struct OneshotArgs {
    std::uint32_t passes;
    std::uint32_t lanes;
    std::uint32_t segment_blocks;
};

struct SegmentArgs {
    std::uint32_t passes;
    std::uint32_t lanes;
    std::uint32_t segment_blocks;
    std::uint32_t pass;
    std::uint32_t slice;
};

struct DispatchShape {
    std::uint32_t lanes_per_block = 1;
    std::uint32_t jobs_per_block = 1;
};

std::string ns_string_to_utf8(NSString* text) {
    if (text == nil) {
        return {};
    }
    const char* utf8 = text.UTF8String;
    return utf8 == nullptr ? std::string() : std::string(utf8);
}

std::string describe_error(NSError* error, NSString* fallback) {
    if (error == nil) {
        return ns_string_to_utf8(fallback);
    }
    NSString* description = error.localizedDescription ?: fallback;
    NSString* failure = error.localizedFailureReason;
    if (failure != nil && failure.length > 0) {
        return ns_string_to_utf8([NSString stringWithFormat:@"%@ (%@)", description, failure]);
    }
    return ns_string_to_utf8(description);
}

NSArray<id<MTLDevice>>* copy_all_devices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    return devices == nil ? @[] : devices;
}

id<MTLDevice> require_device(std::size_t device_index) {
    NSArray<id<MTLDevice>>* devices = copy_all_devices();
    if (device_index >= devices.count) {
        throw std::runtime_error("Metal device index out of range");
    }
    id<MTLDevice> device = devices[device_index];
    if (device == nil) {
        throw std::runtime_error("Metal device is null");
    }
    return device;
}

std::string device_name(id<MTLDevice> device) {
    return ns_string_to_utf8(device.name);
}

std::string device_id_string(id<MTLDevice> device) {
    return "metal:" + std::to_string(static_cast<unsigned long long>(device.registryID));
}

std::uint32_t highest_power_of_two_not_above(std::uint32_t value) {
    if (value == 0) {
        return 0;
    }
    std::uint32_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}

struct PreparedGpuBatch {
    id<MTLDevice> __strong device = nil;
    id<MTLCommandQueue> __strong queue = nil;
    id<MTLLibrary> __strong library = nil;
    id<MTLComputePipelineState> __strong precompute_pipeline = nil;
    id<MTLComputePipelineState> __strong oneshot_pipeline = nil;
    id<MTLComputePipelineState> __strong oneshot_precompute_pipeline = nil;
    id<MTLComputePipelineState> __strong segment_pipeline = nil;
    id<MTLComputePipelineState> __strong segment_precompute_pipeline = nil;
    id<MTLBuffer> __strong memory_buffer = nil;
    id<MTLBuffer> __strong refs_buffer = nil;
    argon2::Argon2Params params;
    std::size_t batch_size = 1;
    bool by_segment = false;
    bool precompute_refs = false;
    std::uint32_t passes = 1;
    std::uint32_t lanes = 1;
    std::uint32_t segment_blocks = 1;
    std::size_t memory_size = 0;
    DispatchShape dispatch;

    PreparedGpuBatch(const Job& job,
                     const SolverConfig& config,
                     std::size_t device_index,
                     std::size_t actual_batch_size)
        : params(kDigestSize,
                 job.seed_bytes().data(),
                 job.seed_bytes().size(),
                 nullptr,
                 0,
                 nullptr,
                 0,
                 job.time_cost(),
                 job.memory_cost_kb(),
                 job.parallelism()) {
        batch_size = actual_batch_size;
        by_segment = config.by_segment;
        precompute_refs = config.precompute_refs;
        passes = job.time_cost();
        lanes = job.parallelism();
        segment_blocks = params.getSegmentBlocks();
        memory_size = params.getMemorySize() * batch_size;

        device = require_device(device_index);
        queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("Metal command queue creation failed");
        }

        NSString* source = [NSString stringWithUTF8String:kArgon2MetalKernelSource];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        NSError* error = nil;
        library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) {
            throw std::runtime_error("Metal library compile failed: " + describe_error(error, @"unknown compile error"));
        }

        precompute_pipeline = build_pipeline(@"argon2_precompute_kernel");
        oneshot_pipeline = build_pipeline(@"argon2_kernel_oneshot");
        oneshot_precompute_pipeline = build_pipeline(@"argon2_kernel_oneshot_precompute");
        segment_pipeline = build_pipeline(@"argon2_kernel_segment");
        segment_precompute_pipeline = build_pipeline(@"argon2_kernel_segment_precompute");

        memory_buffer = [device newBufferWithLength:static_cast<NSUInteger>(memory_size)
                                            options:MTLResourceStorageModeShared];
        if (memory_buffer == nil) {
            throw std::runtime_error("Metal memory buffer allocation failed");
        }

        if (precompute_refs) {
            const std::uint32_t segments = lanes * (argon2::ARGON2_SYNC_POINTS / 2);
            const std::size_t refs_size = static_cast<std::size_t>(segments) * segment_blocks * sizeof(std::uint32_t) * 2;
            refs_buffer = [device newBufferWithLength:static_cast<NSUInteger>(refs_size)
                                              options:MTLResourceStorageModeShared];
            if (refs_buffer == nil) {
                throw std::runtime_error("Metal refs buffer allocation failed");
            }
            encode_precompute_refs();
        }

        dispatch = choose_dispatch();
    }

    id<MTLComputePipelineState> active_pipeline() const {
        if (by_segment) {
            return precompute_refs ? segment_precompute_pipeline : segment_pipeline;
        }
        return precompute_refs ? oneshot_precompute_pipeline : oneshot_pipeline;
    }

    std::size_t warp_count_per_threadgroup() const {
        return static_cast<std::size_t>(dispatch.lanes_per_block) * static_cast<std::size_t>(dispatch.jobs_per_block);
    }

    std::size_t threadgroup_memory_length() const {
        return warp_count_per_threadgroup() * kThreadgroupShuffleBytesPerWarp;
    }

    void* map_input_memory(std::size_t job_id) {
        auto* base = static_cast<std::uint8_t*>(memory_buffer.contents);
        return base + params.getMemorySize() * job_id;
    }

    void* map_output_memory(std::size_t job_id) {
        auto* base = static_cast<std::uint8_t*>(memory_buffer.contents);
        const std::size_t mapped_size = static_cast<std::size_t>(params.getLanes()) * argon2::ARGON2_BLOCK_SIZE;
        const std::size_t mapped_offset = params.getMemorySize() * (job_id + 1) - mapped_size;
        return base + mapped_offset;
    }

    float run_once() {
        const auto started = std::chrono::steady_clock::now();
        @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            if (command_buffer == nil) {
                throw std::runtime_error("Metal command buffer creation failed");
            }
            if (by_segment) {
                encode_segment_passes(command_buffer);
            } else {
                encode_oneshot(command_buffer);
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("Metal command buffer failed: "
                    + describe_error(command_buffer.error, @"unknown command buffer error"));
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            std::chrono::steady_clock::now() - started);
        return elapsed.count();
    }

private:
    id<MTLComputePipelineState> build_pipeline(NSString* name) {
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil) {
            throw std::runtime_error("Metal function not found: " + ns_string_to_utf8(name));
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error("Metal pipeline creation failed for " + ns_string_to_utf8(name) + ": " + describe_error(error, @"unknown pipeline error"));
        }
        return pipeline;
    }

    DispatchShape choose_dispatch() const {
        id<MTLComputePipelineState> pipeline = active_pipeline();
        std::uint32_t lanes_per_block = by_segment ? highest_power_of_two_not_above(lanes) : lanes;
        if (lanes_per_block == 0) {
            lanes_per_block = 1;
        }
        while (by_segment && lanes_per_block > 1 && lanes % lanes_per_block != 0) {
            lanes_per_block /= 2;
        }

        const std::uint32_t threads_per_job = kThreadsPerLane * lanes_per_block;
        const std::uint32_t max_threads = static_cast<std::uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
        if (threads_per_job == 0 || max_threads < threads_per_job) {
            throw std::runtime_error("Metal threadgroup size exceeds pipeline limit");
        }

        const std::size_t max_tg_mem = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
        const std::size_t per_job_tg_mem = kThreadgroupShuffleBytesPerWarp * static_cast<std::size_t>(lanes_per_block);
        if (per_job_tg_mem == 0 || per_job_tg_mem > max_tg_mem) {
            throw std::runtime_error("Metal threadgroup memory exceeds device limit");
        }
        const std::uint32_t jobs_limit_threads = std::max<std::uint32_t>(1, max_threads / threads_per_job);
        const std::uint32_t jobs_limit_memory = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(max_tg_mem / per_job_tg_mem));
        std::uint32_t jobs_per_block = std::min<std::uint32_t>(static_cast<std::uint32_t>(batch_size), std::min(jobs_limit_threads, jobs_limit_memory));
        if (jobs_per_block == 0) {
            jobs_per_block = 1;
        }
        jobs_per_block = highest_power_of_two_not_above(jobs_per_block);
        if (jobs_per_block == 0) {
            jobs_per_block = 1;
        }
        while (jobs_per_block > 1 && batch_size % jobs_per_block != 0) {
            jobs_per_block /= 2;
        }
        return DispatchShape{
            .lanes_per_block = lanes_per_block,
            .jobs_per_block = std::max<std::uint32_t>(1, jobs_per_block),
        };
    }

    void encode_precompute_refs() {
        @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            if (command_buffer == nil) {
                throw std::runtime_error("Metal precompute command buffer creation failed");
            }
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error("Metal precompute encoder creation failed");
            }
            PrecomputeArgs args{.passes = passes, .lanes = lanes, .segment_blocks = segment_blocks};
            const std::uint32_t segment_addr_blocks = (segment_blocks + (argon2::ARGON2_BLOCK_SIZE / (2 * sizeof(std::uint32_t))) - 1)
                / (argon2::ARGON2_BLOCK_SIZE / (2 * sizeof(std::uint32_t)));
            const std::uint32_t segments = lanes * (argon2::ARGON2_SYNC_POINTS / 2);
            const MTLSize threads_per_group = MTLSizeMake(kThreadsPerLane, 1, 1);
            const MTLSize threadgroups = MTLSizeMake(segment_addr_blocks * segments, 1, 1);
            [encoder setComputePipelineState:precompute_pipeline];
            [encoder setThreadgroupMemoryLength:kThreadgroupShuffleBytesPerWarp atIndex:0];
            [encoder setBuffer:refs_buffer offset:0 atIndex:0];
            [encoder setBytes:&args length:sizeof(args) atIndex:1];
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("Metal refs precompute failed: "
                    + describe_error(command_buffer.error, @"unknown command buffer error"));
            }
        }
    }

    void encode_oneshot(id<MTLCommandBuffer> command_buffer) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error("Metal oneshot encoder creation failed");
        }
        const MTLSize threads_per_group = MTLSizeMake(kThreadsPerLane * lanes, dispatch.jobs_per_block, 1);
        const MTLSize threadgroups = MTLSizeMake(1, static_cast<NSUInteger>(batch_size / dispatch.jobs_per_block), 1);
        [encoder setComputePipelineState:active_pipeline()];
        [encoder setThreadgroupMemoryLength:threadgroup_memory_length() atIndex:0];
        [encoder setBuffer:memory_buffer offset:0 atIndex:0];
        if (precompute_refs) {
            OneshotArgs args{.passes = passes, .lanes = lanes, .segment_blocks = segment_blocks};
            [encoder setBuffer:refs_buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        } else {
            OneshotArgs args{.passes = passes, .lanes = lanes, .segment_blocks = segment_blocks};
            [encoder setBytes:&args length:sizeof(args) atIndex:1];
        }
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
    }

    void encode_segment_passes(id<MTLCommandBuffer> command_buffer) {
        const MTLSize threads_per_group = MTLSizeMake(kThreadsPerLane * dispatch.lanes_per_block, dispatch.jobs_per_block, 1);
        const MTLSize threadgroups = MTLSizeMake(lanes / dispatch.lanes_per_block, static_cast<NSUInteger>(batch_size / dispatch.jobs_per_block), 1);
        for (std::uint32_t pass = 0; pass < passes; ++pass) {
            for (std::uint32_t slice = 0; slice < argon2::ARGON2_SYNC_POINTS; ++slice) {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (encoder == nil) {
                    throw std::runtime_error("Metal segment encoder creation failed");
                }
                [encoder setComputePipelineState:active_pipeline()];
                [encoder setThreadgroupMemoryLength:threadgroup_memory_length() atIndex:0];
                [encoder setBuffer:memory_buffer offset:0 atIndex:0];
                SegmentArgs args{.passes = passes, .lanes = lanes, .segment_blocks = segment_blocks, .pass = pass, .slice = slice};
                if (precompute_refs) {
                    [encoder setBuffer:refs_buffer offset:0 atIndex:1];
                    [encoder setBytes:&args length:sizeof(args) atIndex:2];
                } else {
                    [encoder setBytes:&args length:sizeof(args) atIndex:1];
                }
                [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
                [encoder endEncoding];
            }
        }
    }
};

} // namespace

struct SolverSessionState {
    std::unique_ptr<PreparedGpuBatch> prepared;
    std::vector<std::string> passwords;

    SolverSessionState(const Job& job,
                       const SolverConfig& config,
                       std::size_t device_index)
        : prepared(std::make_unique<PreparedGpuBatch>(job, config, device_index, config.batch_size)),
          passwords(config.batch_size) {
        for (auto& password : passwords) {
            password.reserve(job.pass_prefix().size() + 20);
        }
    }
};

SolverSession::SolverSession() = default;
SolverSession::SolverSession(SolverSession&&) noexcept = default;
SolverSession& SolverSession::operator=(SolverSession&&) noexcept = default;
SolverSession::~SolverSession() = default;

namespace {

std::vector<std::uint8_t> compute_reference_digest(const Job& job, std::uint64_t nonce) {
    const auto password = job.password_for_nonce(nonce);
    std::vector<std::uint8_t> digest(kDigestSize);
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
    return digest;
}

std::vector<std::uint8_t> compute_gpu_digest(const Job& job,
                                             const SolverConfig& config,
                                             std::size_t device_index,
                                             std::uint64_t nonce) {
    PreparedGpuBatch prepared(job, config, device_index, 1);
    std::string password;
    password.reserve(job.pass_prefix().size() + 20);
    job.write_password_for_nonce(password, nonce);
    prepared.params.fillFirstBlocks(prepared.map_input_memory(0), password.data(), password.size(), argon2::ARGON2_ID, argon2::ARGON2_VERSION_13);
    prepared.run_once();
    std::vector<std::uint8_t> digest(kDigestSize);
    prepared.params.finalize(digest.data(), prepared.map_output_memory(0));
    return digest;
}

BatchResult mine_batch_gpu(const Job& job,
                           const SolverConfig& config,
                           std::uint64_t start_nonce,
                           std::atomic_bool& stop,
                           std::atomic<std::int64_t>& attempts,
                           PreparedGpuBatch& prepared,
                           std::vector<std::string>& passwords) {
    for (std::size_t i = 0; i < config.batch_size; ++i) {
        if (stop.load(std::memory_order_relaxed)) {
            break;
        }
        job.write_password_for_nonce(passwords[i], start_nonce + i);
        prepared.params.fillFirstBlocks(prepared.map_input_memory(i), passwords[i].data(), passwords[i].size(), argon2::ARGON2_ID, argon2::ARGON2_VERSION_13);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return {};
    }

    prepared.run_once();

    BatchResult result;
    std::int64_t local_attempts = 0;
    for (std::size_t i = 0; i < config.batch_size; ++i) {
        if (stop.load(std::memory_order_relaxed)) {
            break;
        }
        std::array<std::uint8_t, kDigestSize> digest{};
        prepared.params.finalize(digest.data(), prepared.map_output_memory(i));
        ++local_attempts;
        if (meets_difficulty(digest.data(), digest.size(), job.difficulty_bits())) {
            result.found = true;
            result.nonce = start_nonce + i;
            result.digest = digest;
            stop.store(true, std::memory_order_relaxed);
            break;
        }
    }
    if (local_attempts > 0) {
        attempts.fetch_add(local_attempts, std::memory_order_relaxed);
    }
    return result;
}

} // namespace

Solver::Solver(std::size_t device_index) : device_index_(device_index) {
}

SolverConfig Solver::default_config_for(const Job& job) const {
    SolverConfig config;
    config.batch_size = std::min<std::size_t>(estimate_max_batch_size(job), kDefaultRunBatchCap);
    config.by_segment = false;
    config.precompute_refs = false;
    return config;
}

SolveResult Solver::mine_batch(const Job& job,
                               const SolverConfig& config,
                               std::uint64_t start_nonce,
                               std::atomic_bool& stop,
                               std::atomic<std::int64_t>& attempts) const {
    auto session = create_session(job, config, start_nonce);
    return mine_next_batch(job, session, stop, attempts);
}

SolverSession Solver::create_session(const Job& job,
                                     const SolverConfig& config,
                                     std::uint64_t start_nonce) const {
    auto current_config = config;
    if (current_config.batch_size == 0) {
        current_config = default_config_for(job);
    }
    SolverSession session;
    session.config = current_config;
    session.next_nonce = start_nonce;
    session.state = std::make_unique<SolverSessionState>(job, current_config, device_index_);
    return session;
}

SolveResult Solver::mine_next_batch(const Job& job,
                                    SolverSession& session,
                                    std::atomic_bool& stop,
                                    std::atomic<std::int64_t>& attempts) const {
    if (!session.state) {
        throw std::runtime_error("Metal solver session is not initialized");
    }

    SolveResult result;
    const auto batch = mine_batch_gpu(
        job,
        session.config,
        session.next_nonce,
        stop,
        attempts,
        *session.state->prepared,
        session.state->passwords);
    session.next_nonce += session.config.batch_size;
    if (batch.found) {
        result.found = true;
        result.nonce = batch.nonce;
        result.digest = hex_encode(batch.digest.data(), batch.digest.size());
    }
    return result;
}

BenchmarkResult Solver::run_benchmark_case(const Job& job,
                                           const SolverConfig& config,
                                           std::chrono::milliseconds duration) const {
    BenchmarkResult result;
    result.config = config;

    std::atomic_bool stop{false};
    std::atomic<std::int64_t> attempts{0};
    std::uint64_t next_nonce = 1;
    PreparedGpuBatch prepared(job, config, device_index_, config.batch_size);
    std::vector<std::string> passwords(config.batch_size);
    for (auto& password : passwords) {
        password.reserve(job.pass_prefix().size() + 20);
    }
    const auto started_at = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - started_at < duration) {
        mine_batch_gpu(job, config, next_nonce, stop, attempts, prepared, passwords);
        next_nonce += config.batch_size;
    }

    result.attempts = attempts.load();
    result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at);
    if (result.elapsed.count() > 0) {
        result.attempts_per_second = static_cast<double>(result.attempts) * 1000.0 / static_cast<double>(result.elapsed.count());
    }
    return result;
}

void Solver::validate_against_reference(const Job& job, std::uint64_t nonce) const {
    const auto reference_digest = compute_reference_digest(job, nonce);
    const auto gpu_digest = compute_gpu_digest(job,
                                               SolverConfig{.batch_size = 1, .by_segment = false, .precompute_refs = false},
                                               device_index_,
                                               nonce);
    if (reference_digest != gpu_digest) {
        throw std::runtime_error("GPU digest mismatch for nonce=" + std::to_string(nonce)
            + ": expected=" + hex_encode(reference_digest.data(), reference_digest.size())
            + " actual=" + hex_encode(gpu_digest.data(), gpu_digest.size()));
    }
}

BenchmarkResult Solver::find_best_benchmark_config() const {
    Job benchmark_job(JobConfig{
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

    const auto max_batch_size = estimate_max_batch_size(benchmark_job);
    const auto candidates = build_benchmark_candidates(max_batch_size);

    std::cout << "GPU \xE8\x87\xAA\xE5\x8A\xA8\xE8\xB0\x83\xE4\xBC\x98\xE5\xBC\x80\xE5\xA7\x8B\xEF\xBC\x9A\xE5\x85\xB1 " << candidates.size()
              << " \xE7\xBB\x84\xE9\x85\x8D\xE7\xBD\xAE\xEF\xBC\x8C\xE6\xAF\x8F\xE7\xBB\x84\xE6\xB5\x8B\xE9\x80\x9F\xE7\xBA\xA6 " << (kBenchmarkCaseDuration.count() / 1000) << " \xE7\xA7\x92\xE3\x80\x82" << std::endl;

    BenchmarkResult best;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        std::cout << "GPU \xE8\x87\xAA\xE5\x8A\xA8\xE8\xB0\x83\xE4\xBC\x98\xE8\xBF\x9B\xE5\xBA\xA6 " << (index + 1) << "/" << candidates.size()
                  << "\xEF\xBC\x9A\xE6\x89\xB9\xE5\xA4\xA7\xE5\xB0\x8F " << candidate.batch_size
                  << "\xEF\xBC\x8C\xE6\x8C\x89\xE5\x88\x86\xE6\xAE\xB5 " << localized_bool(candidate.by_segment)
                  << "\xEF\xBC\x8C\xE9\xA2\x84\xE8\xAE\xA1\xE7\xAE\x97\xE5\x8F\x82\xE8\x80\x83\xE5\x80\xBC " << localized_bool(candidate.precompute_refs) << "\xE3\x80\x82" << std::endl;
        const auto current = run_benchmark_case(benchmark_job, candidate, kBenchmarkCaseDuration);
        std::cout << "GPU \xE8\x87\xAA\xE5\x8A\xA8\xE8\xB0\x83\xE4\xBC\x98\xE7\xBB\x93\xE6\x9E\x9C " << (index + 1) << "/" << candidates.size()
                  << "\xEF\xBC\x9A\xE6\x89\xB9\xE5\xA4\xA7\xE5\xB0\x8F " << current.config.batch_size
                  << "\xEF\xBC\x8C\xE6\x8C\x89\xE5\x88\x86\xE6\xAE\xB5 " << localized_bool(current.config.by_segment)
                  << "\xEF\xBC\x8C\xE9\xA2\x84\xE8\xAE\xA1\xE7\xAE\x97\xE5\x8F\x82\xE8\x80\x83\xE5\x80\xBC " << localized_bool(current.config.precompute_refs)
                  << "\xEF\xBC\x8C\xE9\x80\x9F\xE5\xBA\xA6\xE7\xBA\xA6 " << std::fixed << std::setprecision(2) << current.attempts_per_second << " \xE6\xAC\xA1/\xE7\xA7\x92\xE3\x80\x82" << std::endl;
        if (current.attempts_per_second > best.attempts_per_second) {
            best = current;
        }
    }

    std::cout << "GPU \xE8\x87\xAA\xE5\x8A\xA8\xE8\xB0\x83\xE4\xBC\x98\xE5\xAE\x8C\xE6\x88\x90\xEF\xBC\x9A\xE6\x8E\xA8\xE8\x8D\x90\xE6\x89\xB9\xE5\xA4\xA7\xE5\xB0\x8F " << best.config.batch_size
              << "\xEF\xBC\x8C\xE6\x8C\x89\xE5\x88\x86\xE6\xAE\xB5 " << localized_bool(best.config.by_segment)
              << "\xEF\xBC\x8C\xE9\xA2\x84\xE8\xAE\xA1\xE7\xAE\x97\xE5\x8F\x82\xE8\x80\x83\xE5\x80\xBC " << localized_bool(best.config.precompute_refs)
              << "\xEF\xBC\x8C\xE9\xA2\x84\xE8\xAE\xA1\xE9\x80\x9F\xE5\xBA\xA6\xE7\xBA\xA6 " << best.attempts_per_second << " \xE6\xAC\xA1/\xE7\xA7\x92\xE3\x80\x82" << std::endl;
    return best;
}

std::size_t Solver::estimate_max_batch_size(const Job& job) const {
    id<MTLDevice> device = require_device(device_index_);
    argon2::Argon2Params params(kDigestSize,
                                job.seed_bytes().data(),
                                job.seed_bytes().size(),
                                nullptr,
                                0,
                                nullptr,
                                0,
                                job.time_cost(),
                                job.memory_cost_kb(),
                                job.parallelism());
    const auto bytes_per_job = params.getMemorySize();
    if (bytes_per_job == 0) {
        return 1;
    }

    const auto segments = static_cast<std::size_t>(job.parallelism())
        * static_cast<std::size_t>(argon2::ARGON2_SYNC_POINTS / 2);
    const auto refs_overhead = segments
        * static_cast<std::size_t>(params.getSegmentBlocks())
        * sizeof(std::uint32_t)
        * 2;

    std::size_t working_set_budget = 0;
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        working_set_budget = static_cast<std::size_t>(device.recommendedMaxWorkingSetSize);
    }
    if (working_set_budget == 0) {
        working_set_budget = static_cast<std::size_t>(device.maxBufferLength);
    }

    const auto usable_budget = static_cast<std::size_t>(static_cast<double>(working_set_budget) * 0.5);
    if (usable_budget <= refs_overhead) {
        return 1;
    }

    const auto memory_buffer_budget = std::min<std::size_t>(
        usable_budget - refs_overhead,
        static_cast<std::size_t>(device.maxBufferLength));
    const auto max_batch = memory_buffer_budget / bytes_per_job;
    return std::max<std::size_t>(1, std::min<std::size_t>(max_batch, 256));
}

std::vector<SolverConfig> Solver::build_benchmark_candidates(std::size_t max_batch_size) {
    std::vector<SolverConfig> candidates;
    for (std::size_t batch_size : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}, std::size_t{256}}) {
        if (batch_size > max_batch_size) {
            continue;
        }
        candidates.push_back(SolverConfig{.batch_size = batch_size, .by_segment = false, .precompute_refs = false});
        candidates.push_back(SolverConfig{.batch_size = batch_size, .by_segment = true, .precompute_refs = false});
        candidates.push_back(SolverConfig{.batch_size = batch_size, .by_segment = true, .precompute_refs = true});
    }
    if (candidates.empty()) {
        candidates.push_back(SolverConfig{.batch_size = 1, .by_segment = false, .precompute_refs = false});
    }
    return candidates;
}

} // namespace app
