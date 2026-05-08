#include "app/solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "argon2-cuda/device.h"
#include "argon2-cuda/globalcontext.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/programcontext.h"
#include "argon2-gpu-common/argon2-common.h"
#include "argon2-gpu-common/argon2params.h"

#include <argon2.h>

namespace app {
namespace {

constexpr std::size_t kDigestSize = 32;
constexpr std::size_t kDefaultRunBatchCap = 32;
constexpr std::chrono::milliseconds kBenchmarkCaseDuration{5000};

const char* localized_bool(bool value) noexcept {
    return value ? "\xE6\x98\xAF" : "\xE5\x90\xA6";
}

struct BatchResult {
    bool found = false;
    std::uint64_t nonce = 0;
    std::array<std::uint8_t, kDigestSize> digest{};
};

struct PreparedGpuBatch {
    argon2::cuda::GlobalContext global;
    argon2::cuda::Device device;
    std::unique_ptr<argon2::cuda::ProgramContext> program_context;
    argon2::Argon2Params params;
    std::unique_ptr<argon2::cuda::ProcessingUnit> unit;

    PreparedGpuBatch(const Job& job,
                     const SolverConfig& config,
                     std::size_t device_index,
                     std::size_t batch_size)
        : global(),
          device(),
          program_context(),
          params(kDigestSize,
                 job.seed_bytes().data(),
                 job.seed_bytes().size(),
                 nullptr,
                 0,
                 nullptr,
                 0,
                 job.time_cost(),
                 job.memory_cost_kb(),
                 job.parallelism()),
          unit() {
        const auto& devices = global.getAllDevices();
        if (device_index >= devices.size()) {
            throw std::runtime_error("CUDA device index out of range");
        }
        device = devices[device_index];
        program_context = std::make_unique<argon2::cuda::ProgramContext>(
            &global,
            std::vector<argon2::cuda::Device>{device},
            argon2::ARGON2_ID,
            argon2::ARGON2_VERSION_13);
        unit = std::make_unique<argon2::cuda::ProcessingUnit>(
            program_context.get(),
            &params,
            &device,
            batch_size,
            config.by_segment,
            config.precompute_refs);
    }
};

} // namespace

struct SolverSessionState {
    std::unique_ptr<PreparedGpuBatch> prepared;
    std::vector<std::string> passwords;
    std::uint64_t prepared_start_nonce = 0;
    bool has_prepared_batch = false;
    bool generated_passwords_on_gpu = false;

    SolverSessionState(const Job& job,
                       const SolverConfig& config,
                       std::size_t device_index,
                       std::uint64_t start_nonce)
        : prepared(std::make_unique<PreparedGpuBatch>(
              job,
              config,
              device_index,
              config.batch_size)),
          passwords(config.batch_size) {
        generated_passwords_on_gpu = config.generate_passwords_on_gpu;
        for (auto& password : passwords) {
            password.reserve(job.pass_prefix().size() + 20);
        }
        if (generated_passwords_on_gpu) {
            prepared->unit->setGeneratedPasswordPrefix(
                job.pass_prefix().data(),
                job.pass_prefix().size());
        } else {
            prepare_batch(job, config, start_nonce);
        }
    }

    void prepare_batch(const Job& job,
                       const SolverConfig& config,
                       std::uint64_t start_nonce) {
        for (std::size_t i = 0; i < config.batch_size; ++i) {
            job.write_password_for_nonce(passwords[i], start_nonce + i);
            prepared->unit->setPassword(i, passwords[i].data(), passwords[i].size());
        }
        prepared_start_nonce = start_nonce;
        has_prepared_batch = true;
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
    prepared.unit->setPassword(0, password.data(), password.size());
    prepared.unit->beginProcessing();
    prepared.unit->endProcessing();

    std::vector<std::uint8_t> digest(kDigestSize);
    prepared.unit->getHash(0, digest.data());
    return digest;
}

} // namespace

Solver::Solver(std::size_t device_index) : device_index_(device_index) {
}

SolverConfig Solver::default_config_for(const Job& job) const {
    SolverConfig config;
    config.batch_size = std::min<std::size_t>(estimate_max_batch_size(job), kDefaultRunBatchCap);
    config.by_segment = false;
    config.precompute_refs = false;
    config.generate_passwords_on_gpu = true;
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
    session.state = std::make_unique<SolverSessionState>(
        job,
        current_config,
        device_index_,
        start_nonce);
    return session;
}

SolveResult Solver::mine_next_batch(const Job& job,
                                    SolverSession& session,
                                    std::atomic_bool& stop,
                                    std::atomic<std::int64_t>& attempts) const {
    if (!session.state) {
        throw std::runtime_error("CUDA solver session is not initialized");
    }

    auto& state = *session.state;
    if (!state.generated_passwords_on_gpu
        && (!state.has_prepared_batch || state.prepared_start_nonce != session.next_nonce)) {
        state.prepare_batch(job, session.config, session.next_nonce);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return {};
    }

    const auto current_start_nonce = session.next_nonce;
    const auto next_start_nonce = current_start_nonce + session.config.batch_size;
    if (state.generated_passwords_on_gpu) {
        state.prepared->unit->beginProcessingWithGeneratedPasswords(
            current_start_nonce,
            job.difficulty_bits());
    } else {
        state.prepared->unit->beginProcessingWithDifficultyCheck(job.difficulty_bits());
    }
    if (!state.generated_passwords_on_gpu && !stop.load(std::memory_order_relaxed)) {
        // The ProcessingUnit API allows staging the next input batch after
        // beginProcessing(), while the CUDA stream works on the current batch.
        state.prepare_batch(job, session.config, next_start_nonce);
    }
    state.prepared->unit->endProcessing();

    BatchResult batch;
    if (!stop.load(std::memory_order_relaxed)) {
        std::size_t hit_index = 0;
        std::array<std::uint8_t, kDigestSize> digest{};
        if (state.prepared->unit->getDifficultyResult(&hit_index, digest.data())) {
            batch.found = true;
            batch.nonce = current_start_nonce + hit_index;
            batch.digest = digest;
            stop.store(true, std::memory_order_relaxed);
        }
    }
    attempts.fetch_add(
        static_cast<std::int64_t>(session.config.batch_size),
        std::memory_order_relaxed);

    session.next_nonce = next_start_nonce;
    SolveResult result;
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
    auto session = create_session(job, config, 1);
    const auto started_at = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - started_at < duration) {
        mine_next_batch(job, session, stop, attempts);
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
    SolverConfig config;
    config.batch_size = 1;
    config.by_segment = false;
    config.precompute_refs = false;
    config.generate_passwords_on_gpu = false;
    const auto gpu_digest = compute_gpu_digest(job,
                                               config,
                                               device_index_,
                                               nonce);
    if (reference_digest != gpu_digest) {
        throw std::runtime_error("GPU digest mismatch for nonce=" + std::to_string(nonce)
            + ": expected=" + hex_encode(reference_digest.data(), reference_digest.size())
            + " actual=" + hex_encode(gpu_digest.data(), gpu_digest.size()));
    }
}

BenchmarkResult Solver::find_best_benchmark_config() const {
    JobConfig benchmark_config;
    benchmark_config.seed = "benchmark-seed-fixed";
    benchmark_config.round_id = 1;
    benchmark_config.visitor_id = "benchmark-visitor-fixed";
    benchmark_config.challenge_id = 1;
    benchmark_config.session_salt = "benchmark-session-salt-fixed";
    benchmark_config.time_cost = 1;
    benchmark_config.memory_cost_mb = 64;
    benchmark_config.parallelism = 1;
    benchmark_config.difficulty_bits = 255;
    Job benchmark_job(std::move(benchmark_config));

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
    using namespace argon2::cuda;

    GlobalContext global;
    const auto& devices = global.getAllDevices();
    if (device_index_ >= devices.size()) {
        throw std::runtime_error("CUDA device index out of range");
    }

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, static_cast<int>(device_index_));

    const auto bytes_per_job = static_cast<std::size_t>(job.memory_cost_kb()) * 1024ULL * static_cast<std::size_t>(job.parallelism());
    if (bytes_per_job == 0) {
        return 1;
    }

    const auto usable = static_cast<std::size_t>(static_cast<double>(properties.totalGlobalMem) * 0.5);
    const auto max_batch = usable / bytes_per_job;
    return std::max<std::size_t>(1, std::min<std::size_t>(max_batch, 256));
}

std::vector<SolverConfig> Solver::build_benchmark_candidates(std::size_t max_batch_size) {
    std::vector<SolverConfig> candidates;
    for (std::size_t batch_size : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}, std::size_t{256}}) {
        if (batch_size > max_batch_size) {
            continue;
        }
        SolverConfig default_config;
        default_config.batch_size = batch_size;
        default_config.by_segment = false;
        default_config.precompute_refs = false;
        default_config.generate_passwords_on_gpu = true;
        candidates.push_back(default_config);

        SolverConfig segmented_config;
        segmented_config.batch_size = batch_size;
        segmented_config.by_segment = true;
        segmented_config.precompute_refs = false;
        segmented_config.generate_passwords_on_gpu = true;
        candidates.push_back(segmented_config);

        SolverConfig precomputed_config;
        precomputed_config.batch_size = batch_size;
        precomputed_config.by_segment = true;
        precomputed_config.precompute_refs = true;
        precomputed_config.generate_passwords_on_gpu = true;
        candidates.push_back(precomputed_config);
    }
    if (candidates.empty()) {
        SolverConfig fallback_config;
        fallback_config.batch_size = 1;
        fallback_config.by_segment = false;
        fallback_config.precompute_refs = false;
        fallback_config.generate_passwords_on_gpu = true;
        candidates.push_back(fallback_config);
    }
    return candidates;
}

namespace {

	struct Rpow2CudaKernelParams {
	    std::uint32_t difficulty_bits;
	    std::uint32_t block_count;
	    std::uint32_t nonce_offset;
	    std::uint32_t padding;
	    std::uint32_t initial_state[8];
	    std::uint32_t template_words[32];
	    unsigned long long start_nonce;
	    unsigned long long batch_size;
	};

struct Rpow2CudaDeviceResult {
    unsigned int found;
    unsigned long long nonce;
    std::uint32_t digest_words[8];
};

	__constant__ std::uint32_t kRpow2Sha256K[64] = {
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

	constexpr std::uint32_t kRpow2Sha256KHost[64] = {
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

	__device__ __forceinline__ std::uint32_t rpow2_rotr32(std::uint32_t value, unsigned int bits) {
	    return __funnelshift_r(value, value, bits);
	}

	__device__ __forceinline__ std::uint32_t rpow2_ch(std::uint32_t x,
	                                                  std::uint32_t y,
	                                                  std::uint32_t z) {
	    return (x & y) ^ ((~x) & z);
	}

	__device__ __forceinline__ std::uint32_t rpow2_maj(std::uint32_t x,
	                                                   std::uint32_t y,
	                                                   std::uint32_t z) {
	    return (x & y) ^ (x & z) ^ (y & z);
	}

	__device__ __forceinline__ std::uint32_t rpow2_big_sigma0(std::uint32_t value) {
	    return rpow2_rotr32(value, 2u) ^ rpow2_rotr32(value, 13u) ^ rpow2_rotr32(value, 22u);
	}

	__device__ __forceinline__ std::uint32_t rpow2_big_sigma1(std::uint32_t value) {
	    return rpow2_rotr32(value, 6u) ^ rpow2_rotr32(value, 11u) ^ rpow2_rotr32(value, 25u);
	}

	__device__ __forceinline__ std::uint32_t rpow2_small_sigma0(std::uint32_t value) {
	    return rpow2_rotr32(value, 7u) ^ rpow2_rotr32(value, 18u) ^ (value >> 3);
	}

		__device__ __forceinline__ std::uint32_t rpow2_small_sigma1(std::uint32_t value) {
		    return rpow2_rotr32(value, 17u) ^ rpow2_rotr32(value, 19u) ^ (value >> 10);
		}

		__device__ __forceinline__ std::uint32_t rpow2_bswap32(std::uint32_t value) {
		    return ((value & 0x000000ffu) << 24)
		        | ((value & 0x0000ff00u) << 8)
		        | ((value & 0x00ff0000u) >> 8)
		        | ((value & 0xff000000u) >> 24);
		}

	std::uint32_t rpow2_rotr32_host(std::uint32_t value, unsigned int bits) {
	    return (value >> bits) | (value << (32u - bits));
	}

	__device__ void rpow2_write_nonce_words(std::uint32_t* w,
	                                        std::uint32_t block,
	                                        std::uint32_t nonce_offset,
	                                        unsigned long long nonce) {
	    for (std::uint32_t i = 0; i < 8; ++i) {
	        const std::uint32_t absolute_byte = nonce_offset + i;
	        if ((absolute_byte >> 6) != block) {
	            continue;
	        }
        const std::uint32_t block_byte = absolute_byte & 63u;
        const std::uint32_t word_index = block_byte >> 2;
        const std::uint32_t byte_index = block_byte & 3u;
        const std::uint32_t shift = (3u - byte_index) * 8u;
        w[word_index] |= static_cast<std::uint32_t>((nonce >> (i * 8u)) & 0xffull) << shift;
    }
}

__device__ std::uint32_t rpow2_trailing_zero_bits(std::uint32_t h0,
                                                  std::uint32_t h1,
                                                  std::uint32_t h2,
                                                  std::uint32_t h3,
                                                  std::uint32_t h4,
                                                  std::uint32_t h5,
                                                  std::uint32_t h6,
                                                  std::uint32_t h7) {
    if (h7 != 0) {
        return static_cast<std::uint32_t>(__ffs(h7) - 1);
    }
    if (h6 != 0) {
        return 32u + static_cast<std::uint32_t>(__ffs(h6) - 1);
    }
    if (h5 != 0) {
        return 64u + static_cast<std::uint32_t>(__ffs(h5) - 1);
    }
    if (h4 != 0) {
        return 96u + static_cast<std::uint32_t>(__ffs(h4) - 1);
    }
    if (h3 != 0) {
        return 128u + static_cast<std::uint32_t>(__ffs(h3) - 1);
    }
    if (h2 != 0) {
        return 160u + static_cast<std::uint32_t>(__ffs(h2) - 1);
    }
    if (h1 != 0) {
        return 192u + static_cast<std::uint32_t>(__ffs(h1) - 1);
    }
    if (h0 != 0) {
        return 224u + static_cast<std::uint32_t>(__ffs(h0) - 1);
    }
    return 256u;
}

	__device__ __forceinline__ void rpow2_compress_block(std::uint32_t* h0,
	                                                     std::uint32_t* h1,
	                                                     std::uint32_t* h2,
	                                                     std::uint32_t* h3,
	                                                     std::uint32_t* h4,
	                                                     std::uint32_t* h5,
	                                                     std::uint32_t* h6,
	                                                     std::uint32_t* h7,
	                                                     std::uint32_t* w) {
	    std::uint32_t a = *h0;
	    std::uint32_t b = *h1;
	    std::uint32_t c = *h2;
	    std::uint32_t d = *h3;
	    std::uint32_t e = *h4;
	    std::uint32_t f = *h5;
	    std::uint32_t g = *h6;
	    std::uint32_t h = *h7;

#pragma unroll 64
	    for (std::uint32_t i = 0; i < 64; ++i) {
	        std::uint32_t schedule_word = w[i & 15u];
	        if (i >= 16) {
	            schedule_word = schedule_word
	                + rpow2_small_sigma0(w[(i + 1u) & 15u])
	                + w[(i + 9u) & 15u]
	                + rpow2_small_sigma1(w[(i + 14u) & 15u]);
	            w[i & 15u] = schedule_word;
	        }
	        const std::uint32_t temp1 = h
	            + rpow2_big_sigma1(e)
	            + rpow2_ch(e, f, g)
	            + kRpow2Sha256K[i]
	            + schedule_word;
	        const std::uint32_t temp2 = rpow2_big_sigma0(a) + rpow2_maj(a, b, c);
	        h = g;
	        g = f;
	        f = e;
	        e = d + temp1;
	        d = c;
	        c = b;
	        b = a;
	        a = temp1 + temp2;
	    }

	    *h0 += a;
	    *h1 += b;
	    *h2 += c;
	    *h3 += d;
	    *h4 += e;
	    *h5 += f;
	    *h6 += g;
	    *h7 += h;
	}

		__device__ __forceinline__ void rpow2_hash_nonce(const Rpow2CudaKernelParams& params,
		                                                 unsigned long long nonce,
		                                                 std::uint32_t* h0,
	                                                 std::uint32_t* h1,
	                                                 std::uint32_t* h2,
	                                                 std::uint32_t* h3,
	                                                 std::uint32_t* h4,
	                                                 std::uint32_t* h5,
	                                                 std::uint32_t* h6,
	                                                 std::uint32_t* h7) {
	    *h0 = params.initial_state[0];
	    *h1 = params.initial_state[1];
	    *h2 = params.initial_state[2];
	    *h3 = params.initial_state[3];
	    *h4 = params.initial_state[4];
	    *h5 = params.initial_state[5];
	    *h6 = params.initial_state[6];
	    *h7 = params.initial_state[7];

#pragma unroll 2
	    for (std::uint32_t block = 0; block < params.block_count; ++block) {
	        std::uint32_t w[16];
#pragma unroll
	        for (std::uint32_t i = 0; i < 16; ++i) {
	            w[i] = params.template_words[block * 16u + i];
	        }
	        rpow2_write_nonce_words(w, block, params.nonce_offset, nonce);
		        rpow2_compress_block(h0, h1, h2, h3, h4, h5, h6, h7, w);
		    }
		}

		__device__ __forceinline__ void rpow2_sha256_round_scalar(std::uint32_t& a,
		                                                          std::uint32_t& b,
		                                                          std::uint32_t& c,
		                                                          std::uint32_t& d,
		                                                          std::uint32_t& e,
		                                                          std::uint32_t& f,
		                                                          std::uint32_t& g,
		                                                          std::uint32_t& h,
		                                                          std::uint32_t schedule_word,
		                                                          std::uint32_t round_constant) {
		    const std::uint32_t temp1 = h
		        + rpow2_big_sigma1(e)
		        + rpow2_ch(e, f, g)
		        + round_constant
		        + schedule_word;
		    const std::uint32_t temp2 = rpow2_big_sigma0(a) + rpow2_maj(a, b, c);
		    h = g;
		    g = f;
		    f = e;
		    e = d + temp1;
		    d = c;
		    c = b;
		    b = a;
		    a = temp1 + temp2;
		}

		__device__ __forceinline__ void rpow2_hash_nonce_prefix16(const Rpow2CudaKernelParams& params,
		                                                         unsigned long long nonce,
		                                                         std::uint32_t* h0,
		                                                         std::uint32_t* h1,
		                                                         std::uint32_t* h2,
		                                                         std::uint32_t* h3,
		                                                         std::uint32_t* h4,
		                                                         std::uint32_t* h5,
		                                                         std::uint32_t* h6,
		                                                         std::uint32_t* h7) {
		    std::uint32_t a = 0x6a09e667u;
		    std::uint32_t b = 0xbb67ae85u;
		    std::uint32_t c = 0x3c6ef372u;
		    std::uint32_t d = 0xa54ff53au;
		    std::uint32_t e = 0x510e527fu;
		    std::uint32_t f = 0x9b05688cu;
		    std::uint32_t g = 0x1f83d9abu;
		    std::uint32_t h = 0x5be0cd19u;
		    std::uint32_t w0 = params.template_words[0];
		    std::uint32_t w1 = params.template_words[1];
		    std::uint32_t w2 = params.template_words[2];
		    std::uint32_t w3 = params.template_words[3];
		    std::uint32_t w4 = rpow2_bswap32(static_cast<std::uint32_t>(nonce));
		    std::uint32_t w5 = rpow2_bswap32(static_cast<std::uint32_t>(nonce >> 32));
		    std::uint32_t w6 = 0x80000000u;
		    std::uint32_t w7 = 0u;
		    std::uint32_t w8 = 0u;
		    std::uint32_t w9 = 0u;
		    std::uint32_t w10 = 0u;
		    std::uint32_t w11 = 0u;
		    std::uint32_t w12 = 0u;
		    std::uint32_t w13 = 0u;
		    std::uint32_t w14 = 0u;
		    std::uint32_t w15 = 192u;

#define RPOW2_ROUND(W, K) rpow2_sha256_round_scalar(a, b, c, d, e, f, g, h, W, K)
#define RPOW2_EXPAND(W, W1, W9, W14) W = W + rpow2_small_sigma0(W1) + W9 + rpow2_small_sigma1(W14)
		    RPOW2_ROUND(w0, 0x428a2f98u);
		    RPOW2_ROUND(w1, 0x71374491u);
		    RPOW2_ROUND(w2, 0xb5c0fbcfu);
		    RPOW2_ROUND(w3, 0xe9b5dba5u);
		    RPOW2_ROUND(w4, 0x3956c25bu);
		    RPOW2_ROUND(w5, 0x59f111f1u);
		    RPOW2_ROUND(w6, 0x923f82a4u);
		    RPOW2_ROUND(w7, 0xab1c5ed5u);
		    RPOW2_ROUND(w8, 0xd807aa98u);
		    RPOW2_ROUND(w9, 0x12835b01u);
		    RPOW2_ROUND(w10, 0x243185beu);
		    RPOW2_ROUND(w11, 0x550c7dc3u);
		    RPOW2_ROUND(w12, 0x72be5d74u);
		    RPOW2_ROUND(w13, 0x80deb1feu);
		    RPOW2_ROUND(w14, 0x9bdc06a7u);
		    RPOW2_ROUND(w15, 0xc19bf174u);
		    RPOW2_EXPAND(w0, w1, w9, w14);
		    RPOW2_ROUND(w0, 0xe49b69c1u);
		    RPOW2_EXPAND(w1, w2, w10, w15);
		    RPOW2_ROUND(w1, 0xefbe4786u);
		    RPOW2_EXPAND(w2, w3, w11, w0);
		    RPOW2_ROUND(w2, 0x0fc19dc6u);
		    RPOW2_EXPAND(w3, w4, w12, w1);
		    RPOW2_ROUND(w3, 0x240ca1ccu);
		    RPOW2_EXPAND(w4, w5, w13, w2);
		    RPOW2_ROUND(w4, 0x2de92c6fu);
		    RPOW2_EXPAND(w5, w6, w14, w3);
		    RPOW2_ROUND(w5, 0x4a7484aau);
		    RPOW2_EXPAND(w6, w7, w15, w4);
		    RPOW2_ROUND(w6, 0x5cb0a9dcu);
		    RPOW2_EXPAND(w7, w8, w0, w5);
		    RPOW2_ROUND(w7, 0x76f988dau);
		    RPOW2_EXPAND(w8, w9, w1, w6);
		    RPOW2_ROUND(w8, 0x983e5152u);
		    RPOW2_EXPAND(w9, w10, w2, w7);
		    RPOW2_ROUND(w9, 0xa831c66du);
		    RPOW2_EXPAND(w10, w11, w3, w8);
		    RPOW2_ROUND(w10, 0xb00327c8u);
		    RPOW2_EXPAND(w11, w12, w4, w9);
		    RPOW2_ROUND(w11, 0xbf597fc7u);
		    RPOW2_EXPAND(w12, w13, w5, w10);
		    RPOW2_ROUND(w12, 0xc6e00bf3u);
		    RPOW2_EXPAND(w13, w14, w6, w11);
		    RPOW2_ROUND(w13, 0xd5a79147u);
		    RPOW2_EXPAND(w14, w15, w7, w12);
		    RPOW2_ROUND(w14, 0x06ca6351u);
		    RPOW2_EXPAND(w15, w0, w8, w13);
		    RPOW2_ROUND(w15, 0x14292967u);
		    RPOW2_EXPAND(w0, w1, w9, w14);
		    RPOW2_ROUND(w0, 0x27b70a85u);
		    RPOW2_EXPAND(w1, w2, w10, w15);
		    RPOW2_ROUND(w1, 0x2e1b2138u);
		    RPOW2_EXPAND(w2, w3, w11, w0);
		    RPOW2_ROUND(w2, 0x4d2c6dfcu);
		    RPOW2_EXPAND(w3, w4, w12, w1);
		    RPOW2_ROUND(w3, 0x53380d13u);
		    RPOW2_EXPAND(w4, w5, w13, w2);
		    RPOW2_ROUND(w4, 0x650a7354u);
		    RPOW2_EXPAND(w5, w6, w14, w3);
		    RPOW2_ROUND(w5, 0x766a0abbu);
		    RPOW2_EXPAND(w6, w7, w15, w4);
		    RPOW2_ROUND(w6, 0x81c2c92eu);
		    RPOW2_EXPAND(w7, w8, w0, w5);
		    RPOW2_ROUND(w7, 0x92722c85u);
		    RPOW2_EXPAND(w8, w9, w1, w6);
		    RPOW2_ROUND(w8, 0xa2bfe8a1u);
		    RPOW2_EXPAND(w9, w10, w2, w7);
		    RPOW2_ROUND(w9, 0xa81a664bu);
		    RPOW2_EXPAND(w10, w11, w3, w8);
		    RPOW2_ROUND(w10, 0xc24b8b70u);
		    RPOW2_EXPAND(w11, w12, w4, w9);
		    RPOW2_ROUND(w11, 0xc76c51a3u);
		    RPOW2_EXPAND(w12, w13, w5, w10);
		    RPOW2_ROUND(w12, 0xd192e819u);
		    RPOW2_EXPAND(w13, w14, w6, w11);
		    RPOW2_ROUND(w13, 0xd6990624u);
		    RPOW2_EXPAND(w14, w15, w7, w12);
		    RPOW2_ROUND(w14, 0xf40e3585u);
		    RPOW2_EXPAND(w15, w0, w8, w13);
		    RPOW2_ROUND(w15, 0x106aa070u);
		    RPOW2_EXPAND(w0, w1, w9, w14);
		    RPOW2_ROUND(w0, 0x19a4c116u);
		    RPOW2_EXPAND(w1, w2, w10, w15);
		    RPOW2_ROUND(w1, 0x1e376c08u);
		    RPOW2_EXPAND(w2, w3, w11, w0);
		    RPOW2_ROUND(w2, 0x2748774cu);
		    RPOW2_EXPAND(w3, w4, w12, w1);
		    RPOW2_ROUND(w3, 0x34b0bcb5u);
		    RPOW2_EXPAND(w4, w5, w13, w2);
		    RPOW2_ROUND(w4, 0x391c0cb3u);
		    RPOW2_EXPAND(w5, w6, w14, w3);
		    RPOW2_ROUND(w5, 0x4ed8aa4au);
		    RPOW2_EXPAND(w6, w7, w15, w4);
		    RPOW2_ROUND(w6, 0x5b9cca4fu);
		    RPOW2_EXPAND(w7, w8, w0, w5);
		    RPOW2_ROUND(w7, 0x682e6ff3u);
		    RPOW2_EXPAND(w8, w9, w1, w6);
		    RPOW2_ROUND(w8, 0x748f82eeu);
		    RPOW2_EXPAND(w9, w10, w2, w7);
		    RPOW2_ROUND(w9, 0x78a5636fu);
		    RPOW2_EXPAND(w10, w11, w3, w8);
		    RPOW2_ROUND(w10, 0x84c87814u);
		    RPOW2_EXPAND(w11, w12, w4, w9);
		    RPOW2_ROUND(w11, 0x8cc70208u);
		    RPOW2_EXPAND(w12, w13, w5, w10);
		    RPOW2_ROUND(w12, 0x90befffau);
		    RPOW2_EXPAND(w13, w14, w6, w11);
		    RPOW2_ROUND(w13, 0xa4506cebu);
		    RPOW2_EXPAND(w14, w15, w7, w12);
		    RPOW2_ROUND(w14, 0xbef9a3f7u);
		    RPOW2_EXPAND(w15, w0, w8, w13);
		    RPOW2_ROUND(w15, 0xc67178f2u);
#undef RPOW2_ROUND
#undef RPOW2_EXPAND

		    *h0 = 0x6a09e667u + a;
		    *h1 = 0xbb67ae85u + b;
		    *h2 = 0x3c6ef372u + c;
		    *h3 = 0xa54ff53au + d;
		    *h4 = 0x510e527fu + e;
		    *h5 = 0x9b05688cu + f;
		    *h6 = 0x1f83d9abu + g;
		    *h7 = 0x5be0cd19u + h;
		}

		template <unsigned int NoncesPerThread>
		__global__ __launch_bounds__(512, 2) void rpow2_mine_kernel(Rpow2CudaKernelParams params,
		                                                            Rpow2CudaDeviceResult* result) {
	    const unsigned long long thread_id = static_cast<unsigned long long>(blockIdx.x)
	        * static_cast<unsigned long long>(blockDim.x)
	        + static_cast<unsigned long long>(threadIdx.x);
	    const unsigned long long thread_stride = static_cast<unsigned long long>(gridDim.x)
	        * static_cast<unsigned long long>(blockDim.x);
	    const unsigned long long group_stride = thread_stride
	        * static_cast<unsigned long long>(NoncesPerThread);
	    const volatile unsigned int* found_flag =
	        reinterpret_cast<const volatile unsigned int*>(&result->found);

	    for (unsigned long long group_offset = thread_id * static_cast<unsigned long long>(NoncesPerThread);
	         group_offset < params.batch_size;
	         group_offset += group_stride) {
	        if (*found_flag != 0u) {
	            return;
	        }
#pragma unroll
	        for (unsigned int item = 0; item < NoncesPerThread; ++item) {
	            const unsigned long long offset = group_offset + static_cast<unsigned long long>(item);
	            if (offset >= params.batch_size) {
	                return;
	            }
	            const unsigned long long nonce = params.start_nonce + offset;
	            std::uint32_t h0;
	            std::uint32_t h1;
	            std::uint32_t h2;
	            std::uint32_t h3;
	            std::uint32_t h4;
	            std::uint32_t h5;
	            std::uint32_t h6;
	            std::uint32_t h7;
	            rpow2_hash_nonce(params, nonce, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7);

	            if (rpow2_trailing_zero_bits(h0, h1, h2, h3, h4, h5, h6, h7) >= params.difficulty_bits) {
	                if (atomicCAS(&result->found, 0u, 1u) == 0u) {
	                    result->nonce = nonce;
	                    result->digest_words[0] = h0;
	                    result->digest_words[1] = h1;
	                    result->digest_words[2] = h2;
	                    result->digest_words[3] = h3;
	                    result->digest_words[4] = h4;
	                    result->digest_words[5] = h5;
	                    result->digest_words[6] = h6;
	                    result->digest_words[7] = h7;
	                }
	                return;
	            }
	        }
		    }
		}

		template <unsigned int NoncesPerThread>
		__global__ __launch_bounds__(512, 2) void rpow2_mine_prefix16_kernel(Rpow2CudaKernelParams params,
		                                                                     Rpow2CudaDeviceResult* result) {
		    const unsigned long long thread_id = static_cast<unsigned long long>(blockIdx.x)
		        * static_cast<unsigned long long>(blockDim.x)
		        + static_cast<unsigned long long>(threadIdx.x);
		    const unsigned long long thread_stride = static_cast<unsigned long long>(gridDim.x)
		        * static_cast<unsigned long long>(blockDim.x);
		    const unsigned long long group_stride = thread_stride
		        * static_cast<unsigned long long>(NoncesPerThread);
		    const volatile unsigned int* found_flag =
		        reinterpret_cast<const volatile unsigned int*>(&result->found);

		    for (unsigned long long group_offset = thread_id * static_cast<unsigned long long>(NoncesPerThread);
		         group_offset < params.batch_size;
		         group_offset += group_stride) {
		        if (*found_flag != 0u) {
		            return;
		        }
#pragma unroll
		        for (unsigned int item = 0; item < NoncesPerThread; ++item) {
		            const unsigned long long offset = group_offset + static_cast<unsigned long long>(item);
		            if (offset >= params.batch_size) {
		                return;
		            }
		            const unsigned long long nonce = params.start_nonce + offset;
		            std::uint32_t h0;
		            std::uint32_t h1;
		            std::uint32_t h2;
		            std::uint32_t h3;
		            std::uint32_t h4;
		            std::uint32_t h5;
		            std::uint32_t h6;
		            std::uint32_t h7;
		            rpow2_hash_nonce_prefix16(params, nonce, &h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7);

		            if (rpow2_trailing_zero_bits(h0, h1, h2, h3, h4, h5, h6, h7) >= params.difficulty_bits) {
		                if (atomicCAS(&result->found, 0u, 1u) == 0u) {
		                    result->nonce = nonce;
		                    result->digest_words[0] = h0;
		                    result->digest_words[1] = h1;
		                    result->digest_words[2] = h2;
		                    result->digest_words[3] = h3;
		                    result->digest_words[4] = h4;
		                    result->digest_words[5] = h5;
		                    result->digest_words[6] = h6;
		                    result->digest_words[7] = h7;
		                }
		                return;
		            }
		        }
		    }
		}

		__global__ void rpow2_empty_kernel() {
		}

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

	void rpow2_compress_host(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
	    std::uint32_t w[16];
	    for (std::uint32_t index = 0; index < 16; ++index) {
	        const auto offset = index * 4;
	        w[index] = (static_cast<std::uint32_t>(block[offset]) << 24)
	            | (static_cast<std::uint32_t>(block[offset + 1]) << 16)
	            | (static_cast<std::uint32_t>(block[offset + 2]) << 8)
	            | static_cast<std::uint32_t>(block[offset + 3]);
	    }

	    std::uint32_t a = state[0];
	    std::uint32_t b = state[1];
	    std::uint32_t c = state[2];
	    std::uint32_t d = state[3];
	    std::uint32_t e = state[4];
	    std::uint32_t f = state[5];
	    std::uint32_t g = state[6];
	    std::uint32_t h = state[7];
	    for (std::uint32_t i = 0; i < 64; ++i) {
	        std::uint32_t schedule_word = w[i & 15u];
	        if (i >= 16) {
	            const auto s0 = rpow2_rotr32_host(w[(i + 1u) & 15u], 7u)
	                ^ rpow2_rotr32_host(w[(i + 1u) & 15u], 18u)
	                ^ (w[(i + 1u) & 15u] >> 3);
	            const auto s1 = rpow2_rotr32_host(w[(i + 14u) & 15u], 17u)
	                ^ rpow2_rotr32_host(w[(i + 14u) & 15u], 19u)
	                ^ (w[(i + 14u) & 15u] >> 10);
	            schedule_word = schedule_word + s0 + w[(i + 9u) & 15u] + s1;
	            w[i & 15u] = schedule_word;
	        }
	        const auto temp1 = h
	            + (rpow2_rotr32_host(e, 6u) ^ rpow2_rotr32_host(e, 11u) ^ rpow2_rotr32_host(e, 25u))
	            + ((e & f) ^ ((~e) & g))
	            + kRpow2Sha256KHost[i]
	            + schedule_word;
	        const auto temp2 =
	            (rpow2_rotr32_host(a, 2u) ^ rpow2_rotr32_host(a, 13u) ^ rpow2_rotr32_host(a, 22u))
	            + ((a & b) ^ (a & c) ^ (b & c));
	        h = g;
	        g = f;
	        f = e;
	        e = d + temp1;
	        d = c;
	        c = b;
	        b = a;
	        a = temp1 + temp2;
	    }
	    state[0] += a;
	    state[1] += b;
	    state[2] += c;
	    state[3] += d;
	    state[4] += e;
	    state[5] += f;
	    state[6] += g;
	    state[7] += h;
	}

	struct Rpow2PreparedTail {
	    std::array<std::uint32_t, 8> initial_state{};
	    std::array<std::uint32_t, 32> template_words{};
	    std::uint32_t block_count = 0;
	    std::uint32_t nonce_offset = 0;
	};

	Rpow2PreparedTail build_rpow2_prepared_tail(const std::uint8_t* prefix,
	                                            std::size_t prefix_len) {
	    const auto message_len = prefix_len + 8;
	    if (prefix_len > 111) {
	        throw std::runtime_error("RPOW2 nonce prefix is too long for CUDA solver");
	    }

	    Rpow2PreparedTail prepared;
	    prepared.initial_state = {
	        0x6a09e667,
	        0xbb67ae85,
	        0x3c6ef372,
	        0xa54ff53a,
	        0x510e527f,
	        0x9b05688c,
	        0x1f83d9ab,
	        0x5be0cd19,
	    };

	    const auto full_prefix_blocks = prefix_len / 64;
	    for (std::size_t block = 0; block < full_prefix_blocks; ++block) {
	        rpow2_compress_host(prepared.initial_state, prefix + block * 64);
	    }

	    const auto tail_prefix_len = prefix_len % 64;
	    const auto tail_message_len = tail_prefix_len + 8;
	    const auto padded_tail_len = ((tail_message_len + 9 + 63) / 64) * 64;
	    if (padded_tail_len > 128) {
	        throw std::runtime_error("RPOW2 padded tail is too long for CUDA solver");
	    }

	    std::array<std::uint8_t, 128> padded{};
	    if (tail_prefix_len > 0) {
	        std::memcpy(padded.data(), prefix + full_prefix_blocks * 64, tail_prefix_len);
	    }
	    padded[tail_message_len] = 0x80;
	    const auto bit_len = static_cast<std::uint64_t>(message_len) * 8;
	    for (std::size_t i = 0; i < 8; ++i) {
	        padded[padded_tail_len - 1 - i] = static_cast<std::uint8_t>((bit_len >> (i * 8)) & 0xff);
	    }

	    prepared.block_count = static_cast<std::uint32_t>(padded_tail_len / 64);
	    prepared.nonce_offset = static_cast<std::uint32_t>(tail_prefix_len);
	    for (std::size_t index = 0; index < padded_tail_len / 4; ++index) {
	        const auto offset = index * 4;
	        prepared.template_words[index] = (static_cast<std::uint32_t>(padded[offset]) << 24)
	            | (static_cast<std::uint32_t>(padded[offset + 1]) << 16)
	            | (static_cast<std::uint32_t>(padded[offset + 2]) << 8)
	            | static_cast<std::uint32_t>(padded[offset + 3]);
	    }
	    return prepared;
	}

void digest_words_to_bytes(const std::uint32_t words[8], std::array<std::uint8_t, 32>& digest) {
    for (std::size_t index = 0; index < 8; ++index) {
        digest[index * 4] = static_cast<std::uint8_t>((words[index] >> 24) & 0xff);
        digest[index * 4 + 1] = static_cast<std::uint8_t>((words[index] >> 16) & 0xff);
        digest[index * 4 + 2] = static_cast<std::uint8_t>((words[index] >> 8) & 0xff);
        digest[index * 4 + 3] = static_cast<std::uint8_t>(words[index] & 0xff);
    }
}

} // namespace

	Rpow2CudaSession::Rpow2CudaSession(std::size_t device_index,
	                                   const std::uint8_t* nonce_prefix,
	                                   std::size_t nonce_prefix_len,
	                                   std::uint32_t difficulty_bits,
	                                   std::uint64_t batch_size,
	                                   std::uint64_t start_nonce,
	                                   std::uint32_t threads_per_block,
	                                   std::uint32_t nonces_per_thread,
	                                   std::uint32_t max_blocks)
	    : device_index_(static_cast<int>(device_index)),
	      prefix_len_(static_cast<std::uint32_t>(nonce_prefix_len)),
	      difficulty_bits_(difficulty_bits),
	      batch_size_(batch_size),
	      next_nonce_(start_nonce) {
    if (nonce_prefix == nullptr && nonce_prefix_len > 0) {
        throw std::runtime_error("RPOW2 nonce prefix pointer is null");
    }
    if (nonce_prefix_len > 111) {
        throw std::runtime_error("RPOW2 nonce prefix is too long for CUDA solver");
    }
    if (batch_size == 0) {
        throw std::runtime_error("RPOW2 CUDA batch size must be greater than zero");
    }
    if (batch_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("RPOW2 CUDA batch size exceeds kernel grid limit");
    }
	    if (std::numeric_limits<std::uint64_t>::max() - start_nonce < batch_size) {
	        throw std::runtime_error("RPOW2 CUDA nonce range exhausted");
	    }

	    check_cuda(cudaSetDevice(device_index_), "cudaSetDevice failed");
	    cudaDeviceProp properties{};
	    check_cuda(cudaGetDeviceProperties(&properties, device_index_),
	               "cudaGetDeviceProperties failed");
	    threads_per_block_ = threads_per_block == 0 ? 256u : threads_per_block;
	    nonces_per_thread_ = nonces_per_thread == 0 ? 4u : nonces_per_thread;
	    if (nonces_per_thread_ != 1u
	        && nonces_per_thread_ != 2u
	        && nonces_per_thread_ != 4u
	        && nonces_per_thread_ != 8u) {
	        throw std::runtime_error("RPOW2 CUDA nonces_per_thread must be one of 1, 2, 4, or 8");
	    }
	    const auto max_threads_per_block = std::min<std::uint32_t>(
	        512u,
	        static_cast<std::uint32_t>(properties.maxThreadsPerBlock));
	    if (threads_per_block_ == 0 || threads_per_block_ > max_threads_per_block) {
	        throw std::runtime_error("RPOW2 CUDA threads_per_block is out of range");
	    }
	    const auto sm_count = static_cast<std::uint32_t>(std::max(properties.multiProcessorCount, 1));
	    max_blocks_ = max_blocks == 0 ? sm_count * 8u : max_blocks;
	    max_blocks_ = std::max<std::uint32_t>(1u, max_blocks_);

	    const auto prepared = build_rpow2_prepared_tail(nonce_prefix, nonce_prefix_len);
	    initial_state_ = prepared.initial_state;
	    template_words_ = prepared.template_words;
	    block_count_ = prepared.block_count;
	    nonce_offset_ = prepared.nonce_offset;

	    check_cuda(cudaMalloc(&device_result_, sizeof(Rpow2CudaDeviceResult)),
	               "cudaMalloc result failed");
	}

Rpow2CudaSession::~Rpow2CudaSession() {
	    if (device_result_ != nullptr) {
	        cudaFree(device_result_);
	        device_result_ = nullptr;
	    }
	}

	namespace {

		template <unsigned int NoncesPerThread>
		void launch_rpow2_kernel(unsigned int blocks,
		                         unsigned int threads_per_block,
		                         const Rpow2CudaKernelParams& params,
		                         Rpow2CudaDeviceResult* result) {
		    rpow2_mine_kernel<NoncesPerThread><<<blocks, threads_per_block>>>(params, result);
		}

		template <unsigned int NoncesPerThread>
		void launch_rpow2_prefix16_kernel(unsigned int blocks,
		                                  unsigned int threads_per_block,
		                                  const Rpow2CudaKernelParams& params,
		                                  Rpow2CudaDeviceResult* result) {
		    rpow2_mine_prefix16_kernel<NoncesPerThread><<<blocks, threads_per_block>>>(params, result);
		}

		bool is_rpow2_prefix16_fast_path(const Rpow2CudaKernelParams& params) {
		    return params.block_count == 1u
		        && params.nonce_offset == 16u
		        && params.initial_state[0] == 0x6a09e667u
		        && params.initial_state[1] == 0xbb67ae85u
		        && params.initial_state[2] == 0x3c6ef372u
		        && params.initial_state[3] == 0xa54ff53au
		        && params.initial_state[4] == 0x510e527fu
		        && params.initial_state[5] == 0x9b05688cu
		        && params.initial_state[6] == 0x1f83d9abu
		        && params.initial_state[7] == 0x5be0cd19u;
		}

		void launch_rpow2_kernel(unsigned int blocks,
		                         unsigned int threads_per_block,
		                         std::uint32_t nonces_per_thread,
		                         const Rpow2CudaKernelParams& params,
		                         Rpow2CudaDeviceResult* result) {
		    const bool prefix16_fast_path = is_rpow2_prefix16_fast_path(params);
		    switch (nonces_per_thread) {
		    case 1:
		        if (prefix16_fast_path) {
		            launch_rpow2_prefix16_kernel<1>(blocks, threads_per_block, params, result);
		        } else {
		            launch_rpow2_kernel<1>(blocks, threads_per_block, params, result);
		        }
		        break;
		    case 2:
		        if (prefix16_fast_path) {
		            launch_rpow2_prefix16_kernel<2>(blocks, threads_per_block, params, result);
		        } else {
		            launch_rpow2_kernel<2>(blocks, threads_per_block, params, result);
		        }
		        break;
		    case 4:
		        if (prefix16_fast_path) {
		            launch_rpow2_prefix16_kernel<4>(blocks, threads_per_block, params, result);
		        } else {
		            launch_rpow2_kernel<4>(blocks, threads_per_block, params, result);
		        }
		        break;
		    case 8:
		        if (prefix16_fast_path) {
		            launch_rpow2_prefix16_kernel<8>(blocks, threads_per_block, params, result);
		        } else {
		            launch_rpow2_kernel<8>(blocks, threads_per_block, params, result);
		        }
		        break;
		    default:
		        throw std::runtime_error("unsupported RPOW2 CUDA nonces_per_thread");
	    }
	}

	} // namespace

	Rpow2CudaProfiledBatchResult Rpow2CudaSession::mine_next_batch_profiled() {
	    if (std::numeric_limits<std::uint64_t>::max() - next_nonce_ < batch_size_) {
	        throw std::runtime_error("RPOW2 CUDA nonce range exhausted");
	    }
	    check_cuda(cudaSetDevice(device_index_), "cudaSetDevice failed");
	    check_cuda(cudaMemset(device_result_, 0, sizeof(Rpow2CudaDeviceResult)),
	               "cudaMemset result failed");

	    Rpow2CudaKernelParams params{};
	    params.difficulty_bits = difficulty_bits_;
	    params.block_count = block_count_;
	    params.nonce_offset = nonce_offset_;
	    params.padding = 0;
	    for (std::size_t index = 0; index < initial_state_.size(); ++index) {
	        params.initial_state[index] = initial_state_[index];
	    }
	    for (std::size_t index = 0; index < template_words_.size(); ++index) {
	        params.template_words[index] = template_words_[index];
	    }
	    params.start_nonce = static_cast<unsigned long long>(next_nonce_);
	    params.batch_size = static_cast<unsigned long long>(batch_size_);

	    const auto work_per_block = static_cast<std::uint64_t>(threads_per_block_)
	        * static_cast<std::uint64_t>(nonces_per_thread_);
	    const auto needed_blocks = static_cast<std::uint64_t>(
	        (batch_size_ + work_per_block - 1) / work_per_block);
	    const auto launch_blocks = static_cast<unsigned int>(
	        std::max<std::uint64_t>(1, std::min<std::uint64_t>(needed_blocks, max_blocks_)));
	    if (launch_blocks == 0) {
	        throw std::runtime_error("RPOW2 CUDA launch grid is empty");
	    }

	    cudaEvent_t kernel_start{};
	    cudaEvent_t kernel_end{};
	    check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate start failed");
	    try {
	        check_cuda(cudaEventCreate(&kernel_end), "cudaEventCreate end failed");
	    } catch (...) {
	        cudaEventDestroy(kernel_start);
	        throw;
	    }

	    const auto wall_started = std::chrono::steady_clock::now();
	    check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord start failed");
	    launch_rpow2_kernel(launch_blocks,
	                        threads_per_block_,
	                        nonces_per_thread_,
	                        params,
	                        static_cast<Rpow2CudaDeviceResult*>(device_result_));
	    check_cuda(cudaGetLastError(), "RPOW2 CUDA kernel launch failed");
	    check_cuda(cudaEventRecord(kernel_end), "cudaEventRecord end failed");
	    check_cuda(cudaEventSynchronize(kernel_end), "RPOW2 CUDA kernel execution failed");
	    float kernel_milliseconds = 0.0f;
	    check_cuda(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end),
	               "cudaEventElapsedTime failed");
	    cudaEventDestroy(kernel_start);
	    cudaEventDestroy(kernel_end);

	    Rpow2CudaDeviceResult host_result{};
	    check_cuda(cudaMemcpy(&host_result,
	                          device_result_,
	                          sizeof(host_result),
	                          cudaMemcpyDeviceToHost),
	               "cudaMemcpy result failed");
	    const auto wall_elapsed = std::chrono::duration<double, std::milli>(
	        std::chrono::steady_clock::now() - wall_started).count();

	    const auto max_attempts = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
	    attempts_ = attempts_ > max_attempts - batch_size_ ? max_attempts : attempts_ + batch_size_;
	    next_nonce_ += batch_size_;

	    Rpow2CudaProfiledBatchResult profiled;
	    profiled.result.found = host_result.found != 0u;
	    profiled.result.nonce = host_result.nonce;
	    profiled.result.attempts = static_cast<std::int64_t>(attempts_);
	    if (profiled.result.found) {
	        digest_words_to_bytes(host_result.digest_words, profiled.result.digest);
	    }
	    profiled.wall_milliseconds = wall_elapsed;
	    profiled.kernel_milliseconds = static_cast<double>(kernel_milliseconds);
	    return profiled;
	}

	Rpow2CudaBatchResult Rpow2CudaSession::mine_next_batch() {
	    if (std::numeric_limits<std::uint64_t>::max() - next_nonce_ < batch_size_) {
	        throw std::runtime_error("RPOW2 CUDA nonce range exhausted");
	    }
	    check_cuda(cudaSetDevice(device_index_), "cudaSetDevice failed");
	    check_cuda(cudaMemset(device_result_, 0, sizeof(Rpow2CudaDeviceResult)),
	               "cudaMemset result failed");

	    Rpow2CudaKernelParams params{};
	    params.difficulty_bits = difficulty_bits_;
	    params.block_count = block_count_;
	    params.nonce_offset = nonce_offset_;
	    params.padding = 0;
	    for (std::size_t index = 0; index < initial_state_.size(); ++index) {
	        params.initial_state[index] = initial_state_[index];
	    }
	    for (std::size_t index = 0; index < template_words_.size(); ++index) {
	        params.template_words[index] = template_words_[index];
	    }
	    params.start_nonce = static_cast<unsigned long long>(next_nonce_);
	    params.batch_size = static_cast<unsigned long long>(batch_size_);

	    const auto work_per_block = static_cast<std::uint64_t>(threads_per_block_)
	        * static_cast<std::uint64_t>(nonces_per_thread_);
	    const auto needed_blocks = static_cast<std::uint64_t>(
	        (batch_size_ + work_per_block - 1) / work_per_block);
	    const auto launch_blocks = static_cast<unsigned int>(
	        std::max<std::uint64_t>(1, std::min<std::uint64_t>(needed_blocks, max_blocks_)));
	    if (launch_blocks == 0) {
	        throw std::runtime_error("RPOW2 CUDA launch grid is empty");
	    }

	    launch_rpow2_kernel(launch_blocks,
	                        threads_per_block_,
	                        nonces_per_thread_,
	                        params,
	                        static_cast<Rpow2CudaDeviceResult*>(device_result_));
	    check_cuda(cudaGetLastError(), "RPOW2 CUDA kernel launch failed");
	    check_cuda(cudaDeviceSynchronize(), "RPOW2 CUDA kernel execution failed");

	    Rpow2CudaDeviceResult host_result{};
	    check_cuda(cudaMemcpy(&host_result,
	                          device_result_,
	                          sizeof(host_result),
	                          cudaMemcpyDeviceToHost),
	               "cudaMemcpy result failed");

	    const auto max_attempts = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
	    attempts_ = attempts_ > max_attempts - batch_size_ ? max_attempts : attempts_ + batch_size_;
	    next_nonce_ += batch_size_;

	    Rpow2CudaBatchResult result;
	    result.found = host_result.found != 0u;
	    result.nonce = host_result.nonce;
	    result.attempts = static_cast<std::int64_t>(attempts_);
	    if (result.found) {
	        digest_words_to_bytes(host_result.digest_words, result.digest);
	    }
	    return result;
	}

	Rpow2CudaBatchResult mine_rpow2_cuda_batch(std::size_t device_index,
	                                           const std::uint8_t* nonce_prefix,
	                                           std::size_t nonce_prefix_len,
	                                           std::uint32_t difficulty_bits,
	                                           std::uint64_t batch_size,
	                                           std::uint64_t start_nonce) {
	    return mine_rpow2_cuda_batch(device_index,
	                                 nonce_prefix,
	                                 nonce_prefix_len,
	                                 difficulty_bits,
	                                 batch_size,
	                                 start_nonce,
	                                 256,
	                                 4,
	                                 0);
	}

	Rpow2CudaBatchResult mine_rpow2_cuda_batch(std::size_t device_index,
	                                           const std::uint8_t* nonce_prefix,
	                                           std::size_t nonce_prefix_len,
	                                           std::uint32_t difficulty_bits,
	                                           std::uint64_t batch_size,
	                                           std::uint64_t start_nonce,
	                                           std::uint32_t threads_per_block,
	                                           std::uint32_t nonces_per_thread,
	                                           std::uint32_t max_blocks) {
	    Rpow2CudaSession session(device_index,
	                             nonce_prefix,
	                             nonce_prefix_len,
	                             difficulty_bits,
	                             batch_size,
	                             start_nonce,
	                             threads_per_block,
	                             nonces_per_thread,
	                             max_blocks);
	    return session.mine_next_batch();
	}

	double benchmark_empty_launch_microseconds(std::size_t device_index) {
	    check_cuda(cudaSetDevice(static_cast<int>(device_index)), "cudaSetDevice failed");
	    for (int index = 0; index < 16; ++index) {
	        rpow2_empty_kernel<<<1, 1>>>();
	    }
	    check_cuda(cudaGetLastError(), "RPOW2 empty kernel warmup failed");
	    check_cuda(cudaDeviceSynchronize(), "RPOW2 empty kernel warmup sync failed");
	    constexpr int iterations = 1000;
	    const auto started = std::chrono::steady_clock::now();
	    for (int index = 0; index < iterations; ++index) {
	        rpow2_empty_kernel<<<1, 1>>>();
	    }
	    check_cuda(cudaGetLastError(), "RPOW2 empty kernel launch failed");
	    check_cuda(cudaDeviceSynchronize(), "RPOW2 empty kernel sync failed");
	    const auto elapsed = std::chrono::duration<double, std::micro>(
	        std::chrono::steady_clock::now() - started).count();
	    return elapsed / static_cast<double>(iterations);
	}

	Rpow2CudaBenchmarkResult benchmark_rpow2_cuda(std::size_t device_index,
	                                             const std::uint8_t* nonce_prefix,
	                                             std::size_t nonce_prefix_len,
	                                             std::uint32_t difficulty_bits,
	                                             std::uint64_t batch_size,
	                                             std::uint32_t duration_ms,
	                                             std::uint32_t threads_per_block,
	                                             std::uint32_t nonces_per_thread,
	                                             std::uint32_t max_blocks) {
	    Rpow2CudaSession session(device_index,
	                             nonce_prefix,
	                             nonce_prefix_len,
	                             difficulty_bits,
	                             batch_size,
	                             0,
	                             threads_per_block,
	                             nonces_per_thread,
	                             max_blocks);
	    Rpow2CudaBenchmarkResult benchmark;
	    benchmark.empty_launch_microseconds = benchmark_empty_launch_microseconds(device_index);
	    const auto started = std::chrono::steady_clock::now();
	    const auto duration = std::chrono::milliseconds(duration_ms);
	    std::int64_t last_attempts = 0;
	    double batch_wall_ms = 0.0;
	    while (std::chrono::steady_clock::now() - started < duration) {
	        const auto batch = session.mine_next_batch_profiled();
	        const auto current_attempts = batch.result.attempts;
	        const auto delta = current_attempts > last_attempts
	            ? static_cast<std::uint64_t>(current_attempts - last_attempts)
	            : 0ull;
	        benchmark.attempts += delta;
	        last_attempts = current_attempts;
	        benchmark.batches += 1;
	        benchmark.kernel_milliseconds += batch.kernel_milliseconds;
	        batch_wall_ms += batch.wall_milliseconds;
	    }
	    benchmark.elapsed_milliseconds = std::chrono::duration<double, std::milli>(
	        std::chrono::steady_clock::now() - started).count();
	    if (benchmark.kernel_milliseconds > 0.0) {
	        benchmark.kernel_hashrate =
	            static_cast<double>(benchmark.attempts) * 1000.0 / benchmark.kernel_milliseconds;
	    }
	    if (benchmark.elapsed_milliseconds > 0.0) {
	        benchmark.effective_hashrate =
	            static_cast<double>(benchmark.attempts) * 1000.0 / benchmark.elapsed_milliseconds;
	    }
	    const auto measured_batch_overhead = batch_wall_ms - benchmark.kernel_milliseconds;
	    (void)measured_batch_overhead;
	    return benchmark;
	}

	} // namespace app
