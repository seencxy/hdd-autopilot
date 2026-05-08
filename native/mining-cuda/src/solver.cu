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
    std::uint32_t prefix_len;
    std::uint32_t difficulty_bits;
    std::uint32_t block_count;
    std::uint32_t padding;
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

__device__ std::uint32_t rpow2_rotr32(std::uint32_t value, unsigned int bits) {
    return (value >> bits) | (value << (32u - bits));
}

__device__ void rpow2_write_nonce_words(std::uint32_t* w,
                                        std::uint32_t block,
                                        std::uint32_t prefix_len,
                                        unsigned long long nonce) {
    for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t absolute_byte = prefix_len + i;
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
    std::uint32_t words[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    std::uint32_t bits = 0;
    for (int i = 7; i >= 0; --i) {
        const std::uint32_t word = words[i];
        if (word == 0) {
            bits += 32;
            continue;
        }
        return bits + static_cast<std::uint32_t>(__ffs(word) - 1);
    }
    return bits;
}

__global__ void rpow2_mine_kernel(const std::uint32_t* template_words,
                                  Rpow2CudaKernelParams params,
                                  Rpow2CudaDeviceResult* result) {
    const unsigned long long gid = static_cast<unsigned long long>(blockIdx.x)
        * static_cast<unsigned long long>(blockDim.x)
        + static_cast<unsigned long long>(threadIdx.x);
    if (gid >= params.batch_size || result->found != 0u) {
        return;
    }

    const unsigned long long nonce = params.start_nonce + gid;
    std::uint32_t h0 = 0x6a09e667;
    std::uint32_t h1 = 0xbb67ae85;
    std::uint32_t h2 = 0x3c6ef372;
    std::uint32_t h3 = 0xa54ff53a;
    std::uint32_t h4 = 0x510e527f;
    std::uint32_t h5 = 0x9b05688c;
    std::uint32_t h6 = 0x1f83d9ab;
    std::uint32_t h7 = 0x5be0cd19;

    for (std::uint32_t block = 0; block < params.block_count; ++block) {
        std::uint32_t w[64];
        for (std::uint32_t i = 0; i < 16; ++i) {
            w[i] = template_words[block * 16u + i];
        }
        rpow2_write_nonce_words(w, block, params.prefix_len, nonce);
        for (std::uint32_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rpow2_rotr32(w[i - 15], 7u)
                ^ rpow2_rotr32(w[i - 15], 18u)
                ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rpow2_rotr32(w[i - 2], 17u)
                ^ rpow2_rotr32(w[i - 2], 19u)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h0;
        std::uint32_t b = h1;
        std::uint32_t c = h2;
        std::uint32_t d = h3;
        std::uint32_t e = h4;
        std::uint32_t f = h5;
        std::uint32_t g = h6;
        std::uint32_t h = h7;
        for (std::uint32_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rpow2_rotr32(e, 6u)
                ^ rpow2_rotr32(e, 11u)
                ^ rpow2_rotr32(e, 25u);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + kRpow2Sha256K[i] + w[i];
            const std::uint32_t s0 = rpow2_rotr32(a, 2u)
                ^ rpow2_rotr32(a, 13u)
                ^ rpow2_rotr32(a, 22u);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;
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
    }
}

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint32_t> build_rpow2_template_words(const std::uint8_t* prefix,
                                                      std::size_t prefix_len) {
    const auto message_len = prefix_len + 8;
    const auto padded_len = ((message_len + 9 + 63) / 64) * 64;
    if (padded_len > 128) {
        throw std::runtime_error("RPOW2 padded message is too long for CUDA solver");
    }

    std::array<std::uint8_t, 128> padded{};
    if (prefix_len > 0) {
        std::memcpy(padded.data(), prefix, prefix_len);
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

void digest_words_to_bytes(const std::uint32_t words[8], std::array<std::uint8_t, 32>& digest) {
    for (std::size_t index = 0; index < 8; ++index) {
        digest[index * 4] = static_cast<std::uint8_t>((words[index] >> 24) & 0xff);
        digest[index * 4 + 1] = static_cast<std::uint8_t>((words[index] >> 16) & 0xff);
        digest[index * 4 + 2] = static_cast<std::uint8_t>((words[index] >> 8) & 0xff);
        digest[index * 4 + 3] = static_cast<std::uint8_t>(words[index] & 0xff);
    }
}

} // namespace

Rpow2CudaBatchResult mine_rpow2_cuda_batch(std::size_t device_index,
                                           const std::uint8_t* nonce_prefix,
                                           std::size_t nonce_prefix_len,
                                           std::uint32_t difficulty_bits,
                                           std::uint64_t batch_size,
                                           std::uint64_t start_nonce) {
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

    check_cuda(cudaSetDevice(static_cast<int>(device_index)), "cudaSetDevice failed");
    const auto template_words = build_rpow2_template_words(nonce_prefix, nonce_prefix_len);

    std::uint32_t* device_template = nullptr;
    Rpow2CudaDeviceResult* device_result = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_template),
                          template_words.size() * sizeof(std::uint32_t)),
               "cudaMalloc template failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result),
                              sizeof(Rpow2CudaDeviceResult)),
                   "cudaMalloc result failed");
        check_cuda(cudaMemcpy(device_template,
                              template_words.data(),
                              template_words.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy template failed");
        check_cuda(cudaMemset(device_result, 0, sizeof(Rpow2CudaDeviceResult)),
                   "cudaMemset result failed");

        Rpow2CudaKernelParams params{};
        params.prefix_len = static_cast<std::uint32_t>(nonce_prefix_len);
        params.difficulty_bits = difficulty_bits;
        params.block_count = static_cast<std::uint32_t>(template_words.size() / 16);
        params.padding = 0;
        params.start_nonce = static_cast<unsigned long long>(start_nonce);
        params.batch_size = static_cast<unsigned long long>(batch_size);

        constexpr unsigned int threads_per_block = 256;
        const auto blocks = static_cast<unsigned int>(
            (batch_size + threads_per_block - 1) / threads_per_block);
        rpow2_mine_kernel<<<blocks, threads_per_block>>>(device_template, params, device_result);
        check_cuda(cudaGetLastError(), "RPOW2 CUDA kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "RPOW2 CUDA kernel execution failed");

        Rpow2CudaDeviceResult host_result{};
        check_cuda(cudaMemcpy(&host_result,
                              device_result,
                              sizeof(host_result),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy result failed");

        Rpow2CudaBatchResult result;
        result.found = host_result.found != 0u;
        result.nonce = host_result.nonce;
        result.attempts = static_cast<std::int64_t>(batch_size);
        if (result.found) {
            digest_words_to_bytes(host_result.digest_words, result.digest);
        }
        cudaFree(device_result);
        cudaFree(device_template);
        return result;
    } catch (...) {
        if (device_result != nullptr) {
            cudaFree(device_result);
        }
        if (device_template != nullptr) {
            cudaFree(device_template);
        }
        throw;
    }
}

} // namespace app
