#include "app/solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
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
        for (auto& password : passwords) {
            password.reserve(job.pass_prefix().size() + 20);
        }
        prepare_batch(job, config, start_nonce);
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
    if (!state.has_prepared_batch || state.prepared_start_nonce != session.next_nonce) {
        state.prepare_batch(job, session.config, session.next_nonce);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return {};
    }

    const auto current_start_nonce = session.next_nonce;
    const auto next_start_nonce = current_start_nonce + session.config.batch_size;
    state.prepared->unit->beginProcessingWithDifficultyCheck(job.difficulty_bits());
    if (!stop.load(std::memory_order_relaxed)) {
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
        candidates.push_back(default_config);

        SolverConfig segmented_config;
        segmented_config.batch_size = batch_size;
        segmented_config.by_segment = true;
        segmented_config.precompute_refs = false;
        candidates.push_back(segmented_config);

        SolverConfig precomputed_config;
        precomputed_config.batch_size = batch_size;
        precomputed_config.by_segment = true;
        precomputed_config.precompute_refs = true;
        candidates.push_back(precomputed_config);
    }
    if (candidates.empty()) {
        SolverConfig fallback_config;
        fallback_config.batch_size = 1;
        fallback_config.by_segment = false;
        fallback_config.precompute_refs = false;
        candidates.push_back(fallback_config);
    }
    return candidates;
}

} // namespace app
