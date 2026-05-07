#ifndef ARGON2_CUDA_KERNELRUNNER_H
#define ARGON2_CUDA_KERNELRUNNER_H

#if HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

/* workaround weird CMake/CUDA bug: */
#ifdef argon2
#undef argon2
#endif

namespace argon2 {
namespace cuda {

struct DifficultyCheckResult;

class KernelRunner
{
private:
    std::uint32_t type, version;
    std::uint32_t passes, lanes, segmentBlocks;
    std::size_t batchSize;
    bool bySegment;
    bool precompute;

    cudaEvent_t start, end, kernelStart, kernelEnd;
    cudaStream_t stream;
    void *memory;
    void *refs;
    DifficultyCheckResult *difficultyResultDevice;
    std::unique_ptr<DifficultyCheckResult> difficultyResultHost;
    void *generatedPasswordPrefix;
    void *generatedSalt;
    void *generatedSecret;
    void *generatedAssocData;
    std::size_t generatedPasswordPrefixSize;
    std::uint32_t generatedOutputLength;
    std::uint32_t generatedMemoryCost;
    std::uint32_t generatedSaltLength;
    std::uint32_t generatedSecretLength;
    std::uint32_t generatedAssocDataLength;
    bool generatedPasswordContextReady;

    std::unique_ptr<std::uint8_t[]> blocksIn;
    std::unique_ptr<std::uint8_t[]> blocksOut;

    void copyInputBlocks();
    void copyOutputBlocks();

    void precomputeRefs();

    void runKernelSegment(std::uint32_t lanesPerBlock,
                          std::size_t jobsPerBlock,
                          std::uint32_t pass, std::uint32_t slice);
    void runKernelOneshot(std::uint32_t lanesPerBlock,
                          std::size_t jobsPerBlock);
    void runDifficultyCheckKernel(int difficultyBits);
    void runGeneratedInputKernel(std::uint64_t startNonce);

public:
    std::uint32_t getMinLanesPerBlock() const { return bySegment ? 1 : lanes; }
    std::uint32_t getMaxLanesPerBlock() const { return lanes; }

    std::size_t getMinJobsPerBlock() const { return 1; }
    std::size_t getMaxJobsPerBlock() const { return batchSize; }

    std::size_t getBatchSize() const { return batchSize; }

    KernelRunner(std::uint32_t type, std::uint32_t version,
                 std::uint32_t passes, std::uint32_t lanes,
                 std::uint32_t segmentBlocks, std::size_t batchSize,
                 bool bySegment, bool precompute);
    ~KernelRunner();

    void *getInputMemory(std::size_t jobId) const;
    const void *getOutputMemory(std::size_t jobId) const;

    void setGeneratedPasswordContext(
            const void *passwordPrefix, std::size_t passwordPrefixSize,
            std::uint32_t outputLength,
            const void *salt, std::uint32_t saltLength,
            const void *secret, std::uint32_t secretLength,
            const void *assocData, std::uint32_t assocDataLength,
            std::uint32_t memoryCost);
    void run(std::uint32_t lanesPerBlock, std::size_t jobsPerBlock);
    void runWithDifficultyCheck(std::uint32_t lanesPerBlock,
                                std::size_t jobsPerBlock,
                                int difficultyBits);
    void runGeneratedWithDifficultyCheck(std::uint32_t lanesPerBlock,
                                         std::size_t jobsPerBlock,
                                         std::uint64_t startNonce,
                                         int difficultyBits);
    float finish();
    bool getDifficultyResult(std::size_t *jobId, void *hash) const;
};

} // cuda
} // argon2

#endif /* HAVE_CUDA */

#endif // ARGON2_CUDA_KERNELRUNNER_H
