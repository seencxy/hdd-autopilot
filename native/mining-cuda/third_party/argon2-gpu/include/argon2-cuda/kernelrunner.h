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

    void run(std::uint32_t lanesPerBlock, std::size_t jobsPerBlock);
    void runWithDifficultyCheck(std::uint32_t lanesPerBlock,
                                std::size_t jobsPerBlock,
                                int difficultyBits);
    float finish();
    bool getDifficultyResult(std::size_t *jobId, void *hash) const;
};

} // cuda
} // argon2

#endif /* HAVE_CUDA */

#endif // ARGON2_CUDA_KERNELRUNNER_H
