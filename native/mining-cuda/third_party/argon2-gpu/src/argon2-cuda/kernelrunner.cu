/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "kernelrunner.h"
#include "cudaexception.h"

#include <stdexcept>
#ifndef NDEBUG
#include <iostream>
#endif

#define ARGON2_D  0
#define ARGON2_I  1
#define ARGON2_ID 2

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

namespace argon2 {
namespace cuda {

using namespace std;

__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

struct u64_shuffle_buf {
    uint32_t lo[THREADS_PER_LANE];
    uint32_t hi[THREADS_PER_LANE];
};

__device__ uint64_t u64_shuffle(uint64_t v, uint32_t thread_src,
                                uint32_t thread, struct u64_shuffle_buf *buf)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);

    buf->lo[thread] = lo;
    buf->hi[thread] = hi;

    __syncthreads();

    lo = buf->lo[thread_src];
    hi = buf->hi[thread_src];

    return u64_build(hi, lo);
}

struct block_g {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_th {
    uint64_t a, b, c, d;
};

__device__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(uint32_t)(test == ref);
    return u64_build(x, x);
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

__device__ void g(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}

template<class shuffle>
__device__ void apply_shuffle(struct block_th *block, uint32_t thread,
                              struct u64_shuffle_buf *buf)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t src_thr = shuffle::apply(thread, i);

        uint64_t v = block_th_get(block, i);
        v = u64_shuffle(v, src_thr, thread, buf);
        block_th_set(block, i, v);
    }
}

__device__ void transpose(struct block_th *block, uint32_t thread,
                          struct u64_shuffle_buf *buf)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr, thread, buf);
        block_th_set(block, idx, v);
    }
}

struct identity_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return thread;
    }
};

struct shift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct unshift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct shift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

struct unshift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

__device__ void shuffle_block(struct block_th *block, uint32_t thread,
                              struct u64_shuffle_buf *buf)
{
    transpose(block, thread, buf);

    g(block);

    apply_shuffle<shift1_shuffle>(block, thread, buf);

    g(block);

    apply_shuffle<unshift1_shuffle>(block, thread, buf);
    transpose(block, thread, buf);

    g(block);

    apply_shuffle<shift2_shuffle>(block, thread, buf);

    g(block);

    apply_shuffle<unshift2_shuffle>(block, thread, buf);
}

__device__ void next_addresses(struct block_th *addr, struct block_th *tmp,
                               uint32_t thread_input, uint32_t thread,
                               struct u64_shuffle_buf *buf)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread, buf);

    addr->a ^= u64_build(0, thread_input);
    move_block(tmp, addr);

    shuffle_block(addr, thread, buf);

    xor_block(addr, tmp);
}

__device__ void compute_ref_pos(
        uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t lane, uint32_t slice, uint32_t offset,
        uint32_t *ref_lane, uint32_t *ref_index)
{
    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    *ref_lane = *ref_lane % lanes;

    uint32_t base;
    if (pass != 0) {
        base = lane_blocks - segment_blocks;
    } else {
        if (slice == 0) {
            *ref_lane = lane;
        }
        base = slice * segment_blocks;
    }

    uint32_t ref_area_size = base + offset - 1;
    if (*ref_lane != lane) {
        ref_area_size = min(ref_area_size, base);
    }

    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

    if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
        *ref_index += (slice + 1) * segment_blocks;
        if (*ref_index >= lane_blocks) {
            *ref_index -= lane_blocks;
        }
    }
}

struct ref {
    uint32_t ref_lane;
    uint32_t ref_index;
};

struct DifficultyCheckResult {
    uint32_t found;
    uint32_t job_id;
    uint8_t digest[32];
};

__device__ __constant__ uint64_t blake2b_iv_device[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

__device__ __constant__ uint8_t blake2b_sigma_device[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

struct Blake2bDeviceState {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t buf[128];
    size_t buf_len;
};

__device__ uint64_t blake2b_load64_device(const uint8_t *src)
{
    uint64_t res = src[0];
    res |= static_cast<uint64_t>(src[1]) << 8;
    res |= static_cast<uint64_t>(src[2]) << 16;
    res |= static_cast<uint64_t>(src[3]) << 24;
    res |= static_cast<uint64_t>(src[4]) << 32;
    res |= static_cast<uint64_t>(src[5]) << 40;
    res |= static_cast<uint64_t>(src[6]) << 48;
    res |= static_cast<uint64_t>(src[7]) << 56;
    return res;
}

__device__ void blake2b_store32_device(uint8_t *dst, uint32_t v)
{
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

__device__ void blake2b_store64_device(uint8_t *dst, uint64_t v)
{
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
    dst[4] = static_cast<uint8_t>(v >> 32);
    dst[5] = static_cast<uint8_t>(v >> 40);
    dst[6] = static_cast<uint8_t>(v >> 48);
    dst[7] = static_cast<uint8_t>(v >> 56);
}

__device__ __forceinline__ void blake2b_g_device(
        const uint64_t *m, uint64_t *v, uint32_t r, uint32_t i,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    a &= 15;
    b &= 15;
    c &= 15;
    d &= 15;
    v[a] = v[a] + v[b] + m[blake2b_sigma_device[r][2 * i + 0]];
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + m[blake2b_sigma_device[r][2 * i + 1]];
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 63);
}

__device__ __forceinline__ void blake2b_round_device(
        const uint64_t *m, uint64_t *v, uint32_t r)
{
    blake2b_g_device(m, v, r, 0, 0, 4, 8, 12);
    blake2b_g_device(m, v, r, 1, 1, 5, 9, 13);
    blake2b_g_device(m, v, r, 2, 2, 6, 10, 14);
    blake2b_g_device(m, v, r, 3, 3, 7, 11, 15);
    blake2b_g_device(m, v, r, 4, 0, 5, 10, 15);
    blake2b_g_device(m, v, r, 5, 1, 6, 11, 12);
    blake2b_g_device(m, v, r, 6, 2, 7, 8, 13);
    blake2b_g_device(m, v, r, 7, 3, 4, 9, 14);
}

__device__ void blake2b_compress_device(
        Blake2bDeviceState *state, const uint8_t *block, uint64_t f0)
{
    uint64_t m[16];
    uint64_t v[16];

    for (uint32_t i = 0; i < 16; ++i) {
        m[i] = blake2b_load64_device(block + i * sizeof(uint64_t));
    }

    v[0] = state->h[0];
    v[1] = state->h[1];
    v[2] = state->h[2];
    v[3] = state->h[3];
    v[4] = state->h[4];
    v[5] = state->h[5];
    v[6] = state->h[6];
    v[7] = state->h[7];
    v[8] = blake2b_iv_device[0];
    v[9] = blake2b_iv_device[1];
    v[10] = blake2b_iv_device[2];
    v[11] = blake2b_iv_device[3];
    v[12] = blake2b_iv_device[4] ^ state->t[0];
    v[13] = blake2b_iv_device[5] ^ state->t[1];
    v[14] = blake2b_iv_device[6] ^ f0;
    v[15] = blake2b_iv_device[7];

    for (uint32_t r = 0; r < 12; ++r) {
        blake2b_round_device(m, v, r);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        state->h[i] ^= v[i] ^ v[i + 8];
    }
}

__device__ void blake2b_increment_counter_device(
        Blake2bDeviceState *state, uint64_t inc)
{
    state->t[0] += inc;
    state->t[1] += state->t[0] < inc;
}

__device__ void blake2b_init_device(Blake2bDeviceState *state, uint32_t out_len)
{
    state->t[0] = 0;
    state->t[1] = 0;
    state->buf_len = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        state->h[i] = blake2b_iv_device[i];
    }
    state->h[0] ^= static_cast<uint64_t>(out_len)
            | (UINT64_C(1) << 16) | (UINT64_C(1) << 24);
}

__device__ void blake2b_update_device(
        Blake2bDeviceState *state, const uint8_t *input, size_t input_len)
{
    if (state->buf_len + input_len > 128) {
        const size_t left = 128 - state->buf_len;
        for (size_t i = 0; i < left; ++i) {
            state->buf[state->buf_len + i] = input[i];
        }
        blake2b_increment_counter_device(state, 128);
        blake2b_compress_device(state, state->buf, 0);

        state->buf_len = 0;
        input_len -= left;
        input += left;

        while (input_len > 128) {
            blake2b_increment_counter_device(state, 128);
            blake2b_compress_device(state, input, 0);
            input_len -= 128;
            input += 128;
        }
    }

    for (size_t i = 0; i < input_len; ++i) {
        state->buf[state->buf_len + i] = input[i];
    }
    state->buf_len += input_len;
}

__device__ void blake2b_final_device(
        Blake2bDeviceState *state, uint8_t *output, uint32_t output_len)
{
    uint8_t buffer[64];
    for (uint32_t i = 0; i < sizeof(buffer); ++i) {
        buffer[i] = 0;
    }

    blake2b_increment_counter_device(state, state->buf_len);
    for (size_t i = state->buf_len; i < 128; ++i) {
        state->buf[i] = 0;
    }
    blake2b_compress_device(state, state->buf, UINT64_C(0xFFFFFFFFFFFFFFFF));

    for (uint32_t i = 0; i < 8; ++i) {
        blake2b_store64_device(buffer + i * sizeof(uint64_t), state->h[i]);
    }
    for (uint32_t i = 0; i < output_len; ++i) {
        output[i] = buffer[i];
    }
}

__device__ bool digest_meets_difficulty_device(
        const uint8_t *digest, uint32_t digest_size, int difficulty_bits)
{
    if (difficulty_bits < 0) {
        return false;
    }

    const uint32_t full_bytes = static_cast<uint32_t>(difficulty_bits / 8);
    for (uint32_t i = 0; i < full_bytes; ++i) {
        if (i >= digest_size || digest[i] != 0) {
            return false;
        }
    }

    const uint32_t remaining_bits = static_cast<uint32_t>(difficulty_bits % 8);
    if (remaining_bits == 0) {
        return true;
    }
    if (full_bytes >= digest_size) {
        return false;
    }

    const uint8_t mask = static_cast<uint8_t>(0xFFu << (8 - remaining_bits));
    return (digest[full_bytes] & mask) == 0;
}

__global__ void argon2_finalize_difficulty_kernel(
        const void *memory, size_t job_size, uint32_t lanes,
        size_t batch_size, int difficulty_bits,
        DifficultyCheckResult *result)
{
    const uint32_t job_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (static_cast<size_t>(job_id) >= batch_size) {
        return;
    }

    Blake2bDeviceState blake;
    uint8_t out_len_bytes[4];
    uint8_t word_bytes[8];
    uint8_t digest[32];
    const auto memory_bytes = static_cast<const uint8_t *>(memory);
    const size_t copy_size = static_cast<size_t>(lanes) * ARGON2_BLOCK_SIZE;
    const uint8_t *final_blocks =
            memory_bytes + static_cast<size_t>(job_id) * job_size
            + job_size - copy_size;

    blake2b_init_device(&blake, sizeof(digest));
    blake2b_store32_device(out_len_bytes, sizeof(digest));
    blake2b_update_device(&blake, out_len_bytes, sizeof(out_len_bytes));
    for (uint32_t word = 0; word < ARGON2_QWORDS_IN_BLOCK; ++word) {
        uint64_t value = 0;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            value ^= blake2b_load64_device(
                    final_blocks
                    + static_cast<size_t>(lane) * ARGON2_BLOCK_SIZE
                    + static_cast<size_t>(word) * sizeof(uint64_t));
        }
        blake2b_store64_device(word_bytes, value);
        blake2b_update_device(&blake, word_bytes, sizeof(word_bytes));
    }
    blake2b_final_device(&blake, digest, sizeof(digest));

    if (!digest_meets_difficulty_device(digest, sizeof(digest), difficulty_bits)) {
        return;
    }

    if (atomicCAS(&result->found, 0u, 1u) == 0u) {
        result->job_id = job_id;
        for (uint32_t i = 0; i < sizeof(digest); ++i) {
            result->digest[i] = digest[i];
        }
    }
}

/*
 * Refs hierarchy:
 * lanes -> passes -> slices -> blocks
 */
template<uint32_t type>
__global__ void argon2_precompute_kernel(
        struct ref *refs, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks)
{
    uint32_t block_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t warp = threadIdx.y;
    uint32_t thread = threadIdx.x;

    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf = &shuffle_bufs[warp];

    uint32_t segment_addr_blocks = (segment_blocks + ARGON2_QWORDS_IN_BLOCK - 1)
            / ARGON2_QWORDS_IN_BLOCK;
    uint32_t block = block_id % segment_addr_blocks;
    uint32_t segment = block_id / segment_addr_blocks;

    uint32_t slice, pass, pass_id, lane;
    if (type == ARGON2_ID) {
        slice = segment % (ARGON2_SYNC_POINTS / 2);
        lane = segment / (ARGON2_SYNC_POINTS / 2);
        pass_id = pass = 0;
    } else {
        slice = segment % ARGON2_SYNC_POINTS;
        pass_id = segment / ARGON2_SYNC_POINTS;

        pass = pass_id % passes;
        lane = pass_id / passes;
    }

    struct block_th addr, tmp;

    uint32_t thread_input;
    switch (thread) {
    case 0:
        thread_input = pass;
        break;
    case 1:
        thread_input = lane;
        break;
    case 2:
        thread_input = slice;
        break;
    case 3:
        thread_input = lanes * segment_blocks * ARGON2_SYNC_POINTS;
        break;
    case 4:
        thread_input = passes;
        break;
    case 5:
        thread_input = type;
        break;
    case 6:
        thread_input = block + 1;
        break;
    default:
        thread_input = 0;
        break;
    }

    next_addresses(&addr, &tmp, thread_input, thread, shuffle_buf);

    refs += segment * segment_blocks;

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos = i * THREADS_PER_LANE + thread;
        uint32_t offset = block * ARGON2_QWORDS_IN_BLOCK + pos;
        if (offset < segment_blocks) {
            uint64_t v = block_th_get(&addr, i);
            uint32_t ref_index = u64_lo(v);
            uint32_t ref_lane  = u64_hi(v);

            compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                            &ref_lane, &ref_index);

            refs[offset].ref_index = ref_index;
            refs[offset].ref_lane  = ref_lane;
        }
    }
}

template<uint32_t version>
__device__ void argon2_core(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp,
        struct u64_shuffle_buf *shuffle_buf, uint32_t lanes,
        uint32_t thread, uint32_t pass, uint32_t ref_index, uint32_t ref_lane)
{
    struct block_g *mem_ref = memory + ref_index * lanes + ref_lane;

    if (version != ARGON2_VERSION_10 && pass != 0) {
        load_block(tmp, mem_curr, thread);
        load_block_xor(prev, mem_ref, thread);
        xor_block(tmp, prev);
    } else {
        load_block_xor(prev, mem_ref, thread);
        move_block(tmp, prev);
    }

    shuffle_block(prev, thread, shuffle_buf);

    xor_block(prev, tmp);

    store_block(mem_curr, prev, thread);
}

template<uint32_t type, uint32_t version>
__device__ void argon2_step_precompute(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp,
        struct u64_shuffle_buf *shuffle_buf, const struct ref **refs,
        uint32_t lanes, uint32_t segment_blocks, uint32_t thread,
        uint32_t lane, uint32_t pass, uint32_t slice, uint32_t offset)
{
    uint32_t ref_index, ref_lane;
    if (type == ARGON2_I || (type == ARGON2_ID && pass == 0 &&
            slice < ARGON2_SYNC_POINTS / 2)) {
        ref_index = (*refs)->ref_index;
        ref_lane = (*refs)->ref_lane;
        (*refs)++;
    } else {
        uint64_t v = u64_shuffle(prev->a, 0, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);

        compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                        &ref_lane, &ref_index);
    }

    argon2_core<version>(memory, mem_curr, prev, tmp, shuffle_buf, lanes,
                         thread, pass, ref_index, ref_lane);
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_segment_precompute(
        struct block_g *memory, const struct ref *refs,
        uint32_t passes, uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t slice)
{
    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf =
            &shuffle_bufs[blockDim.y * threadIdx.z + threadIdx.y];

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += (size_t)job_id * lanes * lane_blocks;

    struct block_th prev, tmp;

    struct block_g *mem_segment =
            memory + slice * segment_blocks * lanes + lane;
    struct block_g *mem_prev, *mem_curr;
    uint32_t start_offset = 0;
    if (pass == 0) {
        if (slice == 0) {
            mem_prev = mem_segment + 1 * lanes;
            mem_curr = mem_segment + 2 * lanes;
            start_offset = 2;
        } else {
            mem_prev = mem_segment - lanes;
            mem_curr = mem_segment;
        }
    } else {
        mem_prev = mem_segment + (slice == 0 ? lane_blocks * lanes : 0) - lanes;
        mem_curr = mem_segment;
    }

    load_block(&prev, mem_prev, thread);

    if (type == ARGON2_ID) {
        if (pass == 0 && slice < ARGON2_SYNC_POINTS / 2) {
            refs += lane * (lane_blocks / 2) + slice * segment_blocks;
            refs += start_offset;
        }
    } else {
        refs += (lane * passes + pass) * lane_blocks + slice * segment_blocks;
        refs += start_offset;
    }

    for (uint32_t offset = start_offset; offset < segment_blocks; ++offset) {
        argon2_step_precompute<type, version>(
                    memory, mem_curr, &prev, &tmp, shuffle_buf, &refs, lanes,
                    segment_blocks, thread, lane, pass, slice, offset);

        mem_curr += lanes;
    }
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_oneshot_precompute(
        struct block_g *memory, const struct ref *refs, uint32_t passes,
        uint32_t lanes, uint32_t segment_blocks)
{
    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf =
            &shuffle_bufs[lanes * threadIdx.z + threadIdx.y];

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += (size_t)job_id * lanes * lane_blocks;

    struct block_th prev, tmp;

    struct block_g *mem_lane = memory + lane;
    struct block_g *mem_prev = mem_lane + 1 * lanes;
    struct block_g *mem_curr = mem_lane + 2 * lanes;

    load_block(&prev, mem_prev, thread);

    if (type == ARGON2_ID) {
        refs += lane * (lane_blocks / 2) + 2;
    } else {
        refs += lane * passes * lane_blocks + 2;
    }

    uint32_t skip = 2;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                argon2_step_precompute<type, version>(
                            memory, mem_curr, &prev, &tmp, shuffle_buf, &refs,
                            lanes, segment_blocks, thread,
                            lane, pass, slice, offset);

                mem_curr += lanes;
            }

            __syncthreads();
        }

        mem_curr = mem_lane;
    }
}

template<uint32_t type, uint32_t version>
__device__ void argon2_step(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp, struct block_th *addr,
        struct u64_shuffle_buf *shuffle_buf, uint32_t lanes,
        uint32_t segment_blocks, uint32_t thread, uint32_t *thread_input,
        uint32_t lane, uint32_t pass, uint32_t slice, uint32_t offset)
{
    uint32_t ref_index, ref_lane;

    if (type == ARGON2_I || (type == ARGON2_ID && pass == 0 &&
            slice < ARGON2_SYNC_POINTS / 2)) {
        uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
        if (addr_index == 0) {
            if (thread == 6) {
                ++*thread_input;
            }
            next_addresses(addr, tmp, *thread_input, thread, shuffle_buf);
        }

        uint32_t thr = addr_index % THREADS_PER_LANE;
        uint32_t idx = addr_index / THREADS_PER_LANE;

        uint64_t v = block_th_get(addr, idx);
        v = u64_shuffle(v, thr, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    } else {
        uint64_t v = u64_shuffle(prev->a, 0, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    }

    compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                    &ref_lane, &ref_index);

    argon2_core<version>(memory, mem_curr, prev, tmp, shuffle_buf, lanes,
                         thread, pass, ref_index, ref_lane);
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_segment(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t pass, uint32_t slice)
{
    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf =
            &shuffle_bufs[blockDim.y * threadIdx.z + threadIdx.y];

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += (size_t)job_id * lanes * lane_blocks;

    struct block_th prev, addr, tmp;
    uint32_t thread_input;

    if (type == ARGON2_I || type == ARGON2_ID) {
        switch (thread) {
        case 0:
            thread_input = pass;
            break;
        case 1:
            thread_input = lane;
            break;
        case 2:
            thread_input = slice;
            break;
        case 3:
            thread_input = lanes * lane_blocks;
            break;
        case 4:
            thread_input = passes;
            break;
        case 5:
            thread_input = type;
            break;
        default:
            thread_input = 0;
            break;
        }

        if (pass == 0 && slice == 0 && segment_blocks > 2) {
            if (thread == 6) {
                ++thread_input;
            }
            next_addresses(&addr, &tmp, thread_input, thread, shuffle_buf);
        }
    }

    struct block_g *mem_segment =
            memory + slice * segment_blocks * lanes + lane;
    struct block_g *mem_prev, *mem_curr;
    uint32_t start_offset = 0;
    if (pass == 0) {
        if (slice == 0) {
            mem_prev = mem_segment + 1 * lanes;
            mem_curr = mem_segment + 2 * lanes;
            start_offset = 2;
        } else {
            mem_prev = mem_segment - lanes;
            mem_curr = mem_segment;
        }
    } else {
        mem_prev = mem_segment + (slice == 0 ? lane_blocks * lanes : 0) - lanes;
        mem_curr = mem_segment;
    }

    load_block(&prev, mem_prev, thread);

    for (uint32_t offset = start_offset; offset < segment_blocks; ++offset) {
        argon2_step<type, version>(
                    memory, mem_curr, &prev, &tmp, &addr, shuffle_buf,
                    lanes, segment_blocks, thread, &thread_input,
                    lane, pass, slice, offset);

        mem_curr += lanes;
    }
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_oneshot(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks)
{
    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf =
            &shuffle_bufs[lanes * threadIdx.z + threadIdx.y];

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += (size_t)job_id * lanes * lane_blocks;

    struct block_th prev, addr, tmp;
    uint32_t thread_input;

    if (type == ARGON2_I || type == ARGON2_ID) {
        switch (thread) {
        case 1:
            thread_input = lane;
            break;
        case 3:
            thread_input = lanes * lane_blocks;
            break;
        case 4:
            thread_input = passes;
            break;
        case 5:
            thread_input = type;
            break;
        default:
            thread_input = 0;
            break;
        }

        if (segment_blocks > 2) {
            if (thread == 6) {
                ++thread_input;
            }
            next_addresses(&addr, &tmp, thread_input, thread, shuffle_buf);
        }
    }

    struct block_g *mem_lane = memory + lane;
    struct block_g *mem_prev = mem_lane + 1 * lanes;
    struct block_g *mem_curr = mem_lane + 2 * lanes;

    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                argon2_step<type, version>(
                            memory, mem_curr, &prev, &tmp, &addr, shuffle_buf,
                            lanes, segment_blocks, thread, &thread_input,
                            lane, pass, slice, offset);

                mem_curr += lanes;
            }

            __syncthreads();

            if (type == ARGON2_I || type == ARGON2_ID) {
                if (thread == 2) {
                    ++thread_input;
                }
                if (thread == 6) {
                    thread_input = 0;
                }
            }
        }
        if (type == ARGON2_I) {
            if (thread == 0) {
                ++thread_input;
            }
            if (thread == 2) {
                thread_input = 0;
            }
        }
        mem_curr = mem_lane;
    }
}

KernelRunner::KernelRunner(uint32_t type, uint32_t version, uint32_t passes,
                           uint32_t lanes, uint32_t segmentBlocks,
                           size_t batchSize, bool bySegment, bool precompute)
    : type(type), version(version), passes(passes), lanes(lanes),
      segmentBlocks(segmentBlocks), batchSize(batchSize), bySegment(bySegment),
      precompute(precompute), stream(), memory(), refs(),
      difficultyResultDevice(),
      difficultyResultHost(new DifficultyCheckResult{}),
      start(), end(), kernelStart(), kernelEnd(),
      blocksIn(new uint8_t[batchSize * lanes * 2 * ARGON2_BLOCK_SIZE]),
      blocksOut(new uint8_t[batchSize * lanes * ARGON2_BLOCK_SIZE])
{
    // FIXME: check overflow:
    size_t memorySize = batchSize * lanes * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;

#ifndef NDEBUG
        std::cerr << "[INFO] Allocating " << memorySize << " bytes for memory..."
                  << std::endl;
#endif

    CudaException::check(cudaMalloc(&memory, memorySize));
    CudaException::check(cudaMalloc(&difficultyResultDevice,
                                    sizeof(DifficultyCheckResult)));

    CudaException::check(cudaEventCreate(&start));
    CudaException::check(cudaEventCreate(&end));
    CudaException::check(cudaEventCreate(&kernelStart));
    CudaException::check(cudaEventCreate(&kernelEnd));

    CudaException::check(cudaStreamCreate(&stream));

    if ((type == ARGON2_I || type == ARGON2_ID) && precompute) {
        uint32_t segments =
                type == ARGON2_ID
                ? lanes * (ARGON2_SYNC_POINTS / 2)
                : passes * lanes * ARGON2_SYNC_POINTS;

        size_t refsSize = segments * segmentBlocks * sizeof(struct ref);

#ifndef NDEBUG
        std::cerr << "[INFO] Allocating " << refsSize << " bytes for refs..."
                  << std::endl;
#endif

        CudaException::check(cudaMalloc(&refs, refsSize));

        precomputeRefs();
        CudaException::check(cudaStreamSynchronize(stream));
    }
}

void KernelRunner::precomputeRefs()
{
    struct ref *refs = (struct ref *)this->refs;

    uint32_t segmentAddrBlocks = (segmentBlocks + ARGON2_QWORDS_IN_BLOCK - 1)
            / ARGON2_QWORDS_IN_BLOCK;
    uint32_t segments =
            type == ARGON2_ID
            ? lanes * (ARGON2_SYNC_POINTS / 2)
            : passes * lanes * ARGON2_SYNC_POINTS;

    dim3 blocks = dim3(1, segments * segmentAddrBlocks);
    dim3 threads = dim3(THREADS_PER_LANE);

    size_t shmemSize = sizeof(struct u64_shuffle_buf);
    if (type == ARGON2_I) {
        argon2_precompute_kernel<ARGON2_I>
            <<<blocks, threads, shmemSize, stream>>>(
                refs, passes, lanes, segmentBlocks);
    } else {
        argon2_precompute_kernel<ARGON2_ID>
            <<<blocks, threads, shmemSize, stream>>>(
                refs, passes, lanes, segmentBlocks);
    }
}

KernelRunner::~KernelRunner()
{
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (end != nullptr) {
        cudaEventDestroy(end);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (memory != nullptr) {
        cudaFree(memory);
    }
    if (refs != nullptr) {
        cudaFree(refs);
    }
    if (difficultyResultDevice != nullptr) {
        cudaFree(difficultyResultDevice);
    }
}

void *KernelRunner::getInputMemory(size_t jobId) const
{
    size_t copySize = lanes * 2 * ARGON2_BLOCK_SIZE;
    return blocksIn.get() + jobId * copySize;
}
const void *KernelRunner::getOutputMemory(size_t jobId) const
{
    size_t copySize = lanes * ARGON2_BLOCK_SIZE;
    return blocksOut.get() + jobId * copySize;
}

void KernelRunner::copyInputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = lanes * 2 * ARGON2_BLOCK_SIZE;

    CudaException::check(cudaMemcpy2DAsync(
                             memory, jobSize,
                             blocksIn.get(), copySize,
                             copySize, batchSize, cudaMemcpyHostToDevice,
                             stream));
}

void KernelRunner::copyOutputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = lanes * ARGON2_BLOCK_SIZE;
    uint8_t *mem = static_cast<uint8_t *>(memory);

    CudaException::check(cudaMemcpy2DAsync(
                             blocksOut.get(), copySize,
                             mem + (jobSize - copySize), jobSize,
                             copySize, batchSize, cudaMemcpyDeviceToHost,
                             stream));
}

void KernelRunner::runDifficultyCheckKernel(int difficultyBits)
{
    const size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    dim3 threads = dim3(128);
    dim3 blocks = dim3((batchSize + threads.x - 1) / threads.x);

    argon2_finalize_difficulty_kernel<<<blocks, threads, 0, stream>>>(
        memory, jobSize, lanes, batchSize, difficultyBits,
        difficultyResultDevice);
}

void KernelRunner::runKernelSegment(uint32_t lanesPerBlock,
                                    size_t jobsPerBlock,
                                    uint32_t pass, uint32_t slice)
{
    if (lanesPerBlock > lanes || lanes % lanesPerBlock != 0) {
        throw std::logic_error("Invalid lanesPerBlock!");
    }

    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
        throw std::logic_error("Invalid jobsPerBlock!");
    }

    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(1, lanes / lanesPerBlock, batchSize / jobsPerBlock);
    dim3 threads = dim3(THREADS_PER_LANE, lanesPerBlock, jobsPerBlock);
    uint32_t blockSize = lanesPerBlock * jobsPerBlock;
    uint32_t shared_size = blockSize * sizeof(struct u64_shuffle_buf);
    if (type == ARGON2_I) {
        if (precompute) {
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_segment_precompute<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2_kernel_segment_precompute<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        } else {
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        }
    } else if (type == ARGON2_ID) {
        if (precompute) {
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_segment_precompute<ARGON2_ID, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2_kernel_segment_precompute<ARGON2_ID, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        } else {
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_segment<ARGON2_ID, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2_kernel_segment<ARGON2_ID, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        }
    } else {
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_10>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks,
                        pass, slice);
        } else {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_13>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks,
                        pass, slice);
        }
    }
}

void KernelRunner::runKernelOneshot(uint32_t lanesPerBlock,
                                    size_t jobsPerBlock)
{
    if (lanesPerBlock != lanes) {
        throw std::logic_error("Invalid lanesPerBlock!");
    }

    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
        throw std::logic_error("Invalid jobsPerBlock!");
    }

    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(1, 1, batchSize / jobsPerBlock);
    dim3 threads = dim3(THREADS_PER_LANE, lanes, jobsPerBlock);
    uint32_t blockSize = lanesPerBlock * jobsPerBlock;
    uint32_t shared_size = blockSize * sizeof(struct u64_shuffle_buf);
    if (type == ARGON2_I) {
        if (precompute) {
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_oneshot_precompute<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            } else {
                argon2_kernel_oneshot_precompute<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            }
        } else {
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_oneshot<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            } else {
                argon2_kernel_oneshot<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            }
        }
    } else if (type == ARGON2_ID) {
        if (precompute) {
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_oneshot_precompute<ARGON2_ID, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            } else {
                argon2_kernel_oneshot_precompute<ARGON2_ID, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            }
        } else {
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_oneshot<ARGON2_ID, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            } else {
                argon2_kernel_oneshot<ARGON2_ID, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            }
        }
    } else {
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_oneshot<ARGON2_D, ARGON2_VERSION_10>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks);
        } else {
            argon2_kernel_oneshot<ARGON2_D, ARGON2_VERSION_13>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks);
        }
    }
}

void KernelRunner::run(uint32_t lanesPerBlock, size_t jobsPerBlock)
{
    CudaException::check(cudaEventRecord(start, stream));

    copyInputBlocks();

    CudaException::check(cudaEventRecord(kernelStart, stream));

    if (bySegment) {
        for (uint32_t pass = 0; pass < passes; pass++) {
            for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
                runKernelSegment(lanesPerBlock, jobsPerBlock, pass, slice);
            }
        }
    } else {
        runKernelOneshot(lanesPerBlock, jobsPerBlock);
    }

    CudaException::check(cudaGetLastError());

    CudaException::check(cudaEventRecord(kernelEnd, stream));

    copyOutputBlocks();

    CudaException::check(cudaEventRecord(end, stream));
}

void KernelRunner::runWithDifficultyCheck(
        uint32_t lanesPerBlock, size_t jobsPerBlock, int difficultyBits)
{
    CudaException::check(cudaEventRecord(start, stream));

    CudaException::check(cudaMemsetAsync(difficultyResultDevice, 0,
                                         sizeof(DifficultyCheckResult),
                                         stream));

    copyInputBlocks();

    CudaException::check(cudaEventRecord(kernelStart, stream));

    if (bySegment) {
        for (uint32_t pass = 0; pass < passes; pass++) {
            for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
                runKernelSegment(lanesPerBlock, jobsPerBlock, pass, slice);
            }
        }
    } else {
        runKernelOneshot(lanesPerBlock, jobsPerBlock);
    }

    runDifficultyCheckKernel(difficultyBits);

    CudaException::check(cudaGetLastError());

    CudaException::check(cudaEventRecord(kernelEnd, stream));

    CudaException::check(cudaMemcpyAsync(
                             difficultyResultHost.get(), difficultyResultDevice,
                             sizeof(DifficultyCheckResult),
                             cudaMemcpyDeviceToHost, stream));

    CudaException::check(cudaEventRecord(end, stream));
}

float KernelRunner::finish()
{
    float time = 0.0;
    CudaException::check(cudaStreamSynchronize(stream));

#ifndef NDEBUG
    CudaException::check(cudaEventElapsedTime(&time, start, kernelStart));
    std::cerr << "[INFO] Copy to device took " << time << " ms." << std::endl;

    CudaException::check(cudaEventElapsedTime(&time, kernelEnd, end));
    std::cerr << "[INFO] Copy from device took " << time << " ms." << std::endl;
#endif

    CudaException::check(cudaEventElapsedTime(&time, kernelStart, kernelEnd));
    return time;
}

bool KernelRunner::getDifficultyResult(size_t *jobId, void *hash) const
{
    if (difficultyResultHost->found == 0) {
        return false;
    }
    if (jobId != nullptr) {
        *jobId = difficultyResultHost->job_id;
    }
    if (hash != nullptr) {
        auto out = static_cast<uint8_t *>(hash);
        for (size_t i = 0; i < sizeof(difficultyResultHost->digest); ++i) {
            out[i] = difficultyResultHost->digest[i];
        }
    }
    return true;
}

} // cuda
} // argon2
