#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int* onceTwice(int* nums, int numsSize, int* returnSize) {
    long long bits[32] = {0};
    for (int i = 0; i < numsSize; ++i) {
        uint32_t x = (uint32_t)nums[i];
        for (int b = 0; b < 32; ++b) bits[b] += (x >> b) & 1u;
    }
    uint32_t xor_u = 0;
    for (int b = 0; b < 32; ++b) if (bits[b] % 3 != 0) xor_u |= (1u << b);
    uint32_t mask = xor_u & (~xor_u + 1u);
    long long bitsg[32] = {0};
    for (int i = 0; i < numsSize; ++i) {
        uint32_t x = (uint32_t)nums[i];
        if (x & mask) for (int b = 0; b < 32; ++b) bitsg[b] += (x >> b) & 1u;
    }
    uint32_t cand_u = 0;
    for (int b = 0; b < 32; ++b) if (bitsg[b] % 3 != 0) cand_u |= (1u << b);
    uint32_t other_u = cand_u ^ xor_u;
    int32_t cand = (int32_t)cand_u;
    int32_t other = (int32_t)other_u;
    int cnt = 0;
    for (int i = 0; i < numsSize; ++i) if (nums[i] == cand) ++cnt;
    int32_t once, twice;
    if (cnt == 1) { once = cand; twice = other; }
    else { once = other; twice = cand; }
    int* res = (int*)malloc(2 * sizeof(int));
    res[0] = once;
    res[1] = twice;
    *returnSize = 2;
    return res;
}
