#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> onceTwice(vector<int>& nums) {
        long long bits[32] = {0};
        for (int num : nums) {
            uint32_t x = (uint32_t)num;
            for (int i = 0; i < 32; ++i) bits[i] += (x >> i) & 1u;
        }
        uint32_t xor_u = 0;
        for (int i = 0; i < 32; ++i) if (bits[i] % 3 != 0) xor_u |= (1u << i);
        uint32_t mask = xor_u & (~xor_u + 1u);
        long long bitsg[32] = {0};
        for (int num : nums) {
            uint32_t x = (uint32_t)num;
            if (x & mask) for (int i = 0; i < 32; ++i) bitsg[i] += (x >> i) & 1u;
        }
        uint32_t cand_u = 0;
        for (int i = 0; i < 32; ++i) if (bitsg[i] % 3 != 0) cand_u |= (1u << i);
        uint32_t other_u = cand_u ^ xor_u;
        int cand = (int)cand_u;
        int other = (int)other_u;
        int cnt = 0;
        for (int num : nums) if (num == cand) ++cnt;
        if (cnt == 1) return {cand, other};
        else return {other, cand};
    }
};
