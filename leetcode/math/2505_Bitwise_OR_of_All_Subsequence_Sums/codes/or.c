long long subsequenceSumOr(int* nums, int numsSize) {
    long long ans = 0, prefix = 0;
    for (int i = 0; i < numsSize; i++) {
        prefix += (long long)nums[i];
        ans |= (long long)nums[i] | prefix;
    }
    return ans;
}
