int missingElement(int* nums, int numsSize, int k) {
    for (int i = 1; i < numsSize; i++) {
        int gap = nums[i] - nums[i - 1] - 1;
        if (k <= gap) {
            return nums[i - 1] + k;
        }
        k -= gap;
    }
    return nums[numsSize - 1] + k;
}
