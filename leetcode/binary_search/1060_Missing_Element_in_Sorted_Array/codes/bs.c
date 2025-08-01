int missingElement(int* nums, int numsSize, int k) {
    int missing;

    int getMissing(int i) {
        return nums[i] - nums[0] - i;
    }

    if (nums[numsSize - 1] - nums[0] - (numsSize - 1) < k) {
        return nums[numsSize - 1] + k - (nums[numsSize - 1] - nums[0] - (numsSize - 1));
    }

    int left = 0, right = numsSize - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        missing = nums[mid] - nums[0] - mid;
        if (missing < k) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    missing = nums[left - 1] - nums[0] - (left - 1);
    return nums[left - 1] + k - missing;
}
