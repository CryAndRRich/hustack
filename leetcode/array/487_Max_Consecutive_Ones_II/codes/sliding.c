int findMaxConsecutiveOnes(int* nums, int numsSize) {
    int left = 0, right = 0, maxLen = 0, zeroCount = 0;

    for (right = 0; right < numsSize; right++) {
        if (nums[right] == 0) zeroCount++;

        while (zeroCount > 1) {
            if (nums[left++] == 0) zeroCount--;
        }

        if (right - left + 1 > maxLen)
            maxLen = right - left + 1;
    }

    return maxLen;
}
