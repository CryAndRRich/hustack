#include <stdbool.h>
#include <math.h>
#include <stdio.h>

bool isValid(int* nums, int numsSize, int k, double mid) {
    double sum = 0;
    for (int i = 0; i < k; i++) {
        sum += nums[i] - mid;
    }
    if (sum >= 0) return true;

    double prevSum = 0, minPrevSum = 0;
    for (int i = k; i < numsSize; i++) {
        sum += nums[i] - mid;
        prevSum += nums[i - k] - mid;
        minPrevSum = fmin(minPrevSum, prevSum);
        if (sum - minPrevSum >= 0)
            return true;
    }
    return false;
}

double findMaxAverage(int* nums, int numsSize, int k) {
    double left = -10000, right = 10000, eps = 1e-5;
    while (right - left > eps) {
        double mid = (left + right) / 2;
        if (isValid(nums, numsSize, k, mid))
            left = mid;
        else
            right = mid;
    }
    return left;
}