#include <stdio.h>

typedef long long ll;

struct Pair {
    int val;
    int idx;
};

ll maxScore(int* nums, int numsSize) {
    struct Pair stack[100005];
    int top = 0;
    stack[top++] = (struct Pair){ nums[numsSize - 1], numsSize - 1 };
    for (int i = numsSize - 2; i >= 0; --i) {
        if (nums[i] > stack[top - 1].val)
            stack[top++] = (struct Pair){ nums[i], i };
    }
    ll res = 0;
    int j = 0;
    for (int k = top - 1; k >= 0; --k) {
        res += (ll)(stack[k].idx - j) * stack[k].val;
        j = stack[k].idx;
    }
    return res;
}
