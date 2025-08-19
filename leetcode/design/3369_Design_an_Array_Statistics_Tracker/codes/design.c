#include <stdlib.h>
#include <limits.h>
#include <string.h>

static int cmpInt(const void* a, const void* b) {
    return (*(int*)a) - (*(int*)b);
}

typedef struct {
    int num;
    int freq;
} FreqPair;

typedef struct {
    long long sum;
    int *nums;
    int cap, size;
    FreqPair *freqArr;
    int freqCap, freqSize;
} StatisticsTracker;

StatisticsTracker* statisticsTrackerCreate() {
    StatisticsTracker *st = malloc(sizeof(*st));
    st -> sum = 0;
    st -> cap = 4;
    st -> nums = malloc(st -> cap * sizeof(int));
    st -> size = 0;
    st -> freqCap = 4;
    st -> freqSize = 0;
    st -> freqArr = malloc(st -> freqCap * sizeof(FreqPair));
    return st;
}

static void ensureNums(StatisticsTracker *st) {
    if (st -> size == st -> cap) {
        st -> cap *= 2;
        st -> nums = realloc(st -> nums, st -> cap * sizeof(int));
    }
}

static int freqFind(StatisticsTracker *st, int num) {
    for (int i = 0; i < st -> freqSize; i++)
        if (st -> freqArr[i].num == num) return i;
    return -1;
}

void statisticsTrackerAddNumber(StatisticsTracker* st, int number) {
    st -> sum += number;
    ensureNums(st);
    st -> nums[st -> size++] = number;

    int idx = freqFind(st, number);
    if (idx >= 0) {
        st -> freqArr[idx].freq++;
    } else {
        if (st -> freqSize == st -> freqCap) {
            st -> freqCap *= 2;
            st -> freqArr = realloc(st -> freqArr, st -> freqCap * sizeof(FreqPair));
        }
        st -> freqArr[st -> freqSize].num = number;
        st -> freqArr[st -> freqSize].freq = 1;
        st -> freqSize++;
    }
}

void statisticsTrackerRemoveFirstAddedNumber(StatisticsTracker* st) {
    if (st -> size == 0) return;
    int number = st -> nums[0];
    st -> sum -= number;
    memmove(st -> nums, st -> nums + 1, (st -> size - 1) * sizeof(int));
    st -> size--;

    int idx = freqFind(st, number);
    if (idx >= 0) {
        if (--st -> freqArr[idx].freq == 0) {
            st -> freqArr[idx] = st -> freqArr[--st -> freqSize];
        }
    }
}

int statisticsTrackerGetMean(StatisticsTracker* st) {
    if (st -> size == 0) return 0;
    return (int)(st -> sum / st -> size);
}

int statisticsTrackerGetMedian(StatisticsTracker* st) {
    if (st -> size == 0) return 0;
    int n = st -> size;
    int *tmp = malloc(n * sizeof(int));
    memcpy(tmp, st -> nums, n * sizeof(int));
    qsort(tmp, n, sizeof(int), cmpInt);
    int mid = tmp[n/2];
    free(tmp);
    return mid;
}

int statisticsTrackerGetMode(StatisticsTracker* st) {
    if (st -> freqSize == 0) return 0;
    int bestNum = st -> freqArr[0].num;
    int bestFreq = st -> freqArr[0].freq;
    for (int i = 1; i < st -> freqSize; i++) {
        int f = st -> freqArr[i].freq;
        int num = st -> freqArr[i].num;
        if (f > bestFreq || (f == bestFreq && num < bestNum)) {
            bestFreq = f;
            bestNum = num;
        }
    }
    return bestNum;
}

void statisticsTrackerFree(StatisticsTracker* st) {
    free(st -> nums);
    free(st -> freqArr);
    free(st);
}

/**
 * Your StatisticsTracker struct will be instantiated and called as such:
 * StatisticsTracker* obj = statisticsTrackerCreate();
 * statisticsTrackerAddNumber(obj, number);
 
 * statisticsTrackerRemoveFirstAddedNumber(obj);
 
 * int param_3 = statisticsTrackerGetMean(obj);
 
 * int param_4 = statisticsTrackerGetMedian(obj);
 
 * int param_5 = statisticsTrackerGetMode(obj);
 
 * statisticsTrackerFree(obj);
*/