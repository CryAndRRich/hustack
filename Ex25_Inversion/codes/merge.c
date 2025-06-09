#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007
#define MAXN 1000005

int temp[MAXN];

long long merge_and_count(int* arr, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;
    long long inv_count = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inv_count += (mid - i + 1);
            if (inv_count >= MOD) inv_count %= MOD;
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int l = left; l <= right; l++) {
        arr[l] = temp[l];
    }

    return inv_count % MOD;
}

long long merge_sort(int* arr, int left, int right) {
    if (left >= right) return 0;

    int mid = (left + right) / 2;
    long long inv_left = merge_sort(arr, left, mid);
    long long inv_right = merge_sort(arr, mid + 1, right);
    long long inv_split = merge_and_count(arr, left, mid, right);

    return (inv_left + inv_right + inv_split) % MOD;
}

int main() {
    int n;
    static int arr[MAXN];

    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    long long res = merge_sort(arr, 0, n - 1);
    printf("%lld", res);

    return 0;
}
