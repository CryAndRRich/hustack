#include <stdio.h>

#define MOD 1000000000LL

long long modExp(long long base, int exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % MOD;
        base = (base * base) % MOD;
        exp /= 2;
    }
    return result;
}

long long lcm(const int *nums, int n) {
    int count[1001] = {0}; 
    for (int i = 0; i < n; i++) {
        int curr = nums[i];
        int t = 2; 
        int c = 0; 
        
        while (curr > 1) {
            if (curr % t == 0) {
                c++;
                curr /= t;
            } else {
                if (c > count[t])
                    count[t] = c;
                c = 0;
                t++;
            }
        }
        if (c > count[t])
            count[t] = c;
    }
    long long ans = 1;
    for (int i = 2; i < 1001; i++) {
        if (count[i] > 0) {
            ans = (ans * modExp(i, count[i])) % MOD;
        }
    }
    
    return ans;
}

int main() {
    int n;
    scanf("%d", &n);
    int arr[n];
    for(int i = 0; i < n; i += 1) {
        scanf("%d", &arr[i]);
    }
    
    printf("%d", lcm(arr, n));
    return 0;
}
