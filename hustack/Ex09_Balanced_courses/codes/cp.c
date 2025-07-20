#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int m, n;             
int *allowed;         
int k;                
int *conflict_mask;   
bool *conflicts;      

void init_conflict() {
    int total = 1 << n;
    conflicts = (bool*)malloc(total * sizeof(bool));
    for (int mask = 0; mask < total; mask++) {
        bool c = true;
        for (int t = 0; t < n; t++) {
            if (mask & (1 << t)) {
                if (mask & conflict_mask[t]) {
                    c = false;
                    break;
                }
            }
        }
        conflicts[mask] = c;
    }
}

bool check(int L) {
    int **valid = (int**)malloc((m + 1) * sizeof(int*));
    int *validCount = (int*)malloc((m + 1) * sizeof(int));
    int *validSize = (int*)malloc((m + 1) * sizeof(int));
    
    for (int i = 1; i <= m; i++) {
        validSize[i] = 16;
        validCount[i] = 0;
        valid[i] = (int*)malloc(validSize[i] * sizeof(int));
        int A = allowed[i];
        for (int sub = A; ; sub = (sub - 1) & A) {
            int cnt = __builtin_popcount(sub);
            if (cnt <= L && conflicts[sub]) {
                if (validCount[i] == validSize[i]) {
                    validSize[i] *= 2;
                    valid[i] = (int*)realloc(valid[i], validSize[i] * sizeof(int));
                }
                valid[i][validCount[i]++] = sub;
            }
            if (sub == 0)
                break;
        }
    }
    
    int total = 1 << n;
    bool *dp = (bool*)calloc(total, sizeof(bool));
    dp[0] = true;
    bool *newdp = (bool*)calloc(total, sizeof(bool));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 0; j < total; j++) {
            newdp[j] = false;
        }
        for (int mask = 0; mask < total; mask++) {
            if (!dp[mask])
                continue;
            for (int j = 0; j < validCount[i]; j++) {
                int subset = valid[i][j];
                int newmask = mask | subset;
                newdp[newmask] = true;
            }
        }
        for (int j = 0; j < total; j++) {
            dp[j] = newdp[j];
        }
    }
    
    bool c = dp[total - 1];  
    
    for (int i = 1; i <= m; i++) {
        free(valid[i]);
    }
    free(valid);
    free(validCount);
    free(validSize);
    free(dp);
    free(newdp);
    
    return c;
}

int main(){
    scanf("%d %d", &m, &n);
    
    allowed = (int*)malloc((m + 1) * sizeof(int));
    for (int i = 1; i <= m; i++) {
        allowed[i] = 0;
    }

    for (int i = 1; i <= m; i++) {
        int count;
        scanf("%d", &count);
        for (int j = 0; j < count; j++) {
            int task;
            scanf("%d", &task);
            if (task >= 1 && task <= n) {
                allowed[i] |= (1 << (task - 1));
            }
        }
    }
    
    scanf("%d", &k);
    conflict_mask = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < k; i++) {
        int a, b;
        scanf("%d %d", &a, &b);
        int t1 = a - 1;
        int t2 = b - 1;
        conflict_mask[t1] |= (1 << t2);
        conflict_mask[t2] |= (1 << t1);
    }
    
    init_conflict();
    
    int lo = 0, hi = n, ans = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (check(mid)) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    
    printf("%d", ans);
    
    free(allowed);
    free(conflict_mask);
    free(conflicts);
    
    return 0;
}
